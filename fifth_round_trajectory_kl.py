"""Trajectory-level KL rate for repeated system-only qubit measurements.

Unlike fourth_round_detectability.py, the environment is not reset between
observations.  A rank-one energy measurement is performed every delta_t.  The
qubit collapses, but the posterior distribution of the unobserved hot/cold
environment is retained and propagated.  This is an exact two-state hidden
Markov filter, even when the qubit is coherently driven between measurements.

The chain rule for relative entropy gives the output-record KL rate as the
stationary average of the conditional Bernoulli divergence between the full
filter prediction and the effective-bath prediction.  Conditional averaging
is used instead of a noisy log-likelihood ratio sample.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fourth_round_detectability import bernoulli_kl, matrix_exponential
from second_round_driven_qubit import (
    I2,
    PROJECT_E,
    ThermalBath,
    conditional_generator,
    liouvillian,
    mat,
    observables,
    stationary_state,
    vec,
)
from third_round_inference import effective_bath, fast_switch_closed_form


PROJECT_G = I2 - PROJECT_E
PROJECTORS = (PROJECT_E, PROJECT_G)


def measured_hidden_kernel(propagator: np.ndarray) -> np.ndarray:
    """K[y_next, env_next, env_now, y_now] for the full model."""
    kernel = np.zeros((2, 2, 2, 2), dtype=float)
    for env_now in range(2):
        for y_now, state in enumerate(PROJECTORS):
            initial = np.zeros(8, dtype=complex)
            initial[4 * env_now : 4 * (env_now + 1)] = vec(state)
            final = propagator @ initial
            for env_next in range(2):
                block = mat(final[4 * env_next : 4 * (env_next + 1)])
                for y_next, effect in enumerate(PROJECTORS):
                    kernel[y_next, env_next, env_now, y_now] = max(
                        0.0, float(np.trace(effect @ block).real)
                    )
            total = kernel[:, :, env_now, y_now].sum()
            kernel[:, :, env_now, y_now] /= total
    return kernel


def measured_effective_kernel(propagator: np.ndarray) -> np.ndarray:
    """T[y_next, y_now] for the effective single-bath model."""
    kernel = np.zeros((2, 2), dtype=float)
    for y_now, state in enumerate(PROJECTORS):
        final = mat(propagator @ vec(state))
        for y_next, effect in enumerate(PROJECTORS):
            kernel[y_next, y_now] = max(
                0.0, float(np.trace(effect @ final).real)
            )
        kernel[:, y_now] /= kernel[:, y_now].sum()
    return kernel


def trajectory_kl_rate(
    full_kernel: np.ndarray,
    effective_kernel: np.ndarray,
    rng: np.random.Generator,
    samples: int = 250_000,
    burn_in: int = 5_000,
    batches: int = 25,
) -> tuple[float, float]:
    """Return KL nats/measurement and a batch standard error."""
    current_y = 1  # start after a ground-state outcome; burn-in removes bias
    environment_posterior = np.array([0.5, 0.5], dtype=float)
    batch_size = samples // batches
    batch_means: list[float] = []
    batch_total = 0.0
    kept_in_batch = 0

    for step in range(burn_in + samples):
        joint = np.empty((2, 2), dtype=float)
        for y_next in range(2):
            joint[y_next] = (
                full_kernel[y_next, :, :, current_y] @ environment_posterior
            )
        prediction = joint.sum(axis=1)
        prediction /= prediction.sum()
        reference = effective_kernel[:, current_y]
        increment = bernoulli_kl(prediction[0], reference[0])

        next_y = 0 if rng.random() < prediction[0] else 1
        environment_posterior = joint[next_y] / prediction[next_y]
        current_y = next_y

        if step < burn_in:
            continue
        batch_total += increment
        kept_in_batch += 1
        if kept_in_batch == batch_size:
            batch_means.append(batch_total / batch_size)
            batch_total = 0.0
            kept_in_batch = 0

    values = np.asarray(batch_means)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, standard_error


def main() -> None:
    hot = ThermalBath(beta=0.6, gamma=1.0)
    cold = ThermalBath(beta=2.0, gamma=0.8)
    delta_t = 1.0
    evidence = np.log(20.0)
    switch_rates = np.logspace(0, 4, 21)
    rng = np.random.default_rng(20260821)
    output = Path(__file__).with_name("fifth_round_trajectory_kl_scan.csv")
    rows: list[list[float]] = []

    for omega_rabi in (0.0, 0.4, 0.8):
        coarse = effective_bath(hot, cold, p_hot=0.5)
        effective_propagator = matrix_exponential(
            liouvillian(coarse, omega_rabi), delta_t
        )
        effective_kernel = measured_effective_kernel(effective_propagator)

        for k in switch_rates:
            full_propagator = matrix_exponential(
                conditional_generator(hot, cold, k, k, omega_rabi), delta_t
            )
            full_kernel = measured_hidden_kernel(full_propagator)
            rate, error = trajectory_kl_rate(
                full_kernel, effective_kernel, rng
            )
            record_length = evidence / rate
            rh, rc = stationary_state(
                conditional_generator(hot, cold, k, k, omega_rabi)
            )
            hidden = observables(
                rh, rc, hot, cold, omega_rabi, k, k
            )["sigma_hidden"]
            rows.append(
                [
                    omega_rabi,
                    k,
                    delta_t,
                    rate,
                    error,
                    record_length,
                    hidden,
                    hidden * delta_t / rate,
                ]
            )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Omega",
                "switch_rate",
                "measurement_interval",
                "trajectory_KL_nats_per_measurement",
                "KL_batch_standard_error",
                "measurements_for_log_likelihood_log20",
                "stationary_hidden_entropy_rate",
                "hidden_entropy_per_KL_nat",
            ]
        )
        writer.writerows(rows)

    print(f"wrote {output}")
    data = np.asarray(rows)
    for omega_rabi in (0.0, 0.4, 0.8):
        subset = data[data[:, 0] == omega_rabi]
        fast = subset[subset[:, 1] >= 100.0]
        kl_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 3]), 1)[0]
        length_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 5]), 1)[0]
        cost_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 7]), 1)[0]
        hidden_limit = fast_switch_closed_form(hot, cold, omega_rabi)[
            "sigma_hidden"
        ]
        print(
            f"Omega={omega_rabi:g}: trajectory KL~k^{kl_slope:.4f}, "
            f"record length~k^{length_slope:.4f}, "
            f"hidden/KL~k^{cost_slope:.4f}, hidden limit={hidden_limit:.6g}"
        )
        for target in (10.0, 100.0, 1000.0, 10000.0):
            row = subset[np.argmin(abs(subset[:, 1] - target))]
            print(
                f"  k={row[1]:g}: KLrate={row[3]:.3e}+/-{row[4]:.1e}, "
                f"N_log20={row[5]:.3e}, hidden={row[6]:.6g}, "
                f"hidden/KL={row[7]:.3e}"
            )


if __name__ == "__main__":
    main()
