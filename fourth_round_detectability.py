"""Fourth-round study: sample complexity of detecting fast bath switching.

The full model is a qubit whose hot/cold thermal Lindbladians are selected by
a symmetric two-state Markov environment with switching rate k.  The null
model is the single effective Lindbladian obtained by stationary averaging.

We use a system-only prepare-evolve-measure protocol.  Each shot prepares a
qubit state, lets it evolve for a fixed time tau, and performs a binary Pauli
measurement.  The environment label is initially stationary and unobserved.
For a finite menu of fixed protocols we compute

  * the largest probability difference |p_k-p_eff|;
  * the associated Bernoulli KL divergence;
  * the number log(20)/D of independent shots needed for log-likelihood
    evidence log(20).

Fast-switching perturbation theory predicts |p_k-p_eff|=O(k^-1), while the
normalization of probabilities cancels the linear term in relative entropy,
so D=O(k^-2).  In contrast, bath-resolved hidden entropy production tends to
a nonzero constant.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from second_round_driven_qubit import (
    I2,
    SX,
    SZ,
    ThermalBath,
    conditional_generator,
    liouvillian,
    observables,
    stationary_state,
    vec,
)
from third_round_inference import effective_bath, fast_switch_closed_form


SY = np.array([[0, -1j], [1j, 0]], dtype=complex)


def matrix_exponential(generator: np.ndarray, time: float) -> np.ndarray:
    """Small dense matrix exponential using scaling-and-squaring Padé."""
    return expm(generator * time)


def pure_state(axis: np.ndarray, sign: int) -> np.ndarray:
    return 0.5 * (I2 + sign * axis)


def probability(
    propagator: np.ndarray,
    rho0: np.ndarray,
    effect: np.ndarray,
    blocks: int,
) -> float:
    if blocks == 2:
        initial = np.concatenate((0.5 * vec(rho0), 0.5 * vec(rho0)))
        final = propagator @ initial
        rho = final[:4].reshape((2, 2), order="F")
        rho += final[4:].reshape((2, 2), order="F")
    else:
        rho = (propagator @ vec(rho0)).reshape((2, 2), order="F")
    return float(np.clip(np.trace(effect @ rho).real, 0.0, 1.0))


def bernoulli_kl(p: float, q: float) -> float:
    tiny = 1e-15
    p = float(np.clip(p, tiny, 1.0 - tiny))
    q = float(np.clip(q, tiny, 1.0 - tiny))
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def undriven_ground_energy_coefficients(
    hot: ThermalBath, cold: ThermalBath, time: float
) -> tuple[float, float]:
    """Analytic coefficients p_k-p_eff=r/k and KL=c/k^2 for Omega=0.

    The qubit starts in the ground state and is measured in the energy basis.
    Symmetric bath switching and a stationary initial bath label are assumed.
    """
    gamma_hot = hot.up + hot.down
    gamma_cold = cold.up + cold.down
    gamma_bar = 0.5 * (gamma_hot + gamma_cold)
    up_bar = 0.5 * (hot.up + cold.up)
    delta_gamma = gamma_hot - gamma_cold
    delta_up = hot.up - cold.up
    f_eff = up_bar / gamma_bar
    q = f_eff * (1.0 - np.exp(-gamma_bar * time))
    amplitude = -(delta_gamma / 8.0) * (
        (delta_up - delta_gamma * f_eff)
        * (1.0 - np.exp(-gamma_bar * time))
        / gamma_bar
        + delta_gamma * f_eff * time * np.exp(-gamma_bar * time)
    )
    kl_coefficient = amplitude**2 / (2.0 * q * (1.0 - q))
    return float(amplitude), float(kl_coefficient)


def protocol_menu() -> list[tuple[str, np.ndarray, np.ndarray]]:
    axes = {"x": SX, "y": SY, "z": SZ}
    preparations = {
        f"{axis}{'+' if sign > 0 else '-'}": pure_state(matrix, sign)
        for axis, matrix in axes.items()
        for sign in (-1, 1)
    }
    effects = {f"{axis}+": pure_state(matrix, 1) for axis, matrix in axes.items()}
    return [
        (f"prep_{prep}_measure_{measurement}", rho0, effect)
        for prep, rho0 in preparations.items()
        for measurement, effect in effects.items()
    ]


def best_protocol(
    full: np.ndarray, effective: np.ndarray
) -> tuple[str, float, float, float, float]:
    candidates = []
    for name, rho0, effect in protocol_menu():
        p = probability(full, rho0, effect, blocks=2)
        q = probability(effective, rho0, effect, blocks=1)
        candidates.append((bernoulli_kl(p, q), name, p, q, abs(p - q)))
    divergence, name, p, q, difference = max(candidates)
    return name, p, q, difference, divergence


def main() -> None:
    hot = ThermalBath(beta=0.6, gamma=1.0)
    cold = ThermalBath(beta=2.0, gamma=0.8)
    tau = 1.0
    evidence = np.log(20.0)
    output = Path(__file__).with_name("fourth_round_detectability_scan.csv")
    rows: list[list[object]] = []
    trace_error = 0.0

    for omega_rabi in (0.0, 0.4, 0.8):
        coarse_bath = effective_bath(hot, cold, p_hot=0.5)
        effective_generator = liouvillian(coarse_bath, omega_rabi)
        effective_propagator = matrix_exponential(effective_generator, tau)

        for k in np.logspace(0, 4, 81):
            full_generator = conditional_generator(
                hot, cold, k, k, omega_rabi
            )
            full_propagator = matrix_exponential(full_generator, tau)
            name, p, q, difference, divergence = best_protocol(
                full_propagator, effective_propagator
            )
            shots = evidence / divergence if divergence > 0.0 else np.inf

            rh, rc = stationary_state(full_generator)
            hidden = observables(
                rh, rc, hot, cold, omega_rabi, k, k
            )["sigma_hidden"]

            # Check trace preservation on the maximally mixed input.
            initial = np.concatenate((0.25 * vec(I2), 0.25 * vec(I2)))
            final = full_propagator @ initial
            trace = np.vdot(
                np.concatenate((vec(I2), vec(I2))), final
            ).real
            trace_error = max(trace_error, abs(trace - 1.0))
            rows.append(
                [
                    omega_rabi,
                    k,
                    tau,
                    name,
                    p,
                    q,
                    difference,
                    divergence,
                    shots,
                    hidden,
                ]
            )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Omega",
                "switch_rate",
                "interrogation_time",
                "best_fixed_menu_protocol",
                "p_full",
                "p_effective",
                "absolute_probability_difference",
                "bernoulli_KL_nats_per_shot",
                "shots_for_log_likelihood_log20",
                "stationary_hidden_entropy_rate",
            ]
        )
        writer.writerows(rows)

    print(f"wrote {output}")
    print(f"max trace-preservation error: {trace_error:.3e}")
    analytic_dp, analytic_kl = undriven_ground_energy_coefficients(hot, cold, tau)
    omega_zero_last = next(
        row for row in reversed(rows) if row[0] == 0.0
    )
    print(
        "Omega=0 analytic fast-limit coefficients: "
        f"k*(p_full-p_eff)={analytic_dp:.9g}, k^2*KL={analytic_kl:.9g}; "
        f"numeric at kmax=({omega_zero_last[1] * (omega_zero_last[4] - omega_zero_last[5]):.9g}, "
        f"{omega_zero_last[1] ** 2 * omega_zero_last[7]:.9g})"
    )
    data = np.asarray(
        [[row[0], row[1], row[6], row[7], row[8], row[9]] for row in rows],
        dtype=float,
    )
    for omega_rabi in (0.0, 0.4, 0.8):
        subset = data[data[:, 0] == omega_rabi]
        fast = subset[subset[:, 1] >= 100.0]
        difference_slope = np.polyfit(
            np.log(fast[:, 1]), np.log(fast[:, 2]), 1
        )[0]
        kl_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 3]), 1)[0]
        shot_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 4]), 1)[0]
        hidden_limit = fast_switch_closed_form(hot, cold, omega_rabi)[
            "sigma_hidden"
        ]
        print(
            f"Omega={omega_rabi:g}: |dp|~k^{difference_slope:.4f}, "
            f"KL~k^{kl_slope:.4f}, shots~k^{shot_slope:.4f}; "
            f"hidden limit={hidden_limit:.6g}, "
            f"hidden(kmax)={subset[-1,5]:.6g}"
        )
        for target in (10.0, 100.0, 1000.0, 10000.0):
            index = int(np.argmin(abs(subset[:, 1] - target)))
            numeric = subset[index]
            row = next(
                item
                for item in rows
                if item[0] == omega_rabi and item[1] == numeric[1]
            )
            print(
                f"  k={numeric[1]:g}: protocol={row[3]}, "
                f"|dp|={numeric[2]:.3e}, KL={numeric[3]:.3e}, "
                f"N_log20={numeric[4]:.3e}, hidden={numeric[5]:.6g}"
            )


if __name__ == "__main__":
    main()
