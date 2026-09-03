"""Third-round study: infer hidden bath switching from qubit correlations.

Compares the bath-resolved 8-dimensional conditional Lindblad generator with
the coarse-grained 4-dimensional effective-bath generator.  The stationary
state and the sigma_z fluctuation spectrum are evaluated using the quantum
regression theorem and an eigenmode representation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from second_round_driven_qubit import (
    I2,
    SZ,
    ThermalBath,
    conditional_generator,
    liouvillian,
    mat,
    observables,
    stationary_state,
    vec,
)


def stationary_vector(generator: np.ndarray, blocks: int) -> np.ndarray:
    trace_row = np.concatenate([vec(I2).conj()] * blocks)
    matrix = generator.copy()
    rhs = np.zeros(4 * blocks, dtype=complex)
    matrix[-1, :] = trace_row
    rhs[-1] = 1.0
    return np.linalg.solve(matrix, rhs)


def correlation_modes(
    generator: np.ndarray, stationary: np.ndarray, observable: np.ndarray, blocks: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return decay eigenvalues and coefficients of connected <O(t)O(0)>."""
    left = np.concatenate([vec(observable.T)] * blocks)
    initial_parts = []
    for block in range(blocks):
        rho = mat(stationary[4 * block : 4 * (block + 1)])
        initial_parts.append(vec(observable @ rho))
    initial = np.concatenate(initial_parts)
    mean = float((left @ stationary).real)

    values, right = np.linalg.eig(generator)
    inverse = np.linalg.inv(right)
    coefficients = (left @ right) * (inverse @ initial)
    # Remove the stationary mode explicitly; this subtracts mean^2.
    zero = int(np.argmin(np.abs(values)))
    coefficients[zero] = 0.0
    mask = np.abs(coefficients) > 1e-12
    return values[mask], coefficients[mask], mean


def spectrum(
    values: np.ndarray, coefficients: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(frequencies, dtype=float)
    for index, frequency in enumerate(frequencies):
        integral = np.sum(coefficients / (-values - 1j * frequency))
        result[index] = 2.0 * integral.real
    return result


def effective_bath(hot: ThermalBath, cold: ThermalBath, p_hot: float) -> ThermalBath:
    up = p_hot * hot.up + (1.0 - p_hot) * cold.up
    down = p_hot * hot.down + (1.0 - p_hot) * cold.down
    beta = float(np.log(down / up))
    gamma = float(down - up)  # bosonic convention: down-up=gamma
    return ThermalBath(beta=beta, gamma=gamma)


def effective_stationary(generator: np.ndarray) -> np.ndarray:
    return stationary_vector(generator, blocks=1)


def fast_switch_closed_form(
    hot: ThermalBath, cold: ThermalBath, omega_rabi: float, p_hot: float = 0.5
) -> dict[str, float]:
    """Resonant optical-Bloch steady state and bath-resolved k->infinity fluxes."""
    p_cold = 1.0 - p_hot
    up = p_hot * hot.up + p_cold * cold.up
    down = p_hot * hot.down + p_cold * cold.down
    gamma1 = up + down
    gamma2 = gamma1 / 2.0
    z_eq = (up - down) / gamma1
    z = z_eq * gamma1 / (gamma1 + omega_rabi**2 / gamma2)
    y = -omega_rabi * z / gamma2
    excited = 0.5 * (1.0 + z)
    jh = 0.5 * p_hot * ((hot.up - hot.down) - (hot.up + hot.down) * z)
    jc = 0.5 * p_cold * ((cold.up - cold.down) - (cold.up + cold.down) * z)
    sigma_full = -hot.beta * jh - cold.beta * jc
    beta_eff = np.log(down / up)
    sigma_coarse = -beta_eff * (jh + jc)
    return {
        "z": float(z),
        "coherence_l1": float(abs(y)),
        "excited": float(excited),
        "J_hot": float(jh),
        "J_cold": float(jc),
        "sigma_full": float(sigma_full),
        "sigma_hidden": float(sigma_full - sigma_coarse),
    }


def spectral_distance(full: np.ndarray, coarse: np.ndarray, frequency: np.ndarray) -> float:
    numerator = np.trapezoid((full - coarse) ** 2, frequency)
    denominator = np.trapezoid(full**2, frequency)
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    hot = ThermalBath(beta=0.6, gamma=1.0)
    cold = ThermalBath(beta=2.0, gamma=0.8)
    frequencies = np.logspace(-4, 2, 500)
    output = Path(__file__).with_name("third_round_inference_scan.csv")
    rows: list[list[float]] = []

    for omega_rabi in (0.0, 0.15, 0.4, 0.8):
        coarse_bath = effective_bath(hot, cold, p_hot=0.5)
        coarse_generator = liouvillian(coarse_bath, omega_rabi)
        coarse_stationary = effective_stationary(coarse_generator)
        coarse_values, coarse_coefficients, coarse_mean = correlation_modes(
            coarse_generator, coarse_stationary, SZ, blocks=1
        )
        coarse_spectrum = spectrum(coarse_values, coarse_coefficients, frequencies)

        for k in np.logspace(-3, 3, 81):
            full_generator = conditional_generator(hot, cold, k, k, omega_rabi)
            full_stationary = stationary_vector(full_generator, blocks=2)
            full_values, full_coefficients, full_mean = correlation_modes(
                full_generator, full_stationary, SZ, blocks=2
            )
            full_spectrum = spectrum(full_values, full_coefficients, frequencies)

            rh, rc = stationary_state(full_generator)
            obs = observables(rh, rc, hot, cold, omega_rabi, k, k)
            rho_full = rh + rc
            rho_coarse = mat(coarse_stationary)
            state_distance = 0.5 * np.sum(
                np.abs(np.linalg.eigvalsh(rho_full - rho_coarse))
            )
            distance = spectral_distance(full_spectrum, coarse_spectrum, frequencies)
            low_ratio = float(full_spectrum[0] / coarse_spectrum[0])
            slow_decay = min(
                (-value.real for value in full_values if value.real < -1e-10),
                default=np.nan,
            )
            rows.append(
                [
                    omega_rabi,
                    k,
                    full_mean,
                    coarse_mean,
                    state_distance,
                    distance,
                    low_ratio,
                    slow_decay,
                    obs["sigma_full"],
                    obs["sigma_hidden"],
                ]
            )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Omega",
                "switch_rate",
                "mean_sigma_z_full",
                "mean_sigma_z_effective",
                "stationary_trace_distance",
                "relative_spectral_distance",
                "zero_frequency_spectrum_ratio",
                "slowest_visible_decay_rate",
                "sigma_full",
                "sigma_hidden",
            ]
        )
        writer.writerows(rows)

    data = np.asarray(rows)
    print(f"wrote {output}")
    for omega_rabi in sorted(set(data[:, 0])):
        subset = data[data[:, 0] == omega_rabi]
        print(
            f"Omega={omega_rabi:g}: state distance "
            f"[{subset[:,4].min():.3e}, {subset[:,4].max():.3e}], "
            f"spectral distance [{subset[:,5].min():.3e}, {subset[:,5].max():.3e}], "
            f"low-f ratio [{subset[:,6].min():.3g}, {subset[:,6].max():.3g}]"
        )
        limit = fast_switch_closed_form(hot, cold, omega_rabi)
        last = subset[-1]
        print(
            f"  closed k->inf: Jhot={limit['J_hot']:.6g}, "
            f"hidden sigma={limit['sigma_hidden']:.6g}; "
            f"k_max hidden error={abs(last[9]-limit['sigma_hidden']):.3e}"
        )
        fast = subset[subset[:, 1] >= 10.0]
        state_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 4]), 1)[0]
        spectrum_slope = np.polyfit(np.log(fast[:, 1]), np.log(fast[:, 5]), 1)[0]
        print(
            f"  fast-switch scaling: state distance ~ k^{state_slope:.3f}, "
            f"spectral distance ~ k^{spectrum_slope:.3f}; "
            f"hidden sigma(k_max)={fast[-1,9]:.5g}"
        )
        for target_k in (1e-3, 1e-1, 1e1, 1e3):
            row = subset[np.argmin(abs(subset[:, 1] - target_k))]
            print(
                f"  k~{row[1]:.3g}: state={row[4]:.3e}, "
                f"spectrum={row[5]:.3e}, low-ratio={row[6]:.3g}, "
                f"hidden sigma={row[9]:.3e}"
            )


if __name__ == "__main__":
    main()
