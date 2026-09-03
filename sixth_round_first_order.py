"""Direct first-order fast-switching coefficients for the Q2 theory upgrade.

This script evaluates the analytic correction

    L1 = -C B R A_R^{-1} R B E

and the corresponding finite-time channel derivative

    Psi1(t) = integral exp((t-s)L_eff) L1 exp(s L_eff) ds.

It then predicts the leading fixed-protocol and no-reset trajectory KL
coefficients without fitting large switching-rate data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fifth_round_trajectory_kl import PROJECTORS
from fourth_round_detectability import (
    matrix_exponential,
    undriven_ground_energy_coefficients,
)
from second_round_driven_qubit import ThermalBath, liouvillian, mat, vec
from third_round_inference import effective_bath


def fast_slow_matrices(
    hot: ThermalBath,
    cold: ThermalBath,
    omega_rabi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return L_eff and its first fast-switching correction L1."""
    identity4 = np.eye(4, dtype=complex)
    zero4 = np.zeros((4, 4), dtype=complex)
    q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    a = np.kron(q, identity4)
    b = np.block(
        [
            [liouvillian(hot, omega_rabi), zero4],
            [zero4, liouvillian(cold, omega_rabi)],
        ]
    )
    embedding = np.vstack((0.5 * identity4, 0.5 * identity4))
    marginal = np.hstack((identity4, identity4))
    slow = embedding @ marginal
    fast = np.eye(8, dtype=complex) - slow

    # For this symmetric two-state Q, the Moore-Penrose inverse equals the
    # group inverse: it vanishes on Ran(P) and inverts A on Ran(R).
    a_group_inverse = np.linalg.pinv(a)
    l_eff = marginal @ b @ embedding
    l1 = -marginal @ b @ fast @ a_group_inverse @ fast @ b @ embedding
    return l_eff, l1


def first_order_channel(
    l_eff: np.ndarray,
    l1: np.ndarray,
    time: float,
    quadrature_order: int = 96,
) -> np.ndarray:
    """Evaluate the Duhamel integral for Psi1 by Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    times = 0.5 * time * (nodes + 1.0)
    result = np.zeros_like(l_eff)
    for sample_time, weight in zip(times, weights):
        result += weight * (
            matrix_exponential(l_eff, time - sample_time)
            @ l1
            @ matrix_exponential(l_eff, sample_time)
        )
    return 0.5 * time * result


def measurement_coefficients(psi1: np.ndarray) -> np.ndarray:
    """Return a[y_next,y_now] for repeated rank-one energy measurements."""
    coefficients = np.zeros((2, 2), dtype=float)
    for y_now, state in enumerate(PROJECTORS):
        correction = mat(psi1 @ vec(state))
        for y_next, effect in enumerate(PROJECTORS):
            coefficients[y_next, y_now] = float(
                np.trace(effect @ correction).real
            )
    return coefficients


def effective_measurement_kernel(l_eff: np.ndarray, time: float) -> np.ndarray:
    propagator = matrix_exponential(l_eff, time)
    kernel = np.zeros((2, 2), dtype=float)
    for y_now, state in enumerate(PROJECTORS):
        final = mat(propagator @ vec(state))
        for y_next, effect in enumerate(PROJECTORS):
            kernel[y_next, y_now] = float(np.trace(effect @ final).real)
    return kernel


def stationary_distribution(column_stochastic: np.ndarray) -> np.ndarray:
    matrix = column_stochastic - np.eye(column_stochastic.shape[0])
    rhs = np.zeros(column_stochastic.shape[0])
    matrix[-1] = 1.0
    rhs[-1] = 1.0
    return np.linalg.solve(matrix, rhs)


def trajectory_kl_coefficient(
    effective_kernel: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    stationary = stationary_distribution(effective_kernel)
    fisher_rate = 0.0
    for y_now in range(2):
        fisher_rate += stationary[y_now] * np.sum(
            coefficients[:, y_now] ** 2 / effective_kernel[:, y_now]
        )
    return 0.5 * float(fisher_rate)


def empirical_fast_coefficients(path: Path) -> dict[float, float]:
    """Read k^2 times the trajectory KL rate at the largest stored k."""
    best: dict[float, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            omega = float(row["Omega"])
            k = float(row["switch_rate"])
            rate = float(row["trajectory_KL_nats_per_measurement"])
            if omega not in best or k > best[omega][0]:
                best[omega] = (k, k * k * rate)
    return {omega: value for omega, (_, value) in best.items()}


def main() -> None:
    hot = ThermalBath(beta=0.6, gamma=1.0)
    cold = ThermalBath(beta=2.0, gamma=0.8)
    delta_t = 1.0
    empirical = empirical_fast_coefficients(
        Path(__file__).with_name("fifth_round_trajectory_kl_scan.csv")
    )

    for omega_rabi in (0.0, 0.4, 0.8):
        l_eff, l1 = fast_slow_matrices(hot, cold, omega_rabi)
        direct_l_eff = liouvillian(
            effective_bath(hot, cold, p_hot=0.5), omega_rabi
        )
        psi1 = first_order_channel(l_eff, l1, delta_t)
        coefficients = measurement_coefficients(psi1)
        kernel = effective_measurement_kernel(l_eff, delta_t)
        predicted = trajectory_kl_coefficient(kernel, coefficients)

        print(f"Omega={omega_rabi:g}")
        print(f"  ||L_eff-direct||={np.linalg.norm(l_eff-direct_l_eff):.3e}")
        print(f"  max column-sum(a)={np.max(abs(coefficients.sum(axis=0))):.3e}")
        print(f"  a[y_next,y_now]=\n{coefficients}")
        print(f"  predicted lim k^2*d_k={predicted:.10g}")
        print(f"  empirical at largest k={empirical[omega_rabi]:.10g}")
        print(
            "  relative discrepancy="
            f"{abs(predicted-empirical[omega_rabi])/predicted:.3e}"
        )

        if omega_rabi == 0.0:
            analytic_amplitude, _ = undriven_ground_energy_coefficients(
                hot, cold, delta_t
            )
            print(
                "  undriven ground->excited a: "
                f"matrix={coefficients[0,1]:.10g}, "
                f"closed form={analytic_amplitude:.10g}"
            )


if __name__ == "__main__":
    main()
