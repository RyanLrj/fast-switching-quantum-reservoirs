"""Feasibility calculations for a qubit coupled to randomly switched baths.

The undriven qubit reduces to a four-state continuous-time Markov chain:
environment h/c times qubit excited/ground.  This script validates the closed
form stationary heat current and writes a compact parameter scan to CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Bath:
    gamma: float
    excited_fraction: float

    @property
    def up(self) -> float:
        return self.gamma * self.excited_fraction

    @property
    def down(self) -> float:
        return self.gamma * (1.0 - self.excited_fraction)


def generator(hot: Bath, cold: Bath, a: float, b: float) -> np.ndarray:
    """Column-stochastic generator for [e,h; g,h; e,c; g,c]."""
    q = np.zeros((4, 4), dtype=float)

    def add(src: int, dst: int, rate: float) -> None:
        q[dst, src] += rate
        q[src, src] -= rate

    add(0, 1, hot.down)
    add(1, 0, hot.up)
    add(2, 3, cold.down)
    add(3, 2, cold.up)
    add(0, 2, a)
    add(1, 3, a)
    add(2, 0, b)
    add(3, 1, b)
    return q


def stationary_state(q: np.ndarray) -> np.ndarray:
    a = q.copy()
    rhs = np.zeros(4)
    a[-1, :] = 1.0
    rhs[-1] = 1.0
    p = np.linalg.solve(a, rhs)
    return p


def analytic_current(
    hot: Bath, cold: Bath, a: float, b: float, gap: float = 1.0
) -> float:
    gh, gc = hot.gamma, cold.gamma
    delta = gh * gc + gh * b + gc * a
    return (
        gap
        * a
        * b
        * gh
        * gc
        * (hot.excited_fraction - cold.excited_fraction)
        / ((a + b) * delta)
    )


def numerical_currents(
    p: np.ndarray, hot: Bath, cold: Bath, gap: float = 1.0
) -> tuple[float, float]:
    jh = gap * (hot.up * p[1] - hot.down * p[0])
    jc = gap * (cold.up * p[3] - cold.down * p[2])
    return jh, jc


def tilted_generator(
    hot: Bath, cold: Bath, a: float, b: float, counting_field: float
) -> np.ndarray:
    """Tilted generator for net energy quanta absorbed from the hot bath.

    A hot-bath excitation carries +1 and a hot-bath relaxation carries -1.
    Escape rates stay untilted, as required for a moment-generating operator.
    """
    q = generator(hot, cold, a, b)
    q[0, 1] = hot.up * np.exp(counting_field)
    q[1, 0] = hot.down * np.exp(-counting_field)
    return q


def scgf(hot: Bath, cold: Bath, a: float, b: float, field: float) -> float:
    vals = np.linalg.eigvals(tilted_generator(hot, cold, a, b, field))
    return float(vals[np.argmax(vals.real)].real)


def current_cumulants(
    hot: Bath, cold: Bath, a: float, b: float, step: float = 1e-4
) -> tuple[float, float]:
    lm = scgf(hot, cold, a, b, -step)
    l0 = scgf(hot, cold, a, b, 0.0)
    lp = scgf(hot, cold, a, b, step)
    mean = (lp - lm) / (2.0 * step)
    variance_rate = (lp - 2.0 * l0 + lm) / (step * step)
    return mean, variance_rate


def main() -> None:
    hot = Bath(gamma=1.3, excited_fraction=0.38)
    cold = Bath(gamma=0.7, excited_fraction=0.12)
    out = Path(__file__).with_name("switched_bath_scan.csv")
    rows = []
    max_error = 0.0
    max_balance_error = 0.0
    for k in np.logspace(-4, 4, 161):
        q = generator(hot, cold, a=k, b=k)
        p = stationary_state(q)
        jh, jc = numerical_currents(p, hot, cold)
        exact = analytic_current(hot, cold, a=k, b=k)
        fcs_mean, fcs_variance = current_cumulants(hot, cold, a=k, b=k)
        fano = fcs_variance / abs(fcs_mean) if abs(fcs_mean) > 1e-14 else np.nan
        sigma = jh * (1.0 / 0.5 - 1.0 / 2.0)  # illustrative Tc=0.5, Th=2
        max_error = max(max_error, abs(jh - exact))
        max_balance_error = max(max_balance_error, abs(jh + jc))
        rows.append((k, *p, jh, jc, exact, fcs_mean, fcs_variance, fano, sigma))

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "switch_rate",
                "p_excited_hot",
                "p_ground_hot",
                "p_excited_cold",
                "p_ground_cold",
                "J_hot_numeric",
                "J_cold_numeric",
                "J_hot_exact",
                "FCS_mean_hot_quanta",
                "FCS_variance_rate",
                "Fano_factor",
                "entropy_production_example",
            ]
        )
        writer.writerows(rows)

    print(f"wrote {out}")
    print(f"max analytic error: {max_error:.3e}")
    print(f"max energy-balance error: {max_balance_error:.3e}")
    print(f"minimum probability: {min(min(row[1:5]) for row in rows):.6g}")
    print(f"Fano range: {min(row[-2] for row in rows):.6g} to {max(row[-2] for row in rows):.6g}")


if __name__ == "__main__":
    main()
