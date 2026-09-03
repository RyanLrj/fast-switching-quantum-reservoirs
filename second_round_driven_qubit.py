"""Second-round feasibility study: driven qubit with Markov-switched baths.

Units: hbar = k_B = omega_0 = 1.  The coherent drive is represented in the
rotating frame by H = detuning*sigma_z/2 + Omega*sigma_x/2.  Bath jumps act
on the bare transition and obey local detailed balance at omega_0.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SP = np.array([[0, 1], [0, 0]], dtype=complex)
SM = SP.conj().T
PROJECT_E = SP @ SM


@dataclass(frozen=True)
class ThermalBath:
    beta: float
    gamma: float

    @property
    def nbar(self) -> float:
        return 1.0 / np.expm1(self.beta)

    @property
    def up(self) -> float:
        return self.gamma * self.nbar

    @property
    def down(self) -> float:
        return self.gamma * (self.nbar + 1.0)


def vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(4, order="F")


def mat(vector: np.ndarray) -> np.ndarray:
    return vector.reshape((2, 2), order="F")


def dissipator(jump: np.ndarray, rho: np.ndarray) -> np.ndarray:
    jj = jump.conj().T @ jump
    return jump @ rho @ jump.conj().T - 0.5 * (jj @ rho + rho @ jj)


def liouvillian(
    bath: ThermalBath, omega_rabi: float, detuning: float = 0.0
) -> np.ndarray:
    hamiltonian = 0.5 * detuning * SZ + 0.5 * omega_rabi * SX

    def action(rho: np.ndarray) -> np.ndarray:
        coherent = -1j * (hamiltonian @ rho - rho @ hamiltonian)
        return (
            coherent
            + bath.down * dissipator(SM, rho)
            + bath.up * dissipator(SP, rho)
        )

    result = np.empty((4, 4), dtype=complex)
    for column in range(4):
        basis = np.zeros(4, dtype=complex)
        basis[column] = 1.0
        result[:, column] = vec(action(mat(basis)))
    return result


def conditional_generator(
    hot: ThermalBath,
    cold: ThermalBath,
    switch_hc: float,
    switch_ch: float,
    omega_rabi: float,
    detuning: float = 0.0,
) -> np.ndarray:
    lh = liouvillian(hot, omega_rabi, detuning)
    lc = liouvillian(cold, omega_rabi, detuning)
    return np.block(
        [
            [lh - switch_hc * np.eye(4), switch_ch * np.eye(4)],
            [switch_hc * np.eye(4), lc - switch_ch * np.eye(4)],
        ]
    )


def stationary_state(generator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trace_row = np.concatenate((vec(I2), vec(I2))).conj()
    matrix = generator.copy()
    rhs = np.zeros(8, dtype=complex)
    matrix[-1, :] = trace_row
    rhs[-1] = 1.0
    solution = np.linalg.solve(matrix, rhs)
    rh, rc = mat(solution[:4]), mat(solution[4:])
    return 0.5 * (rh + rh.conj().T), 0.5 * (rc + rc.conj().T)


def bath_action(bath: ThermalBath, rho: np.ndarray) -> np.ndarray:
    return bath.down * dissipator(SM, rho) + bath.up * dissipator(SP, rho)


def observables(
    rh: np.ndarray,
    rc: np.ndarray,
    hot: ThermalBath,
    cold: ThermalBath,
    omega_rabi: float,
    switch_hc: float,
    switch_ch: float,
) -> dict[str, float]:
    rho = rh + rc
    # Heat into the qubit: each excitation adds +1, each relaxation removes 1.
    jh = float(np.trace(PROJECT_E @ bath_action(hot, rh)).real)
    jc = float(np.trace(PROJECT_E @ bath_action(cold, rc)).real)
    # Bare-energy power supplied by the coherent drive.
    hdrive = 0.5 * omega_rabi * SX
    power = float((-1j * np.trace(PROJECT_E @ (hdrive @ rho - rho @ hdrive))).real)
    coherence = float(2.0 * abs(rho[0, 1]))
    sigma_full = float(-hot.beta * jh - cold.beta * jc)
    p_hot = switch_ch / (switch_hc + switch_ch)
    p_cold = switch_hc / (switch_hc + switch_ch)
    up_eff = p_hot * hot.up + p_cold * cold.up
    down_eff = p_hot * hot.down + p_cold * cold.down
    beta_eff = float(np.log(down_eff / up_eff))
    sigma_coarse = float(-beta_eff * (jh + jc))
    sigma_hidden = sigma_full - sigma_coarse
    return {
        "excited_population": float(rho[0, 0].real),
        "coherence_l1": coherence,
        "J_hot": jh,
        "J_cold": jc,
        "drive_power": power,
        "first_law_residual": jh + jc + power,
        "sigma_full": sigma_full,
        "beta_effective": beta_eff,
        "sigma_coarse": sigma_coarse,
        "sigma_hidden": sigma_hidden,
        "min_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
    }


def tilted_generator(
    hot: ThermalBath,
    cold: ThermalBath,
    switch_hc: float,
    switch_ch: float,
    omega_rabi: float,
    field: float,
) -> np.ndarray:
    """Count net bare-energy quanta absorbed from the hot bath."""
    base = conditional_generator(hot, cold, switch_hc, switch_ch, omega_rabi)
    # Hot block only. With column-major vectorization, jump superoperator is
    # kron(L.conj(), L). Escape/anticommutator terms remain untilted.
    hot_excitation = hot.up * np.kron(SP.conj(), SP)
    hot_relaxation = hot.down * np.kron(SM.conj(), SM)
    base[:4, :4] += (np.exp(field) - 1.0) * hot_excitation
    base[:4, :4] += (np.exp(-field) - 1.0) * hot_relaxation
    return base


def entropy_tilted_generator(
    hot: ThermalBath,
    cold: ThermalBath,
    switch_hc: float,
    switch_ch: float,
    omega_rabi: float,
    field: float,
) -> np.ndarray:
    """Tilt by entropy delivered to the two thermal reservoirs."""
    base = conditional_generator(hot, cold, switch_hc, switch_ch, omega_rabi)
    for offset, bath in ((0, hot), (4, cold)):
        excitation = bath.up * np.kron(SP.conj(), SP)
        relaxation = bath.down * np.kron(SM.conj(), SM)
        # Excitation removes one energy quantum from the bath; relaxation adds one.
        base[offset : offset + 4, offset : offset + 4] += (
            np.exp(-field * bath.beta) - 1.0
        ) * excitation
        base[offset : offset + 4, offset : offset + 4] += (
            np.exp(field * bath.beta) - 1.0
        ) * relaxation
    return base


def entropy_scgf(
    hot: ThermalBath,
    cold: ThermalBath,
    switch_hc: float,
    switch_ch: float,
    omega_rabi: float,
    field: float,
) -> float:
    values = np.linalg.eigvals(
        entropy_tilted_generator(
            hot, cold, switch_hc, switch_ch, omega_rabi, field
        )
    )
    return float(values[np.argmax(values.real)].real)


def scgf(*args: object, field: float) -> float:
    values = np.linalg.eigvals(tilted_generator(*args, field=field))
    return float(values[np.argmax(values.real)].real)


def cumulants(
    hot: ThermalBath,
    cold: ThermalBath,
    switch_hc: float,
    switch_ch: float,
    omega_rabi: float,
    step: float = 2e-4,
) -> tuple[float, float]:
    args = (hot, cold, switch_hc, switch_ch, omega_rabi)
    lm = scgf(*args, field=-step)
    l0 = scgf(*args, field=0.0)
    lp = scgf(*args, field=step)
    return (lp - lm) / (2 * step), (lp - 2 * l0 + lm) / step**2


def main() -> None:
    hot = ThermalBath(beta=0.6, gamma=1.0)
    cold = ThermalBath(beta=2.0, gamma=0.8)
    output = Path(__file__).with_name("second_round_scan.csv")
    rows: list[list[float]] = []
    max_first_law_error = 0.0
    min_sigma = np.inf
    min_eigenvalue = np.inf
    max_fcs_error = 0.0

    for omega_rabi in (0.0, 0.15, 0.4, 0.8, 1.5):
        for k in np.logspace(-3, 3, 121):
            gen = conditional_generator(hot, cold, k, k, omega_rabi)
            rh, rc = stationary_state(gen)
            obs = observables(rh, rc, hot, cold, omega_rabi, k, k)
            fcs_mean, fcs_variance = cumulants(hot, cold, k, k, omega_rabi)
            fano = fcs_variance / abs(fcs_mean) if abs(fcs_mean) > 1e-12 else np.nan
            max_first_law_error = max(max_first_law_error, abs(obs["first_law_residual"]))
            min_sigma = min(min_sigma, obs["sigma_full"])
            min_eigenvalue = min(min_eigenvalue, obs["min_eigenvalue"])
            max_fcs_error = max(max_fcs_error, abs(fcs_mean - obs["J_hot"]))
            rows.append(
                [
                    omega_rabi,
                    k,
                    obs["excited_population"],
                    obs["coherence_l1"],
                    obs["J_hot"],
                    obs["J_cold"],
                    obs["drive_power"],
                    obs["sigma_full"],
                    obs["sigma_coarse"],
                    obs["sigma_hidden"],
                    fcs_mean,
                    fcs_variance,
                    fano,
                ]
            )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Omega",
                "switch_rate",
                "excited_population",
                "coherence_l1",
                "J_hot",
                "J_cold",
                "drive_power",
                "sigma_full",
                "sigma_coarse",
                "sigma_hidden",
                "FCS_mean_hot",
                "FCS_variance_hot",
                "Fano_hot",
            ]
        )
        writer.writerows(rows)

    data = np.asarray(rows)
    print(f"wrote {output}")
    print(f"max first-law residual: {max_first_law_error:.3e}")
    print(f"minimum full entropy production: {min_sigma:.6g}")
    print(f"minimum state eigenvalue: {min_eigenvalue:.6g}")
    print(f"max FCS/current mismatch: {max_fcs_error:.3e}")
    for omega_rabi in sorted(set(data[:, 0])):
        subset = data[data[:, 0] == omega_rabi]
        sigma = subset[:, 7]
        sigma_hidden = subset[:, 9]
        coherence = subset[:, 3]
        jhot = subset[:, 4]
        variance = subset[:, 11]
        fano = subset[:, 12]
        crossings = []
        for left, right in zip(subset[:-1], subset[1:]):
            if left[4] == 0 or left[4] * right[4] < 0:
                crossings.append(float(np.sqrt(left[1] * right[1])))
        print(
            f"Omega={omega_rabi:>4g}: sigma [{sigma.min():.5g}, {sigma.max():.5g}], "
            f"hidden [{sigma_hidden.min():.5g}, {sigma_hidden.max():.5g}], "
            f"coherence [{coherence.min():.5g}, {coherence.max():.5g}], "
            f"Jhot [{jhot.min():.5g}, {jhot.max():.5g}], "
            f"variance [{variance.min():.5g}, {variance.max():.5g}], "
            f"Fano [{np.nanmin(fano):.5g}, {np.nanmax(fano):.5g}], "
            f"zero-crossings~{crossings}"
        )

    # Test the Gallavotti-Cohen-type symmetry for bath entropy. A coherent
    # work source changes the appropriate reverse process, so failure of the
    # same-protocol symmetry in the driven case is diagnostic, not a bug.
    for omega_rabi in (0.0, 0.4):
        symmetry_error = 0.0
        for field in np.linspace(-1.0, 0.0, 21):
            lhs = entropy_scgf(hot, cold, 0.7, 0.7, omega_rabi, field)
            rhs = entropy_scgf(hot, cold, 0.7, 0.7, omega_rabi, -1.0 - field)
            symmetry_error = max(symmetry_error, abs(lhs - rhs))
        print(
            f"Omega={omega_rabi:g}: same-protocol entropy-SCGF symmetry error "
            f"{symmetry_error:.3e}"
        )


if __name__ == "__main__":
    main()
