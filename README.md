# Fast-Switching Quantum Thermal Reservoirs

Code, numerical data, and figures supporting the research project
*Finite Bath-Resolved Entropy Flow and Quadratic Detection Cost under Fast
Quantum Reservoir Switching*.

This is a theoretical and computational study. It considers a driven qubit
whose thermal Lindblad generator is selected by a hidden continuous-time
Markov process. The repository tests the fast-switching limit and the
separation between system-only statistical distinguishability and
bath-resolved entropy-flow bookkeeping.

## Main results represented by the code

- The reduced finite-time channel approaches the stationary averaged
  Lindblad channel with an `O(k^-1)` correction.
- For fixed system-only protocols with full-support reference probabilities,
  the output Kullback--Leibler divergence is `O(k^-2)`.
- When the first-order observable coefficient is nonzero, the divergence is
  `Theta(k^-2)` and fixed-error independent-shot discrimination requires
  `Theta(k^2)` samples.
- The same quadratic information scaling is recovered for correlated,
  no-reset rank-one measurement trajectories.
- The bath-resolved entropy-flow difference can retain a finite positive
  fast-switching limit even while system-only evidence disappears.

## Repository contents

| File | Purpose |
|---|---|
| `prereearch_switched_baths.py` | Initial switched-bath scan |
| `second_round_driven_qubit.py` | Driven-qubit steady-state and thermodynamic checks |
| `third_round_inference.py` | State and spectral inference checks |
| `fourth_round_detectability.py` | Independent-shot detectability calculations |
| `fifth_round_trajectory_kl.py` | No-reset trajectory relative-entropy-rate calculation |
| `sixth_round_first_order.py` | First-order asymptotic coefficients |
| `plot_manuscript_figures.py` | Generation of the four manuscript figures |
| `*_scan.csv` | Frozen numerical outputs |
| `figures/` | Reproducible figures in PDF, SVG, and PNG formats |
| `provenance/` | Parameter manifest, audit record, hashes, and reproduction scripts |

All CSV files contain theoretical or simulated results; no experimental data
are included or claimed.

## Quick start

The frozen snapshot used Python 3.14.3. Create an isolated environment and
install either the bounded dependencies or the exact frozen environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

To reproduce the full numerical workflow on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File provenance\reproduce.ps1 `
  -Python .\.venv\Scripts\python.exe
```

The no-reset trajectory calculation uses a fixed random seed but is the most
computationally expensive stage. Reproduction on a different NumPy, SciPy, or
BLAS/LAPACK build can change last-place floating-point digits. Scientific
comparison should therefore use the reported numerical tolerances in
`provenance/audit_record_2026-08-23.md`, not only byte-for-byte equality.

To verify that the checked-in snapshot has not changed:

```powershell
powershell -ExecutionPolicy Bypass -File provenance\verify_manifest.ps1
```

## Scope and interpretation

The reported entropy-flow difference compares bath-resolved bookkeeping with
the effective-bath description. It excludes the energetic and informational
cost of externally generating the classical switching process and is not
claimed to be the universal total entropy production of a complete laboratory
device.

The rotating-frame driven-qubit equation is used as a phenomenological GKLS
model. The largest drive value, `Omega = 0.8` at `omega_0 = 1`, should be read
as a model parameter study unless a microscopic beyond-rotating-wave or
Floquet validation is supplied.

## Citation and license

Citation metadata are provided in `CITATION.cff`. Until a journal article or
archival DOI is available, cite the software release and its GitHub URL.

The code and repository text are released under the MIT License. The paper
itself is intentionally not included in this repository.
