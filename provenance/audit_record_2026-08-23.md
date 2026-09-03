# Physics and numerical integrity audit — 2026-08-23

Release note: this archived audit covered the broader working archive. The
GitHub package is a curated subset containing the complete numerical workflow,
the frozen data, and the canonical figures. The paper source is excluded.

## Scope

The audit covered the English and Chinese manuscripts, all root numerical
programs and CSV scans, the canonical plotting program, four figure families,
and the two final PDFs. It was a static-code, numerical-reproduction, and
physics-consistency audit; it was not external peer review.

## Provenance findings

- The root fourth- and fifth-round CSV files were byte-identical to the copies
  shipped in the English LaTeX package before this snapshot was created.
- Each canonical figure had identical copies in the English and Chinese
  manuscript packages before this snapshot was created.
- The plotting program reads the recorded CSVs. No manually inserted data
  points, deleted outliers, or axis truncations that reverse the trends were
  found.
- Analytic asymptotic lines are explicit in the plotting source and were
  independently reproduced by `sixth_round_first_order.py` or the closed-form
  fast-switching calculation.

## Numerical checks

- All five CSV scans contained no NaN or infinity values.
- Maximum steady-state first-law residual: `2.925e-14`.
- Minimum full entropy-production value in the scanned second-round grid:
  `1.64394148e-4`.
- Minimum steady-state qubit eigenvalue in the main audit grid: `0.299628556`.
- Maximum stationary linear-system residual in the main grid: `3.216e-13`.
- Minimum raw measurement probability before clipping: `0.0`; no negative raw
  probability was found.
- Maximum raw probability-normalization error: `6.516e-13`.
- Fourth-round independent reproduction: maximum relative KL difference
  `1.533e-6` across the scan.
- Fifth-round fixed-seed reproduction: maximum relative trajectory-KL
  difference `1.485e-6` across the scan.
- Third-round spectral quantities reproduced to approximately `3e-13`
  relative to their scan ranges.
- The largest range-normalized difference in the finite-difference FCS
  variance was `7.45e-6`; FCS data are not used in the final figures.

Different SciPy/NumPy builds produced different raw CSV hashes because of
last-place matrix-exponential and eigenvalue differences, while the numerical
results agreed within the tolerances above.

## Physics checks

- The Lindblad rates obey local detailed balance, `u/d = exp(-beta)`.
- The conditional classical-quantum generator is trace preserving and yields
  positive stationary states on the scanned parameter grid.
- Heat is consistently defined relative to the bare Hamiltonian, with coherent
  drive power separated as work.
- The manuscript explicitly excludes switching-device cost and does not label
  its resolution-dependent entropy-flow difference as universal total entropy
  production.
- The main model-risk item is the microscopic interpretation of the strong
  drive `Omega=0.8` at `omega_0=1` under a rotating-wave Hamiltonian. This is a
  limitation of physical applicability, not evidence of fabricated data or a
  failure of the abstract GKLS dynamics.

## Integrity conclusion

No evidence of fabricated numerical data, concealed negative probabilities,
non-positive states, selective point removal, or manipulated figures was
found. Static files alone cannot prove that no historical manual edit ever
occurred. The Git snapshot, annotated tag, manifests, and future run logs are
therefore required for prospective traceability.
