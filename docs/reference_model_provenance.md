# Reference-model reconstruction provenance

## RR scope

The RR retains nine reference identities:

1. Persistence
2. Weekly seasonal naive
3. Comparable-history empirical change
4. Bayesian random walk
5. Original Ridge regression
6. Gaussian linear
7. Student-t linear
8. Empirical residual
9. AEMO P5MIN external benchmark

The first eight are 30-minute endpoint references. AEMO is evaluated across
5, 10, 15, 20, 25 and 30 minutes. The endpoint-only models are never presented
as six-horizon models.

## Historical evidence consulted

Historical source repositories and preserved reconstruction artifacts were
consulted read-only to recover the established reference definitions and
prediction-level evidence. Retained evidence was filtered to evaluation years
2017–2020 before entering RR. Integrity hashes are recorded in
`outputs/probabilistic_reference_models/probabilistic_reference_assets_manifest.json`.

The source review also confirmed that Gaussian linear, Student-t linear and
empirical-residual references belong to the intended comparison inventory. A
temporary flattened-TCN Ridge staging design was rejected because it did not
represent the recovered original 55-feature Ridge endpoint contract.

## Material used and adapted

- The original five-model RR notebook and its 2017–2020 results remain the
  authoritative endpoint workflow already present in RR.
- Preserved probabilistic predictions, fold metrics, calibration tables,
  fitted fold parameters, feature contract and parity audit were copied into
  RR and filtered to the public boundary.
- Pooled probabilistic summaries were recomputed from the retained 2017–2020
  prediction rows only; no model was refitted.
- The RR experiment declarations were rewritten to describe the original
  55-feature Ridge and the three established probabilistic extensions.
- The final evaluation was rebuilt as two explicit comparisons: six-horizon
  TCN/AEMO results and a separate 30-minute endpoint table.

## Material rejected or superseded

- The interim flattened-TCN Ridge staging specification and its memory blocker.
- Six-horizon representations of the original statistical and probabilistic
  endpoint references.
- Post-2020 prediction rows and summaries from historical source artifacts.
- Source-workspace paths, caches, notebook metadata noise and material outside
  the RR boundary.

## Newly written in RR

- `notebooks/05b_probabilistic_reference_models.ipynb`
- `notebooks/10_final_2020_evaluation.ipynb`
- `tools/build_reference_section.py`
- Corrected experiment contracts, preflight checks and reference tests
- Publication-ready 2017–2020 probabilistic summaries and 2020 comparison
  tables/figures

Historical source evidence was used read-only throughout this reconstruction;
only evidence serving the RR's public 2015–2020 purpose was retained.
