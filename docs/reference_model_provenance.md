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

## Sources consulted

### DREAM — primary source

- `notebooks/final_notebooks/05a_reference_models_expanding_window.ipynb`
- `notebooks/final_notebooks/05b_probabilistic_reference_models.ipynb`
- `redundancy/pre_v6_workspace_20260905/outputs/probabilistic_reference_models/`

DREAM supplied the established probabilistic-reference definitions and the
preserved prediction-level evidence. The imported artifacts were filtered to
evaluation years 2017–2020 before entering RR. Exact source hashes are recorded
in `outputs/probabilistic_reference_models/probabilistic_reference_assets_manifest.json`.

### REALITY — lineage and implementation cross-check

- `notebooks/05a_reference_models_expanding_window.ipynb`
- `notebooks/05b_probabilistic_reference_models.ipynb`

REALITY confirmed the historical five-model reference workflow and the
probabilistic-reference implementation. No REALITY file was modified or copied
wholesale.

### FINAL — scope cross-check

- `notebooks/04_reference_models.ipynb`
- `docs/RR_RECONSTRUCTION_WORKPLAN_PRIVATE.md`

FINAL confirmed that Gaussian linear, Student-t linear and empirical-residual
references belong to the intended RR comparison inventory. Its temporary
flattened-Ridge RR staging material was rejected because it did not represent
the recovered original Ridge contract.

## Material used and adapted

- The original five-model RR notebook and its 2017–2020 results remain the
  authoritative endpoint workflow already present in RR.
- DREAM's saved probabilistic predictions, fold metrics, calibration tables,
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
- Source-repository paths, caches, notebook metadata noise and unrelated later
  experiments.

## Newly written in RR

- `notebooks/05b_probabilistic_reference_models.ipynb`
- `notebooks/10_final_2020_evaluation.ipynb`
- `tools/build_reference_section.py`
- Corrected experiment contracts, preflight checks and reference tests
- Publication-ready 2017–2020 probabilistic summaries and 2020 comparison
  tables/figures

DREAM, REALITY and FINAL were used read-only throughout this reconstruction.
