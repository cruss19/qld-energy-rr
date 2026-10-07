# Notebook reconstruction provenance

## Purpose

RR originally moved directly from reference models to the frozen ensemble comparison.  This reconstruction inserts the missing model-development layer without importing later-period results or rerunning a model.

## Sources consulted

| Source repository | Commit | Material reviewed | RR decision |
|---|---|---|---|
| `qld-demand-forecasting-dream` | `fb76fa6b57bbba4b7e94da013b271af28694dfaa` | `notebooks/06a_tcn_1.ipynb`, `06b_tcn_2.ipynb`, `06c_tcn_3.ipynb`, `06d_v6_tcn_results_analysis.ipynb`, `07_TCN_star.ipynb`, `08_TCN_starNLL.ipynb`; `src/v6/tcn_models.py`, `tcn_star.py`, `tcn_star_nll.py` | Architecture descriptions and the model-development/diagnostic structure were adapted. Numerical claims were replaced with RR's own completed 2017–2019 artifacts. |
| `qld-demand-forecasting-final` | `f824b85012e4d3e5c2b732198cbdc8a28d9d52d7` | `notebooks/05_tcn_base_model_experiments.ipynb`, `06_tcn2_vs_ridge.ipynb`, `07_tcn_star.ipynb`, `08_tcn_starNLL - competing models.ipynb` | Used as an outline of the intended reader journey. Most cells were placeholders, so no numerical result was copied. |
| `qld-demand-forecasting-reality` | `fb76fa6b57bbba4b7e94da013b271af28694dfaa` | `notebooks/06_tcn_modelling.ipynb`, `06a_six_channel_tcn.ipynb`, `06b_probabilistic_six_channel_tcn.ipynb` | General language about chronological evaluation and fair comparison was considered. Model code, later-year results, and the different six-channel lineage were rejected. |

## Accepted and adapted

- The DREAM progression from consolidated TCN1, through branched TCN2 and shared-regional TCN3, to deep-lookback TCN_star and probabilistic TCN_starNLL.
- Architecture explanations verified directly against the authoritative DREAM source modules and RR run identities.
- Results-analysis ideas limited to fold comparison, horizon profiles, convergence metadata, and an explicit decision boundary.
- FINAL's intended separation between compact prototypes, TCN_star, and TCN_starNLL.

## Rejected or superseded

- Every result after 2020, all 2021–2024 validation material, event/Callide studies, spike-threshold studies, weight-decay experiments, seven-step experiments, later LNN variants, and private feature research.
- REALITY's later six-channel models because they are not the RR model lineage.
- Source-notebook outputs tied to historical local paths or different datasets.
- Training-launch and optional execution cells; the RR notebooks are results-only.

## Newly written for RR

- `06a`–`06d`, `07`, and `08` notebooks that read only curated RR evidence under `outputs/06_tcn_development/`.
- Development tables generated from completed seed-42 folds for validation years 2017, 2018, and 2019.
- Compact cross-model figures and explicit explanations of why objective values cannot be compared between MSE and NLL models.

The frozen 2020 ensemble and final-evaluation artifacts are neither recomputed nor used to justify development choices.
