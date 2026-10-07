# Notebook reconstruction provenance

## Purpose

RR originally moved directly from reference models to the frozen ensemble comparison.  This reconstruction inserts the missing model-development layer without importing later-period results or rerunning a model.

## Historical evidence consulted

Historical source repositories and reconstruction artifacts were consulted
read-only. The review was limited to architecture definitions, the intended
model-development sequence, and presentation structure relevant to the RR's
2015–2020 boundary. Private workspace names, commits and internal paths are not
needed to establish the public result and are deliberately omitted.

## Accepted and adapted

- The progression from consolidated TCN1, through branched TCN2 and shared-regional TCN3, to deep-lookback TCN_star and probabilistic TCN_starNLL.
- Architecture explanations verified against preserved source evidence and RR run identities.
- Results-analysis ideas limited to fold comparison, horizon profiles, convergence metadata, and an explicit decision boundary.
- Separation between compact prototypes, TCN_star, and TCN_starNLL.

## Rejected or superseded

- Material outside the RR's declared 2015–2020 public boundary.
- Alternative model lineages that do not belong to the RR reconstruction.
- Source-notebook outputs tied to historical local paths or different datasets.
- Training-launch and optional execution cells; the RR notebooks are results-only.

## Newly written for RR

- `06a`–`06d`, `07`, and `08` notebooks that read only curated RR evidence under `outputs/06_tcn_development/`.
- Development tables generated from completed seed-42 folds for validation years 2017, 2018, and 2019.
- Compact cross-model figures and explicit explanations of why objective values cannot be compared between MSE and NLL models.

Only evidence relevant to the 2015–2020 RR boundary was retained. Numerical
claims were checked against retained RR evidence, and no result outside that
boundary entered the public reconstruction.
