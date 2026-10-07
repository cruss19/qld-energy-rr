# Queensland Electricity Demand Forecasting

**Leakage-aware, multi-horizon probabilistic forecasting for Queensland's five-minute electricity market**

![Platform](https://img.shields.io/badge/reference%20platform-Windows%2011-0078D4)
![Python](https://img.shields.io/badge/Python-3.11.15-3776AB)
![Environment](https://img.shields.io/badge/environment-reconstruction%20verified-2E8B57)
![Forecasts](https://img.shields.io/badge/horizons-5%E2%80%9330%20minutes-6A5ACD)
![ONNX](https://img.shields.io/badge/ONNX-opset%2018-005CED)

This repository is the public, reproducible portfolio edition of an end-to-end
machine-learning project for forecasting Queensland operational electricity
demand. It demonstrates how I turned public energy-market, weather, calendar,
population, and distributed-energy data into a chronologically evaluated
forecasting system, progressing from transparent baselines to deterministic
and probabilistic Temporal Convolutional Networks (TCNs).

The public study covers **2015–2020**. Development and model selection are
restricted to **2015–2019**, and **2020 is reserved as the final unseen
evaluation gate**. The public model lineage culminates in `TCN_starNLL`: a
long-context TCN with horizon-specific bounded Student-t distribution heads.

> **Evidence status:** the bounded 2015–2020 feature frame reproduces all 96
> columns of the frozen model frame exactly. The three final `TCN_starNLL`
> members, their equal-weight Student-t mixture ensemble, the fixed 2020
> evaluation, the comparison notebook, and an opset-18 checked ONNX artifact
> are complete. Later-year private-research results are not used here.

## At a glance

| Item | Public RR contract |
|---|---|
| Forecast target | Queensland operational demand change, reconstructed to demand in MW |
| Cadence | Five minutes |
| Horizons | 5, 10, 15, 20, 25, and 30 minutes |
| Development data | 2015–2019 only |
| Final evaluation | One unseen 2020 gate |
| Principal architecture | Multi-branch causal TCN with late context fusion |
| Probabilistic output | Student-t location, scale, and degrees of freedom per horizon |
| Main point metrics | MAE, RMSE, and signed bias in MW |
| Distribution metrics | Negative log likelihood and calibration diagnostics |
| External reference | AEMO five-minute demand forecast, aligned and scored separately |
| Reproducibility | Pinned environment, exact Windows lock, deterministic contracts, tests, manifests, and checksums |

## Why this problem matters

Queensland demand moves at several interacting time scales. Recent load
momentum matters over minutes; weather and rooftop solar alter regional demand
shapes; calendar effects change the relationship between time and consumption;
and slow structural variables change the long-run level of the system. A useful
forecast therefore has to preserve short-term chronology without allowing
future information, revised records, or test-period statistics to leak into
training.

The project addresses two related questions:

1. How accurately can demand be forecast along the complete next-30-minute
   path rather than at only one endpoint?
2. Can the model express useful forecast uncertainty during volatile and
   heavy-tailed demand changes?

At forecast origin \(t\), the deterministic target at horizon \(h\) is the
demand change

\[
\Delta D_{t,h}=D_{t+h}-D_t,
\]

and the demand-level forecast is reconstructed as

\[
\widehat D_{t+h}=D_t+\widehat{\Delta D}_{t,h},
\qquad h\in\{5,10,15,20,25,30\}\text{ minutes}.
\]

Predicting change avoids presenting persistence in the demand level as if it
were modelling skill. Reporting in MW keeps the final error measures directly
interpretable.

## Results

The six-horizon table uses the exact 105,400 forecast origins common to the five
TCN ensembles and AEMO. The locked 2020 gate was not used for model selection.

| Rank | Model | MAE (MW) | RMSE (MW) | 30-minute MAE (MW) | NLL |
|---:|---|---:|---:|---:|---:|
| 1 | `TCN_starNLL` three-seed ensemble | **39.305** | **52.676** | **46.986** | 5.293 |
| 2 | `TCN_star` three-seed ensemble | 39.339 | 52.738 | 47.481 | — |
| 3 | `TCN3` three-seed ensemble | 43.821 | 58.149 | 54.356 | — |
| 4 | `TCN2` three-seed ensemble | 44.294 | 58.762 | 55.051 | — |
| 5 | `TCN1` three-seed ensemble | 44.371 | 58.758 | 55.196 | — |
| 6 | AEMO P5MIN external benchmark | 49.740 | 65.794 | 61.056 | — |

On matched origins, `TCN_starNLL` lowers pooled MAE by **10.434 MW
(20.98%)** relative to the AEMO benchmark. Its 95% predictive interval covers
94.55% of observations.

The original statistical and probabilistic linear references forecast the
30-minute endpoint only. They are included in a separate 14-model endpoint
table on the same 105,400-origin intersection. At 30 minutes, `TCN_starNLL`
records 46.986 MW MAE; the strongest linear reference, Student-t linear,
records 56.523 MW; the original Ridge records 56.721 MW; and AEMO records
61.056 MW. Full horizon, endpoint, sample-count, and probabilistic results are
in [`notebooks/10_final_2020_evaluation.ipynb`](notebooks/10_final_2020_evaluation.ipynb).

## Model evolution

The public repository tells the modelling story rather than exposing only the
last network:

| Stage | Purpose | What it establishes |
|---|---|---|
| Persistence and weekly seasonal naive | Transparent minimum baselines | Whether a learned model beats simple temporal carry-forward rules |
| Comparable-history empirical model | Non-parametric change reference | Whether similar historical conditions contain useful local signal |
| Ridge regression | Regularised linear benchmark | Value of the engineered information set without a deep network |
| Gaussian, Student-t and empirical-residual linear references | Probabilistic endpoint benchmarks | Whether residual uncertainty is better represented by Gaussian, heavy-tailed or empirical distributions |
| `TCN_1`–`TCN_3` | Controlled architecture development | Value of causal convolutions, branch structure, and temporal depth |
| `TCN_star` | Long-context deterministic model | Value of a 42-hour input history and repeated dilation cycle |
| `TCN_starNLL` | Probabilistic extension | Dynamic location, scale, and tail thickness at all six horizons |
| Three-seed ensemble | Replication and variance reduction | Performance that is not dependent on one random initialization |

Each retained comparison must use an explicitly recorded feature contract,
chronological split, seed, training configuration, checkpoint rule, and scoring
implementation. The 2020 gate is not used to select among these stages.

## Architecture

The checked artifact is documented in the
[`TCN_starNLL` model card](docs/model_card_tcn_star_nll.md). Its static
[architecture rendering](outputs/model_artifacts/TCN_starNLL_architecture.png)
can be reviewed directly, while the
[opset-18 ONNX file](outputs/model_artifacts/TCN_starNLL_2020_seed_42.onnx)
can be opened in Netron.

```mermaid
flowchart LR
    A[AEMO and public contextual data] --> B[Schema and availability validation]
    B --> C[Causal feature engineering]
    C --> D1[Demand history]
    C --> D2[System and climate history]
    C --> D3[Five-region history]
    C --> D4[Known-origin calendar and structural context]
    D1 --> E1[Causal TCN branch]
    D2 --> E2[Causal TCN branch]
    D3 --> E3[Regional TCN branch]
    E1 --> F[Late fusion]
    E2 --> F
    E3 --> F
    D4 --> F
    F --> G[Student-t location / scale / df heads]
    G --> H[Six forecast horizons]
    H --> I[Point and probabilistic evaluation]
```

The verified `TCN_star` lineage uses:

- **505 five-minute states**, covering the forecast origin and preceding 42
  hours;
- kernel size **5**;
- twelve residual blocks with dilations
  `1, 2, 4, 8, 16, 32` repeated twice;
- a theoretical causal receptive field of **1,009 five-minute steps**;
- separate demand, system/climate, and regional temporal branches;
- static forecast-origin context joined only at late fusion; and
- six simultaneous output horizons.

`TCN_starNLL` retains that backbone and predicts three values for every
horizon:

- `mu`: location and point forecast in standardized target space;
- `scale`: strictly positive predictive scale; and
- `df`: degrees of freedom bounded above 2, preventing undefined variance and
  limiting uncontrolled tail behaviour.

Training minimizes Student-t negative log likelihood. Point forecasts use the
location output and are transformed back to MW for evaluation.

The checked export accepts 7 demand, 16 system, 2 climate, and 5 × 10 regional
temporal channels, each with 505 states, plus 55 encoded calendar values and
one statewide-population scalar. Later post-2020 research features are not
carried into this public model.

## Information set

The candidate public-era inputs are grouped by how and when they are available:

| Context | Examples | Model route |
|---|---|---|
| Demand history | Current demand and causal short/daily/weekly lags | Demand TCN branch |
| System history | Available generation, net interchange, UIGF, supply cushion, and causal lags | System/context TCN branch |
| Climate history | Lagged broad-scale climate indicators, only where publication timing is defensible | Context branch or late fusion according to the verified contract |
| Regional history | Temperature, humidity/thermal, cloud/rainfall, solar-capacity, and daylight timing variables | Regional temporal branch |
| Calendar | Day/week/year cycles, weekend, public- and school-holiday context | Forecast-origin late fusion |
| Structural | Original annual Queensland population estimate | Forecast-origin late fusion |

Weather and regional structural data are organised around five representative
Queensland nodes:

1. Brisbane;
2. Cairns / Atherton;
3. Dalby / Chinchilla;
4. Emerald / Gladstone; and
5. Townsville / Burdekin.

The public feature table documents, for every field, its provider,
formula, timestamp meaning, publication assumption, raw range, transformation,
fitted statistics, tensor context, and missing-value rule. Features are not
admitted merely because they exist in one of the research repositories.

## Data sources and provenance

| Provider | Public product | Project role | Redistribution policy |
|---|---|---|---|
| Australian Energy Market Operator (AEMO) | NEM `DISPATCHREGIONSUM` | Queensland demand, generation, interchange, UIGF, record metadata, and operational benchmark fields | Source and processed datasets are rebuilt locally, not committed |
| Open-Meteo | Historical Weather API | Hourly gridded weather for the five project nodes | No complete dataset committed; request metadata and attribution required |
| Australian Bureau of Statistics | ASGS 2016 POA boundaries | Auditable postcode-to-region allocation for distributed solar capacity | Frozen source archive remains local |
| Clean Energy Regulator | SRES postcode capacity data | Regional distributed-energy capacity | Source files remain local; derived use requires attribution |
| State of Queensland | Public-holiday and school-calendar information | Known-ahead calendar context | Original generation code is published; source terms remain applicable |
| Queensland Government Statistician's Office | Historical annual Queensland population estimates | Statewide structural late-fusion context | Bounded source extract and provenance are retained |

The repository will distribute acquisition and transformation code, sanitized
manifests, schema contracts, checksums, and deliberately small licensed or
synthetic fixtures—not an unreviewed copy of third-party research data.

### AEMO record handling

AEMO rows retain the complete record key during ingestion. Multiple record
versions are resolved using `LASTCHANGED`, and intervention solutions are
handled before reducing the data to one physical series per timestamp. Units,
continuity, duplicates, and coverage are checked before features are built.

`DEMANDFORECAST` is deliberately **excluded from model inputs**. AEMO defines
it as a five-minute forecast adjustment rather than a standalone demand-level
forecast. The external five-minute reference is therefore reconstructed from
the previous five-minute `TOTALDEMAND` and the applicable forecast adjustment,
then aligned and scored independently. This avoids both target ambiguity and
information redundancy from feeding another forecaster's output into the
research model.

### Temporal availability

- Every lag and rolling statistic is backward-looking.
- Scalers, imputers, vocabularies, thresholds, and other fitted transforms use
  training-fold rows only.
- Targets near a fold boundary are excluded unless the complete future horizon
  remains inside that same partition.
- Slow-moving data use explicit effective dates and backward as-of joins.
- Hourly weather may be forward-filled only after its source timestamp; future
  observations are never back-filled into earlier five-minute rows.
- Unknown categorical levels receive an explicit fallback category rather than
  changing the encoded width at evaluation time.
- Raw and derived tables are checked for unique, sorted, continuous timestamps.

## Experimental design

Random row-level splitting is unsuitable because nearby electricity records
are strongly dependent. The RR uses chronological development, expanding
training windows where appropriate, and one final hold-out year.

| Component | Public RR rule |
|---|---|
| Study period | 2015–2020, plus only the pre-period history required for causal lags |
| Development | 2015–2019 |
| Final gate | 1 January–31 December 2020 |
| Model selection | Validation data inside the development period only |
| Transform fitting | Training partition only |
| Checkpoint selection | Minimum validation objective under the verified run rule |
| Replication | Seeds 42, 142, and 242 where the final experiment contract requires three seeds |
| Ensemble | Prediction-level combination of the official best checkpoint from each verified member |
| Final reporting | Aggregate plus horizon-level point and probabilistic metrics |

The 2020 result is evaluated once after the public specification is frozen. If
historical artifacts cannot prove that discipline, the result will be rebuilt
from the public pipeline rather than relabelled.

## What this repository demonstrates

- Translating an operational energy question into a testable forecasting
  contract.
- Building multi-source data pipelines with explicit provenance and licensing.
- Resolving versioned AEMO records and aligning mixed-frequency inputs.
- Engineering causal temporal, calendar, energy-system, and regional-weather
  features.
- Designing multi-branch, long-receptive-field causal TCNs.
- Extending deterministic forecasts to heteroscedastic, heavy-tailed predictive
  distributions.
- Preventing leakage through fold-contained preprocessing and boundary tests.
- Managing exact experiment identity across configurations, seeds,
  checkpoints, predictions, metrics, and ensembles.
- Exporting and validating portable ONNX architecture artifacts.
- Communicating technical results with explicit limitations rather than
  overstating incomplete evidence.

## Reproducibility

### Reference platform

- Windows 11, x86-64
- Python 3.11.15
- CPU reference build of PyTorch 2.13.0
- ONNX opset 18

The readable environment is defined in [`environment.yml`](environment.yml).
The exact reconstruction-tested Windows solution and artifact hashes are in
[`conda-lock.yml`](conda-lock.yml).

### Create and verify the environment

```powershell
git clone https://github.com/cruss19/qld-energy-rr.git
cd qld-energy-rr
conda env create -f environment.yml
conda activate qld-energy-rr
python tools\verify_environment.py --expected-environment-name qld-energy-rr
python -m pytest -q
```

The verification utility checks package identity and imports, a Pandas/PyArrow
Parquet round trip, a scikit-learn transformation, a PyTorch forward/backward
pass, ONNX export and structural checking, TensorBoard event creation,
headless plotting, and YAML parsing. The lockfile was also reconstructed in a
disposable environment before acceptance.

Other operating systems are not currently claimed as supported. A Windows
GitHub Actions workflow is planned so that the public repository can recreate
and test the environment independently of the author's workstation.

## Repository map

| Location | Purpose |
|---|---|
| `config/` | Public experiment definitions and immutable contracts |
| `data/` | Manifests, schemas, local rebuild targets, and synthetic fixtures |
| `docs/` | Architecture, provenance, methodology, model cards, and evaluation notes |
| `notebooks/` | Curated analyses that support the published narrative |
| `outputs/` | Reproducible public figures, tables, and model artifacts |
| `src/` | Reusable acquisition, feature, model, and evaluation code |
| `tests/` | Unit, contract, leakage, and reproducibility tests |
| `tools/` | Command-line build, verification, training, and evaluation utilities |
| `tensorboard/` | TensorBoard launch and comparison helpers |
| `tensor_logs/` | Local event streams; ignored by Git unless deliberately curated |
| `training_output/` | Local checkpoints and run artifacts; ignored by Git by default |

### Notebook sequence

1. `01_data_acquisition_and_provenance.ipynb`
2. `02_data_validation_and_preparation.ipynb`
3. `03_feature_development_and_reference_contracts.ipynb`
4. `04_development_data_quality_and_eda.ipynb`
5. `05_reference_models.ipynb`, with `05b_probabilistic_reference_models.ipynb`
   as its probabilistic companion
6. `06a_tcn1_development.ipynb`, `06b_tcn2_development.ipynb`, and
   `06c_tcn3_development.ipynb` introduce the three compact TCN prototypes;
   `06d_tcn_prototype_comparison.ipynb` compares their development folds
7. `07_tcn_star_development.ipynb`
8. `08_tcn_star_nll_development.ipynb`
9. `09_tcn_ensemble_comparison.ipynb`
10. `10_final_2020_evaluation.ipynb`
11. `11_extension_modelling.ipynb` is an explicitly post-project extension

The source comparison and reconstruction decisions are recorded in [`docs/notebook_reconstruction_provenance.md`](docs/notebook_reconstruction_provenance.md).

Superseded comparison notebooks are retained under `notebooks/archive/` only
as reconstruction history and are not part of the public reader journey.

## Public evidence trail

The retained release evidence includes:

1. source, attribution, and licensing documentation;
2. sanitized source manifests with retrieval metadata and checksums;
3. deterministic acquisition and preprocessing commands;
4. feature schemas, value ranges, transformations, and leakage tests;
5. exact baseline and TCN experiment configurations;
6. model cards and a static architecture diagram;
7. a checked ONNX model that can be inspected with
   [Netron](https://netron.app/);
8. seed and best-checkpoint manifests;
9. reproducible aggregate and horizon-level result tables;
10. forecast-versus-observed and calibration figures; and
11. a carefully qualified AEMO comparison.

GitHub does not natively turn an ONNX file into an architecture diagram. The
README will therefore show a static, reviewable rendering linked to a detailed
architecture page; that page will link to the checked `.onnx` artifact for
interactive inspection in Netron.

## Public/private boundary

This repository covers the reproducible **2015–2020 development phase through
the `TCN_starNLL` model family**. It is intentionally not a mirror of the full
research workspace.

Excluded from this public repository are:

- post-2020 experimental results and event-specific redesigns;
- later model families, private ablations, and proprietary feature research;
- locked future-test data or results;
- deployment and commercial product engineering;
- machine-specific paths, credentials, caches, and raw provider archives; and
- artifacts whose provenance, licensing, or experiment identity is not yet
  sufficiently defensible.

Nothing enters the RR merely because it already exists. It enters only because
it serves the RR's public purpose and passes relevance, provenance, licensing,
privacy, and reproducibility review.

## Limitations

- Historical performance does not establish future performance under changing
  market, technology, climate, or demand regimes.
- The five regional nodes approximate statewide spatial conditions.
- Historical weather is gridded reanalysis/model data rather than direct
  observation at every named location.
- Exact operational publication latency must be documented for any feature
  presented as forecast-origin available.
- AEMO comparisons are sensitive to forecast origin, target timestamp,
  intervention handling, revisions, aggregation, and metric definition.
- Third-party data remain subject to provider terms and are not automatically
  redistributable because they are publicly accessible.
- The original five statistical references and the three probabilistic linear
  references forecast the 30-minute endpoint only. They are compared with the
  TCN ensembles and AEMO at that endpoint, not represented as six-horizon
  models.

## Project status

| Area | Status |
|---|---|
| Repository scaffold | Complete |
| Windows environment and exact lock | Complete and reconstruction-tested |
| Environment verification tool | Complete |
| Public data contracts and source manifests | Complete for the published 2015–2020 workflow |
| Curated source-code migration | Complete for the published model lineage |
| Public feature contract | Exact 96-column parity verified |
| 2020 reference reconstruction | Five original and three probabilistic 30-minute references complete; AEMO retained as a six-horizon external benchmark |
| 2020 `TCN_starNLL` three-seed ensemble | Complete |
| ONNX model and architecture rendering | Complete; opset-18 full checker passed |
| Final results and AEMO comparison | Separate six-horizon and 30-minute common-origin comparisons complete |

## Author and contact

Developed by **cruss19** as an independent applied machine-learning and energy
forecasting project.

Professional résumé, LinkedIn, and contact links will be added before the
repository is published.
