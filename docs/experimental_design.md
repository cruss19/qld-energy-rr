# Experimental Design and Evaluation Protocol

## 1. Purpose

This document records the experimental design used for the public Queensland
electricity-demand forecasting study presented in the Resume Repository (RR).
It defines the forecasting task, chronological data partitions, information
boundary, model-development procedure, reference comparisons, replication
strategy, ensemble construction, evaluation metrics, and reproducibility
controls applied to the 2015–2020 study period.

The design was adapted from the broader research project's experimental
framework and restricted to the public model lineage ending with
`TCN_starNLL`. Later private research, post-2020 model development, proprietary
feature investigations, and commercial implementation were outside the scope
of this study.

The governing methodological principle was chronological honesty. Inputs,
transformations, model decisions, thresholds, checkpoint rules, and comparison
methods were determined without using future target information. The 2020
period was treated as the final chronological RR benchmark and was excluded
from feature development, hyperparameter selection, architecture selection,
and training-duration selection.

## 2. Forecasting task

The study modelled Queensland operational electricity demand at five-minute
resolution. At each eligible forecast origin \(t\), a model predicted the next
six consecutive five-minute demand changes:

\[
\Delta D_{t+5},\;\Delta D_{t+10},\;\Delta D_{t+15},\;
\Delta D_{t+20},\;\Delta D_{t+25},\;\Delta D_{t+30}.
\]

Cumulatively adding the predicted changes to observed demand at the forecast
origin reconstructed the demand-level path from \(D_t\) to \(D_{t+30}\). This
representation supported six forecast horizons, the 30-minute endpoint, the
complete forecast path, and the 21 future subinterval ramps with durations of
5, 10, 15, 20, 25, or 30 minutes.

An eligible sample satisfied all of the following conditions:

- the input sequence ended at the forecast origin and retained chronological
  order;
- every input was available at or before the forecast origin under its declared
  availability rule;
- all six future target observations existed and remained within the relevant
  validation or evaluation boundary;
- future target values were absent from model inputs and fitted
  transformations;
- lags, differences, rolling statistics, ramps, and derived variables were
  strictly backward-looking; and
- all directly compared models used the same forecast origins, target values,
  horizons, eligibility mask, and metric implementation.

The AEMO `DEMANDFORECAST` field was used only as an external reference forecast.
It was not included as a model input.

## 3. Data acquisition, permissions, and contracts

### 3.1 Source acquisition

The modelling dataset was reconstructed from documented public or
authoritative sources. Acquisition was performed by source-specific utilities
that preserved the original provider fields before any modelling
transformation. Requests, source filenames, retrieval dates, file sizes,
coverage, and SHA-256 checksums were recorded in immutable manifests.

Acquisition was bounded at the source boundary: admitted observations ended on
31 December 2020 and versioned source products were released no later than 31
December 2020. Later rolling provider files were not downloaded and then
trimmed. Fixed ABS, ASGS and CER artifacts were verified by exact byte size and
SHA-256 before admission.

The principal source families were:

| Provider | Source used in the RR | Role in the study |
|---|---|---|
| Australian Energy Market Operator (AEMO) | National Electricity Market `DISPATCHREGIONSUM` records | Queensland `TOTALDEMAND`, available generation, net interchange, UIGF, dispatch metadata, record-version fields, intervention state, and the operational forecast benchmark fields |
| Open-Meteo | Historical Weather API | Hourly gridded weather history for the five representative Queensland regional nodes |
| Queensland Government Statistician's Office | Annual Queensland estimated resident population, historical 2013-2020 extract | Original TCN_starNLL statewide population scalar at late fusion, changing on 1 April and standardized within each training fold |
| Australian Bureau of Statistics (ABS) | ASGS 2016 SA2/POA boundaries | Spatial allocation support for the distributed-solar source |
| Clean Energy Regulator | Historical 2015-2018 workbooks and the official all-data file current at 31 October 2020 | Regional distributed-solar capacity context without post-2020 revisions |
| State of Queensland | Published Queensland public-holiday and school-calendar records | Known-ahead calendar context |

The five regional weather nodes were Brisbane, Cairns/Atherton,
Dalby/Chinchilla, Emerald/Gladstone, and Townsville/Burdekin. Coordinates,
requested weather variables, units, hourly coverage, retry behaviour, and cached
response identity were retained with the Open-Meteo acquisition record.

AEMO ingestion retained the full record key and relevant metadata until
multiple versions had been resolved. `LASTCHANGED`, run type, and intervention
state were applied before the table was reduced to one auditable physical
series per dispatch interval. `DEMANDFORECAST` was excluded from model inputs
and reconstructed only as a separately scored external benchmark.

### 3.2 Permissions, licensing, and redistribution

Public accessibility was not treated as permission to redistribute a complete
provider dataset. Provider terms, attribution requirements, copyright notices,
and dataset-specific licences were reviewed and recorded for every source.

Raw AEMO extracts, complete Open-Meteo responses, ABS workbooks and boundary
files, Clean Energy Regulator source files, and government calendar source
tables were rebuilt or retained locally rather than copied wholesale into the
public repository where redistribution permission was absent or uncertain.
The RR instead contained:

- deterministic acquisition and transformation code;
- source URLs, request parameters, retrieval metadata, and attribution;
- immutable sanitized manifests and SHA-256 checksums;
- schema and temporal-availability contracts;
- deliberately small lawful or synthetic fixtures used by automated tests; and
- derived aggregate outputs only where their publication was permitted.

No credentials, access tokens, personal filesystem paths, private endpoints,
or provider caches were included in the RR. A reader could reproduce the
source acquisition under the provider's current terms without the repository
redistributing an unreviewed copy of third-party data.

### 3.3 Data contracts

The reconstruction used explicit contracts at each boundary of the data
pipeline:

1. **Source acquisition contract** — identified the provider, product,
   endpoint or file family, request parameters, date coverage, expected files,
   checksum procedure, retrieval metadata, and permitted use.
2. **Raw schema contract** — fixed source keys, column names, dtypes, units,
   timestamp meaning, missing-value codes, valid ranges, revision fields, and
   duplicate-record rules.
3. **Temporal-availability contract** — defined when each field became usable
   relative to a forecast origin, including publication delay, effective date,
   revision selection, backward-as-of joins, and operational versus oracle
   status.
4. **Processed-table contract** — fixed the Australia/Brisbane timezone,
   five-minute cadence, sorting and uniqueness requirements, source-to-grid
   alignment, allowed forward filling, continuity tests, and missingness
   indicators.
5. **Feature contract** — linked every model input to its source field, causal
   formula, lag or rolling window, unit, range, transformation, fitted
   statistics, missing-value treatment, tensor context, and channel position.
6. **Split and eligibility contract** — fixed training, validation, and final
   evaluation boundaries; warm-up history; complete-target requirements; and
   the common eligible-origin mask used for paired comparisons.
7. **Model-input contract** — fixed feature order, dtype, tensor geometry,
   sequence orientation, forecast-origin alignment, category width, and scaler
   identity for each model family.
8. **Artifact contract** — connected each processed dataset, fitted transform,
   run, checkpoint, prediction table, ensemble, metric table, and figure to its
   configuration, source revision, and data fingerprint.

Contract validation occurred before model fitting. Files with an unexpected
schema, missing coverage, incorrect units, duplicate keys, discontinuous
timestamps, checksum mismatch, or incompatible feature geometry failed the
build rather than being silently coerced.

### 3.4 Causal availability

The data lineage connected every model feature to a processed column,
transformation, raw source field, source artifact, and availability rule.
Slow-moving structural variables used backward-as-of joins against explicit
release or effective dates. Retrospective interpolation across future releases
was not used. Hourly weather observations aligned to the five-minute grid were
forward-filled only after their source timestamp and were never back-filled
from a future observation.

Features were classified as:

1. **Operational** — demonstrably available at the forecast origin under a
   documented publication, revision, and effective-date rule.
2. **Retrospective/oracle** — finalised observations retained for explanatory
   analysis or explicitly labelled upper-bound experiments.
3. **Quarantined** — targets, future realised values, inadmissible
   contemporaneous outcomes, `DEMANDFORECAST` as an input, and sources without
   a defensible availability rule.

Only features admitted by the declared contract entered an operational model.
Oracle analyses were reported separately and did not select the operational
model.

## 4. Chronological partitions

Random row-level splitting was not used because adjacent five-minute records
and overlapping forecast horizons are strongly dependent. Development used
expanding chronological windows:

| Fold | Training period | Validation period |
|---|---|---|
| 1 | 1 January 2015–31 December 2016 | 1 January–31 December 2017 |
| 2 | 1 January 2015–31 December 2017 | 1 January–31 December 2018 |
| 3 | 1 January 2015–31 December 2018 | 1 January–31 December 2019 |

Pre-period observations were used only when required to construct causal lags
for otherwise eligible samples. They were not treated as additional targets or
as evidence from a later partition.

Within each fold, preprocessing and all learned transformations were fitted on
the training partition only and applied unchanged to validation. Validation
rows did not influence imputation parameters, category vocabularies, scaler
statistics, clipping limits, dimensionality reduction, feature selection,
event thresholds, residual calibration, or model weights.

The final models were refitted on all eligible observations from 1 January 2015
through 31 December 2019. Their frozen predictions were evaluated on eligible
origins from 1 January through 31 December 2020.

Because 2020 had been encountered during the broader historical project, the
public result was described as a **held-out chronological RR benchmark
reconstructed under a frozen protocol**, rather than as a previously unseen
test set. This qualification did not alter its experimental role: 2020 target
outcomes were not used to change the public reconstruction.

## 5. Preprocessing and leakage control

Every chronological fold owned an independent fitted preprocessing state. The
following quantities were estimated from the fold's training observations
only:

- imputation parameters;
- category vocabularies and encoded widths;
- means, standard deviations, extrema, quantiles, and clipping limits;
- target and residual scalers;
- dimensionality-reduction components;
- feature-selection statistics;
- calibration mappings; and
- event or regime thresholds.

The pipeline enforced stable feature names, ordering, dtypes, units, tensor
geometry, and category width. Unknown categories used an explicit fallback
level rather than altering the encoded dimension. Timestamps were checked for
timezone consistency, ordering, uniqueness, five-minute continuity, and
duplicate forecast origins. Samples whose targets crossed a fold or calendar
boundary were excluded.

Automated contract tests verified that input windows contained no future
timestamps, transformations were fitted only to training rows, paired models
shared the same evaluation mask, and model inputs and outputs remained finite.
Prediction artifacts retained the forecast origin, target timestamps, horizon,
actual value, prediction, model identity, seed, fold, and model-state identity.

## 6. Model-development sequence

The study introduced model complexity in stages:

1. transparent deterministic references;
2. probabilistic statistical references;
3. `TCN1`;
4. `TCN2`;
5. `TCN3`;
6. `TCN_star`; and
7. `TCN_starNLL`.

Each TCN stage retained its own documented feature, architecture, objective,
preprocessing, and training contract. Later implementations were not
retroactively substituted for earlier stages. This preserved the evidential
value of the model progression rather than presenting only the final model.

Architecture and training decisions were based on the three pre-2020 folds.
The controlled decisions included the information set, input length, receptive
field, convolutional width, dilation schedule, kernel size, dropout,
activation, optimiser, learning-rate schedule, batch size, gradient controls,
loss, predictive distribution, numerical parameter bounds, calibration, and
ensemble method.

Development runs used seed 42 as the common comparison seed. The search space
and completed trials were retained with the experiment record, including
negative results. Point models were selected primarily through macro-averaged
horizon MAE, with RMSE, signed bias, fold stability, and regime performance as
guardrails. Probabilistic models were selected primarily through
macro-averaged validation negative log-likelihood, with MAE, RMSE, CRPS,
coverage, interval width, PIT behaviour, fold stability, and numerical
reliability as guardrails. No model was declared superior from one metric
alone.

## 7. Training-duration and model-state selection

Development-fold training used the declared validation objective to identify
the official best epoch within each fold. Final 2020 model selection did not use
2020 loss. Instead, each model family's final training duration was fixed from
the three pre-2020 development folds:

1. the official best validation epoch was recorded independently for the 2017,
   2018, and 2019 folds;
2. the median of the three epoch numbers defined the final duration; and
3. all three final seeds completed that same fixed number of epochs.

The fixed-duration rule was applied after preprocessing and hyperparameters had
been frozen. The final state was therefore determined before 2020 scoring and
was not selected from a sequence of 2020 losses.

Development checkpoints preserved model, optimiser, scheduler, scaler, random
number generator, epoch, and best-metric state. Final runs preserved the same
state for reproducibility even though their reported model state was the frozen
fixed epoch rather than a checkpoint selected on 2020 outcomes.

## 8. Replication and ensembles

Each final TCN family was fitted on the 2015–2019 partition with seeds 42, 142,
and 242. This produced three independently initialised members for each of
`TCN1`, `TCN2`, `TCN3`, `TCN_star`, and `TCN_starNLL`.

Deterministic ensemble predictions were the equal-weight arithmetic mean of the
three member predictions at every origin and horizon.

The `TCN_starNLL` probabilistic ensemble was an equal-weight mixture of the
three member Student-t predictive distributions. Its point prediction was the
mean of member predictive locations. Mixture density was evaluated with a
numerically stable log-sum-exp calculation, and prediction intervals were
obtained from the mixture distribution. The ensemble was not collapsed into a
single fitted Student-t distribution.

The ensemble record retained member run identities, seeds, fixed-epoch model
states, the combination formula, aligned member predictions, ensemble
predictions, and aggregate and horizon-level metrics. Ensemble construction did
not refit members or optimise weights against 2020 outcomes.

## 9. Reference models and information parity

The comparison set comprised:

- persistence;
- weekly seasonal naive;
- comparable-history empirical change;
- Bayesian random walk;
- Ridge;
- Gaussian linear;
- Student-t linear;
- empirical-residual probabilistic reference; and
- the aligned AEMO five-minute demand forecast benchmark.

Deterministic references were fitted once under their declared deterministic
procedure rather than being given artificial seed ensembles. Any stochastic
reference preserved its random state.

The original Ridge is the established 30-minute reference: 55 base predictors,
two deterministic calendar encodings, fold-fitted preprocessing, three
chronological inner splits with a six-row target-horizon gap, 17 candidate
alpha values, and the original `sparse_cg` implementation. It is not a
flattened copy of the TCN's 505-step tensors.

The Gaussian, Student-t and empirical-residual references retain that
30-minute information contract. Their distributional parameters are estimated
from a chronological 180-day training-only calibration block. The Gaussian
and empirical models use the original Ridge location; the Student-t model uses
the established five-step penalised IRLS location fit.

Two comparison scopes are therefore reported. The six-horizon table contains
the five TCN ensembles and AEMO. A separate 30-minute endpoint table contains
those models plus all eight statistical references. Every table uses its own
explicit common-origin intersection.

The AEMO comparison used the intersection of eligible forecast origins and
matched target timestamps, horizons, intervention treatment, record-vintage
policy, aggregation, exclusions, and metric definitions. The AEMO forecast
remained an external benchmark rather than an input or model-selection signal.

## 10. Evaluation measures

All models were evaluated on identical eligible origins. Metrics were reported
for the complete evaluation matrix and separately at 5, 10, 15, 20, 25, and
30 minutes.

### 10.1 Point performance

- mean absolute error (MAE), in MW;
- root mean squared error (RMSE), in MW;
- signed bias, in MW;
- reconstructed demand-level error;
- 30-minute endpoint error; and
- error across the 21 future subinterval ramps.

### 10.2 Probabilistic performance

- negative log-likelihood (NLL);
- continuous ranked probability score (CRPS);
- central prediction-interval coverage;
- mean prediction-interval width;
- probability integral transform (PIT) diagnostics; and
- point accuracy derived from the declared predictive centre.

### 10.3 Conditional performance

The evaluation included results by season, time of day, weekday/weekend,
demand regime, and ramp regime. Conditional strata and thresholds were fixed
before the final evaluation and were not searched after viewing 2020 outcomes.

### 10.4 Paired uncertainty analysis

Uncertainty in paired metric differences was estimated with a seven-day moving
block bootstrap, preserving serial and weekly dependence. The analysis used
10,000 paired replicates, a fixed random seed of 42, and two-sided 95% percentile
intervals. Resampling operated on aligned blocks of forecast origins; individual
five-minute rows were not treated as independent observations.

## 11. Ramp-event analysis

Significant-ramp thresholds were estimated separately for each ramp duration
from the applicable training partition. The threshold was the training-fold
95th percentile of absolute ramp magnitude calculated from unique ramp starts,
not from duplicated appearances of the same ramp in overlapping windows.

Upward and downward events were separate and non-exclusive. Event evaluation
reported precision-recall AUC, Brier score, log loss, reliability, recall,
false-alarm rate, and warning lead time where supported by the model output.
All thresholds were frozen before the corresponding validation or final
evaluation period was scored.

## 12. Treatment of 2020 conditions

The 2020 period was evaluated as observed. Pandemic-related changes and other
unusual demand conditions were not removed, retrospectively imputed, or given a
special feature devised after viewing benchmark outcomes. Any exogenous feature
used during 2020 followed the same pre-declared availability and transformation
contract used in development.

This treatment assessed how models developed on preceding years generalized to
a structurally unusual chronological period. It did not claim that the observed
2020 regime represented ordinary future operation.

## 13. Run identity and retained evidence

Each model run retained:

- the resolved configuration and unique run identifier;
- model family, variant, fold, seed, and fixed training duration;
- source revision and working-tree state;
- data, feature, split, and environment fingerprints;
- feature names, order, units, and tensor contract;
- hardware, numerical precision, and dependency versions;
- training and validation histories;
- learning-rate and stopping histories;
- resumable checkpoint state;
- official development-fold best state or final fixed-epoch state;
- TensorBoard event data;
- aligned predictions and evaluation masks;
- aggregate and horizon-level metrics; and
- normal-completion or failure status.

Artifacts were written to unique, non-overwriting run directories. Final
prediction and metric products were treated as immutable evidence and were
linked to the configuration, data fingerprint, source revision, and model-state
identity that produced them.

## 14. Reproducibility and presentation

The software environment was defined by the readable Conda specification and
the exact Windows lockfile. Public commands reconstructed acquisition,
preprocessing, references, model runs, ensembles, evaluation tables, and
figures from declared repository paths rather than machine-specific locations.

Reusable logic resided in `src/`, experiment parameters in `config/`, command
entry points in `tools/`, and contract and leakage checks in `tests/`. Curated
notebooks communicated and audited the study but did not contain the only
implementation of preprocessing, training, ensemble construction, or metrics.

The public notebook sequence covered:

1. data and split auditing;
2. reference models;
3. chronological model-development results;
4. final 2020 evaluation; and
5. model comparison and conclusions.

Every notebook executed from a clean kernel against repository code and
declared inputs. Published tables and figures were generated from retained
prediction artifacts rather than manually transcribed values.

## 15. Interpretation boundary

The study established historical performance under the declared 2015–2020
contract. It did not establish uninterrupted performance under later market,
technology, climate, demographic, or demand regimes. Regional weather nodes
approximated statewide spatial conditions, and finalised historical weather was
distinguished from authentic issue-time weather forecasts.

The RR documented the public foundation of a larger research programme without
disclosing later private model families, post-2020 experimental results,
proprietary feature work, or commercial implementation. This boundary did not
alter the reported RR methodology or results; it defined the evidence to which
the public claims applied.
