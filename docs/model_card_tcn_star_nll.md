# Model card: TCN_starNLL

## Summary

`TCN_starNLL` is the final public probabilistic model in the Queensland Energy
Resume Repository. It predicts Queensland operational-demand change at 5, 10,
15, 20, 25 and 30 minutes. The public result is an equal-weight mixture of
three independently initialized Student-t models (seeds 42, 142 and 242).

## Intended use

The artifact is a reproducible research and portfolio demonstration of
short-horizon electricity-demand forecasting. It is not an operational AEMO
dispatch product and must not be used as the sole basis for market, safety or
grid-control decisions.

## Data and evaluation

- Study period: 2015–2020.
- Development and model selection: 2015–2019 only.
- Final fixed evaluation: held-out chronological 2020 RR benchmark reconstructed
  under a frozen protocol; it is not described as previously unseen because the
  year had been encountered in the broader historical project.
- Final refit: 2015–2019 for seven epochs, selected as the median official
  development best epoch before observing 2020 results.
- Target: `TOTALDEMAND(t+h) - TOTALDEMAND(t)` for six five-minute horizons.
- Transformations are fitted on training-fold data only.
- Historical maintainer verification found exact parity for all 96 compared
  frozen-frame columns over 629,273 eligible origins. The frozen comparison
  cache is deliberately not distributed; the sanitized result is retained in
  `outputs/03_feature_contract/public_parity_manifest.json`. Six terminal 2020
  timestamps are excluded because a complete 30-minute future target cannot
  exist.

## Inputs and architecture

The model receives 505 causal five-minute states (42 hours including the
forecast origin) for 7 demand, 16 system, 2 climate, and 5 × 10 regional
channels. Fifty-five encoded calendar values and one annual statewide
population value join at late fusion. The backbone uses kernel size 5 and the
dilation sequence `1, 2, 4, 8, 16, 32` twice. Each horizon returns Student-t
location, positive scale and degrees of freedom bounded to `[2.1, 20]`.

## Ensemble rule

Member weights are fixed at one third. The point forecast is the arithmetic
mean of member locations. Predictive density is the equal-weight Student-t
mixture evaluated with log-sum-exp; intervals are mixture quantiles. There is
no 2020 weight fitting or member refitting.

## 2020 performance

The complete ensemble contains 105,402 forecast origins. The all-model ranking
uses the 105,400 origins shared by every model and benchmark.

| Scope | MAE (MW) | RMSE (MW) | Bias (MW) | NLL | 95% coverage |
|---|---:|---:|---:|---:|---:|
| Full ensemble artifact | 39.308 | 52.682 | 2.322 | 5.293 | 94.55% |
| Exact common comparison | 39.305 | 52.676 | 2.325 | 5.293 | 94.55% |

| Horizon | MAE (MW) | RMSE (MW) | Bias (MW) | NLL | 95% coverage |
|---:|---:|---:|---:|---:|---:|
| 5 min | 31.323 | 39.896 | 0.823 | 5.088 | 95.44% |
| 10 min | 34.472 | 44.555 | 2.050 | 5.180 | 94.77% |
| 15 min | 37.854 | 49.691 | 3.110 | 5.268 | 94.51% |
| 20 min | 41.114 | 54.704 | 3.307 | 5.345 | 94.25% |
| 25 min | 44.095 | 59.404 | 1.625 | 5.409 | 94.01% |
| 30 min | 46.990 | 63.913 | 3.016 | 5.466 | 94.33% |

On common origins the model ranks first and reduces pooled MAE by 10.434 MW
(20.98%) relative to the AEMO P5MIN benchmark. The full ranking, conditional
slices, ramp analysis and calibration plots are in
`notebooks/10_final_2020_evaluation.ipynb`.

## Portable artifact

- ONNX: `outputs/model_artifacts/TCN_starNLL_2020_seed_42.onnx`
- Opset: 18
- Structural validation: `onnx.checker.check_model(full_check=True)` passed
- ONNX SHA-256: `00707426b62b80826802c765761326e4c340155b46cfb9a381c6c22d69f55303`
- Source checkpoint SHA-256: `1424ece584844e39764929ee79a5f4cd0e20717a29d90ac5d2cf7c39761b80c0`

The ONNX file represents seed 42 for architecture inspection and portable
single-member inference; the reported champion is the three-member mixture.

## Limitations

- Five regional nodes approximate statewide spatial conditions.
- Historical weather is gridded data and is carried forward from its timestamp.
- Annual statewide population is a coarse structural covariate.
- Performance is specific to the 2020 regime and does not establish future
  performance under market, technology or climate change.
- The historical statistical and probabilistic linear references forecast only
  the 30-minute endpoint. Their comparison with `TCN_starNLL` is therefore
  reported separately from the six-horizon TCN/AEMO comparison.
