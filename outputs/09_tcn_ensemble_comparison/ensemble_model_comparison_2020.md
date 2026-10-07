# Frozen 2020 TCN ensemble comparison

| overall_point_rank | model | ensemble_type | seeds | forecast_origins | origin_horizon_pairs | mae_mw | rmse_mw | bias_mw | mape_percent | mae_30m_mw | rmse_30m_mw | nll | coverage_95 | mean_interval_width_95_mw | pit_mean | pit_variance | mae_mw_rank | rmse_mw_rank | mae_30m_mw_rank | multi_metric_mean_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TCN_starNLL | equal_weight_student_t_mixture | 42,142,242 | 105402 | 632412 | 39.3081 | 52.6821 | 2.322 | 0.6533 | 46.9899 | 63.9135 | 5.2926 | 0.9455 | 190.3808 | 0.4823 | 0.0867 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | TCN_star | equal_weight_arithmetic_mean | 42,142,242 | 105402 | 632412 | 39.3413 | 52.7444 | 0.9495 | 0.6541 | 47.4853 | 64.4157 | nan | nan | nan | nan | nan | 2.0 | 2.0 | 2.0 | 2.0 |
| 3 | TCN3 | equal_weight_arithmetic_mean | 42,142,242 | 105402 | 632412 | 43.8238 | 58.1554 | 6.1005 | 0.7277 | 54.3601 | 72.1277 | nan | nan | nan | nan | nan | 3.0 | 3.0 | 3.0 | 3.0 |
| 4 | TCN2 | equal_weight_arithmetic_mean | 42,142,242 | 105402 | 632412 | 44.2971 | 58.7673 | 1.6051 | 0.7317 | 55.0556 | 72.961 | nan | nan | nan | nan | nan | 4.0 | 5.0 | 4.0 | 4.3333 |
| 5 | TCN1 | equal_weight_arithmetic_mean | 42,142,242 | 105402 | 632412 | 44.374 | 58.7642 | -0.9627 | 0.7341 | 55.2004 | 72.8889 | nan | nan | nan | nan | nan | 5.0 | 4.0 | 5.0 | 4.6667 |
