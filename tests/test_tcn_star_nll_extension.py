from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.tcn_star_nll_extension_data import (
    ExtensionDataset,
    build_demand_helpers,
    build_radiation_helpers,
)
from src.tcn_star_nll_extension_model import (
    TCNStarNLLCausalFeatureEstimates,
    TCNStarNLLCausalFeatureEstimatesNoSD,
)


def _demand_source() -> pd.DataFrame:
    index = pd.date_range("2014-12-01", "2015-03-31 23:55", freq="5min")
    values = np.arange(len(index), dtype=float)
    holiday = np.zeros(len(index), dtype=np.int8)
    holiday[index.normalize() == pd.Timestamp("2015-02-05")] = 1
    return pd.DataFrame({
        "SETTLEMENTDATE": index,
        "TOTALDEMAND": values,
        "is_public_holiday_qld": holiday,
    })


def test_demand_weekday_holiday_donor_uses_nearest_sunday():
    source = _demand_source()
    timestamp = pd.DatetimeIndex([pd.Timestamp("2015-02-26 12:00")])
    result = build_demand_helpers(timestamp, source)
    series = source.set_index("SETTLEMENTDATE")["TOTALDEMAND"]
    # The t-21d donor is Thursday 5 February, a holiday; its nearest Sunday is
    # 8 February. The other donors remain 12 and 19 February.
    donors = series.reindex(pd.to_datetime([
        "2015-02-19 12:00", "2015-02-12 12:00", "2015-02-08 12:00",
    ])).to_numpy()
    assert result.iloc[0]["demand_estimated"] == np.mean(donors)
    assert result.iloc[0]["demand_estimated_sd"] == np.std(donors, ddof=1)


def test_public_holiday_uses_three_previous_sundays():
    source = _demand_source()
    timestamp = pd.DatetimeIndex([pd.Timestamp("2015-02-05 12:00")])
    result = build_demand_helpers(timestamp, source)
    series = source.set_index("SETTLEMENTDATE")["TOTALDEMAND"]
    donors = series.reindex(pd.to_datetime([
        "2015-02-01 12:00", "2015-01-25 12:00", "2015-01-18 12:00",
    ])).to_numpy()
    assert result.iloc[0]["demand_estimated"] == np.mean(donors)
    assert result.iloc[0]["demand_estimated_sd"] == np.std(donors, ddof=1)


def test_radiation_helpers_use_same_clock_previous_three_days():
    hours = pd.date_range("2015-01-01", "2015-01-10 23:00", freq="h")
    rows = []
    labels = {
        "Brisbane", "Cairns / Atherton", "Dalby / Chinchilla",
        "Emerald / Gladstone", "Townsville / Burdekin",
    }
    for label in labels:
        rows.append(pd.DataFrame({
            "region": label,
            "datetime": hours,
            "direct_radiation": hours.day.to_numpy(float) * 10 + hours.hour.to_numpy(float),
        }))
    weather = pd.concat(rows, ignore_index=True)
    timestamps = pd.DatetimeIndex([pd.Timestamp("2015-01-10 12:35")])
    result = build_radiation_helpers(timestamps, weather)
    assert result.iloc[0]["brisbane__direct_radiation"] == 111.0
    assert result.iloc[0]["brisbane__direct_radiation_estimated"] == np.mean([102.0, 92.0, 82.0])
    assert result.iloc[0]["brisbane__direct_radiation_estimated_sd"] == np.std([102.0, 92.0, 82.0], ddof=1)


def test_dataset_replaces_only_terminal_values_and_aligns_sd_channels():
    length = 510
    arrays = {
        "demand": torch.zeros(length, 8),
        "demand_terminal_estimated": torch.arange(length, dtype=torch.float32) + 1000,
        "system": torch.zeros(length, 16),
        "climate": torch.zeros(length, 2),
        "regional": torch.zeros(length, 5, 12),
        "radiation_terminal_estimated": torch.arange(length * 5, dtype=torch.float32).reshape(length, 5),
        "calendar": torch.zeros(length, 3),
        "population": torch.zeros(length, 1),
        "targets": torch.zeros(length, 6),
    }
    arrays["demand"][:, 0] = torch.arange(length)
    arrays["demand"][:, 7] = torch.arange(length) / 10
    arrays["regional"][:, :, 10] = 5
    arrays["regional"][:, :, 11] = torch.arange(length)[:, None] / 100
    endpoint = 504
    sample = ExtensionDataset(arrays, np.array([endpoint]))[0]
    demand, regional = sample[0], sample[3]
    assert demand[0, -2] == endpoint - 1
    assert demand[0, -1] == endpoint + 1000
    assert torch.equal(demand[7], arrays["demand"][:endpoint + 1, 7])
    assert torch.all(regional[:, 10, -2] == 5)
    assert torch.equal(regional[:, 10, -1], arrays["radiation_terminal_estimated"][endpoint])
    assert torch.equal(regional[0, 11], arrays["regional"][:endpoint + 1, 0, 11])


def test_extension_model_tensor_contract_and_output():
    model = TCNStarNLLCausalFeatureEstimates(calendar_dim=4)
    output = model(
        torch.zeros(2, 8, 505),
        torch.zeros(2, 16, 505),
        torch.zeros(2, 2, 505),
        torch.zeros(2, 5, 12, 505),
        torch.zeros(2, 4),
        torch.zeros(2, 1),
    )
    assert [tuple(value.shape) for value in output] == [(2, 6), (2, 6), (2, 6)]
    assert model.receptive_field_steps == 1009


def test_no_sd_extension_contract_uses_estimates_without_sd_channels():
    length = 505
    arrays = {
        "demand": torch.zeros(length, 7),
        "demand_terminal_estimated": torch.arange(length, dtype=torch.float32) + 1000,
        "system": torch.zeros(length, 16),
        "climate": torch.zeros(length, 2),
        "regional": torch.zeros(length, 5, 11),
        "radiation_terminal_estimated": torch.arange(length * 5, dtype=torch.float32).reshape(length, 5),
        "calendar": torch.zeros(length, 4),
        "population": torch.zeros(length, 1),
        "targets": torch.zeros(length, 6),
    }
    sample = ExtensionDataset(arrays, np.array([504]))[0]
    assert sample[0].shape == (7, 505)
    assert sample[3].shape == (5, 11, 505)
    assert sample[0][0, -1] == 1504
    assert torch.equal(sample[3][:, 10, -1], arrays["radiation_terminal_estimated"][504])

    model = TCNStarNLLCausalFeatureEstimatesNoSD(calendar_dim=4)
    output = model(*(value.unsqueeze(0) for value in sample[:6]))
    assert [tuple(value.shape) for value in output] == [(1, 6), (1, 6), (1, 6)]
    assert model.DEMAND_CHANNELS == 7
    assert model.REGIONAL_FEATURES == 11
