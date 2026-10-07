import pandas as pd
import pytest

from src.population_pipeline import (
    align_annual_population_to_forecast_origins,
    fit_population_standardizer,
)
from src.rr_data_preparation import build_calendar, causal_weather_alignment_audit, development_only


def test_calendar_never_changes_timestamp_count() -> None:
    timestamps = pd.Series(pd.date_range("2020-01-01", periods=24, freq="5min"))
    calendar = build_calendar(timestamps)
    assert len(calendar) == len(timestamps)
    assert calendar["SETTLEMENTDATE"].equals(timestamps)


def test_calendar_rejects_post_2020_rows() -> None:
    with pytest.raises(ValueError, match="escapes"):
        build_calendar(pd.Series([pd.Timestamp("2021-01-01")]))


def test_weather_alignment_is_backward_only() -> None:
    origins = pd.Series(pd.to_datetime(["2020-01-01 00:05", "2020-01-01 01:05"]))
    parts = []
    for region in ("Brisbane", "Cairns / Atherton", "Dalby / Chinchilla", "Emerald / Gladstone", "Townsville / Burdekin"):
        parts.append(pd.DataFrame({"region": region, "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"])}))
    audit = causal_weather_alignment_audit(origins, pd.concat(parts, ignore_index=True))
    assert audit["future_weather_rows"].eq(0).all()
    assert audit["maximum_age_minutes"].eq(5).all()


def test_development_eda_excludes_locked_2020() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2019-12-31 23:55", "2020-01-01 00:00"])})
    result = development_only(frame, "timestamp")
    assert result["timestamp"].tolist() == [pd.Timestamp("2019-12-31 23:55")]


def test_original_annual_population_changes_on_first_april() -> None:
    annual = pd.DataFrame({
        "reference_year": [2013, 2014, 2015],
        "total_qld_population": [4_652_824, 4_719_653, 4_777_692],
        "reference_date": pd.to_datetime(["2013-06-30", "2014-06-30", "2015-06-30"]),
        "available_from": pd.to_datetime(["2014-04-01", "2015-04-01", "2016-04-01"]),
    })
    origins = pd.Series(pd.to_datetime(["2015-03-31 23:55", "2015-04-01 00:00", "2016-04-01 00:00"]))
    result = align_annual_population_to_forecast_origins(origins, annual)
    assert result["total_qld_population"].tolist() == [4_652_824, 4_719_653, 4_777_692]
    assert (result["available_from"] <= result["SETTLEMENTDATE"]).all()


def test_population_standardizer_uses_only_training_rows() -> None:
    frame = pd.DataFrame({"total_qld_population": [4_652_824.0, 4_719_653.0, 9_000_000.0]})
    standardizer = fit_population_standardizer(frame, pd.Series([True, True, False]))
    transformed = standardizer.transform(frame)
    assert transformed.loc[:1, "total_qld_population_z"].mean() == pytest.approx(0.0)
    assert transformed.loc[2, "total_qld_population_z"] > 1.0
