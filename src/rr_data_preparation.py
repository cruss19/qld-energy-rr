"""Validated 2015-2020 source tables for the Resume Repository."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter

from src.population_pipeline import (
    align_annual_population_to_forecast_origins,
    load_annual_qld_population,
)


STUDY_START = pd.Timestamp("2015-01-01 00:00:00")
STUDY_END = pd.Timestamp("2020-12-31 23:59:59")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2020-01-01 00:00:00")

REGIONS = (
    "Brisbane",
    "Cairns / Atherton",
    "Dalby / Chinchilla",
    "Emerald / Gladstone",
    "Townsville / Burdekin",
)

QLD_SCHOOL_TERMS = {
    2015: (("2015-01-27", "2015-04-02"), ("2015-04-20", "2015-06-26"), ("2015-07-13", "2015-09-18"), ("2015-10-06", "2015-12-11")),
    2016: (("2016-01-27", "2016-03-24"), ("2016-04-11", "2016-06-24"), ("2016-07-11", "2016-09-16"), ("2016-10-04", "2016-12-09")),
    2017: (("2017-01-23", "2017-03-31"), ("2017-04-18", "2017-06-23"), ("2017-07-10", "2017-09-15"), ("2017-10-03", "2017-12-08")),
    2018: (("2018-01-22", "2018-03-29"), ("2018-04-17", "2018-06-29"), ("2018-07-16", "2018-09-21"), ("2018-10-08", "2018-12-14")),
    2019: (("2019-01-29", "2019-04-05"), ("2019-04-23", "2019-06-28"), ("2019-07-15", "2019-09-20"), ("2019-10-08", "2019-12-13")),
    2020: (("2020-01-28", "2020-04-03"), ("2020-04-20", "2020-06-26"), ("2020-07-13", "2020-09-18"), ("2020-10-06", "2020-12-11")),
}


def _first_monday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(7 - first.weekday()) % 7)


def _observed_monday(day: date) -> date:
    if day.weekday() == 5:
        return day + timedelta(days=2)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def qld_public_holidays_2015_2020() -> set[date]:
    """Statewide full-day Queensland holidays; local show holidays are excluded."""
    days: set[date] = set()
    for year in range(2015, 2021):
        new_year = date(year, 1, 1)
        days.update((new_year, _observed_monday(new_year)))
        days.add(_observed_monday(date(year, 1, 26)))
        easter_sunday = easter(year)
        days.update((easter_sunday - timedelta(days=2), easter_sunday - timedelta(days=1), easter_sunday + timedelta(days=1)))
        if year >= 2017:
            days.add(easter_sunday)
        anzac = date(year, 4, 25)
        days.add(anzac)
        if anzac.weekday() == 6:
            days.add(anzac + timedelta(days=1))
        if year == 2015:
            days.update((date(2015, 6, 8), date(2015, 10, 5)))
        else:
            days.update((_first_monday(year, 5), _first_monday(year, 10)))
        christmas = date(year, 12, 25)
        boxing_day = date(year, 12, 26)
        days.update((christmas, boxing_day))
        if christmas.weekday() in (5, 6):
            days.add(date(year, 12, 27))
        if boxing_day.weekday() in (5, 6):
            days.add(date(year, 12, 28))
    return days


def assert_study_bounds(values: pd.Series, label: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"{label} contains missing timestamps")
    if timestamps.min() < STUDY_START or timestamps.max() > STUDY_END:
        raise ValueError(f"{label} escapes the RR 2015-2020 boundary")
    return timestamps


def development_only(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Return the 2015-2019 development slice; 2020 is the locked test year."""
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    mask = timestamps.ge(STUDY_START) & timestamps.lt(DEVELOPMENT_END_EXCLUSIVE)
    result = frame.loc[mask].copy()
    if result.empty or pd.to_datetime(result[timestamp_column]).dt.year.ge(2020).any():
        raise ValueError("Development-only slice contains no rows or admits locked 2020 data")
    return result


def prepare_aemo(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "SETTLEMENTDATE", "REGIONID", "DISPATCHINTERVAL", "RUNNO", "INTERVENTION",
        "LASTCHANGED", "TOTALDEMAND", "DEMANDFORECAST", "AVAILABLEGENERATION",
        "NETINTERCHANGE", "UIGF",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"AEMO source is missing columns: {missing}")
    result = frame.copy()
    result["SETTLEMENTDATE"] = assert_study_bounds(result["SETTLEMENTDATE"], "AEMO")
    if not result["REGIONID"].eq("QLD1").all():
        raise ValueError("AEMO source contains a region other than QLD1")
    if result["SETTLEMENTDATE"].duplicated().any():
        raise ValueError("AEMO source contains duplicate physical timestamps")
    if not result["SETTLEMENTDATE"].is_monotonic_increasing:
        result = result.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    numeric = ["TOTALDEMAND", "DEMANDFORECAST", "AVAILABLEGENERATION", "NETINTERCHANGE", "UIGF"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("AEMO numeric fields contain non-finite values")
    return result


def prepare_weather(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"region", "datetime", "latitude", "longitude", "temperature_2m", "relative_humidity_2m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Weather source is missing columns: {missing}")
    result = frame.copy()
    result["datetime"] = assert_study_bounds(result["datetime"], "weather")
    if set(result["region"].dropna().unique()) != set(REGIONS):
        raise ValueError("Weather source does not contain the exact five-region contract")
    if result.duplicated(["region", "datetime"]).any():
        raise ValueError("Weather source contains duplicate region-hour keys")
    return result.sort_values(["region", "datetime"]).reset_index(drop=True)


def build_calendar(timestamps: pd.Series) -> pd.DataFrame:
    ts = assert_study_bounds(timestamps, "calendar spine")
    result = pd.DataFrame({"SETTLEMENTDATE": ts})
    result["hour"] = ts.dt.hour.astype("int8")
    result["day_of_week"] = ts.dt.dayofweek.astype("int8")
    result["month"] = ts.dt.month.astype("int8")
    result["is_weekend"] = ts.dt.dayofweek.ge(5).astype("int8")
    public_days = qld_public_holidays_2015_2020()
    full_day = ts.dt.date.isin(public_days)
    christmas_eve_evening = ts.dt.year.ge(2019) & ts.dt.month.eq(12) & ts.dt.day.eq(24) & ts.dt.hour.ge(18)
    result["is_public_holiday_qld"] = (full_day | christmas_eve_evening).astype("int8")
    in_term = pd.Series(False, index=result.index)
    dates = ts.dt.normalize()
    for terms in QLD_SCHOOL_TERMS.values():
        for start, end in terms:
            in_term |= dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    result["is_school_holiday_qld"] = (~in_term).astype("int8")
    return result


def causal_weather_alignment_audit(aemo_times: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Audit a backward-only hourly-to-five-minute join without materialising a wide table."""
    left = pd.DataFrame({"forecast_origin": pd.to_datetime(aemo_times)}).sort_values("forecast_origin")
    rows = []
    for region in REGIONS:
        right = weather.loc[weather["region"].eq(region), ["datetime"]].sort_values("datetime")
        joined = pd.merge_asof(
            left,
            right,
            left_on="forecast_origin",
            right_on="datetime",
            direction="backward",
            allow_exact_matches=True,
        )
        lag = joined["forecast_origin"] - joined["datetime"]
        rows.append({
            "region": region,
            "forecast_origins": len(joined),
            "unmatched": int(joined["datetime"].isna().sum()),
            "future_weather_rows": int(lag.dropna().lt(pd.Timedelta(0)).sum()),
            "maximum_age_minutes": float(lag.dropna().dt.total_seconds().max() / 60),
        })
    return pd.DataFrame(rows)


def write_validated_tables(project_root: Path, aemo: pd.DataFrame, weather: pd.DataFrame) -> dict[str, Path]:
    """Write source-faithful interim tables and the causal five-minute base table."""
    interim = project_root / "data" / "interim"
    processed = project_root / "data" / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    paths = {
        "aemo": interim / "aemo_qld_5min_validated_2015_2020.parquet",
        "weather": interim / "regional_weather_hourly_validated_2015_2020.parquet",
        "annual_population": interim / "qgso_qld_annual_population_2013_2020.parquet",
        "base": processed / "rr_demand_calendar_base_5min_2015_2020.parquet",
    }
    aemo.to_parquet(paths["aemo"], index=False)
    weather.to_parquet(paths["weather"], index=False)
    annual_population = load_annual_qld_population(project_root)
    annual_population.to_parquet(paths["annual_population"], index=False)
    base = aemo.merge(build_calendar(aemo["SETTLEMENTDATE"]), on="SETTLEMENTDATE", validate="one_to_one")
    population = align_annual_population_to_forecast_origins(
        aemo["SETTLEMENTDATE"], annual_population
    )
    base = base.merge(population, on="SETTLEMENTDATE", validate="one_to_one")
    base.to_parquet(paths["base"], index=False)
    return paths
