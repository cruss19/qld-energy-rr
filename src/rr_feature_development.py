"""Build the exact public RR TCN_starNLL feature contract from RR sources.

All joins are causal and every source is bounded at 2020. Fold-fitted scaling
and categorical vocabularies remain outside this source-frame builder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .tcn_star_nll_sources import REGION_NODES, build_publication_safe_soi


HORIZON_MINUTES = (5, 10, 15, 20, 25, 30)
REGION_LABELS = {
    "brisbane": "Brisbane",
    "cairns_atherton": "Cairns / Atherton",
    "dalby_chinchilla": "Dalby / Chinchilla",
    "emerald_gladstone": "Emerald / Gladstone",
    "townsville_burdekin": "Townsville / Burdekin",
}
REGIONAL_FEATURES = (
    "dry_bulb_temp", "dry_bulb_temp_lag_5m", "dry_bulb_temp_lag_15m",
    "dry_bulb_temp_lag_30m", "dew_point_2m", "wet_bulb_temperature_2m",
    "cloud_cover", "rainfall_24h_lagged", "regional_installed_capacity_mw", "tts",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    frame_path: Path
    contract_path: Path
    summary_path: Path
    contract_sha256: str
    rows: int
    eligible_development_rows: int
    eligible_2020_rows: int


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_feature_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "rr_public_feature_contract_v1":
        raise ValueError("Unexpected public feature contract identity")
    return contract, _canonical_sha256(contract)


def _time_to_sunset_index(index: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    from pvlib.solarposition import sun_rise_set_transit_spa

    aware = index.tz_localize("Australia/Brisbane") if index.tz is None else index.tz_convert("Australia/Brisbane")
    # pandas 3 may retain microsecond resolution while pvlib returns
    # nanoseconds; normalize units before integer arithmetic.
    aware = aware.as_unit("ns")
    days = aware.normalize()
    event_days = pd.date_range(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1), freq="D", tz=aware.tz)
    events = sun_rise_set_transit_spa(event_days + pd.Timedelta(hours=12), latitude=latitude, longitude=longitude, how="numpy")
    events.index = event_days
    position = event_days.get_indexer(days)
    sunrise = pd.DatetimeIndex(events["sunrise"].iloc[position]).asi8
    sunset = pd.DatetimeIndex(events["sunset"].iloc[position]).asi8
    previous_sunset = pd.DatetimeIndex(events["sunset"].iloc[position - 1]).asi8
    next_sunrise = pd.DatetimeIndex(events["sunrise"].iloc[position + 1]).asi8
    values = aware.asi8
    result = np.empty(len(aware), dtype=np.float64)
    daylight = (values >= sunrise) & (values <= sunset)
    result[daylight] = (sunset[daylight] - values[daylight]) / (sunset[daylight] - sunrise[daylight])
    before = values < sunrise
    before_mid = previous_sunset + (sunrise - previous_sunset) // 2
    first = before & (values <= before_mid)
    result[first] = -(values[first] - previous_sunset[first]) / (before_mid[first] - previous_sunset[first])
    second = before & ~first
    result[second] = -(sunrise[second] - values[second]) / (sunrise[second] - before_mid[second])
    after = values > sunset
    after_mid = sunset + (next_sunrise - sunset) // 2
    first = after & (values <= after_mid)
    result[first] = -(values[first] - sunset[first]) / (after_mid[first] - sunset[first])
    second = after & ~first
    result[second] = -(next_sunrise[second] - values[second]) / (next_sunrise[second] - after_mid[second])
    return np.clip(result, -1.0, 1.0)


def _holiday_features(index: pd.DatetimeIndex, path: Path) -> pd.DataFrame:
    periods = pd.read_csv(path, parse_dates=["start", "end"])
    result = pd.DataFrame(index=index)
    result["holiday_type"] = "None"
    for row in periods.itertuples(index=False):
        result.loc[(index >= row.start) & (index <= row.end), "holiday_type"] = row.holiday_type
    result["is_public_holiday_qld"] = result["holiday_type"].ne("None").astype("int8")
    holiday_dates = set(index[result["is_public_holiday_qld"].eq(1)].normalize())
    # Deterministic right-boundary support only: this preserves the 2020-12-31
    # day-before-holiday flag without admitting any 2021 observation row.
    holiday_dates.add(pd.Timestamp("2021-01-01"))
    dates = index.normalize()
    result["is_day_before_public_holiday"] = pd.Index(dates + pd.Timedelta(days=1)).isin(holiday_dates).astype("int8")
    result["is_day_after_public_holiday"] = pd.Index(dates - pd.Timedelta(days=1)).isin(holiday_dates).astype("int8")
    return result


def _capacity(index: pd.DatetimeIndex, source: pd.DataFrame, region: str) -> np.ndarray:
    rows = source.loc[source["region"].eq(region)].sort_values("Date").copy()
    first = rows.iloc[0]
    history = pd.DataFrame({
        "available_from": pd.concat([pd.Series([pd.Timestamp(first["Date"]).to_period("M").to_timestamp()]), rows["available_from"]], ignore_index=True),
        "capacity": pd.concat([pd.Series([(first["Cumulative_Capacity_kW"] - first["Rated_Power_Output_kW"]) / 1000.0]), rows["Cumulative_Capacity_kW"] / 1000.0], ignore_index=True),
    }).sort_values("available_from")
    aligned = pd.merge_asof(pd.DataFrame({"datetime": index}), history, left_on="datetime", right_on="available_from", direction="backward")
    if aligned["capacity"].isna().any():
        raise ValueError(f"Capacity does not cover {region}")
    return aligned["capacity"].to_numpy(float)


def _regional_features(index: pd.DatetimeIndex, weather: pd.DataFrame, capacity: pd.DataFrame) -> list[pd.DataFrame]:
    parts = []
    for region, label in REGION_LABELS.items():
        station = weather.loc[weather["region"].eq(label)].sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
        expanded = station.index.union(index).sort_values()
        aligned = station.reindex(expanded).ffill().reindex(index)
        temperature = pd.to_numeric(aligned["temperature_2m"], errors="raise")
        hourly_rain = pd.to_numeric(station["precipitation"], errors="raise")
        rainfall = hourly_rain.rolling(24, min_periods=24).sum().shift(1)
        rainfall = rainfall.reindex(rainfall.index.union(index).sort_values()).ffill().reindex(index)
        latitude, longitude = REGION_NODES[region]
        node = pd.DataFrame({
            "dry_bulb_temp": temperature.to_numpy(float),
            "dry_bulb_temp_lag_5m": temperature.shift(1).to_numpy(float),
            "dry_bulb_temp_lag_15m": temperature.shift(3).to_numpy(float),
            "dry_bulb_temp_lag_30m": temperature.shift(6).to_numpy(float),
            "dew_point_2m": aligned["dew_point_2m"].to_numpy(float),
            "wet_bulb_temperature_2m": aligned["wet_bulb_temperature_2m"].to_numpy(float),
            "cloud_cover": aligned["cloud_cover"].to_numpy(float),
            "rainfall_24h_lagged": rainfall.to_numpy(float),
            "regional_installed_capacity_mw": _capacity(index, capacity, region),
            "tts": _time_to_sunset_index(index, latitude, longitude),
        }, index=index)
        node.columns = [f"{region}__{column}" for column in REGIONAL_FEATURES]
        parts.append(node)
    return parts


def build_origin_frame(base: pd.DataFrame, contract: dict[str, Any], *, project_root: Path | None = None,
                       weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the exact original RR model frame; ``project_root`` supplies auxiliary sources."""
    base = base.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE").copy()
    index = pd.DatetimeIndex(pd.to_datetime(base["SETTLEMENTDATE"]))
    out = pd.DataFrame(index=index)
    out["totaldemand_mw"] = base["TOTALDEMAND"].to_numpy(float)
    for name, steps in (("demand_lag_5min", 1), ("demand_lag_15min", 3), ("demand_lag_30min", 6), ("demand_lag_24h", 288), ("demand_lag_7d", 2016)):
        out[name] = out["totaldemand_mw"].shift(steps)
    out["callide_c_status"] = np.ones(len(out), dtype=np.float32)  # no Callide outage occurs in 2015--2020
    mapping = {"availablegeneration_mw": "AVAILABLEGENERATION", "netinterchange_mw": "NETINTERCHANGE", "uigf_mw": "UIGF"}
    for feature, source in mapping.items():
        out[feature] = base[source].to_numpy(float)
    # Preserve the original information timing: the generation availability
    # term is the last closed five-minute interval, while demand is current.
    out["supply_cushion_mw"] = out["availablegeneration_mw"].shift(1) - out["totaldemand_mw"]
    for feature in (*mapping, "supply_cushion_mw"):
        for suffix, steps in (("lag_5m", 1), ("lag_15m", 3), ("lag_30m", 6)):
            out[f"{feature}_{suffix}"] = out[feature].shift(steps)

    for horizon in HORIZON_MINUTES:
        out[f"target_t_plus_{horizon}m"] = out["totaldemand_mw"].shift(-(horizon // 5)) - out["totaldemand_mw"]

    if project_root is None:
        # Retained for unit tests that exercise the target and causal base.
        target_columns = [f"target_t_plus_{horizon}m" for horizon in HORIZON_MINUTES]
        out["eligible_common"] = (
            out["demand_lag_7d"].notna()
            & out[target_columns].notna().all(axis=1)
            & pd.Series(base["INTERVENTION"].to_numpy(), index=index).eq(0)
        )
        return out.reset_index(names="forecast_origin")
    safe_soi = build_publication_safe_soi(index, project_root / contract["sources"]["monthly_soi"])
    out[["soi_lag_1m", "soi_3m_mean_lag_1m"]] = safe_soi.to_numpy(float)
    holiday = _holiday_features(index, project_root / contract["sources"]["public_holiday_periods"])
    out["day_of_week"] = pd.Categorical(index.day_name())
    out["is_weekend"] = (index.dayofweek >= 5).astype("int8")
    out["is_public_holiday_qld"] = holiday["is_public_holiday_qld"].to_numpy("int8")
    out["is_school_holiday_qld"] = base["is_school_holiday_qld"].to_numpy("int8")
    out["month"] = pd.Categorical(index.month_name())
    out["season"] = pd.Categorical(pd.Series(index.month, index=index).map({12:"Summer",1:"Summer",2:"Summer",3:"Autumn",4:"Autumn",5:"Autumn",6:"Winter",7:"Winter",8:"Winter",9:"Spring",10:"Spring",11:"Spring"}))
    out["is_day_before_public_holiday"] = holiday["is_day_before_public_holiday"].to_numpy("int8")
    out["is_day_after_public_holiday"] = holiday["is_day_after_public_holiday"].to_numpy("int8")
    out["holiday_type"] = pd.Categorical(holiday["holiday_type"])
    minute = index.hour * 60 + index.minute
    out["time_of_day_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    out["time_of_day_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    out["is_workday"] = ((index.dayofweek < 5) & out["is_public_holiday_qld"].eq(0)).astype("int8")
    out["is_school_day"] = (out["is_workday"].eq(1) & out["is_school_holiday_qld"].eq(0)).astype("int8")
    out["total_qld_population"] = base["total_qld_population"].to_numpy(float)

    capacity = pd.read_parquet(project_root / contract["sources"]["regional_solar_capacity"])
    out = out.join(_regional_features(index, weather, capacity))
    required = [feature for group in contract["feature_groups"].values() for feature in group["features"]]
    required += [f"{region}__{feature}" for region in contract["regional_weather"]["regions"] for feature in contract["regional_weather"]["variables"]]
    required += [f"target_t_plus_{horizon}m" for horizon in HORIZON_MINUTES]
    valid = out[required].notna().all(axis=1)
    first, last = np.flatnonzero(valid)[[0, -1]]
    if not valid.iloc[first:last + 1].all():
        raise ValueError("Exact public feature frame contains internal missing values")
    out = out.iloc[first:last + 1][required]
    out.insert(0, "datetime", out.index)
    out["eligible_common"] = True
    return out.reset_index(drop=True)


def expanded_contract_rows(contract: dict[str, Any], sha256: str) -> pd.DataFrame:
    rows = []
    for context, group in contract["feature_groups"].items():
        for position, feature in enumerate(group["features"]):
            rows.append({"contract_id": contract["contract_id"], "contract_sha256": sha256, "context": context,
                         "position": position, "feature": feature, "route": group["route"],
                         "availability": group["availability"], "fold_transformation": group["fold_transformation"]})
    for region_position, region in enumerate(contract["regional_weather"]["regions"]):
        for position, feature in enumerate(contract["regional_weather"]["variables"]):
            rows.append({"contract_id": contract["contract_id"], "contract_sha256": sha256,
                         "context": "regional_weather", "region_position": region_position, "position": position,
                         "feature": f"{region}__{feature}", "route": contract["regional_weather"]["route"],
                         "availability": contract["regional_weather"]["availability"],
                         "fold_transformation": contract["regional_weather"]["fold_transformation"]})
    return pd.DataFrame(rows)


def build_public_feature_artifacts(project_root: Path) -> FeatureBuildResult:
    contract_path = project_root / "config/public_feature_contract.yml"
    contract, sha256 = load_feature_contract(contract_path)
    base = pd.read_parquet(project_root / contract["sources"]["demand_calendar_base"])
    weather = pd.read_parquet(project_root / contract["sources"]["regional_weather"])
    if pd.DatetimeIndex(base["SETTLEMENTDATE"]).max() > pd.Timestamp("2020-12-31 23:55"):
        raise ValueError("Demand source exceeds 2020")
    if pd.DatetimeIndex(weather["datetime"]).max() > pd.Timestamp("2020-12-31 23:00"):
        raise ValueError("Weather source exceeds 2020")
    frame = build_origin_frame(base, contract, project_root=project_root, weather=weather)
    processed = project_root / "data/processed/rr_original_model_frame_5min_2015_2020.parquet"
    table = project_root / "data/contracts/rr_public_feature_contract_v1.csv"
    summary_path = project_root / "outputs/03_feature_contract/feature_preflight_summary.json"
    for path in (processed, table, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed, index=False)
    expanded_contract_rows(contract, sha256).to_csv(table, index=False)
    development = frame["datetime"] < pd.Timestamp("2020-01-01")
    final = frame["datetime"].dt.year.eq(2020)
    summary = {"status": "PASS", "contract_id": contract["contract_id"], "contract_sha256": sha256,
               "rows": len(frame), "columns": len(frame.columns), "first_origin": frame["datetime"].min().isoformat(),
               "last_origin": frame["datetime"].max().isoformat(), "eligible_development_rows": int(development.sum()),
               "eligible_2020_rows": int(final.sum()), "post_2020_rows": 0,
               "population_2_present": any("population_2" in column.lower() for column in frame.columns)}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return FeatureBuildResult(processed, table, summary_path, sha256, len(frame), int(development.sum()), int(final.sum()))
