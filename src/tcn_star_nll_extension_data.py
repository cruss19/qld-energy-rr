"""Upstream causal helpers and tensors for the RR final model extension.

The donor calculations in this module occur before any TCN sequence is built.
The dataset only replaces each sequence's terminal demand/radiation value with
the already-prepared estimate and routes the aligned estimate-SD series as an
ordinary temporal feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/rr_original_model_frame_5min_2015_2020.parquet"
DEMAND_SOURCE = ROOT / "data/processed/rr_demand_calendar_base_5min_2015_2020.parquet"
WEATHER_SOURCE = ROOT / "data/interim/regional_weather_hourly_validated_2015_2020.parquet"
SEQUENCE_STEPS = 505
HORIZONS = (5, 10, 15, 20, 25, 30)
NODES = (
    "brisbane", "cairns_atherton", "dalby_chinchilla",
    "emerald_gladstone", "townsville_burdekin",
)
REGION_LABELS = {
    "brisbane": "Brisbane",
    "cairns_atherton": "Cairns / Atherton",
    "dalby_chinchilla": "Dalby / Chinchilla",
    "emerald_gladstone": "Emerald / Gladstone",
    "townsville_burdekin": "Townsville / Burdekin",
}
DEMAND_FEATURES = (
    "totaldemand_mw", "demand_lag_5min", "demand_lag_15min",
    "demand_lag_30min", "demand_lag_24h", "demand_lag_7d", "callide_c_status",
)
SYSTEM_FEATURES = (
    "availablegeneration_mw", "availablegeneration_mw_lag_5m",
    "availablegeneration_mw_lag_15m", "availablegeneration_mw_lag_30m",
    "netinterchange_mw", "netinterchange_mw_lag_5m",
    "netinterchange_mw_lag_15m", "netinterchange_mw_lag_30m",
    "uigf_mw", "uigf_mw_lag_5m", "uigf_mw_lag_15m", "uigf_mw_lag_30m",
    "supply_cushion_mw", "supply_cushion_mw_lag_5m",
    "supply_cushion_mw_lag_15m", "supply_cushion_mw_lag_30m",
)
CLIMATE_FEATURES = ("soi_lag_1m", "soi_3m_mean_lag_1m")
REGIONAL_BASE_FEATURES = (
    "dry_bulb_temp", "dry_bulb_temp_lag_5m", "dry_bulb_temp_lag_15m",
    "dry_bulb_temp_lag_30m", "dew_point_2m", "wet_bulb_temperature_2m",
    "cloud_cover", "rainfall_24h_lagged", "regional_installed_capacity_mw", "tts",
)
CALENDAR_FEATURES = (
    "day_of_week", "is_weekend", "is_public_holiday_qld",
    "is_school_holiday_qld", "month", "season",
    "is_day_before_public_holiday", "is_day_after_public_holiday",
    "holiday_type", "time_of_day_sin", "time_of_day_cos",
    "day_of_year_sin", "day_of_year_cos", "is_workday", "is_school_day",
)
CATEGORICAL_CALENDAR = {"day_of_week", "month", "season", "holiday_type"}
TARGETS = tuple(f"target_t_plus_{minutes}m" for minutes in HORIZONS)
REGIONAL_BASE_COLUMNS = tuple(
    f"{node}__{feature}" for node in NODES for feature in REGIONAL_BASE_FEATURES
)


def _same_clock_values(series: pd.Series, timestamps: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(timestamps).to_numpy(dtype=np.float64)


def _nearest_sunday(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    weekday = timestamps.dayofweek.to_numpy()
    previous = (weekday - 6) % 7
    following = (6 - weekday) % 7
    offsets = np.where(previous <= following, -previous, following)
    return timestamps + pd.to_timedelta(offsets, unit="D")


def build_demand_helpers(
    timestamps: pd.DatetimeIndex,
    demand_source: pd.DataFrame,
) -> pd.DataFrame:
    """Build demand_estimated and demand_estimated_sd from three causal donors."""
    source = demand_source.copy()
    source["SETTLEMENTDATE"] = pd.to_datetime(source["SETTLEMENTDATE"])
    source = source.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")
    demand = source.set_index("SETTLEMENTDATE")["TOTALDEMAND"].astype(float)
    holiday = source.set_index("SETTLEMENTDATE")["is_public_holiday_qld"].astype(bool)
    current_holiday = holiday.reindex(timestamps).fillna(False).to_numpy(bool)
    current_weekend = timestamps.dayofweek.to_numpy() >= 5
    donors: list[np.ndarray] = []

    for number in (1, 2, 3):
        weekly = timestamps - pd.Timedelta(days=7 * number)
        # Ordinary weekdays retain the same weekday/time donor. If that donor
        # is a public holiday, the nearest Sunday at the same clock replaces it.
        donor_holiday = holiday.reindex(weekly).fillna(False).to_numpy(bool)
        weekday_donor = pd.DatetimeIndex(
            np.where(donor_holiday, _nearest_sunday(weekly).to_numpy(), weekly.to_numpy())
        )
        # A current public holiday uses the preceding three Sundays, regardless
        # of the weekday on which the holiday itself falls.
        days_since_sunday = (timestamps.dayofweek.to_numpy() - 6) % 7
        previous_sunday = timestamps - pd.to_timedelta(days_since_sunday, unit="D")
        previous_sunday = previous_sunday - pd.to_timedelta(
            np.where(days_since_sunday == 0, 7, 0), unit="D"
        )
        holiday_donor = previous_sunday - pd.Timedelta(days=7 * (number - 1))
        chosen = pd.DatetimeIndex(
            np.where(current_holiday, holiday_donor.to_numpy(), weekday_donor.to_numpy())
        )
        # Non-holiday weekends already use the same weekday/time from each of
        # the preceding three weeks; ``weekly`` makes that rule explicit.
        chosen = pd.DatetimeIndex(
            np.where((~current_holiday & current_weekend), weekly.to_numpy(), chosen.to_numpy())
        )
        donors.append(_same_clock_values(demand, chosen))

    donor_matrix = np.column_stack(donors)
    complete = np.isfinite(donor_matrix).all(axis=1)
    estimated = np.full(len(timestamps), np.nan, dtype=np.float64)
    estimated_sd = np.full(len(timestamps), np.nan, dtype=np.float64)
    estimated[complete] = donor_matrix[complete].mean(axis=1)
    estimated_sd[complete] = donor_matrix[complete].std(axis=1, ddof=1)
    return pd.DataFrame(
        {"demand_estimated": estimated, "demand_estimated_sd": estimated_sd},
        index=timestamps,
    )


def build_radiation_helpers(
    timestamps: pd.DatetimeIndex,
    weather_source: pd.DataFrame,
) -> pd.DataFrame:
    """Build actual, estimated and estimate-SD radiation series per region."""
    weather = weather_source.copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    output = pd.DataFrame(index=timestamps)
    completed_hour = timestamps.floor("h") - pd.Timedelta(hours=1)
    target_hour = timestamps.floor("h")
    for node in NODES:
        rows = weather.loc[weather["region"].eq(REGION_LABELS[node])]
        if rows["datetime"].duplicated().any():
            raise ValueError(f"Duplicate hourly radiation timestamps for {node}")
        radiation = rows.set_index("datetime")["direct_radiation"].astype(float).sort_index()
        actual = _same_clock_values(radiation, completed_hour)
        donors = np.column_stack([
            _same_clock_values(radiation, target_hour - pd.Timedelta(days=days))
            for days in (1, 2, 3)
        ])
        complete = np.isfinite(donors).all(axis=1)
        estimated = np.full(len(timestamps), np.nan, dtype=np.float64)
        estimated_sd = np.full(len(timestamps), np.nan, dtype=np.float64)
        estimated[complete] = donors[complete].mean(axis=1)
        estimated_sd[complete] = donors[complete].std(axis=1, ddof=1)
        output[f"{node}__direct_radiation"] = actual
        output[f"{node}__direct_radiation_estimated"] = estimated
        output[f"{node}__direct_radiation_estimated_sd"] = estimated_sd
    return output


def load_frame(
    source: str | Path = SOURCE,
    demand_source: str | Path = DEMAND_SOURCE,
    weather_source: str | Path = WEATHER_SOURCE,
) -> pd.DataFrame:
    """Load the frozen RR frame and append complete upstream helper series."""
    frame = pd.read_parquet(source)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    if not frame.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("RR extension frame is not a continuous five-minute grid")
    demand_raw = pd.read_parquet(
        demand_source,
        columns=["SETTLEMENTDATE", "TOTALDEMAND", "is_public_holiday_qld"],
    )
    weather_raw = pd.read_parquet(
        weather_source,
        columns=["region", "datetime", "direct_radiation"],
    )
    frame = frame.join(build_demand_helpers(frame.index, demand_raw))
    frame = frame.join(build_radiation_helpers(frame.index, weather_raw))
    helper_columns = ["demand_estimated", "demand_estimated_sd"] + [
        f"{node}__{suffix}"
        for node in NODES
        for suffix in (
            "direct_radiation", "direct_radiation_estimated",
            "direct_radiation_estimated_sd",
        )
    ]
    complete = frame[helper_columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError("No timestamp has a complete causal-helper contract")
    first = int(np.flatnonzero(complete.to_numpy())[0])
    frame = frame.iloc[first:].copy()
    if frame[helper_columns].isna().any().any():
        raise ValueError("Causal helper series contain an interior missing value")
    return frame


def _standardize(frame: pd.DataFrame, columns: tuple[str, ...], fit: np.ndarray):
    values = frame.loc[:, columns].to_numpy(dtype=np.float32)
    mean = values[fit].mean(axis=0)
    scale = values[fit].std(axis=0)
    scale[scale < 1e-6] = 1.0
    transformed = (values - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite value after fold standardisation")
    return transformed.astype(np.float32), {"mean": mean.tolist(), "scale": scale.tolist()}


def _calendar(frame: pd.DataFrame, fit: np.ndarray):
    blocks, state = [], {}
    for feature in CALENDAR_FEATURES:
        if feature in CATEGORICAL_CALENDAR:
            source = frame[feature].astype(str)
            vocabulary = sorted(source.iloc[np.flatnonzero(fit)].unique().tolist())
            known = source.where(source.isin(vocabulary), "__UNKNOWN__")
            for value in [*vocabulary, "__UNKNOWN__"]:
                blocks.append(known.eq(value).to_numpy(dtype=np.float32)[:, None])
            state[feature] = {"vocabulary": vocabulary, "unknown": "__UNKNOWN__"}
        else:
            blocks.append(pd.to_numeric(frame[feature]).to_numpy(dtype=np.float32)[:, None])
            state[feature] = {"representation": "numeric"}
    return np.concatenate(blocks, axis=1), state


def prepare_fold(
    frame: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    *,
    include_sd_channels: bool = True,
):
    """Fit every transformation on the training fold and construct tensors."""
    fit = np.asarray((frame.index >= train_start) & (frame.index < train_end))
    if not fit.any():
        raise ValueError("Training-only scaler partition is empty")
    demand_base, demand_state = _standardize(frame, DEMAND_FEATURES, fit)
    demand_scale = float(demand_state["scale"][0])
    demand_mean = float(demand_state["mean"][0])
    demand_estimated = (
        (frame["demand_estimated"].to_numpy(np.float32) - demand_mean) / demand_scale
    ).astype(np.float32)
    demand_sd = (
        frame["demand_estimated_sd"].to_numpy(np.float32) / demand_scale
    ).astype(np.float32)
    demand = (
        np.column_stack([demand_base, demand_sd])
        if include_sd_channels
        else demand_base
    ).astype(np.float32)

    system, system_state = _standardize(frame, SYSTEM_FEATURES, fit)
    climate, climate_state = _standardize(frame, CLIMATE_FEATURES, fit)
    regional_base, regional_state = _standardize(frame, REGIONAL_BASE_COLUMNS, fit)
    regional_base = regional_base.reshape(len(frame), len(NODES), len(REGIONAL_BASE_FEATURES))
    radiation_actual = frame[[f"{node}__direct_radiation" for node in NODES]].to_numpy(np.float32)
    radiation_estimated = frame[[f"{node}__direct_radiation_estimated" for node in NODES]].to_numpy(np.float32)
    radiation_sd = frame[[f"{node}__direct_radiation_estimated_sd" for node in NODES]].to_numpy(np.float32)
    minimum = radiation_actual[fit].min(axis=0)
    maximum = radiation_actual[fit].max(axis=0)
    span = maximum - minimum
    if np.any(span <= 0):
        raise ValueError("Training-fold radiation range is non-positive")
    radiation_actual = np.clip((radiation_actual - minimum) / span, 0.0, 1.0)
    radiation_estimated = np.clip((radiation_estimated - minimum) / span, 0.0, 1.0)
    radiation_sd = radiation_sd / span
    regional_parts = [regional_base, radiation_actual[:, :, None]]
    if include_sd_channels:
        regional_parts.append(radiation_sd[:, :, None])
    regional = np.concatenate(regional_parts, axis=2).astype(np.float32)

    population, population_state = _standardize(frame, ("total_qld_population",), fit)
    target_fit = np.asarray(fit & (frame.index + pd.Timedelta(minutes=30) < train_end))
    targets, target_state = _standardize(frame, TARGETS, target_fit)
    calendar, calendar_state = _calendar(frame, fit)
    values = {
        "demand": demand,
        "demand_terminal_estimated": demand_estimated,
        "system": system,
        "climate": climate,
        "regional": regional,
        "radiation_terminal_estimated": radiation_estimated.astype(np.float32),
        "calendar": calendar,
        "population": population,
        "targets": targets,
    }
    arrays = {
        key: torch.from_numpy(np.ascontiguousarray(value)).share_memory_()
        for key, value in values.items()
    }
    state = {
        "fit_start": str(frame.index[fit].min()),
        "fit_end": str(frame.index[fit].max()),
        "fit_end_exclusive": str(train_end),
        "helper_contract": {
            "calculated_upstream": True,
            "standard_deviation_ddof": 1,
            "demand_terminal": "demand_estimated[t]",
            "demand_sd_channel": "demand_estimated_sd aligned at every sequence timestamp",
            "radiation_terminal": "regional direct_radiation_estimated[t]",
            "radiation_sd_channel": "regional direct_radiation_estimated_sd aligned at every sequence timestamp",
            "sd_helpers_calculated_upstream": True,
            "sd_channels_in_model": include_sd_channels,
        },
        "demand": demand_state,
        "demand_estimated_scaling": "same mean and scale as totaldemand_mw",
        "demand_estimated_sd_scaling": "divide by totaldemand_mw training scale; no centering",
        "system": system_state,
        "climate": climate_state,
        "regional": regional_state,
        "direct_radiation": {
            "minimum": minimum.tolist(), "maximum": maximum.tolist(),
            "scaling": "per-region training-fold min-max, clipped to [0,1]",
            "estimated_sd_scaling": "divide by matching per-region training range; no centering",
        },
        "calendar": calendar_state,
        "population": population_state,
        "targets": target_state,
        "calendar_encoded_width": int(calendar.shape[1]),
    }
    return arrays, state


class ExtensionDataset(Dataset):
    def __init__(self, arrays: dict[str, torch.Tensor], endpoints: np.ndarray):
        self.arrays = arrays
        self.endpoints = np.asarray(endpoints, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, item: int):
        end = int(self.endpoints[item])
        start = end - SEQUENCE_STEPS + 1
        demand = self.arrays["demand"][start:end + 1].T.clone()
        demand[0, -1] = self.arrays["demand_terminal_estimated"][end]
        regional = self.arrays["regional"][start:end + 1].permute(1, 2, 0).clone()
        regional[:, 10, -1] = self.arrays["radiation_terminal_estimated"][end]
        return (
            demand,
            self.arrays["system"][start:end + 1].T,
            self.arrays["climate"][start:end + 1].T,
            regional,
            self.arrays["calendar"][end],
            self.arrays["population"][end],
            self.arrays["targets"][end],
            torch.tensor(end, dtype=torch.int64),
        )


def fold_endpoints(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    endpoints = np.arange(SEQUENCE_STEPS - 1, len(frame))
    times = frame.index[endpoints]
    eligible = (times >= start) & (times + pd.Timedelta(minutes=30) < end)
    result = endpoints[np.asarray(eligible)]
    if not len(result):
        raise ValueError("Fold partition has no eligible extension endpoints")
    return result


def make_loader(arrays, endpoints, *, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        ExtensionDataset(arrays, endpoints),
        batch_size=128,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


__all__ = [
    "NODES", "SEQUENCE_STEPS", "HORIZONS", "ExtensionDataset",
    "build_demand_helpers", "build_radiation_helpers", "load_frame",
    "prepare_fold", "fold_endpoints", "make_loader",
]
