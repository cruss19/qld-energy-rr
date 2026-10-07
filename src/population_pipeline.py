"""Original TCN_starNLL annual Queensland population preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def load_annual_qld_population(project_root: Path) -> pd.DataFrame:
    path = project_root / "data/external/qgso/qld_annual_population_2013_2020.csv"
    frame = pd.read_csv(path, parse_dates=["reference_date", "available_from"])
    required = {"reference_year", "total_qld_population", "reference_date", "available_from"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Annual Queensland population source is missing: {missing}")
    if frame["reference_year"].duplicated().any() or not frame["available_from"].is_monotonic_increasing:
        raise ValueError("Annual Queensland population source is not unique and ordered")
    return frame


def align_annual_population_to_forecast_origins(
    forecast_origins: pd.Series,
    annual_population: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-only join using the original 1-April annual availability rule."""
    left = pd.DataFrame({"SETTLEMENTDATE": pd.to_datetime(forecast_origins)}).sort_values("SETTLEMENTDATE")
    right = annual_population.sort_values("available_from")
    result = pd.merge_asof(
        left,
        right,
        left_on="SETTLEMENTDATE",
        right_on="available_from",
        direction="backward",
        allow_exact_matches=True,
    )
    if result["total_qld_population"].isna().any():
        raise ValueError("Annual Queensland population is unavailable for one or more RR origins")
    if (result["available_from"] > result["SETTLEMENTDATE"]).any():
        raise ValueError("Annual Queensland population alignment admitted a future value")
    return result


@dataclass(frozen=True)
class PopulationStandardizer:
    mean: float
    standard_deviation: float

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["total_qld_population_z"] = (
            result["total_qld_population"] - self.mean
        ) / self.standard_deviation
        return result


def fit_population_standardizer(frame: pd.DataFrame, training_mask: pd.Series) -> PopulationStandardizer:
    training = frame.loc[training_mask, "total_qld_population"].dropna()
    mean = float(training.mean())
    standard_deviation = float(training.std(ddof=0))
    if not standard_deviation > 0:
        raise ValueError("Population standardizer requires variable training-fold data")
    return PopulationStandardizer(mean, standard_deviation)
