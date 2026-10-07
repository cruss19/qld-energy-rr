"""Build the RR four-year weekday seasonal demand-profile figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/rr_demand_calendar_base_5min_2015_2020.parquet"
OUTPUT = ROOT / "outputs/04_development_eda"
PROFILE_PATH = OUTPUT / "weekday_duck_belly_profiles_2016_2017_2019_2020.csv"
FIGURE_PATH = OUTPUT / "weekday_duck_belly_2016_2017_2019_2020.png"
YEARS = (2016, 2017, 2019, 2020)
SEASONS = ("Summer", "Winter")
PALETTE = {
    (2016, "Summer"): "#E69F00",
    (2016, "Winter"): "#56B4E9",
    (2017, "Summer"): "#D1492E",
    (2017, "Winter"): "#21618C",
    (2019, "Summer"): "#9B4A9A",
    (2019, "Winter"): "#168765",
    (2020, "Summer"): "#8C2D04",
    (2020, "Winter"): "#264653",
}
LINESTYLES = {2016: ":", 2017: "-", 2019: "--", 2020: "-."}


def season_window(year: int, season: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if season == "Summer":
        return (
            pd.Timestamp(year - 1, 12, 1),
            pd.Timestamp(year, 3, 1),
            f"Dec {year - 1}–Feb {year}",
        )
    return pd.Timestamp(year, 6, 1), pd.Timestamp(year, 9, 1), f"Jun–Aug {year}"


def build_profiles(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["SETTLEMENTDATE"] = pd.to_datetime(frame["SETTLEMENTDATE"])
    if frame["SETTLEMENTDATE"].duplicated().any():
        raise ValueError("Duplicate Queensland demand timestamps")
    frame["date"] = frame["SETTLEMENTDATE"].dt.normalize()
    frame["minute_of_day"] = (
        frame["SETTLEMENTDATE"].dt.hour * 60 + frame["SETTLEMENTDATE"].dt.minute
    )
    frame = frame.loc[frame["SETTLEMENTDATE"].dt.weekday.lt(5)]

    profiles: list[pd.DataFrame] = []
    for year in YEARS:
        for season in SEASONS:
            start, end, label = season_window(year, season)
            part = frame.loc[
                frame["SETTLEMENTDATE"].ge(start)
                & frame["SETTLEMENTDATE"].lt(end)
            ]
            weekdays = int(part["date"].nunique())
            observations = part.groupby("minute_of_day").size()
            if weekdays < 60 or len(observations) != 288 or not observations.eq(weekdays).all():
                raise ValueError(f"Incomplete five-minute weekday coverage in {label}")
            profile = part.groupby("minute_of_day", as_index=False)["TOTALDEMAND"].mean()
            profile = profile.rename(columns={"TOTALDEMAND": "mean_operational_demand_mw"})
            profile["year"] = year
            profile["season"] = season
            profile["season_dates"] = label
            profile["weekdays"] = weekdays
            profiles.append(profile)
    result = pd.concat(profiles, ignore_index=True)
    if not result.groupby(["year", "season"]).size().eq(288).all():
        raise ValueError("Expected 288 five-minute values for every profile")
    return result


def save_figure(profiles: pd.DataFrame) -> None:
    lower = np.floor(profiles["mean_operational_demand_mw"].min() / 500) * 500 - 250
    upper = np.ceil(profiles["mean_operational_demand_mw"].max() / 500) * 500 + 250
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for year in YEARS:
        for season in SEASONS:
            part = profiles.loc[
                profiles["year"].eq(year) & profiles["season"].eq(season)
            ]
            ax.plot(
                part["minute_of_day"] / 60,
                part["mean_operational_demand_mw"],
                color=PALETTE[(year, season)],
                linewidth=2.3,
                linestyle=LINESTYLES[year],
                label=f"{year} {season.lower()} ({part['season_dates'].iloc[0]})",
            )
    ax.set(
        title="Queensland average weekday duck-belly profiles: 2016, 2017, 2019 and 2020",
        xlabel="Time of day (AEST)",
        ylabel="Mean operational demand (MW)",
        xlim=(0, 24),
        ylim=(lower, upper),
    )
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> int:
    raw = pd.read_parquet(SOURCE, columns=["SETTLEMENTDATE", "TOTALDEMAND"])
    profiles = build_profiles(raw)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILE_PATH, index=False)
    save_figure(profiles)
    coverage = profiles.groupby(["year", "season", "season_dates"], as_index=False).agg(
        weekdays=("weekdays", "first"),
        minimum_mean_mw=("mean_operational_demand_mw", "min"),
        maximum_mean_mw=("mean_operational_demand_mw", "max"),
    )
    print(coverage.to_string(index=False))
    print(FIGURE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
