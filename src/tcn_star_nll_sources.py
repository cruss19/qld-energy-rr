"""Source-faithful auxiliary inputs for the RR TCN_starNLL reconstruction.

This module contains only the two source families that were missing from the
initial RR build: BOM monthly SOI and CER postcode-level small-scale solar
capacity.  It never reads or writes observations after the RR 2020 boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


REGION_NODES = {
    "brisbane": (-27.3842, 153.1175),
    "cairns_atherton": (-16.8736, 145.7458),
    "dalby_chinchilla": (-27.1553, 151.2675),
    "emerald_gladstone": (-23.5675, 148.1792),
    "townsville_burdekin": (-19.2525, 146.7653),
}
MONTH_PATTERN = re.compile(r"^([A-Za-z]{3}\s+\d{4}).*Rated Output In kW$", re.I)


def build_publication_safe_soi(
    forecast_origins: pd.DatetimeIndex,
    source_csv: Path,
) -> pd.DataFrame:
    """Return the original two SOI channels on a five-minute origin spine.

    The preserved original contract admits a source-month value at 19:00
    Brisbane time on the first day of the following month. This reproduces the
    historical one-month-lagged SOI alignment used by the frozen public runs.
    """
    source = pd.read_csv(source_csv, parse_dates=["month"])
    source["month"] = pd.to_datetime(source["month"]).astype("datetime64[ns]")
    if list(source.columns) != ["month", "soi"]:
        raise ValueError("Unexpected BOM SOI persisted schema")
    source = source.sort_values("month").drop_duplicates("month", keep="last")
    if source["month"].max() > pd.Timestamp("2020-11-01"):
        raise ValueError("SOI source exceeds the contracted final source month")
    source["soi_lag_1m"] = source["soi"].astype("float64")
    source["soi_3m_mean_lag_1m"] = source["soi"].rolling(3, min_periods=3).mean()
    source["available_from"] = source["month"] + pd.offsets.MonthBegin(1) + pd.Timedelta(hours=19)
    source["available_from"] = pd.to_datetime(source["available_from"]).astype("datetime64[ns]")
    aligned = pd.merge_asof(
        pd.DataFrame({"forecast_origin": pd.DatetimeIndex(forecast_origins).astype("datetime64[ns]")}),
        source[["available_from", "soi_lag_1m", "soi_3m_mean_lag_1m"]],
        left_on="forecast_origin",
        right_on="available_from",
        direction="backward",
        allow_exact_matches=True,
    )
    result = aligned[["soi_lag_1m", "soi_3m_mean_lag_1m"]]
    result.index = forecast_origins
    if result.isna().any().any():
        raise ValueError("SOI alignment leaves missing values")
    return result


def _normalise_postcode(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\.0$", "", str(value).strip())
    digits = re.sub(r"\D", "", text)
    return digits.zfill(4) if digits else None


def build_poa_region_crosswalk(poa_zip: Path) -> pd.DataFrame:
    """Assign each Queensland 2016 POA to its nearest original project node."""
    import geopandas as gpd
    from shapely.geometry import Point

    poa = gpd.read_file(f"zip://{poa_zip}")
    code_fields = [c for c in poa.columns if c.upper().startswith("POA_CODE")]
    if len(code_fields) != 1:
        raise ValueError(f"Expected one POA code field, found {code_fields}")
    poa["postcode"] = poa[code_fields[0]].map(_normalise_postcode).astype("string")
    poa = poa.loc[poa["postcode"].str.fullmatch(r"4\d{3}", na=False)].copy()
    poa = poa.to_crs("EPSG:3577")
    nodes = gpd.GeoDataFrame(
        {"region": list(REGION_NODES)},
        geometry=[Point(longitude, latitude) for latitude, longitude in REGION_NODES.values()],
        crs="EPSG:4326",
    ).to_crs("EPSG:3577")
    representatives = poa.geometry.representative_point()
    node_xy = np.column_stack((nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()))
    poa_xy = np.column_stack((representatives.x.to_numpy(), representatives.y.to_numpy()))
    distances = ((poa_xy[:, None, :] - node_xy[None, :, :]) ** 2).sum(axis=2)
    poa["region"] = nodes.iloc[np.argmin(distances, axis=1)]["region"].to_numpy()
    result = poa[["postcode", "region"]].drop_duplicates("postcode")
    if result["postcode"].duplicated().any() or set(result["region"]) != set(REGION_NODES):
        raise ValueError("POA-to-region crosswalk failed its identity checks")
    return result.sort_values("postcode").reset_index(drop=True)


def _solar_sheet_name(workbook: Path) -> str:
    names = pd.ExcelFile(workbook).sheet_names
    matches = [name for name in names if "solar" in name.lower() and "sgu" in name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one SGU solar sheet in {workbook.name}: {matches}")
    return matches[0]


def _read_solar_workbook(workbook: Path) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name=_solar_sheet_name(workbook), header=None, dtype=object)
    first = raw.iloc[:, 0].astype("string").str.strip().str.lower()
    rows = raw.index[first.str.contains("postcode", na=False)]
    if len(rows) != 1:
        raise ValueError(f"Could not identify one postcode header in {workbook.name}")
    header = int(rows[0])
    populated = raw.loc[header].notna()
    frame = raw.loc[header + 1 :, populated].copy()
    frame.columns = raw.loc[header, populated].astype(str).str.strip()
    postcode = next(column for column in frame if "postcode" in column.lower())
    frame = frame.rename(columns={postcode: "postcode"}).dropna(how="all")
    frame["postcode"] = frame["postcode"].map(_normalise_postcode).astype("string")
    return frame.loc[frame["postcode"].str.fullmatch(r"4\d{3}", na=False)].reset_index(drop=True)


def _monthly_capacity_by_postcode(frame: pd.DataFrame) -> pd.DataFrame:
    monthly: dict[pd.Timestamp, str] = {}
    for column in frame.columns:
        match = MONTH_PATTERN.match(str(column).strip())
        if match:
            month = pd.to_datetime(match.group(1), format="%b %Y") + pd.offsets.MonthEnd(0)
            monthly[month] = column
    if not monthly:
        raise ValueError("CER solar workbook contains no monthly capacity columns")
    blocks = []
    for month, column in monthly.items():
        block = frame[["postcode"]].copy()
        block["Date"] = month
        block["Rated_Power_Output_kW"] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def build_regional_solar_capacity(project_root: Path) -> pd.DataFrame:
    """Build the original monthly five-region cumulative-capacity channel."""
    cer = project_root / "data" / "external" / "cer"
    vintages = {
        2015: cer / "cer_sres_2016_all_data.xlsx",
        2016: cer / "cer_sres_2017_all_data.xlsx",
        2017: cer / "cer_sres_2018_all_data.xlsx",
        2018: cer / "cer_sres_2018_all_data.xlsx",
        2019: cer / "cer_sres_all_data_as_at_2020-10-31.xlsx",
        2020: cer / "cer_sres_all_data_as_at_2020-10-31.xlsx",
    }
    crosswalk = build_poa_region_crosswalk(
        project_root / "data" / "external" / "abs" / "asgs" / "POA_2016_AUST_SHP.zip"
    )
    overrides = pd.read_csv(
        project_root / "config" / "cer_postcode_overrides.csv", dtype="string"
    )[["postcode", "region"]]
    overrides["postcode"] = overrides["postcode"].map(_normalise_postcode).astype("string")
    crosswalk = pd.concat([crosswalk, overrides], ignore_index=True)
    crosswalk = crosswalk.drop_duplicates("postcode", keep="last")
    source_frames = {path: _read_solar_workbook(path) for path in set(vintages.values())}

    # The 2016 release's explicit 2001-2014 aggregate is the pre-study baseline.
    baseline_frame = source_frames[vintages[2015]]
    baseline_columns = [
        column for column in baseline_frame
        if str(column).lower().startswith("previous years") and "rated output" in str(column).lower()
    ]
    if len(baseline_columns) != 1:
        raise ValueError(f"Expected one pre-2015 baseline column: {baseline_columns}")
    baseline = baseline_frame[["postcode"]].copy()
    baseline["baseline_kw"] = pd.to_numeric(
        baseline_frame[baseline_columns[0]], errors="coerce"
    ).fillna(0.0)
    baseline = baseline.merge(crosswalk, on="postcode", how="left", validate="many_to_one")
    if baseline.loc[baseline["region"].isna(), "baseline_kw"].abs().sum() > 1e-6:
        raise ValueError("Positive pre-2015 CER capacity has no POA allocation")
    baseline_by_region = baseline.groupby("region")["baseline_kw"].sum()

    parts = []
    for year, workbook in vintages.items():
        monthly = _monthly_capacity_by_postcode(source_frames[workbook])
        monthly = monthly.loc[monthly["Date"].dt.year.eq(year)].copy()
        monthly = monthly.merge(crosswalk, on="postcode", how="left", validate="many_to_one")
        unmatched = monthly.loc[monthly["region"].isna(), "Rated_Power_Output_kW"].abs().sum()
        if unmatched > 1e-6:
            raise ValueError(f"Positive CER capacity lacks POA allocation for {year}: {unmatched}")
        parts.append(monthly.groupby(["region", "Date"], as_index=False)["Rated_Power_Output_kW"].sum())
    additions = pd.concat(parts, ignore_index=True)
    complete_index = pd.MultiIndex.from_product(
        [list(REGION_NODES), pd.date_range("2015-01-31", "2020-12-31", freq="ME")],
        names=["region", "Date"],
    )
    result = additions.set_index(["region", "Date"]).reindex(complete_index, fill_value=0.0).reset_index()
    result["Cumulative_Capacity_kW"] = (
        result.groupby("region")["Rated_Power_Output_kW"].cumsum()
        + result["region"].map(baseline_by_region).fillna(0.0)
    )
    result["available_from"] = result["Date"] + pd.offsets.Day(1)
    result["Date"] = pd.to_datetime(result["Date"]).astype("datetime64[ns]")
    result["available_from"] = pd.to_datetime(result["available_from"]).astype("datetime64[ns]")
    if result["Date"].max() > pd.Timestamp("2020-12-31") or len(result) != 5 * 72:
        raise ValueError("Regional solar capacity violates the 2015-2020 contract")
    return result.sort_values(["region", "Date"]).reset_index(drop=True)


def write_regional_solar_capacity(project_root: Path) -> Path:
    output = project_root / "data" / "processed" / "SGU_Solar_monthly_2015_2020.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    build_regional_solar_capacity(project_root).to_parquet(output, index=False)
    return output


__all__ = [
    "REGION_NODES",
    "build_publication_safe_soi",
    "build_regional_solar_capacity",
    "write_regional_solar_capacity",
]
