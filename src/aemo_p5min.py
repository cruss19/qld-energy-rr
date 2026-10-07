"""Acquire and validate the public AEMO P5MIN QLD forecast vintages."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


RAW_COLUMNS = (
    "RUN_DATETIME",
    "INTERVAL_DATETIME",
    "REGIONID",
    "TOTALDEMAND",
    "DEMANDFORECAST",
    "LASTCHANGED",
)
PROCESSED_COLUMNS = (
    "forecast_origin",
    "target_timestamp",
    "horizon_minutes",
    "region_id",
    "forecast_demand_mw",
    "demandforecast_delta_mw",
    "last_changed",
    "source_month",
    "source_url",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_urls(config: dict[str, Any], year: int, month: int) -> list[str]:
    values = {"year": year, "month": f"{month:02d}"}
    return [
        config["historical_table_url_template"].format(**values),
        config["table_archive_url_template"].format(**values),
        config["legacy_archive_url_template"].format(**values),
    ]


def download_first_available(urls: list[str], destination: Path) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            with requests.get(
                url,
                stream=True,
                timeout=300,
                headers={"User-Agent": "qld-energy-rr/1.0"},
            ) as response:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            return url
        except requests.RequestException as error:
            last_error = error
            if destination.exists():
                destination.unlink()
    if last_error is not None:
        raise RuntimeError(f"No AEMO archive URL succeeded: {urls}") from last_error
    raise FileNotFoundError(f"No AEMO archive URL exists: {urls}")


def extract_qld_rows(archive: Path, region_id: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {archive.name}; found {members}")
        with zipped.open(members[0]) as binary:
            reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
            indices: dict[str, int] | None = None
            for record in reader:
                if len(record) < 5 or record[1:3] != ["P5MIN", "REGIONSOLUTION"]:
                    continue
                if record[0] == "I":
                    indices = {name.upper(): index for index, name in enumerate(record)}
                    missing = set(RAW_COLUMNS).difference(indices)
                    if missing:
                        raise ValueError(f"{archive.name} lacks AEMO fields {sorted(missing)}")
                    continue
                if record[0] != "D" or indices is None:
                    continue
                if record[indices["REGIONID"]] != region_id:
                    continue
                rows.append({column: record[indices[column]] for column in RAW_COLUMNS})
    if not rows:
        raise ValueError(f"No {region_id} P5MIN_REGIONSOLUTION rows in {archive.name}")
    frame = pd.DataFrame(rows)
    for column in ("RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in ("TOTALDEMAND", "DEMANDFORECAST"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["RUN_DATETIME", "INTERVAL_DATETIME", "TOTALDEMAND"])


def filter_contract_rows(
    frame: pd.DataFrame,
    *,
    source_month: str,
    source_url: str,
    region_id: str,
    horizons_minutes: tuple[int, ...],
) -> pd.DataFrame:
    result = frame.copy()
    result["horizon_minutes"] = (
        (result["INTERVAL_DATETIME"] - result["RUN_DATETIME"]).dt.total_seconds() / 60.0
    ).round().astype("int16")
    result = result.loc[
        result["REGIONID"].eq(region_id)
        & result["horizon_minutes"].isin(horizons_minutes)
        & result["RUN_DATETIME"].between("2015-01-01", "2020-12-31 23:55:00")
        & result["INTERVAL_DATETIME"].between("2015-01-01", "2020-12-31 23:55:00")
    ].copy()
    result = (
        result.sort_values("LASTCHANGED")
        .drop_duplicates(["RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"], keep="last")
        .sort_values(["RUN_DATETIME", "INTERVAL_DATETIME"])
    )
    result = result.rename(
        columns={
            "RUN_DATETIME": "forecast_origin",
            "INTERVAL_DATETIME": "target_timestamp",
            "REGIONID": "region_id",
            "TOTALDEMAND": "forecast_demand_mw",
            "DEMANDFORECAST": "demandforecast_delta_mw",
            "LASTCHANGED": "last_changed",
        }
    )
    result["source_month"] = source_month
    result["source_url"] = source_url
    return result[list(PROCESSED_COLUMNS)].reset_index(drop=True)


def acquire_all_months(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_dir = project_root / config["monthly_directory"]
    monthly_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / config["source_manifest"]
    existing = pd.read_csv(manifest_path, dtype="string") if manifest_path.exists() else pd.DataFrame()
    manifest_by_month = (
        {str(row["month"]): row for row in existing.to_dict("records")} if not existing.empty else {}
    )
    frames: list[pd.DataFrame] = []
    horizons = tuple(int(value) for value in config["horizons_minutes"])
    for year in range(int(config["start_year"]), int(config["end_year"]) + 1):
        for month in range(1, 13):
            key = f"{year}-{month:02d}"
            destination = monthly_dir / f"p5min_regionsolution_qld1_{year}{month:02d}.parquet"
            if destination.exists():
                monthly = pd.read_parquet(destination)
                record = manifest_by_month.get(key)
                if record is None or str(record.get("filtered_parquet_sha256")) != sha256_file(destination):
                    raise ValueError(f"Existing P5MIN month is unregistered or changed: {destination}")
            else:
                with tempfile.TemporaryDirectory(prefix="rr_aemo_p5min_") as temporary:
                    archive = Path(temporary) / f"p5min_{year}{month:02d}.zip"
                    url = download_first_available(archive_urls(config, year, month), archive)
                    archive_sha256 = sha256_file(archive)
                    raw = extract_qld_rows(archive, config["region_id"])
                    monthly = filter_contract_rows(
                        raw,
                        source_month=key,
                        source_url=url,
                        region_id=config["region_id"],
                        horizons_minutes=horizons,
                    )
                if monthly.empty:
                    raise ValueError(f"AEMO P5MIN month {key} produced no contracted rows")
                monthly.to_parquet(destination, index=False)
                manifest_by_month[key] = {
                    "month": key,
                    "source_url": url,
                    "source_zip_sha256": archive_sha256,
                    "filtered_parquet": destination.relative_to(project_root).as_posix(),
                    "filtered_parquet_sha256": sha256_file(destination),
                    "filtered_rows": len(monthly),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                # Persist progress month-by-month so an interrupted six-year
                # acquisition can resume without accepting an unregistered file.
                pd.DataFrame(list(manifest_by_month.values())).sort_values("month").to_csv(
                    manifest_path, index=False
                )
            frames.append(monthly)
            time.sleep(0.05)
    manifest = pd.DataFrame(list(manifest_by_month.values())).sort_values("month")
    if set(manifest["month"]) != {f"{y}-{m:02d}" for y in range(2015, 2021) for m in range(1, 13)}:
        raise ValueError("P5MIN monthly manifest is incomplete")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values("last_changed")
        .drop_duplicates(["forecast_origin", "target_timestamp", "region_id"], keep="last")
        .sort_values(["forecast_origin", "target_timestamp"])
        .reset_index(drop=True)
    )
    return combined, manifest


def write_canonical_outputs(
    project_root: Path, config: dict[str, Any], frame: pd.DataFrame
) -> tuple[Path, Path]:
    canonical = project_root / config["canonical_output"]
    processed = project_root / config["processed_output"]
    canonical.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(canonical, index=False)
    frame.to_parquet(processed, index=False)
    return canonical, processed


def validate_processed(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    missing = sorted(set(PROCESSED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"P5MIN processed artifact lacks columns: {missing}")
    if frame.duplicated(["forecast_origin", "target_timestamp", "region_id"]).any():
        raise ValueError("P5MIN processed artifact contains duplicate origin-target rows")
    if set(frame["region_id"].astype(str)) != {config["region_id"]}:
        raise ValueError("P5MIN processed artifact is not QLD1-only")
    horizons = set(int(value) for value in config["horizons_minutes"])
    if set(frame["horizon_minutes"].astype(int).unique()) != horizons:
        raise ValueError("P5MIN processed artifact does not contain exactly six horizons")
    for column in ("forecast_origin", "target_timestamp"):
        values = pd.to_datetime(frame[column], errors="raise")
        if values.min() < pd.Timestamp("2015-01-01") or values.max() > pd.Timestamp(
            "2020-12-31 23:55:00"
        ):
            raise ValueError(f"{column} breaches the RR 2015-2020 boundary")
