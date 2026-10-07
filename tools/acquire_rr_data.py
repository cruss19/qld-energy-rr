"""Acquire or verify RR source data without silently overwriting artifacts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_provenance import (  # noqa: E402
    append_manifest_records,
    artifact_record,
    audit_manifest,
    load_yaml,
    source_inventory,
    sha256_file,
    stable_fingerprint,
    validate_registry_policy,
)
from src.aemo_p5min import (  # noqa: E402
    acquire_all_months as acquire_aemo_p5min_months,
    validate_processed as validate_aemo_p5min,
    write_canonical_outputs as write_aemo_p5min_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("verify", "missing"),
        default=os.environ.get("RR_ACQUISITION_MODE", "verify"),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=(
            "aemo_dispatchregionsum",
            "aemo_p5min_regionsolution",
            "open_meteo_hourly",
            "abs_asgs_boundaries",
            "cer_sres_postcode_capacity",
            "bom_soi_monthly",
        ),
        default=(
            "aemo_dispatchregionsum",
            "aemo_p5min_regionsolution",
            "open_meteo_hourly",
            "abs_asgs_boundaries",
            "cer_sres_postcode_capacity",
            "bom_soi_monthly",
        ),
    )
    return parser.parse_args()


def assert_network_authorised(mode: str) -> None:
    if mode == "verify":
        return
    if os.environ.get("RR_ALLOW_NETWORK") != "1":
        raise RuntimeError(
            "Network acquisition is disabled. Set RR_ALLOW_NETWORK=1 explicitly "
            "after reviewing the source requests."
        )


def destination_for(canonical: Path, mode: str) -> Path:
    if mode == "missing":
        if canonical.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {canonical}")
        return canonical
    raise ValueError(f"No acquisition destination for mode={mode}")


def acquire_pinned_artifact(artifact: dict[str, Any]) -> Path:
    """Download one exact pre-2021 artifact and admit it only after identity checks."""
    destination = PROJECT_ROOT / artifact["relative_path"]
    if destination.exists():
        if (
            destination.stat().st_size == int(artifact["byte_size"])
            and sha256_file(destination).lower() == str(artifact["sha256"]).lower()
        ):
            return destination
        raise FileExistsError(f"Existing artifact does not match its frozen contract: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        with requests.get(artifact["url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        actual_size = temporary.stat().st_size
        actual_sha = sha256_file(temporary).lower()
        if actual_size != int(artifact["byte_size"]) or actual_sha != str(artifact["sha256"]).lower():
            raise ValueError(
                f"Frozen-source identity mismatch for {artifact['relative_path']}: "
                f"size={actual_size}, sha256={actual_sha}"
            )
        temporary.replace(destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_aemo_csvs(cache_directory: Path, selected_columns: list[str], start: str, end: str) -> pd.DataFrame:
    csv_files = sorted(
        path
        for path in cache_directory.rglob("*.csv")
        if "DISPATCHREGIONSUM" in path.name.upper()
    )
    if not csv_files:
        raise FileNotFoundError("No cached AEMO DISPATCHREGIONSUM CSV files were found.")
    frames = []
    for file_path in csv_files:
        header_row_number = None
        report_columns = None
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
            for row_number, row in enumerate(csv.reader(source)):
                if len(row) >= 5 and row[:3] == ["I", "DISPATCH", "REGIONSUM"]:
                    header_row_number = row_number
                    report_columns = row[4:]
                    break
        if report_columns is None or header_row_number is None:
            raise ValueError(f"AEMO header not found in {file_path.name}")
        missing = sorted(set(selected_columns).difference(report_columns))
        if missing:
            raise ValueError(f"{file_path.name} is missing AEMO fields: {missing}")
        positions = {0: "record_type"}
        positions.update({4 + report_columns.index(column): column for column in selected_columns})
        use_positions = sorted(positions)
        monthly = pd.read_csv(
            file_path,
            header=None,
            skiprows=header_row_number + 1,
            usecols=use_positions,
            dtype="string",
            low_memory=False,
        )
        monthly.columns = [positions[position] for position in use_positions]
        frames.append(
            monthly.loc[
                monthly["record_type"].eq("D") & monthly["REGIONID"].eq("QLD1"),
                selected_columns,
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    result["SETTLEMENTDATE"] = pd.to_datetime(result["SETTLEMENTDATE"], errors="raise")
    return result.loc[
        result["SETTLEMENTDATE"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    ].copy()


def acquire_aemo(config: dict[str, Any], destination: Path, cache_directory: Path) -> pd.DataFrame:
    from nemosis import dynamic_data_compiler

    selected = list(config["selected_columns"])
    cache_directory.mkdir(parents=True, exist_ok=True)
    dynamic_data_compiler(
        start_time=config["start_time"],
        end_time=config["end_time"],
        table_name=config["table"],
        raw_data_location=str(cache_directory),
        select_columns=["SETTLEMENTDATE", "REGIONID"],
        filter_cols=["REGIONID"],
        filter_values=([config["region_id"]],),
        keep_csv=True,
    )
    frame = _read_aemo_csvs(
        cache_directory,
        selected,
        config["start_time"],
        config["end_time"],
    )
    frame["LASTCHANGED"] = pd.to_datetime(frame["LASTCHANGED"], errors="coerce")
    numeric = [column for column in selected if column not in {"SETTLEMENTDATE", "REGIONID", "LASTCHANGED"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    full_key = ["SETTLEMENTDATE", "REGIONID", "DISPATCHINTERVAL", "RUNNO", "INTERVENTION"]
    frame = frame.sort_values("LASTCHANGED", na_position="first").drop_duplicates(full_key, keep="last")
    frame["physical_priority"] = frame["INTERVENTION"].eq(1).astype("int8")
    frame = (
        frame.sort_values(["SETTLEMENTDATE", "physical_priority", "LASTCHANGED"], na_position="first")
        .drop_duplicates("SETTLEMENTDATE", keep="last")
        .drop(columns="physical_priority")
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )
    if frame["SETTLEMENTDATE"].duplicated().any() or not frame["REGIONID"].eq(config["region_id"]).all():
        raise ValueError("AEMO physical-series selection failed its uniqueness or region contract.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame


def fetch_open_meteo(endpoint: str, params: dict[str, Any], max_retries: int) -> dict[str, Any]:
    for attempt in range(max_retries):
        response = requests.get(endpoint, params=params, timeout=180)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(15 * 2**attempt, 300)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Open-Meteo rate limiting persisted after all configured retries.")


def acquire_open_meteo(config: dict[str, Any], destination: Path) -> pd.DataFrame:
    variables = list(config["hourly_variables"])
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])
    frames = []
    for node in config["project_nodes"]:
        node_frames = []
        for year in range(start.year, end.year + 1):
            chunk_start = max(start, pd.Timestamp(year, 1, 1))
            chunk_end = min(end, pd.Timestamp(year, 12, 31))
            params = {
                "latitude": node["latitude"],
                "longitude": node["longitude"],
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": ",".join(variables),
                "timezone": config["timezone"],
            }
            payload = fetch_open_meteo(config["endpoint"], params, int(config["max_retries"]))
            hourly = payload["hourly"]
            chunk = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"], errors="raise")})
            for variable in variables:
                chunk[variable] = hourly[variable]
            chunk.insert(0, "region", node["region"])
            chunk.insert(1, "latitude", node["latitude"])
            chunk.insert(2, "longitude", node["longitude"])
            chunk.insert(3, "open_meteo_elevation_m", payload.get("elevation"))
            node_frames.append(chunk)
            time.sleep(float(config["request_pause_seconds"]))
        frames.append(pd.concat(node_frames, ignore_index=True))
    result = pd.concat(frames, ignore_index=True).sort_values(["region", "datetime"])
    if result.duplicated(["region", "datetime"]).any():
        raise ValueError("Open-Meteo result contains duplicate region-hour keys.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return result


def acquire_bom_soi(config: dict[str, Any], destination: Path) -> pd.DataFrame:
    """Persist only the contracted through-2020 subset of the official BOM series.

    The response is parsed in memory.  Post-cutoff observations are rejected from
    the persisted artifact rather than being downloaded to a raw RR file and then
    deleted.  This keeps the RR data tree within its public 2020 boundary.
    """
    response = requests.get(
        config["endpoint"],
        headers={"User-Agent": "qld-energy-rr/1.0 (public reproducibility research)"},
        timeout=180,
    )
    response.raise_for_status()
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in response.text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or len(fields[0]) != 6:
            continue
        month = pd.to_datetime(fields[0], format="%Y%m", errors="raise")
        value = float(fields[1])
        rows.append((month, value))
    result = pd.DataFrame(rows, columns=config["persisted_columns"])
    start = pd.Timestamp(config["observation_start"])
    end = pd.Timestamp(config["observation_end"])
    result = result.loc[result["month"].between(start, end, inclusive="both")].copy()
    result = result.sort_values("month").drop_duplicates("month", keep="last")
    if result.empty or result["month"].max() > pd.Timestamp("2020-12-31"):
        raise ValueError("BOM SOI persisted artifact violates the RR 2020 cutoff")
    if result["month"].min() > pd.Timestamp("2014-10-01"):
        raise ValueError("BOM SOI does not contain enough pre-2015 history for lagging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return result


def main() -> int:
    args = parse_args()
    assert_network_authorised(args.mode)
    config = load_yaml(PROJECT_ROOT / "config" / "data_acquisition.yml")
    registry = load_yaml(PROJECT_ROOT / config["paths"]["source_registry_path"])
    validate_registry_policy(registry)
    manifest_path = PROJECT_ROOT / config["paths"]["manifest_path"]

    print(source_inventory(PROJECT_ROOT, registry, manifest_path).to_string(index=False))
    if args.mode == "verify":
        audit = audit_manifest(PROJECT_ROOT, manifest_path)
        print("\nManifest audit")
        print(audit.to_string(index=False) if not audit.empty else "No artifact manifest has been created yet.")
        return 0

    records = []
    source_lookup = {source["source_id"]: source for source in registry["sources"]}
    for source_id in args.sources:
        if source_id == "aemo_dispatchregionsum":
            source_config = config["aemo_dispatchregionsum"]
            canonical = PROJECT_ROOT / source_config["canonical_output"]
            if args.mode == "missing" and canonical.exists():
                destination = canonical
                frame = pd.read_parquet(destination)
                print(f"Verifying present artifact: {canonical}")
            else:
                destination = destination_for(canonical, args.mode)
                cache = PROJECT_ROOT / source_config["cache_directory"]
                frame = acquire_aemo(source_config, destination, cache)
            coverage_column = "SETTLEMENTDATE"
            fingerprint = stable_fingerprint(source_config)
        elif source_id == "aemo_p5min_regionsolution":
            source_config = config["aemo_p5min_regionsolution"]
            canonical = PROJECT_ROOT / source_config["canonical_output"]
            processed = PROJECT_ROOT / source_config["processed_output"]
            if args.mode == "missing" and canonical.exists() and processed.exists():
                destination = canonical
                frame = pd.read_parquet(canonical)
                processed_frame = pd.read_parquet(processed)
                validate_aemo_p5min(frame, source_config)
                validate_aemo_p5min(processed_frame, source_config)
                if len(frame) != len(processed_frame):
                    raise ValueError("Canonical and processed P5MIN artifacts differ in row count")
                print(f"Verifying present artifacts: {canonical} and {processed}")
            else:
                if canonical.exists() != processed.exists():
                    raise FileExistsError(
                        "P5MIN canonical/processed pair is incomplete; refusing silent replacement"
                    )
                frame, monthly_manifest = acquire_aemo_p5min_months(PROJECT_ROOT, source_config)
                validate_aemo_p5min(frame, source_config)
                destination, processed = write_aemo_p5min_outputs(
                    PROJECT_ROOT, source_config, frame
                )
                print(f"Registered {len(monthly_manifest):,} AEMO P5MIN source months")
            coverage_column = "forecast_origin"
            fingerprint = stable_fingerprint(source_config)
        elif source_id == "open_meteo_hourly":
            source_config = config["open_meteo"]
            canonical = PROJECT_ROOT / source_config["canonical_output"]
            if args.mode == "missing" and canonical.exists():
                destination = canonical
                frame = pd.read_parquet(destination)
                print(f"Verifying present artifact: {canonical}")
            else:
                destination = destination_for(canonical, args.mode)
                frame = acquire_open_meteo(source_config, destination)
            coverage_column = "datetime"
            fingerprint = stable_fingerprint(source_config)
        elif source_id == "bom_soi_monthly":
            source_config = config["bom_soi"]
            canonical = PROJECT_ROOT / source_config["canonical_output"]
            if args.mode == "missing" and canonical.exists():
                destination = canonical
                frame = pd.read_csv(destination, parse_dates=["month"])
                print(f"Verifying present artifact: {canonical}")
            else:
                destination = destination_for(canonical, args.mode)
                frame = acquire_bom_soi(source_config, destination)
            coverage_column = "month"
            fingerprint = stable_fingerprint(source_config)
        else:
            source = source_lookup[source_id]
            fingerprint = stable_fingerprint(source)
            for artifact in source["artifacts"]:
                destination = acquire_pinned_artifact(artifact)
                records.append(
                    artifact_record(
                        source_id=source_id,
                        path=destination,
                        project_root=PROJECT_ROOT,
                        acquisition_mode="pinned_download",
                        coverage_start=str(source["coverage_start"]),
                        coverage_end=str(source["coverage_end"]),
                        request_fingerprint=fingerprint,
                        notes=f"Frozen source release: {source['source_release_date']}",
                    )
                )
                print(f"Verified {destination.relative_to(PROJECT_ROOT)}")
            continue
        timestamps = pd.to_datetime(frame[coverage_column], errors="raise")
        if source_id == "bom_soi_monthly":
            if timestamps.max() > pd.Timestamp(source_config["observation_end"]):
                raise ValueError("BOM SOI persisted artifact exceeds its contracted source month")
        elif timestamps.min() < pd.Timestamp("2015-01-01") or timestamps.max() > pd.Timestamp("2020-12-31 23:59:59"):
            raise ValueError(f"{source_id} contains observations outside 2015-2020")
        records.append(
            artifact_record(
                source_id=source_id,
                path=destination,
                project_root=PROJECT_ROOT,
                acquisition_mode=args.mode,
                coverage_start=str(frame[coverage_column].min()),
                coverage_end=str(frame[coverage_column].max()),
                request_fingerprint=fingerprint,
            )
        )
        if source_id == "aemo_p5min_regionsolution":
            records.append(
                artifact_record(
                    source_id=source_id,
                    path=PROJECT_ROOT / source_config["processed_output"],
                    project_root=PROJECT_ROOT,
                    acquisition_mode=args.mode,
                    coverage_start=str(frame[coverage_column].min()),
                    coverage_end=str(frame[coverage_column].max()),
                    request_fingerprint=fingerprint,
                    notes="Processed six-horizon QLD1 origin-vintage benchmark artifact",
                )
            )
        print(f"Created {destination.relative_to(PROJECT_ROOT)} ({len(frame):,} rows)")

    if records:
        manifest = append_manifest_records(manifest_path, records)
        print(f"Updated {manifest_path.relative_to(PROJECT_ROOT)} ({len(manifest):,} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
