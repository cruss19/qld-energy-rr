"""Small, dependency-light helpers for RR data provenance and contract audits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


MANIFEST_COLUMNS = [
    "source_id",
    "relative_path",
    "retrieved_at_utc",
    "byte_size",
    "sha256",
    "coverage_start",
    "coverage_end",
    "acquisition_mode",
    "request_fingerprint",
    "notes",
]


def validate_registry_policy(registry: dict[str, Any]) -> None:
    """Reject any declared source release or coverage that exceeds 2020."""
    policy = registry.get("policy", {})
    cutoff = pd.Timestamp(policy.get("latest_source_release", "2020-12-31"))
    observation_end = pd.Timestamp(policy.get("observation_end", "2020-12-31"))
    errors: list[str] = []
    for source in registry.get("sources", []):
        source_id = source.get("source_id", "<missing>")
        release = source.get("source_release_date")
        if release is not None and pd.Timestamp(release) > cutoff:
            errors.append(f"{source_id}: source release {release} exceeds {cutoff.date()}")
        coverage_end = source.get("coverage_end")
        if coverage_end is not None and pd.Timestamp(coverage_end) > observation_end:
            errors.append(f"{source_id}: coverage end {coverage_end} exceeds {observation_end.date()}")
        for artifact in source.get("artifacts", []):
            relative_path = str(artifact.get("relative_path", ""))
            if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
                errors.append(f"{source_id}: unsafe artifact path {relative_path!r}")
            if not artifact.get("sha256") or artifact.get("byte_size") is None:
                errors.append(f"{source_id}: artifact lacks fixed size/SHA-256: {relative_path}")
    if errors:
        raise ValueError("RR 2020 provenance policy failed:\n- " + "\n- ".join(errors))


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root from a notebook, script, or shell."""
    candidate = (start or Path.cwd()).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "environment.yml").is_file() and (path / "data").is_dir():
            return path
    raise FileNotFoundError("Could not locate the RR repository root.")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    # PyYAML resolves ISO date scalars to ``datetime.date``. Converting those
    # deterministic scalar values to their ISO string representation keeps the
    # request fingerprint stable while still rejecting non-serializable data
    # structures through json.dumps.
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_to_root(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def artifact_record(
    *,
    source_id: str,
    path: Path,
    project_root: Path,
    acquisition_mode: str,
    retrieved_at_utc: str | None = None,
    coverage_start: str | None = None,
    coverage_end: str | None = None,
    request_fingerprint: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_id": source_id,
        "relative_path": relative_to_root(path, project_root),
        "retrieved_at_utc": retrieved_at_utc or datetime.now(timezone.utc).isoformat(),
        "byte_size": stat.st_size,
        "sha256": sha256_file(path),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "acquisition_mode": acquisition_mode,
        "request_fingerprint": request_fingerprint,
        "notes": notes,
    }


def append_manifest_records(path: Path, records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Append identities without deleting earlier acquisition evidence."""
    new = pd.DataFrame(list(records), columns=MANIFEST_COLUMNS)
    if path.exists():
        existing = pd.read_csv(path, dtype="string")
        missing_columns = set(MANIFEST_COLUMNS).difference(existing.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
        combined = pd.concat([existing[MANIFEST_COLUMNS], new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(
        ["source_id", "relative_path", "sha256"], keep="last"
    ).sort_values(["source_id", "relative_path", "retrieved_at_utc"])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined.reset_index(drop=True)


def source_inventory(
    project_root: Path,
    registry: dict[str, Any],
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Report presence and current-checksum registration for every source."""
    if manifest_path is not None and manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype="string")
        missing_columns = set(MANIFEST_COLUMNS).difference(manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    else:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)

    validate_registry_policy(registry)
    rows: list[dict[str, Any]] = []
    for source in registry.get("sources", []):
        acquisition = source["acquisition"]
        expected = source.get("artifacts", [])
        if acquisition == "automatic_api":
            path = project_root / source["canonical_path"]
            matches = [path] if path.is_file() else []
            expected_paths = [source["canonical_path"]]
        elif acquisition in {"pinned_download", "pinned_manual"}:
            expected_paths = [artifact["relative_path"] for artifact in expected]
            matches = [project_root / path for path in expected_paths if (project_root / path).is_file()]
        else:
            raise ValueError(f"Unknown acquisition method: {acquisition}")

        registered_files = 0
        for path in matches:
            relative_path = relative_to_root(path, project_root)
            checksum = sha256_file(path)
            artifact_contract = next(
                (artifact for artifact in expected if artifact["relative_path"] == relative_path),
                None,
            )
            contract_matches = artifact_contract is None or (
                int(artifact_contract["byte_size"]) == path.stat().st_size
                and str(artifact_contract["sha256"]).lower() == checksum.lower()
            )
            registered = manifest.loc[
                manifest["source_id"].eq(source["source_id"])
                & manifest["relative_path"].eq(relative_path)
                & manifest["sha256"].eq(checksum)
            ]
            registered_files += int(not registered.empty and contract_matches)

        if not matches:
            status = "not present locally"
        elif len(matches) == len(expected_paths) and registered_files == len(matches):
            status = "registered and verified"
        else:
            status = "incomplete, unregistered, or contract mismatch"
        rows.append(
            {
                "source_id": source["source_id"],
                "provider": source["provider"],
                "acquisition": acquisition,
                "files_present": len(matches),
                "files_expected": len(expected_paths),
                "files_registered_current": registered_files,
                "bytes_present": sum(p.stat().st_size for p in matches),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def audit_manifest(project_root: Path, manifest_path: Path) -> pd.DataFrame:
    """Recompute size and SHA-256 for every manifest row."""
    if not manifest_path.exists():
        return pd.DataFrame(
            columns=["source_id", "relative_path", "exists", "size_matches", "sha256_matches"]
        )
    manifest = pd.read_csv(manifest_path, dtype="string")
    missing_columns = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    rows = []
    for record in manifest.to_dict("records"):
        path = project_root / record["relative_path"]
        exists = path.is_file()
        size_matches = exists and path.stat().st_size == int(record["byte_size"])
        checksum_matches = exists and sha256_file(path) == record["sha256"]
        coverage_end = pd.to_datetime(record.get("coverage_end"), errors="coerce")
        coverage_within_2020 = pd.isna(coverage_end) or coverage_end <= pd.Timestamp("2020-12-31 23:59:59")
        rows.append(
            {
                "source_id": record["source_id"],
                "relative_path": record["relative_path"],
                "exists": exists,
                "size_matches": size_matches,
                "sha256_matches": checksum_matches,
                "coverage_within_2020": coverage_within_2020,
            }
        )
    return pd.DataFrame(rows)


def duplicate_and_gap_audit(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    frequency: str,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Count duplicate keys and absent timestamps without imputing either."""
    groups = group_columns or []
    working = frame.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="raise")
    key = [*groups, timestamp_column]
    duplicate_rows = int(working.duplicated(key, keep=False).sum())
    summaries = []
    iterator = working.groupby(groups, dropna=False) if groups else [("all", working)]
    for group_name, group in iterator:
        timestamps = pd.DatetimeIndex(group[timestamp_column].dropna().unique()).sort_values()
        if timestamps.empty:
            missing = 0
            start = end = None
        else:
            expected = pd.date_range(timestamps.min(), timestamps.max(), freq=frequency)
            missing = len(expected.difference(timestamps))
            start, end = timestamps.min(), timestamps.max()
        summaries.append(
            {
                "group": group_name,
                "rows": len(group),
                "start": start,
                "end": end,
                "duplicate_key_rows": duplicate_rows if not groups else int(group.duplicated(key).sum()),
                "missing_expected_timestamps": missing,
            }
        )
    return pd.DataFrame(summaries)
