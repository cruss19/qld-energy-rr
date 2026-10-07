"""Register locally obtained RR source files in the immutable artifact manifest."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_provenance import (  # noqa: E402
    append_manifest_records,
    artifact_record,
    audit_manifest,
    load_yaml,
    sha256_file,
    source_inventory,
    stable_fingerprint,
    validate_registry_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--retrieved-at-utc",
        help=(
            "ISO-8601 retrieval timestamp with UTC offset. If omitted, the current "
            "UTC registration time is used and disclosed in the notes."
        ),
    )
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def normalise_retrieval_timestamp(value: str | None) -> tuple[str, str | None]:
    if value is None:
        return (
            datetime.now(timezone.utc).isoformat(),
            "Exact retrieval time not supplied; retrieved_at_utc records manual registration time.",
        )
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("--retrieved-at-utc must include Z or an explicit UTC offset.")
    return timestamp.tz_convert("UTC").isoformat(), None


def main() -> int:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / "config" / "data_acquisition.yml")
    registry_path = PROJECT_ROOT / config["paths"]["source_registry_path"]
    manifest_path = PROJECT_ROOT / config["paths"]["manifest_path"]
    registry = load_yaml(registry_path)
    validate_registry_policy(registry)

    matches = [
        source
        for source in registry.get("sources", [])
        if source.get("source_id") == args.source_id
    ]
    if len(matches) != 1:
        valid = sorted(
            source["source_id"]
            for source in registry.get("sources", [])
            if source.get("acquisition") == "pinned_manual"
        )
        raise ValueError(f"Unknown source_id {args.source_id!r}. Manual choices: {valid}")
    source = matches[0]
    if source.get("acquisition") != "pinned_manual":
        raise ValueError(
            f"{args.source_id!r} is not a manual source; use acquire_rr_data.py instead."
        )

    expected = source.get("artifacts", [])
    files = sorted(
        (PROJECT_ROOT / artifact["relative_path"]).resolve()
        for artifact in expected
        if (PROJECT_ROOT / artifact["relative_path"]).is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"No frozen artifacts are present for {args.source_id}."
        )
    if len(files) != len(expected):
        raise FileNotFoundError(
            f"Only {len(files)} of {len(expected)} frozen artifacts are present for {args.source_id}."
        )
    root = PROJECT_ROOT.resolve()
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not accepted as source artifacts: {path}")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Source artifact escaped the RR root: {path}") from exc

    retrieved_at_utc, timestamp_note = normalise_retrieval_timestamp(
        args.retrieved_at_utc
    )
    notes = "; ".join(
        part for part in [args.notes.strip(), timestamp_note] if part
    ) or None
    registry_fingerprint = stable_fingerprint(source)
    records = []
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        contract = next(item for item in expected if item["relative_path"] == relative_path)
        if path.stat().st_size != int(contract["byte_size"]) or sha256_file(path).lower() != str(contract["sha256"]).lower():
            raise ValueError(f"Frozen source identity mismatch: {relative_path}")
        records.append(artifact_record(
            source_id=args.source_id,
            path=path,
            project_root=PROJECT_ROOT,
            acquisition_mode="pinned_manual",
            retrieved_at_utc=retrieved_at_utc,
            coverage_start=str(source["coverage_start"]),
            coverage_end=str(source["coverage_end"]),
            request_fingerprint=registry_fingerprint,
            notes=notes,
        ))
    manifest = append_manifest_records(manifest_path, records)

    audit = audit_manifest(PROJECT_ROOT, manifest_path)
    source_audit = audit.loc[audit["source_id"].eq(args.source_id)]
    if source_audit.empty or not source_audit[
        ["exists", "size_matches", "sha256_matches"]
    ].all().all():
        raise RuntimeError("Manual source registration failed its immediate checksum audit.")

    inventory = source_inventory(PROJECT_ROOT, registry, manifest_path)
    source_inventory_row = inventory.loc[inventory["source_id"].eq(args.source_id)]
    if source_inventory_row.iloc[0]["status"] != "registered and verified":
        raise RuntimeError("Not every current file for the manual source is registered.")

    print(f"Registered {len(files)} file(s) for {args.source_id}.")
    print(source_inventory_row.to_string(index=False))
    print("\nChecksum audit")
    print(source_audit.to_string(index=False))
    print(f"\nManifest records: {len(manifest):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
