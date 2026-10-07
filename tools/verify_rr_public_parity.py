"""Maintainer-only historical parity verification for the RR public frame.

This utility requires the deliberately undistributed frozen reconstruction
cache under ``training_output/private_cache``. A clean clone cannot run it.
Successful execution writes only a small, sanitized public parity manifest;
it never publishes the frozen comparison frame.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rr_feature_development import load_feature_contract


EXPECTED_PUBLIC_ONLY_COLUMNS = {"callide_c_status", "eligible_common"}
EXPECTED_COMPARED_COLUMNS = 96
EXPECTED_ELIGIBLE_ORIGINS = 629_273


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = ROOT
    public_path = root / "data/processed/rr_original_model_frame_5min_2015_2020.parquet"
    frozen_path = root / "training_output/private_cache/rr_original_model_frame_2015_2020.parquet"
    if not frozen_path.exists():
        raise FileNotFoundError(
            "Maintainer-only frozen comparison cache is unavailable. "
            "Use outputs/03_feature_contract/public_parity_manifest.json "
            "for the retained historical verification result."
        )

    public_raw = pd.read_parquet(public_path)
    frozen_raw = pd.read_parquet(frozen_path)
    if list(public_raw.columns).count("datetime") != 1 or list(frozen_raw.columns).count("datetime") != 1:
        raise RuntimeError("Both parity frames must contain exactly one datetime key")
    public = public_raw.set_index("datetime")
    frozen = frozen_raw.set_index("datetime")

    public_only = set(public.columns).difference(frozen.columns)
    frozen_only_columns = set(frozen.columns).difference(public.columns)
    compared_columns = [column for column in public.columns if column in frozen.columns]
    if public_only != EXPECTED_PUBLIC_ONLY_COLUMNS or frozen_only_columns:
        raise RuntimeError(
            f"Unexpected parity schema difference: public_only={sorted(public_only)}, "
            f"frozen_only={sorted(frozen_only_columns)}"
        )
    if compared_columns != list(frozen.columns):
        raise RuntimeError("Frozen comparison columns or ordering changed")
    if len(compared_columns) != EXPECTED_COMPARED_COLUMNS:
        raise RuntimeError(f"Expected {EXPECTED_COMPARED_COLUMNS} compared columns")

    shared_rows = public.index.intersection(frozen.index)
    if not shared_rows.equals(public.index) or len(shared_rows) != EXPECTED_ELIGIBLE_ORIGINS:
        raise RuntimeError("Public parity row identity changed")
    frozen_only_rows = frozen.index.difference(public.index)
    expected_terminal = pd.date_range("2020-12-31 23:30", "2020-12-31 23:55", freq="5min")
    if not frozen_only_rows.equals(expected_terminal):
        raise RuntimeError(f"Unexpected timestamp difference: {list(frozen_only_rows)}")

    mismatches: list[tuple[str, int, float]] = []
    for column in compared_columns:
        left = public.loc[shared_rows, column]
        right = frozen.loc[shared_rows, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(
                left.to_numpy(float), right.to_numpy(float),
                equal_nan=True, rtol=1e-10, atol=1e-10,
            )
        else:
            equal = (
                left.astype("string").fillna("<NA>").to_numpy()
                == right.astype("string").fillna("<NA>").to_numpy()
            )
        if not bool(np.all(equal)):
            mismatches.append((column, int((~equal).sum()), float((~equal).mean())))
    if mismatches:
        raise RuntimeError(f"RR public feature parity failed: {mismatches}")

    contract, contract_hash = load_feature_contract(root / "config/public_feature_contract.yml")
    manifest = {
        "verification_status": "PASS",
        "verification_scope": (
            "Historical maintainer verification against an undistributed frozen "
            "reconstruction cache; not independently rerunnable from a clean clone."
        ),
        "verification_date": date.today().isoformat(),
        "authoritative_contract_id": contract["contract_id"],
        "authoritative_contract_sha256": contract_hash,
        "public_frame_rows": int(len(public)),
        "public_frame_columns_including_key": int(len(public_raw.columns)),
        "eligible_origin_count": int(len(shared_rows)),
        "compared_columns": len(compared_columns),
        "mismatched_columns": 0,
        "expected_terminal_rows_excluded_for_incomplete_targets": 6,
        "public_only_columns_not_part_of_historical_comparison": sorted(public_only),
        "public_frame_sha256": sha256(public_path),
        "frozen_comparison_artifact_sha256": sha256(frozen_path),
        "frozen_comparison_artifact_distributed": False,
    }
    output = root / "outputs/03_feature_contract/public_parity_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
