"""Prepare public, 2015--2020-only assets for the RR reference-model notebook.

This utility is deliberately a migration/preparation tool, not a model trainer.
It filters the authoritative historical feature frame and preserved reference
results in memory before writing anything to the Resume Repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


START = pd.Timestamp("2015-01-01")
END_EXCLUSIVE = pd.Timestamp("2021-01-01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-source", type=Path, required=True)
    parser.add_argument("--results-source", type=Path, required=True)
    parser.add_argument("--rr-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_source = args.results_source / "expanding_window_model_feature_contract.csv"
    contract = pd.read_csv(contract_source)
    included = contract["include_full_ridge"].astype(str).str.lower().isin({"true", "1"})
    contract = contract.loc[included].copy()
    if len(contract) != 55:
        raise RuntimeError(f"Expected 55 original Ridge features, found {len(contract)}")

    # The private source contract labels the origin demand with a later event-
    # study description. That event is outside the RR period. The numerical
    # series is unchanged; only the public metadata is made period-appropriate.
    demand_row = contract["canonical_feature_name"].eq("totaldemand_mw")
    contract.loc[demand_row, "required_transformation"] = (
        "Observed Queensland operational demand at forecast origin"
    )

    features = contract["canonical_feature_name"].tolist()
    source_columns = set(pq.read_schema(args.feature_source).names)
    required = [
        "datetime",
        "totaldemand_mw",
        "target_ramp_next_30min",
        "interval_of_day",
        "is_weekend",
        "is_public_holiday_qld",
        "month",
        "season",
        *features,
    ]
    required = list(dict.fromkeys(required))
    missing = sorted(set(required) - source_columns)
    if missing:
        raise RuntimeError(f"Authoritative feature source is missing: {missing}")

    frame = pd.read_parquet(args.feature_source, columns=required)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame = frame.loc[frame.datetime.ge(START) & frame.datetime.lt(END_EXCLUSIVE)].copy()
    frame = frame.sort_values("datetime").reset_index(drop=True)
    if frame.empty or frame.datetime.min() < START or frame.datetime.max() >= END_EXCLUSIVE:
        raise RuntimeError("Public feature-frame temporal boundary failed")
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise RuntimeError("Public feature frame must have unique, sorted timestamps")

    data_dir = args.rr_root / "data" / "processed"
    config_dir = args.rr_root / "config"
    output_dir = args.rr_root / "outputs" / "reference_models"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_output = data_dir / "rr_reference_model_frame_5min_2015_2020.parquet"
    contract_output = config_dir / "reference_model_feature_contract.csv"
    frame.to_parquet(feature_output, index=False)
    contract.to_csv(contract_output, index=False)

    csv_files = [
        "expanding_window_metrics_long.csv",
        "expanding_window_fold_sample_sizes.csv",
        "expanding_window_ridge_alphas.csv",
        "expanding_window_ridge_inner_scores.csv",
        "expanding_window_bayesian_diagnostics.csv",
        "expanding_window_comparable_history_fallbacks.csv",
    ]
    for filename in csv_files:
        source = args.results_source / filename
        table = pd.read_csv(source)
        if "evaluation_year" in table.columns:
            table = table.loc[table.evaluation_year.le(2020)].copy()
        if "fold" in table.columns:
            table = table.loc[table.fold.le(4)].copy()
        table.to_csv(output_dir / filename, index=False)

    predictions_source = args.results_source / "expanding_window_oof_predictions.parquet"
    predictions = pd.read_parquet(predictions_source)
    predictions = predictions.loc[predictions.evaluation_year.le(2020)].copy()
    predictions["forecast_origin"] = pd.to_datetime(predictions["forecast_origin"], errors="raise")
    if predictions.forecast_origin.max() >= END_EXCLUSIVE:
        raise RuntimeError("Post-2020 prediction escaped the public filter")
    predictions.to_parquet(output_dir / "expanding_window_oof_predictions.parquet", index=False)

    manifest = {
        "contract": "Original five 30-minute reference models",
        "public_window": {"start": str(START), "end_exclusive": str(END_EXCLUSIVE)},
        "feature_rows": int(len(frame)),
        "feature_columns": int(frame.shape[1]),
        "ridge_base_features": int(len(features)),
        "evaluation_years": sorted(int(v) for v in predictions.evaluation_year.unique()),
        "prediction_rows": int(len(predictions)),
        "source_checksums": {
            "feature_source_sha256": sha256(args.feature_source),
            "feature_contract_source_sha256": sha256(contract_source),
            "predictions_source_sha256": sha256(predictions_source),
        },
        "privacy_boundary": "All written data and results are restricted to 2015-01-01 through 2020-12-31.",
    }
    (output_dir / "reference_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
