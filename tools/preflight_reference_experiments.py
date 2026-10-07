"""Maintainer-only validation of reference contracts and preserved evidence.

This utility requires undistributed processed data and row-level historical
prediction artifacts. It is not part of the public clean-clone execution path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def preflight(root: Path) -> dict:
    config_dir = root / "config" / "experiments"
    paths = sorted(config_dir.glob("reference_[0-9][0-9]_*.yml"))
    jobs = [load_yaml(path) for path in paths]
    if len(jobs) != 9 or len({job["job_id"] for job in jobs}) != 9:
        raise ValueError("Expected nine unique reference job declarations")

    endpoint_jobs = [job for job in jobs if job["model_name"] != "aemo_p5min_external_benchmark"]
    if any(job["major_parameters"]["horizon_minutes"] != 30 for job in endpoint_jobs):
        raise ValueError("Original and probabilistic references must remain 30-minute models")

    contract = pd.read_csv(root / "config" / "reference_model_feature_contract.csv")
    included = contract["include_full_ridge"].astype(str).str.lower().isin({"true", "1"})
    if int(included.sum()) != 55:
        raise ValueError("Original Ridge contract must contain exactly 55 base predictors")

    frame_path = root / "data" / "processed" / "rr_reference_model_frame_5min_2015_2020.parquet"
    required = {"datetime", "totaldemand_mw", "target_ramp_next_30min"}
    missing = sorted(required.difference(pq.read_schema(frame_path).names))
    if missing:
        raise ValueError(f"Reference frame missing required columns: {missing}")

    original_path = root / "outputs" / "reference_models" / "expanding_window_oof_predictions.parquet"
    probabilistic_path = root / "outputs" / "probabilistic_reference_models" / "probabilistic_oof_predictions.parquet"
    original = pd.read_parquet(original_path, columns=["forecast_origin", "target_timestamp", "evaluation_year", "model", "actual_change_30min", "predicted_change_30min"])
    probabilistic = pd.read_parquet(probabilistic_path, columns=["forecast_origin", "target_timestamp", "evaluation_year", "model", "actual_change_30min", "location_change_30min"])
    if original["evaluation_year"].max() > 2020 or probabilistic["evaluation_year"].max() > 2020:
        raise ValueError("Post-2020 reference evidence is forbidden in RR")
    if pd.to_datetime(original["target_timestamp"]).max() >= pd.Timestamp("2021-01-01"):
        raise ValueError("Original-reference target escaped the public boundary")
    if pd.to_datetime(probabilistic["target_timestamp"]).max() >= pd.Timestamp("2021-01-01"):
        raise ValueError("Probabilistic-reference target escaped the public boundary")

    ridge = original.loc[original["model"].eq("Ridge regression")].sort_values(["evaluation_year", "forecast_origin"])
    gaussian = probabilistic.loc[probabilistic["model"].eq("Gaussian linear")].sort_values(["evaluation_year", "forecast_origin"])
    if not ridge["forecast_origin"].reset_index(drop=True).equals(gaussian["forecast_origin"].reset_index(drop=True)):
        raise ValueError("Gaussian and original Ridge forecast origins differ")
    if not (ridge["predicted_change_30min"].to_numpy() == gaussian["location_change_30min"].to_numpy()).all():
        raise ValueError("Gaussian location does not equal the original Ridge location")

    report = {
        "configuration_count": len(jobs),
        "original_reference_models": 5,
        "probabilistic_linear_references": 3,
        "external_benchmarks": 1,
        "original_prediction_rows": int(len(original)),
        "probabilistic_prediction_rows": int(len(probabilistic)),
        "latest_evaluation_year": 2020,
        "ridge_base_predictors": 55,
        "gaussian_location_matches_original_ridge": True,
        "status": "PASS",
        "full_training_launched": False,
    }
    output = root / "outputs" / "03_feature_contract" / "reference_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(preflight(args.project_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
