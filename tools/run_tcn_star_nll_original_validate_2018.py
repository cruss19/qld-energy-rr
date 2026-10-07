"""Run the frozen original PR TCN_starNLL on 2015-2017 -> 2018, seed 42."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tcn_star_nll_tensors import fold_endpoints, load_frame, prepare_fold
from src.tcn_star_nll_training import atomic_json, fit_development

CONFIG_PATH = ROOT / "config/experiments/tcn_star_nll_original_validate_2018_seed42.yml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split = config["split"]
    seed = int(split["seed"])
    train_start = pd.Timestamp(split["train_start"])
    train_end = pd.Timestamp(split["train_end_exclusive"])
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["validation_end_exclusive"])
    if validation_start != train_end:
        raise ValueError("Validation must begin exactly at the exclusive training boundary")

    data_path = ROOT / config["dataset"]
    frame = load_frame(data_path)
    train_endpoints = fold_endpoints(frame, train_start, train_end)
    validation_endpoints = fold_endpoints(frame, validation_start, validation_end)
    run_dir = ROOT / config["output_directory"]
    metrics_table = ROOT / config["epoch_metrics_table"]

    print(
        "RUN_IDENTITY "
        f"model={config['model_name']} variant={config['variant_name']} seed={seed} "
        f"train=2015-01-01..2017-12-31 validation=2018-01-01..2018-12-31 "
        f"dataset={data_path} output={run_dir} "
        "lr=0.0005 weight_decay=0.01 dropout=0.33 micro_batch=128 "
        "accumulation=4 effective_batch=512 max_epochs=50 "
        "checkpoint=min_validation_nll",
        flush=True,
    )
    if args.preflight_only:
        from src.tcn_star_nll_model import TCNStarNLL

        arrays, scaler = prepare_fold(frame, train_start, train_end)
        model = TCNStarNLL(calendar_dim=int(scaler["calendar_encoded_width"]))
        assert len(train_endpoints) > 0 and len(validation_endpoints) > 0
        assert model.receptive_field_steps == 1009
        print(
            "PASS "
            f"runtime_device_check_deferred_to_launch=true train_origins={len(train_endpoints)} "
            f"validation_origins={len(validation_endpoints)} parameters={sum(p.numel() for p in model.parameters())}",
            flush=True,
        )
        return 0

    if (run_dir / "completed_run.json").exists():
        completed = json.loads((run_dir / "completed_run.json").read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            raise RuntimeError(f"Completed output already exists: {run_dir}")
    if (run_dir / "latest_complete_epoch.pt").exists() and not args.resume:
        raise RuntimeError(f"Partial output exists; use --resume only: {run_dir}")

    arrays, scaler = prepare_fold(frame, train_start, train_end)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "preprocessing_state.json", scaler)
    fit_development(
        arrays=arrays,
        train_endpoints=train_endpoints,
        validation_endpoints=validation_endpoints,
        scaler=scaler,
        run_dir=run_dir,
        seed=seed,
        fold_id="validation_2018",
        resume=args.resume,
        metrics_table=metrics_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
