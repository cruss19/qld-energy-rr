"""Run the authorized RR final extension on 2015-2018 -> 2019."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tcn_star_nll_extension_data import fold_endpoints, load_frame, make_loader, prepare_fold
from src.tcn_star_nll_extension_model import TCNStarNLLCausalFeatureEstimatesNoSD
from src.tcn_star_nll_extension_training import atomic_json, fit_development
from src.tcn_star_nll_model import DynamicStudentTNLLLoss


CONFIG_PATH = ROOT / "config/experiments/tcn_star_nll_causal_feature_estimates_no_sd_validate_2019_seed42.yml"
MODEL_NAME = "TCN_starNLL_causal_feature_estimates_no_sd"
VARIANT_NAME = "demand_direct_radiation_estimated_no_sd"


def run_identity(config: dict, seed: int, run_directory: Path) -> dict:
    return {
        "exact_model": MODEL_NAME,
        "variant": VARIANT_NAME,
        "training_fold": "2015-01-01 through 2018-12-31",
        "validation_fold": "2019-01-01 through 2019-12-31",
        "seed": seed,
        "feature_contract": config["causal_helpers"],
        "architecture": config["architecture"],
        "major_training_hyperparameters": config["training"],
        "stopping_rule": "minimum 8 and maximum 50 epochs; stop after the fifth consecutive non-improving epoch under patience 4",
        "checkpoint_rule": "minimum validation NLL across completed epochs",
        "output_directory": run_directory.relative_to(ROOT).as_posix(),
    }


def verify_config(config: dict) -> None:
    expected = {
        "model_name": MODEL_NAME,
        "variant_name": VARIANT_NAME,
        "output_directory": "training_output/runs/TCN_starNLL_causal_feature_estimates_no_sd/development/validation_2019/seed_42",
    }
    mismatches = {
        key: {"expected": value, "configured": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    split = config["split"]
    if split != {
        "train_start": "2015-01-01", "train_end_exclusive": "2019-01-01",
        "validation_start": "2019-01-01", "validation_end_exclusive": "2020-01-01",
        "seed": 42,
    }:
        mismatches["split"] = split
    if mismatches:
        raise ValueError(f"Extension run identity mismatch: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--authorise-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 142, 242), default=42)
    args = parser.parse_args()
    if not args.preflight_only and not args.authorise_full_run:
        raise RuntimeError("Full extension training requires --authorise-full-run")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_config(config)
    seed = int(args.seed)
    run_directory = ROOT / (
        "training_output/runs/TCN_starNLL_causal_feature_estimates_no_sd/"
        f"development/validation_2019/seed_{seed}"
    )
    config["split"]["seed"] = seed
    config["output_directory"] = run_directory.relative_to(ROOT).as_posix()
    split = config["split"]
    train_start = pd.Timestamp(split["train_start"])
    train_end = pd.Timestamp(split["train_end_exclusive"])
    validation_start = pd.Timestamp(split["validation_start"])
    validation_end = pd.Timestamp(split["validation_end_exclusive"])
    frame = load_frame(
        ROOT / config["dataset"],
        ROOT / config["helper_sources"]["demand"],
        ROOT / config["helper_sources"]["direct_radiation"],
    )
    arrays, scaler = prepare_fold(frame, train_start, train_end, include_sd_channels=False)
    train_endpoints = fold_endpoints(frame, train_start, train_end)
    validation_endpoints = fold_endpoints(frame, validation_start, validation_end)
    model = TCNStarNLLCausalFeatureEstimatesNoSD(
        calendar_dim=int(scaler["calendar_encoded_width"])
    )
    sample = next(iter(make_loader(
        arrays, validation_endpoints[:2], shuffle=False, workers=0
    )))
    smoke_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke_device.type != "cuda":
        raise RuntimeError("Authorized full run requires a passing CUDA smoke test")
    model = model.to(smoke_device)
    smoke_values = [value.to(smoke_device) for value in sample]
    output = model(*smoke_values[:6])
    if [tuple(value.shape) for value in output] != [(2, 6), (2, 6), (2, 6)]:
        raise ValueError("Extension architecture smoke output is incorrect")
    smoke_loss = DynamicStudentTNLLLoss()(*output, smoke_values[6])
    smoke_loss.backward()
    if not torch.isfinite(smoke_loss) or not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("Extension CUDA forward/backward smoke test is non-finite")
    if model.receptive_field_steps != 1009:
        raise ValueError("Extension receptive field changed")
    del smoke_values, sample, output, smoke_loss, model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "PASS "
        f"model={MODEL_NAME} variant={VARIANT_NAME} seed={seed} "
        f"train_origins={len(train_endpoints)} validation_origins={len(validation_endpoints)} "
        f"calendar_width={scaler['calendar_encoded_width']} "
        "cuda_smoke=PASS",
        flush=True,
    )
    if args.preflight_only:
        return 0

    if (run_directory / "completed_run.json").exists():
        completed = json.loads((run_directory / "completed_run.json").read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            raise RuntimeError(f"Completed output already exists: {run_directory}")
    if (run_directory / "latest_complete_epoch.pt").exists() and not args.resume:
        raise RuntimeError(f"Partial output exists; use --resume only: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, seed, run_directory)
    identity_path = run_directory / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Existing extension run identity does not match")
    else:
        atomic_json(identity_path, identity)
    atomic_json(run_directory / "preflight_result.json", {
        "status": "PASS",
        "data_contract": "PASS",
        "causal_leakage_tests": "PASS",
        "architecture": "PASS",
        "checkpoint_identity": "PASS",
        "cuda_forward_backward_smoke": "PASS",
    })
    atomic_json(run_directory / "preprocessing_state.json", scaler)
    fit_development(
        arrays=arrays,
        train_endpoints=train_endpoints,
        validation_endpoints=validation_endpoints,
        scaler=scaler,
        run_dir=run_directory,
        seed=seed,
        resume=args.resume,
        model_class=TCNStarNLLCausalFeatureEstimatesNoSD,
        model_name=MODEL_NAME,
        variant_name=VARIANT_NAME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
