"""Run one approved fixed-epoch RR no-SD final refit and 2020 evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tcn_star_nll_extension_data import fold_endpoints, load_frame, make_loader, prepare_fold
from src.tcn_star_nll_extension_final import fit_fixed_epochs, score_frozen
from src.tcn_star_nll_extension_model import TCNStarNLLCausalFeatureEstimatesNoSD
from src.tcn_star_nll_extension_training import atomic_json
from src.tcn_star_nll_model import DynamicStudentTNLLLoss


CONFIG_PATH = ROOT / "config/experiments/tcn_star_nll_causal_feature_estimates_no_sd_final_2020.yml"
MODEL = "TCN_starNLL_causal_feature_estimates_no_sd"
VARIANT = "demand_direct_radiation_estimated_no_sd"
SEEDS = (42, 142, 242)
FIXED_EPOCHS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_directory(seed: int) -> Path:
    return ROOT / "training_output" / "runs" / MODEL / "final" / "train_2015_2019" / f"seed_{seed}"


def identity(config: dict, seed: int) -> dict:
    return {
        "exact_model": MODEL,
        "variant": VARIANT,
        "stage": "final_fixed_duration_refit_and_frozen_2020_evaluation",
        "train_start": "2015-01-01",
        "train_end_exclusive": "2020-01-01",
        "evaluation_start": "2020-01-01",
        "evaluation_end_exclusive": "2021-01-01",
        "seed": seed,
        "feature_contract": config["causal_helpers"],
        "architecture": config["architecture"],
        "major_training_hyperparameters": config["training"],
        "fixed_epochs": FIXED_EPOCHS,
        "fixed_epoch_derivation": config["fixed_epoch_derivation"],
        "stopping_rule": "exactly 10 epochs; no 2020 early stopping",
        "checkpoint_rule": "final epoch-10 state frozen before 2020 scoring",
        "output_directory": run_directory(seed).relative_to(ROOT).as_posix(),
    }


def verify_config(config: dict) -> None:
    expected = {
        "model_name": MODEL,
        "variant_name": VARIANT,
        "seeds": list(SEEDS),
        "fixed_epochs": FIXED_EPOCHS,
        "execution_order": list(SEEDS),
        "output_directory_template": (
            "training_output/runs/TCN_starNLL_causal_feature_estimates_no_sd/"
            "final/train_2015_2019/seed_{seed}"
        ),
    }
    bad = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
    if config.get("split") != {
        "train_start": "2015-01-01",
        "train_end_exclusive": "2020-01-01",
        "evaluation_start": "2020-01-01",
        "evaluation_end_exclusive": "2021-01-01",
    }:
        bad["split"] = config.get("split")
    if config["causal_helpers"].get("standard_deviation_channels_in_model") is not False:
        bad["standard_deviation_channels_in_model"] = "must be false"
    if bad:
        raise ValueError(f"Final no-SD run identity mismatch: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--authorise-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and not args.authorise_full_run:
        raise RuntimeError("Full final training requires --authorise-full-run")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_config(config)
    seed = int(args.seed)
    output = run_directory(seed)
    run_id = identity(config, seed)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != run_id:
        raise RuntimeError(f"Existing final identity differs: {identity_path}")
    atomic_json(identity_path, run_id)

    split = config["split"]
    frame = load_frame(
        ROOT / config["dataset"],
        ROOT / config["helper_sources"]["demand"],
        ROOT / config["helper_sources"]["direct_radiation"],
    )
    train_start = pd.Timestamp(split["train_start"])
    train_end = pd.Timestamp(split["train_end_exclusive"])
    evaluation_start = pd.Timestamp(split["evaluation_start"])
    evaluation_end = pd.Timestamp(split["evaluation_end_exclusive"])
    arrays, scaler = prepare_fold(frame, train_start, train_end, include_sd_channels=False)
    train_endpoints = fold_endpoints(frame, train_start, train_end)
    evaluation_endpoints = fold_endpoints(frame, evaluation_start, evaluation_end)
    train_times = frame.index[train_endpoints]
    evaluation_times = frame.index[evaluation_endpoints]
    if train_times.max() + pd.Timedelta(minutes=30) >= train_end:
        raise RuntimeError("Training targets cross into the 2020 evaluation interval")
    if evaluation_times.min() < evaluation_start or evaluation_times.max() + pd.Timedelta(minutes=30) >= evaluation_end:
        raise RuntimeError("Evaluation endpoints violate the declared 2020 interval")
    if scaler["helper_contract"]["sd_channels_in_model"] is not False:
        raise RuntimeError("Preprocessing state unexpectedly includes SD channels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Authorized final run requires a passing CUDA smoke test")
    model = TCNStarNLLCausalFeatureEstimatesNoSD(
        calendar_dim=int(scaler["calendar_encoded_width"])
    ).to(device)
    sample = next(iter(make_loader(arrays, train_endpoints[:2], shuffle=False, workers=0)))
    values = [value.to(device) for value in sample]
    result = model(*values[:6])
    if [tuple(value.shape) for value in result] != [(2, 6), (2, 6), (2, 6)]:
        raise ValueError("Final extension architecture smoke output is incorrect")
    loss = DynamicStudentTNLLLoss()(*result, values[6])
    loss.backward()
    if not torch.isfinite(loss) or model.receptive_field_steps != 1009:
        raise FloatingPointError("Final extension CUDA smoke test failed")
    del model, sample, values, result, loss
    gc.collect()
    torch.cuda.empty_cache()

    preflight = {
        "status": "PASS",
        "model": MODEL,
        "variant": VARIANT,
        "seed": seed,
        "fixed_epochs": FIXED_EPOCHS,
        "training_origins": len(train_endpoints),
        "evaluation_origins": len(evaluation_endpoints),
        "data_contract": "PASS",
        "causal_leakage_boundary": "PASS",
        "architecture": "PASS",
        "checkpoint_identity": "PASS",
        "cuda_forward_backward_smoke": "PASS",
    }
    atomic_json(output / "preflight_result.json", preflight)
    atomic_json(output / "preprocessing_state.json", scaler)
    print(json.dumps(preflight), flush=True)
    if args.preflight_only:
        return 0

    completion = output / "completed_run.json"
    if completion.exists():
        saved = json.loads(completion.read_text(encoding="utf-8"))
        if saved.get("status") == "complete" and saved.get("identity") == run_id:
            print(f"already_complete={output}", flush=True)
            return 0
        raise RuntimeError(f"Unexpected completion artifact: {completion}")
    partial = output / "latest_complete_epoch.pt"
    if partial.exists() and not args.resume:
        raise RuntimeError(f"Partial output exists; use --resume: {output}")

    model, device = fit_fixed_epochs(
        arrays=arrays,
        endpoints=train_endpoints,
        scaler=scaler,
        run_dir=output,
        seed=seed,
        epochs=FIXED_EPOCHS,
        identity=run_id,
        resume=args.resume,
    )
    frozen = torch.load(output / "frozen_model.pt", map_location=device, weights_only=False)
    if frozen.get("identity") != run_id or int(frozen.get("epoch", -1)) != FIXED_EPOCHS:
        raise RuntimeError("Frozen final state identity is invalid")
    model.load_state_dict(frozen["model"])
    evaluation_loader = make_loader(arrays, evaluation_endpoints, shuffle=False, workers=1)
    metrics, predictions = score_frozen(
        model, evaluation_loader, device, scaler["targets"], frame.index
    )
    pd.DataFrame(metrics).to_csv(output / "evaluation_2020_horizon_metrics.csv", index=False)
    predictions.to_parquet(output / "evaluation_2020_predictions.parquet", index=False)
    atomic_json(completion, {
        "status": "complete",
        "model": MODEL,
        "variant": VARIANT,
        "stage": "final_fixed_duration_refit_and_frozen_2020_evaluation",
        "seed": seed,
        "fixed_epochs": FIXED_EPOCHS,
        "metrics": metrics,
        "data_path": config["dataset"],
        "data_sha256": sha256(ROOT / config["dataset"]),
        "identity": run_id,
    })
    atomic_json(output / "live_progress.json", {
        "status": "complete",
        "phase": "frozen_2020_evaluation",
        "seed": seed,
        "fixed_epochs": FIXED_EPOCHS,
        "evaluation_origins": len(predictions),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
