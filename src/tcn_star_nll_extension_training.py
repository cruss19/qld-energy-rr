"""Training loop for the authorized RR causal-feature-estimate extension."""

from __future__ import annotations

import gc
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tcn_star_nll_extension_data import make_loader
from .tcn_star_nll_extension_model import TCNStarNLLCausalFeatureEstimates
from .tcn_star_nll_model import DynamicStudentTNLLLoss


MODEL_NAME = "TCN_starNLL_causal_feature_estimates"
VARIANT_NAME = "demand_direct_radiation_estimated_sd"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _replace_with_retry(temporary: str | Path, destination: Path) -> None:
    """Tolerate brief Windows sharing locks from live progress readers."""
    last_error: PermissionError | None = None
    for _ in range(100):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05)
    if last_error is not None:
        raise last_error


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        _replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_epoch(
    model,
    loader,
    device,
    target_scale,
    optimizer=None,
    *,
    progress_path: Path,
    phase: str,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = DynamicStudentTNLLLoss().to(device)
    scale_mw = torch.as_tensor(target_scale, dtype=torch.float32, device=device).view(1, -1)
    total_nll = total_mae = total_squared_mw = total_rows = 0.0
    total_batches = len(loader)
    started = time.perf_counter()
    accumulation = 4
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for number, batch in enumerate(loader, start=1):
            values = [item.to(device, non_blocking=True) for item in batch]
            mu, scale, df = model(*values[:6])
            target = values[6]
            loss = criterion(mu.float(), scale.float(), df.float(), target.float())
            if training:
                (loss / accumulation).backward()
                boundary = number % accumulation == 0 or number == total_batches
                if boundary:
                    remainder = number % accumulation
                    if number == total_batches and remainder:
                        correction = accumulation / remainder
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            rows = len(target)
            total_nll += float(loss.detach()) * rows
            total_mae += float(torch.mean(torch.abs(mu.detach() - target) * scale_mw)) * rows
            total_squared_mw += float(torch.mean(((mu.detach() - target) * scale_mw) ** 2)) * rows
            total_rows += rows
            elapsed = time.perf_counter() - started
            atomic_json(progress_path, {
                "status": "active", "phase": phase, "epoch": epoch,
                "current_batch": number, "total_batches": total_batches,
                "progress_fraction": number / total_batches,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / number * (total_batches - number),
                "running_nll": total_nll / total_rows,
                "running_mae_mw": total_mae / total_rows,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    return {
        "nll": total_nll / total_rows,
        "mae_mw": total_mae / total_rows,
        "rmse_mw": float(np.sqrt(total_squared_mw / total_rows)),
    }


def fit_development(
    *, arrays, train_endpoints, validation_endpoints, scaler, run_dir: Path,
    seed: int, resume: bool = False,
) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"runtime_device={device} "
        f"device_name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    model = TCNStarNLLCausalFeatureEstimates(
        calendar_dim=int(scaler["calendar_encoded_width"])
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5,
        threshold=0.0, threshold_mode="abs",
    )
    train_loader = make_loader(arrays, train_endpoints, shuffle=True, workers=2)
    valid_loader = make_loader(arrays, validation_endpoints, shuffle=False, workers=1)
    identity = {
        "model": MODEL_NAME, "variant": VARIANT_NAME, "stage": "development",
        "fold": "train_2015_2018_validate_2019", "seed": seed,
    }
    best_nll, best_epoch, bad_epochs, history, start_epoch = float("inf"), 0, 0, [], 1
    best_path = run_dir / "best_validation_weights.pt"
    latest_path = run_dir / "latest_complete_epoch.pt"
    if latest_path.exists():
        if not resume:
            raise RuntimeError(f"Partial output exists; use --resume only: {run_dir}")
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Extension resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_nll = float(saved["best_validation_nll"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["epochs_no_improve"])
        history = saved["history"]
        start_epoch = int(saved["completed_epoch"]) + 1

    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "live_progress.json"
    for epoch in range(start_epoch, 51):
        lr_used = float(optimizer.param_groups[0]["lr"])
        train = run_epoch(
            model, train_loader, device, scaler["targets"]["scale"], optimizer,
            progress_path=progress_path, phase="training", epoch=epoch,
        )
        valid = run_epoch(
            model, valid_loader, device, scaler["targets"]["scale"],
            progress_path=progress_path, phase="validation", epoch=epoch,
        )
        # Mandatory cleanup occurs immediately after validation and before the
        # scheduler/checkpoint work leading into the next epoch.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scheduler.step(valid["nll"])
        if valid["nll"] < best_nll:
            best_nll, best_epoch, bad_epochs = valid["nll"], epoch, 0
            atomic_torch_save(best_path, {
                "identity": identity, "epoch": epoch,
                "validation_nll": best_nll, "model": model.state_dict(),
            })
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "training_nll": train["nll"],
            "training_mae_mw": train["mae_mw"],
            "training_rmse_mw": train["rmse_mw"],
            "validation_nll": valid["nll"],
            "validation_mae_mw": valid["mae_mw"],
            "validation_rmse_mw": valid["rmse_mw"],
            "learning_rate_used": lr_used,
            "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            "best_epoch": best_epoch,
            "epochs_no_improve": bad_epochs,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "epoch_log.csv", index=False)
        atomic_torch_save(latest_path, {
            "identity": identity, "completed_epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_validation_nll": best_nll,
            "best_epoch": best_epoch, "epochs_no_improve": bad_epochs,
            "history": history,
        })
        print(
            f"model={MODEL_NAME} seed={seed} epoch={epoch}/50 "
            f"training_nll={train['nll']:.6f} training_mae_mw={train['mae_mw']:.3f} "
            f"validation_nll={valid['nll']:.6f} validation_mae_mw={valid['mae_mw']:.3f}",
            flush=True,
        )
        atomic_json(progress_path, {
            "status": "epoch_complete", "phase": "validation", "epoch": epoch,
            "current_batch": len(valid_loader), "total_batches": len(valid_loader),
            "progress_fraction": 1.0, "eta_seconds": 0.0, **row,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        if epoch >= 8 and bad_epochs > 4:
            break
    result = {
        "status": "complete", "model": MODEL_NAME, "variant": VARIANT_NAME,
        "fold": "train_2015_2018_validate_2019", "seed": seed,
        "best_epoch": best_epoch, "best_validation_nll": best_nll,
        "completed_epochs": len(history),
    }
    atomic_json(run_dir / "completed_run.json", result)
    return result


__all__ = ["fit_development", "atomic_json", "MODEL_NAME", "VARIANT_NAME"]
