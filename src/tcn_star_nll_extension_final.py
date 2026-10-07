"""Fixed-duration final refit and frozen evaluation for the RR no-SD extension."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tcn_star_nll_extension_data import HORIZONS, make_loader
from .tcn_star_nll_extension_model import TCNStarNLLCausalFeatureEstimatesNoSD
from .tcn_star_nll_extension_training import (
    atomic_json,
    atomic_torch_save,
    run_epoch,
    seed_everything,
)


def fit_fixed_epochs(
    *, arrays, endpoints, scaler: dict, run_dir: Path, seed: int,
    epochs: int, identity: dict, resume: bool,
) -> tuple[torch.nn.Module, torch.device]:
    """Fit the approved final model for an exact duration without test selection."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Authorized final refit requires CUDA")
    model = TCNStarNLLCausalFeatureEstimatesNoSD(
        calendar_dim=int(scaler["calendar_encoded_width"])
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    loader = make_loader(arrays, endpoints, shuffle=True, workers=2)
    history: list[dict] = []
    start_epoch = 1
    latest = run_dir / "latest_complete_epoch.pt"
    if latest.exists():
        if not resume:
            raise RuntimeError(f"Partial output exists; exact resume required: {run_dir}")
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Final-refit resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved["history"])
        start_epoch = int(saved["completed_epoch"]) + 1

    progress = run_dir / "live_progress.json"
    for epoch in range(start_epoch, epochs + 1):
        metrics = run_epoch(
            model, loader, device, scaler["targets"]["scale"], optimizer,
            progress_path=progress, phase="final_refit_training", epoch=epoch,
        )
        # Required cleanup immediately after the epoch and before the next one.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = {
            "epoch": epoch,
            "training_nll": metrics["nll"],
            "training_mae_mw": metrics["mae_mw"],
            "training_rmse_mw": metrics["rmse_mw"],
            "learning_rate": 0.0005,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "epoch_log.csv", index=False)
        atomic_torch_save(latest, {
            "identity": identity,
            "completed_epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        })
        atomic_json(progress, {
            "status": "epoch_complete",
            "phase": "final_refit_training",
            "epoch": epoch,
            "fixed_epochs": epochs,
            "progress_fraction": epoch / epochs,
            **row,
        })
        print(
            f"final_refit seed={seed} epoch={epoch}/{epochs} "
            f"training_nll={metrics['nll']:.6f} "
            f"training_mae_mw={metrics['mae_mw']:.3f}",
            flush=True,
        )

    atomic_torch_save(run_dir / "frozen_model.pt", {
        "identity": identity,
        "epoch": epochs,
        "seed": seed,
        "model": model.state_dict(),
    })
    return model, device


@torch.inference_mode()
def score_frozen(model, loader, device, target_state: dict, frame_index) -> tuple[list[dict], pd.DataFrame]:
    """Score only the already-frozen state on the authorized 2020 interval."""
    model.eval()
    mean = np.asarray(target_state["mean"], dtype=np.float64)
    scale_state = np.asarray(target_state["scale"], dtype=np.float64)
    mus, scales, dfs, targets, endpoints = [], [], [], [], []
    for batch in loader:
        values = [item.to(device, non_blocking=True) for item in batch]
        mu, scale, df = model(*values[:6])
        mus.append(mu.cpu().numpy())
        scales.append(scale.cpu().numpy())
        dfs.append(df.cpu().numpy())
        targets.append(values[6].cpu().numpy())
        endpoints.append(values[7].cpu().numpy())
    mu = np.concatenate(mus).astype(np.float64) * scale_state + mean
    sigma = np.concatenate(scales).astype(np.float64) * scale_state
    df = np.concatenate(dfs).astype(np.float64)
    actual = np.concatenate(targets).astype(np.float64) * scale_state + mean
    endpoint = np.concatenate(endpoints).astype(np.int64)
    if np.any(sigma <= 0.0) or np.any(df <= 2.0):
        raise FloatingPointError("Frozen model produced invalid Student-t parameters")

    predictions = pd.DataFrame({"forecast_origin": frame_index[endpoint]})
    rows = []
    for index, horizon in enumerate(HORIZONS):
        error = mu[:, index] - actual[:, index]
        rows.append({
            "horizon_minutes": horizon,
            "forecast_origins": len(endpoint),
            "mae_mw": float(np.mean(np.abs(error))),
            "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
            "bias_mw": float(np.mean(error)),
            "mean_scale_mw": float(np.mean(sigma[:, index])),
            "mean_df": float(np.mean(df[:, index])),
        })
        predictions[f"actual_change_{horizon}m_mw"] = actual[:, index]
        predictions[f"predicted_change_{horizon}m_mw"] = mu[:, index]
        predictions[f"predicted_scale_{horizon}m_mw"] = sigma[:, index]
        predictions[f"predicted_df_{horizon}m"] = df[:, index]
    return rows, predictions


__all__ = ["fit_fixed_epochs", "score_frozen"]
