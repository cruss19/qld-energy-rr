"""Build the authorized three-seed no-SD extension ensemble on 2019.

This utility performs inference only. It requires normally completed, exactly
matching development runs for seeds 42, 142 and 242 and combines their saved
minimum-validation-NLL checkpoints as an equal-weight Student-t mixture.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import logsumexp
from scipy.stats import t as student_t


RR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RR_ROOT))

from src.tcn_star_nll_extension_data import HORIZONS, fold_endpoints, load_frame, make_loader, prepare_fold
from src.tcn_star_nll_extension_model import TCNStarNLLCausalFeatureEstimatesNoSD


MODEL = "TCN_starNLL_causal_feature_estimates_no_sd"
VARIANT = "demand_direct_radiation_estimated_no_sd"
SEEDS = (42, 142, 242)
RUN_BASE = RR_ROOT / "training_output" / "runs" / MODEL / "development" / "validation_2019"
DESTINATION = (
    RR_ROOT
    / "training_output"
    / "ensembles"
    / f"{MODEL}_ensemble_validation_2019_seeds_42_142_242"
)
CONFIG = RR_ROOT / "config" / "experiments" / "tcn_star_nll_causal_feature_estimates_no_sd_validate_2019_seed42.yml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_identity(identity: dict) -> dict:
    result = json.loads(json.dumps(identity))
    result.pop("seed", None)
    result.pop("output_directory", None)
    return result


def verify_members() -> list[dict]:
    members: list[dict] = []
    canonical: dict | None = None
    preprocessing: dict | None = None
    for seed in SEEDS:
        run = RUN_BASE / f"seed_{seed}"
        required = {
            "identity": run / "run_identity.json",
            "completion": run / "completed_run.json",
            "checkpoint": run / "best_validation_weights.pt",
            "preprocessing": run / "preprocessing_state.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} is not ensemble-ready; missing: {missing}")
        identity = json.loads(required["identity"].read_text(encoding="utf-8"))
        completion = json.loads(required["completion"].read_text(encoding="utf-8"))
        state = json.loads(required["preprocessing"].read_text(encoding="utf-8"))
        if identity.get("exact_model") != MODEL or identity.get("variant") != VARIANT:
            raise ValueError(f"Seed {seed} has the wrong model identity")
        if int(identity.get("seed")) != seed:
            raise ValueError(f"Seed {seed} identity records seed {identity.get('seed')}")
        if completion.get("status") != "complete" or int(completion.get("seed")) != seed:
            raise RuntimeError(f"Seed {seed} did not complete normally")
        normalized = canonical_identity(identity)
        if canonical is None:
            canonical = normalized
            preprocessing = state
        elif normalized != canonical:
            raise ValueError(f"Seed {seed} does not match the shared run identity")
        if state != preprocessing:
            raise ValueError(f"Seed {seed} preprocessing state differs from the other members")
        members.append(
            {
                "seed": seed,
                "run_directory": run.relative_to(RR_ROOT).as_posix(),
                "best_epoch": int(completion["best_epoch"]),
                "best_validation_nll": float(completion["best_validation_nll"]),
                "completed_epochs": int(completion["completed_epochs"]),
                "checkpoint_sha256": sha256(required["checkpoint"]),
                "identity_sha256": sha256(required["identity"]),
                "preprocessing_sha256": sha256(required["preprocessing"]),
            }
        )
    return members


def mixture_quantile(probability: float, locations: np.ndarray, scales: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    lower = np.min(locations + scales * student_t.ppf(1e-8, df=dfs), axis=0)
    upper = np.max(locations + scales * student_t.ppf(1.0 - 1e-8, df=dfs), axis=0)
    for _ in range(48):
        middle = (lower + upper) / 2.0
        cdf = np.mean(student_t.cdf((middle[None, :] - locations) / scales, df=dfs), axis=0)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    return (lower + upper) / 2.0


def member_predictions(arrays, endpoints: np.ndarray, calendar_width: int, device: torch.device) -> dict[int, dict[str, np.ndarray]]:
    loader = make_loader(arrays, endpoints, shuffle=False, workers=1)
    results: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        checkpoint_path = RUN_BASE / f"seed_{seed}" / "best_validation_weights.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected = {
            "model": MODEL,
            "variant": VARIANT,
            "stage": "development",
            "fold": "train_2015_2018_validate_2019",
            "seed": seed,
        }
        if checkpoint.get("identity") != expected:
            raise ValueError(f"Seed {seed} checkpoint identity mismatch")
        model = TCNStarNLLCausalFeatureEstimatesNoSD(calendar_dim=calendar_width).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        mu_parts: list[np.ndarray] = []
        scale_parts: list[np.ndarray] = []
        df_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                values = [value.to(device, non_blocking=True) for value in batch]
                mu, scale, df = model(*values[:6])
                mu_parts.append(mu.cpu().numpy())
                scale_parts.append(scale.cpu().numpy())
                df_parts.append(df.cpu().numpy())
                target_parts.append(values[6].cpu().numpy())
                index_parts.append(values[7].cpu().numpy())
        results[seed] = {
            "mu": np.concatenate(mu_parts).astype(np.float64),
            "scale": np.concatenate(scale_parts).astype(np.float64),
            "df": np.concatenate(df_parts).astype(np.float64),
            "target": np.concatenate(target_parts).astype(np.float64),
            "index": np.concatenate(index_parts).astype(np.int64),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_validation_nll": float(checkpoint["validation_nll"]),
        }
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    reference = results[SEEDS[0]]
    for seed in SEEDS[1:]:
        if not np.array_equal(results[seed]["index"], reference["index"]):
            raise ValueError(f"Seed {seed} forecast origins do not align")
        if not np.array_equal(results[seed]["target"], reference["target"]):
            raise ValueError(f"Seed {seed} validation targets do not align")
    return results


def main() -> int:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing ensemble: {DESTINATION}")
    members = verify_members()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    split = config["split"]
    frame = load_frame(
        RR_ROOT / config["dataset"],
        RR_ROOT / config["helper_sources"]["demand"],
        RR_ROOT / config["helper_sources"]["direct_radiation"],
    )
    arrays, scaler = prepare_fold(
        frame,
        pd.Timestamp(split["train_start"]),
        pd.Timestamp(split["train_end_exclusive"]),
        include_sd_channels=False,
    )
    stored_scaler = json.loads((RUN_BASE / "seed_42" / "preprocessing_state.json").read_text(encoding="utf-8"))
    if scaler != stored_scaler:
        raise ValueError("Reconstructed preprocessing state does not match the frozen member state")
    endpoints = fold_endpoints(
        frame,
        pd.Timestamp(split["validation_start"]),
        pd.Timestamp(split["validation_end_exclusive"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} validation_origins={len(endpoints)}", flush=True)
    predictions = member_predictions(
        arrays,
        endpoints,
        int(scaler["calendar_encoded_width"]),
        device,
    )

    target_mean = np.asarray(scaler["targets"]["mean"], dtype=np.float64)
    target_scale = np.asarray(scaler["targets"]["scale"], dtype=np.float64)
    target_z = predictions[SEEDS[0]]["target"]
    actual_mw = target_z * target_scale + target_mean
    locations_z = np.stack([predictions[seed]["mu"] for seed in SEEDS])
    scales_z = np.stack([predictions[seed]["scale"] for seed in SEEDS])
    dfs = np.stack([predictions[seed]["df"] for seed in SEEDS])
    locations_mw = locations_z * target_scale[None, None, :] + target_mean[None, None, :]
    scales_mw = scales_z * target_scale[None, None, :]
    ensemble_location_mw = locations_mw.mean(axis=0)

    timestamps = frame.index[predictions[SEEDS[0]]["index"]]
    ensemble = pd.DataFrame({"forecast_origin": timestamps})
    aligned = pd.DataFrame({"forecast_origin": timestamps})
    horizon_rows: list[dict] = []
    pooled_errors: list[np.ndarray] = []
    pooled_nll_z: list[np.ndarray] = []
    pooled_nll_mw: list[np.ndarray] = []
    pooled_covered: list[np.ndarray] = []
    pooled_width: list[np.ndarray] = []
    pooled_pit: list[np.ndarray] = []
    for column, horizon in enumerate(HORIZONS):
        actual = actual_mw[:, column]
        location = ensemble_location_mw[:, column]
        member_locations = locations_mw[:, :, column]
        member_scales = scales_mw[:, :, column]
        member_dfs = dfs[:, :, column]
        standardized_z = (target_z[:, column][None, :] - locations_z[:, :, column]) / scales_z[:, :, column]
        member_log_density_z = student_t.logpdf(standardized_z, df=member_dfs) - np.log(scales_z[:, :, column])
        log_density_z = logsumexp(member_log_density_z, axis=0) - math.log(len(SEEDS))
        standardized_mw = (actual[None, :] - member_locations) / member_scales
        member_log_density_mw = student_t.logpdf(standardized_mw, df=member_dfs) - np.log(member_scales)
        log_density_mw = logsumexp(member_log_density_mw, axis=0) - math.log(len(SEEDS))
        pit = np.mean(student_t.cdf(standardized_mw, df=member_dfs), axis=0)
        lower = mixture_quantile(0.025, member_locations, member_scales, member_dfs)
        upper = mixture_quantile(0.975, member_locations, member_scales, member_dfs)
        covered = (actual >= lower) & (actual <= upper)
        error = location - actual
        horizon_rows.append(
            {
                "model": MODEL,
                "variant": VARIANT,
                "ensemble_type": "equal_weight_student_t_mixture",
                "validation_year": 2019,
                "horizon_minutes": horizon,
                "forecast_origins": len(timestamps),
                "nll_standardized": float(-np.mean(log_density_z)),
                "nll_mw_density": float(-np.mean(log_density_mw)),
                "mae_mw": float(np.mean(np.abs(error))),
                "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
                "bias_mw": float(np.mean(error)),
                "coverage_95": float(np.mean(covered)),
                "mean_interval_width_95_mw": float(np.mean(upper - lower)),
                "pit_mean": float(np.mean(pit)),
                "pit_variance": float(np.var(pit)),
            }
        )
        ensemble[f"actual_change_{horizon}m_mw"] = actual
        ensemble[f"predicted_change_{horizon}m_mw"] = location
        ensemble[f"lower_95_{horizon}m_mw"] = lower
        ensemble[f"upper_95_{horizon}m_mw"] = upper
        ensemble[f"pit_{horizon}m"] = pit
        ensemble[f"log_density_standardized_{horizon}m"] = log_density_z
        ensemble[f"log_density_mw_{horizon}m"] = log_density_mw
        aligned[f"actual_change_{horizon}m_mw"] = actual
        for member_index, seed in enumerate(SEEDS):
            aligned[f"seed_{seed}_predicted_change_{horizon}m_mw"] = member_locations[member_index]
            aligned[f"seed_{seed}_predicted_scale_{horizon}m_mw"] = member_scales[member_index]
            aligned[f"seed_{seed}_predicted_df_{horizon}m"] = member_dfs[member_index]
        pooled_errors.append(error)
        pooled_nll_z.append(-log_density_z)
        pooled_nll_mw.append(-log_density_mw)
        pooled_covered.append(covered.astype(np.float64))
        pooled_width.append(upper - lower)
        pooled_pit.append(pit)

    errors = np.concatenate(pooled_errors)
    pit_all = np.concatenate(pooled_pit)
    aggregate = {
        "model": MODEL,
        "variant": VARIANT,
        "ensemble_type": "equal_weight_student_t_mixture",
        "training_period": "2015-01-01 through 2018-12-31",
        "validation_year": 2019,
        "seeds": "42,142,242",
        "forecast_origins": len(timestamps),
        "origin_horizon_pairs": len(timestamps) * len(HORIZONS),
        "nll_standardized": float(np.mean(np.concatenate(pooled_nll_z))),
        "nll_mw_density": float(np.mean(np.concatenate(pooled_nll_mw))),
        "mae_mw": float(np.mean(np.abs(errors))),
        "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_mw": float(np.mean(errors)),
        "coverage_95": float(np.mean(np.concatenate(pooled_covered))),
        "mean_interval_width_95_mw": float(np.mean(np.concatenate(pooled_width))),
        "pit_mean": float(np.mean(pit_all)),
        "pit_variance": float(np.var(pit_all)),
    }

    member_rows = []
    for member, seed in zip(members, SEEDS):
        errors_seed = locations_mw[SEEDS.index(seed)] - actual_mw
        standardized = (target_z - predictions[seed]["mu"]) / predictions[seed]["scale"]
        log_density = student_t.logpdf(standardized, df=predictions[seed]["df"]) - np.log(predictions[seed]["scale"])
        member_rows.append(
            {
                **member,
                "recomputed_nll_standardized": float(-np.mean(log_density)),
                "recomputed_mae_mw": float(np.mean(np.abs(errors_seed))),
                "recomputed_rmse_mw": float(np.sqrt(np.mean(np.square(errors_seed)))),
            }
        )

    DESTINATION.mkdir(parents=True, exist_ok=False)
    ensemble.to_parquet(DESTINATION / "ensemble_predictions.parquet", index=False)
    aligned.to_parquet(DESTINATION / "aligned_member_predictions.parquet", index=False)
    horizon_metrics = pd.DataFrame(horizon_rows)
    horizon_metrics.to_csv(DESTINATION / "ensemble_horizon_metrics.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(DESTINATION / "ensemble_aggregate_metrics.csv", index=False)
    pd.DataFrame(member_rows).to_csv(DESTINATION / "member_metrics.csv", index=False)
    atomic_json(
        DESTINATION / "ensemble_manifest.json",
        {
            "status": "complete",
            "model": MODEL,
            "variant": VARIANT,
            "stage": "development_extension",
            "training_period": "2015-01-01 through 2018-12-31",
            "validation_period": "2019-01-01 through 2019-12-31",
            "seeds": list(SEEDS),
            "members": members,
            "combination_formula": (
                "equal-weight mixture of the three saved Student-t predictive distributions; "
                "point prediction is the arithmetic mean of member locations; mixture density "
                "uses log-sum-exp; 95% intervals are numerical mixture quantiles"
            ),
            "weight_selection": "none; fixed at one-third per member",
            "member_refitting": "none",
            "checkpoint_rule": "each member's minimum validation NLL checkpoint",
            "forecast_origins": len(timestamps),
            "horizons_minutes": list(HORIZONS),
            "aggregate_metrics": aggregate,
        },
    )
    print(pd.DataFrame(member_rows).to_string(index=False), flush=True)
    print(pd.DataFrame([aggregate]).to_string(index=False), flush=True)
    print(f"ensemble_directory={DESTINATION}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
