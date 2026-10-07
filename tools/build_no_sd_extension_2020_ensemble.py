"""Build the approved 2020 three-seed no-SD Student-t mixture ensemble."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
MODEL = "TCN_starNLL_causal_feature_estimates_no_sd"
PUBLIC_LABEL = "TCN_starNLL_noSD"
VARIANT = "demand_direct_radiation_estimated_no_sd"
SEEDS = (42, 142, 242)
HORIZONS = (5, 10, 15, 20, 25, 30)
RUN_BASE = ROOT / "training_output" / "runs" / MODEL / "final" / "train_2015_2019"
DESTINATION = ROOT / "training_output" / "ensembles" / f"{PUBLIC_LABEL}_ensemble_2020_seeds_42_142_242"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical(identity: dict) -> dict:
    result = json.loads(json.dumps(identity))
    result.pop("seed", None)
    result.pop("output_directory", None)
    return result


def mixture_quantile(probability, locations, scales, dfs):
    lower = np.min(locations + scales * student_t.ppf(1e-8, df=dfs), axis=0)
    upper = np.max(locations + scales * student_t.ppf(1 - 1e-8, df=dfs), axis=0)
    for _ in range(48):
        middle = (lower + upper) / 2
        cdf = np.mean(student_t.cdf((middle[None, :] - locations) / scales, df=dfs), axis=0)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    return (lower + upper) / 2


def main() -> int:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        manifest = DESTINATION / "ensemble_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete":
            print(f"already_complete={DESTINATION}")
            return 0
        raise FileExistsError(f"Refusing to overwrite incomplete ensemble: {DESTINATION}")

    frames, members = [], []
    reference_identity = reference_state = reference_origins = None
    reference_actual = {}
    for seed in SEEDS:
        run = RUN_BASE / f"seed_{seed}"
        paths = {
            "completion": run / "completed_run.json",
            "identity": run / "run_identity.json",
            "frozen": run / "frozen_model.pt",
            "predictions": run / "evaluation_2020_predictions.parquet",
            "metrics": run / "evaluation_2020_horizon_metrics.csv",
            "preprocessing": run / "preprocessing_state.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} is not ensemble-ready: {missing}")
        completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
        identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
        state = json.loads(paths["preprocessing"].read_text(encoding="utf-8"))
        if completion.get("status") != "complete" or completion.get("identity") != identity:
            raise RuntimeError(f"Seed {seed} completion identity mismatch")
        if identity.get("exact_model") != MODEL or identity.get("variant") != VARIANT:
            raise RuntimeError(f"Seed {seed} model identity mismatch")
        if int(identity.get("fixed_epochs", -1)) != 10 or int(identity.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed {seed} fixed-duration identity mismatch")
        normalized = canonical(identity)
        if reference_identity is None:
            reference_identity, reference_state = normalized, state
        elif normalized != reference_identity or state != reference_state:
            raise RuntimeError(f"Seed {seed} does not match the other final members")
        frame = pd.read_parquet(paths["predictions"]).sort_values("forecast_origin").reset_index(drop=True)
        origins = pd.to_datetime(frame["forecast_origin"]).to_numpy()
        if reference_origins is None:
            reference_origins = origins
            reference_actual = {h: frame[f"actual_change_{h}m_mw"].to_numpy(float) for h in HORIZONS}
        elif not np.array_equal(origins, reference_origins):
            raise RuntimeError(f"Seed {seed} forecast origins differ")
        for horizon in HORIZONS:
            np.testing.assert_allclose(
                frame[f"actual_change_{horizon}m_mw"].to_numpy(float),
                reference_actual[horizon], rtol=0, atol=1e-9,
            )
        frames.append(frame)
        members.append({
            "seed": seed,
            "run_directory": run.relative_to(ROOT).as_posix(),
            "fixed_epochs": 10,
            "prediction_sha256": sha256(paths["predictions"]),
            "frozen_model_sha256": sha256(paths["frozen"]),
        })

    ensemble = pd.DataFrame({"forecast_origin": pd.to_datetime(reference_origins)})
    aligned = pd.DataFrame({"forecast_origin": pd.to_datetime(reference_origins)})
    rows, pooled_error, pooled_nll, pooled_covered, pooled_width, pooled_pit = [], [], [], [], [], []
    for horizon in HORIZONS:
        actual = reference_actual[horizon]
        locations = np.stack([f[f"predicted_change_{horizon}m_mw"].to_numpy(float) for f in frames])
        scales = np.stack([f[f"predicted_scale_{horizon}m_mw"].to_numpy(float) for f in frames])
        dfs = np.stack([f[f"predicted_df_{horizon}m"].to_numpy(float) for f in frames])
        predicted = locations.mean(axis=0)
        standardized = (actual[None, :] - locations) / scales
        log_density = logsumexp(student_t.logpdf(standardized, df=dfs) - np.log(scales), axis=0) - math.log(3)
        pit = np.mean(student_t.cdf(standardized, df=dfs), axis=0)
        lower = mixture_quantile(0.025, locations, scales, dfs)
        upper = mixture_quantile(0.975, locations, scales, dfs)
        covered = (actual >= lower) & (actual <= upper)
        error = predicted - actual
        rows.append({
            "model": PUBLIC_LABEL,
            "horizon_minutes": horizon,
            "forecast_origins": len(actual),
            "mae_mw": float(np.mean(np.abs(error))),
            "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
            "bias_mw": float(np.mean(error)),
            "nll": float(-np.mean(log_density)),
            "coverage_95": float(np.mean(covered)),
            "mean_interval_width_95_mw": float(np.mean(upper - lower)),
            "pit_mean": float(np.mean(pit)),
        })
        ensemble[f"actual_change_{horizon}m_mw"] = actual
        ensemble[f"predicted_change_{horizon}m_mw"] = predicted
        ensemble[f"lower_95_{horizon}m_mw"] = lower
        ensemble[f"upper_95_{horizon}m_mw"] = upper
        ensemble[f"pit_{horizon}m"] = pit
        ensemble[f"log_density_{horizon}m"] = log_density
        aligned[f"actual_change_{horizon}m_mw"] = actual
        for index, seed in enumerate(SEEDS):
            aligned[f"seed_{seed}_predicted_change_{horizon}m_mw"] = locations[index]
            aligned[f"seed_{seed}_predicted_scale_{horizon}m_mw"] = scales[index]
            aligned[f"seed_{seed}_predicted_df_{horizon}m"] = dfs[index]
        pooled_error.append(error)
        pooled_nll.append(-log_density)
        pooled_covered.append(covered)
        pooled_width.append(upper - lower)
        pooled_pit.append(pit)

    error = np.concatenate(pooled_error)
    aggregate = {
        "model": PUBLIC_LABEL,
        "exact_model": MODEL,
        "variant": VARIANT,
        "ensemble_type": "equal_weight_student_t_mixture",
        "training_period": "2015-01-01 through 2019-12-31",
        "evaluation_year": 2020,
        "seeds": "42,142,242",
        "forecast_origins": len(ensemble),
        "origin_horizon_pairs": len(error),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
        "bias_mw": float(np.mean(error)),
        "mae_30m_mw": rows[-1]["mae_mw"],
        "nll": float(np.mean(np.concatenate(pooled_nll))),
        "coverage_95": float(np.mean(np.concatenate(pooled_covered))),
        "mean_interval_width_95_mw": float(np.mean(np.concatenate(pooled_width))),
        "pit_mean": float(np.mean(np.concatenate(pooled_pit))),
    }
    DESTINATION.mkdir(parents=True, exist_ok=False)
    ensemble.to_parquet(DESTINATION / "ensemble_predictions.parquet", index=False)
    aligned.to_parquet(DESTINATION / "aligned_member_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(DESTINATION / "ensemble_horizon_metrics.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(DESTINATION / "ensemble_aggregate_metrics.csv", index=False)
    write_json(DESTINATION / "ensemble_manifest.json", {
        "status": "complete",
        "model": PUBLIC_LABEL,
        "exact_model": MODEL,
        "variant": VARIANT,
        "stage": "final_extension",
        "training_period": "2015-01-01 through 2019-12-31",
        "evaluation_period": "2020-01-01 through 2020-12-31",
        "seeds": list(SEEDS),
        "members": members,
        "combination_formula": "equal-weight Student-t mixture; mean member locations for point prediction; log-sum-exp density; numerical mixture quantiles",
        "weight_selection": "none; fixed at one-third before 2020 scoring",
        "member_refitting": "none after frozen final states",
        "forecast_origins": len(ensemble),
        "horizons_minutes": list(HORIZONS),
        "aggregate_metrics": aggregate,
    })
    print(pd.DataFrame([aggregate]).to_string(index=False), flush=True)
    print(f"ensemble_directory={DESTINATION}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
