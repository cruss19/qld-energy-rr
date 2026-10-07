"""Build the five RR 2020 three-seed TCN ensembles and comparison tables.

This is an RR-only evidence builder.  It reads the frozen 2020 predictions from
the completed RR final refits and applies the ensemble rules declared in the RR
experimental design:

* deterministic TCNs: equal-weight arithmetic mean of point predictions;
* TCN_starNLL: equal-weight mixture of the three Student-t distributions.

No member is refitted and no ensemble weight is estimated from 2020 outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t


RR_ROOT = Path(__file__).resolve().parents[1]
RUNS = RR_ROOT / "training_output" / "runs"
ENSEMBLES = RR_ROOT / "training_output" / "ensembles"
OUTPUT = RR_ROOT / "outputs" / "09_tcn_ensemble_comparison"
NOTEBOOK = RR_ROOT / "notebooks" / "09_tcn_ensemble_comparison.ipynb"
MODEL_FRAME = (
    RR_ROOT
    / "training_output"
    / "private_cache"
    / "rr_original_model_frame_2015_2020.parquet"
)

FAMILIES = ("TCN1", "TCN2", "TCN3", "TCN_star", "TCN_starNLL")
PROBABILISTIC_FAMILY = "TCN_starNLL"
SEEDS = (42, 142, 242)
HORIZONS = (5, 10, 15, 20, 25, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact GitHub Markdown table without optional dependencies."""
    display = frame.copy()
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def point_metrics(actual: np.ndarray, predicted: np.ndarray, origin: np.ndarray) -> dict:
    error = predicted - actual
    future_level = origin + actual
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
        "bias_mw": float(np.mean(error)),
        "mape_percent": float(
            np.mean(np.abs(error) / np.maximum(np.abs(future_level), 1.0)) * 100.0
        ),
    }


def mixture_quantile(
    probability: float,
    locations: np.ndarray,
    scales: np.ndarray,
    dfs: np.ndarray,
) -> np.ndarray:
    lower = np.min(
        locations + scales * student_t.ppf(1e-8, df=dfs), axis=0
    )
    upper = np.max(
        locations + scales * student_t.ppf(1.0 - 1e-8, df=dfs), axis=0
    )
    for _ in range(48):
        middle = (lower + upper) / 2.0
        cdf = np.mean(
            student_t.cdf((middle[None, :] - locations) / scales, df=dfs),
            axis=0,
        )
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    return (lower + upper) / 2.0


def aligned_member_frames(family: str) -> tuple[list[pd.DataFrame], list[dict]]:
    frames: list[pd.DataFrame] = []
    members: list[dict] = []
    canonical_origins: np.ndarray | None = None
    canonical_actuals: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        run = RUNS / family / "final" / "train_2015_2019" / f"seed_{seed}"
        completion_path = run / "completed_run.json"
        prediction_path = run / "evaluation_2020_predictions.parquet"
        metric_path = run / "evaluation_2020_horizon_metrics.csv"
        frozen_path = run / "frozen_model.pt"
        for required in (completion_path, prediction_path, metric_path, frozen_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing required ensemble member artifact: {required}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            raise RuntimeError(f"Run is not normally complete: {completion_path}")
        frame = pd.read_parquet(prediction_path).sort_values("forecast_origin").reset_index(drop=True)
        origins = pd.to_datetime(frame["forecast_origin"]).to_numpy()
        if canonical_origins is None:
            canonical_origins = origins
            canonical_actuals = {
                horizon: frame[f"actual_change_{horizon}m_mw"].to_numpy(dtype=np.float64)
                for horizon in HORIZONS
            }
        elif not np.array_equal(origins, canonical_origins):
            raise RuntimeError(f"Forecast-origin mismatch for {family} seed {seed}")
        for horizon in HORIZONS:
            actual = frame[f"actual_change_{horizon}m_mw"].to_numpy(dtype=np.float64)
            if not np.allclose(actual, canonical_actuals[horizon], rtol=0.0, atol=1e-9):
                raise RuntimeError(f"Target mismatch for {family} seed {seed}, horizon {horizon}")
        frames.append(frame)
        members.append(
            {
                "seed": seed,
                "run_directory": str(run),
                "fixed_epochs": int(completion["fixed_epochs"]),
                "run_identity": completion["identity"],
                "prediction_sha256": sha256(prediction_path),
                "frozen_model_sha256": sha256(frozen_path),
            }
        )
    return frames, members


def build_family(family: str, origin_levels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, members = aligned_member_frames(family)
    origins = pd.to_datetime(frames[0]["forecast_origin"])
    origin_demand = origin_levels.reindex(origins).to_numpy(dtype=np.float64)
    if np.isnan(origin_demand).any():
        raise RuntimeError(f"Missing origin demand while scoring {family}")

    ensemble = pd.DataFrame({"forecast_origin": origins})
    aligned = pd.DataFrame({"forecast_origin": origins})
    horizon_rows: list[dict] = []
    pooled_actual: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    pooled_origin: list[np.ndarray] = []
    pooled_nll: list[np.ndarray] = []
    pooled_covered: list[np.ndarray] = []
    pooled_width: list[np.ndarray] = []
    pooled_pit: list[np.ndarray] = []

    for horizon in HORIZONS:
        actual = frames[0][f"actual_change_{horizon}m_mw"].to_numpy(dtype=np.float64)
        locations = np.stack(
            [frame[f"predicted_change_{horizon}m_mw"].to_numpy(dtype=np.float64) for frame in frames]
        )
        predicted = locations.mean(axis=0)
        ensemble[f"actual_change_{horizon}m_mw"] = actual
        ensemble[f"predicted_change_{horizon}m_mw"] = predicted
        aligned[f"actual_change_{horizon}m_mw"] = actual
        for seed, locations_seed in zip(SEEDS, locations):
            aligned[f"seed_{seed}_predicted_change_{horizon}m_mw"] = locations_seed

        row = {
            "model": family,
            "ensemble_type": (
                "equal_weight_student_t_mixture"
                if family == PROBABILISTIC_FAMILY
                else "equal_weight_arithmetic_mean"
            ),
            "horizon_minutes": horizon,
            "forecast_origins": len(origins),
            **point_metrics(actual, predicted, origin_demand),
        }

        if family == PROBABILISTIC_FAMILY:
            scales = np.stack(
                [frame[f"predicted_scale_{horizon}m_mw"].to_numpy(dtype=np.float64) for frame in frames]
            )
            dfs = np.stack(
                [frame[f"predicted_df_{horizon}m"].to_numpy(dtype=np.float64) for frame in frames]
            )
            if np.any(scales <= 0.0) or np.any(dfs <= 2.0):
                raise RuntimeError(f"Invalid Student-t parameters for horizon {horizon}")
            standardized = (actual[None, :] - locations) / scales
            member_log_density = student_t.logpdf(standardized, df=dfs) - np.log(scales)
            log_density = logsumexp(member_log_density, axis=0) - math.log(len(SEEDS))
            pit = np.mean(student_t.cdf(standardized, df=dfs), axis=0)
            lower = mixture_quantile(0.025, locations, scales, dfs)
            upper = mixture_quantile(0.975, locations, scales, dfs)
            covered = (actual >= lower) & (actual <= upper)
            width = upper - lower
            ensemble[f"lower_95_{horizon}m_mw"] = lower
            ensemble[f"upper_95_{horizon}m_mw"] = upper
            ensemble[f"pit_{horizon}m"] = pit
            ensemble[f"log_density_{horizon}m"] = log_density
            for seed, scale, df in zip(SEEDS, scales, dfs):
                aligned[f"seed_{seed}_predicted_scale_{horizon}m_mw"] = scale
                aligned[f"seed_{seed}_predicted_df_{horizon}m"] = df
            row.update(
                {
                    "nll": float(-np.mean(log_density)),
                    "coverage_95": float(np.mean(covered)),
                    "mean_interval_width_95_mw": float(np.mean(width)),
                    "pit_mean": float(np.mean(pit)),
                    "pit_variance": float(np.var(pit)),
                }
            )
            pooled_nll.append(-log_density)
            pooled_covered.append(covered.astype(np.float64))
            pooled_width.append(width)
            pooled_pit.append(pit)
        else:
            row.update(
                {
                    "nll": np.nan,
                    "coverage_95": np.nan,
                    "mean_interval_width_95_mw": np.nan,
                    "pit_mean": np.nan,
                    "pit_variance": np.nan,
                }
            )
        horizon_rows.append(row)
        pooled_actual.append(actual)
        pooled_predicted.append(predicted)
        pooled_origin.append(origin_demand)

    horizon_metrics = pd.DataFrame(horizon_rows)
    pooled_point = point_metrics(
        np.concatenate(pooled_actual),
        np.concatenate(pooled_predicted),
        np.concatenate(pooled_origin),
    )
    aggregate = {
        "model": family,
        "ensemble_type": horizon_metrics["ensemble_type"].iloc[0],
        "seeds": "42,142,242",
        "forecast_origins": len(origins),
        "origin_horizon_pairs": len(origins) * len(HORIZONS),
        **pooled_point,
        "mae_30m_mw": float(
            horizon_metrics.loc[horizon_metrics.horizon_minutes == 30, "mae_mw"].iloc[0]
        ),
        "rmse_30m_mw": float(
            horizon_metrics.loc[horizon_metrics.horizon_minutes == 30, "rmse_mw"].iloc[0]
        ),
        "nll": float(np.mean(np.concatenate(pooled_nll))) if pooled_nll else np.nan,
        "coverage_95": (
            float(np.mean(np.concatenate(pooled_covered))) if pooled_covered else np.nan
        ),
        "mean_interval_width_95_mw": (
            float(np.mean(np.concatenate(pooled_width))) if pooled_width else np.nan
        ),
        "pit_mean": float(np.mean(np.concatenate(pooled_pit))) if pooled_pit else np.nan,
        "pit_variance": float(np.var(np.concatenate(pooled_pit))) if pooled_pit else np.nan,
    }

    destination = ENSEMBLES / f"{family}_ensemble_2020_seeds_42_142_242"
    destination.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(destination / "ensemble_predictions.parquet", index=False)
    aligned.to_parquet(destination / "aligned_member_predictions.parquet", index=False)
    horizon_metrics.to_csv(destination / "ensemble_horizon_metrics.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(destination / "ensemble_aggregate_metrics.csv", index=False)
    write_json(
        destination / "ensemble_manifest.json",
        {
            "status": "complete",
            "model": family,
            "evaluation_year": 2020,
            "training_period": "2015-01-01 through 2019-12-31",
            "seeds": list(SEEDS),
            "members": members,
            "combination_formula": (
                "equal-weight mixture of member Student-t predictive distributions; "
                "point prediction is the arithmetic mean of member locations; mixture "
                "density uses log-sum-exp; intervals are mixture quantiles"
                if family == PROBABILISTIC_FAMILY
                else "equal-weight arithmetic mean of the three member point predictions"
            ),
            "weight_selection": "none; weights fixed at 1/3 before 2020 evaluation",
            "member_refitting": "none",
            "forecast_origins": len(origins),
            "horizons_minutes": list(HORIZONS),
            "aggregate_metrics": aggregate,
        },
    )
    return horizon_metrics, pd.DataFrame([aggregate])


def build_notebook(aggregate: pd.DataFrame, horizon: pd.DataFrame) -> None:
    best = aggregate.sort_values(
        ["multi_metric_mean_rank", "mae_mw", "rmse_mw"], kind="stable"
    ).iloc[0]
    display_columns = [
        "overall_point_rank",
        "model",
        "ensemble_type",
        "mae_mw",
        "rmse_mw",
        "bias_mw",
        "mape_percent",
        "mae_30m_mw",
        "nll",
        "coverage_95",
        "mean_interval_width_95_mw",
        "multi_metric_mean_rank",
    ]
    aggregate_markdown = markdown_table(aggregate[display_columns].round(4))
    cells = [
        nbf.v4.new_markdown_cell(
            "# 09 — TCN ensemble comparison on frozen 2020 evaluation\n\n"
            "This notebook compares the five completed three-seed RR TCN families. "
            "All members were fitted on 2015–2019 for durations fixed from the median "
            "official best epoch across the 2017, 2018 and 2019 development folds. "
            "The 2020 period was scored once after the member states and ensemble rules "
            "were frozen."
        ),
        nbf.v4.new_markdown_cell(
            "## Ensemble contract\n\n"
            "- Seeds: 42, 142 and 242.\n"
            "- Deterministic TCN1, TCN2, TCN3 and TCN_star: equal-weight arithmetic "
            "mean of member point predictions.\n"
            "- TCN_starNLL: equal-weight mixture of the three Student-t predictive "
            "distributions. Its density uses stable log-sum-exp and its 95% intervals "
            "are numerical mixture quantiles. The mixture is not collapsed to a single "
            "Student-t distribution.\n"
            "- No ensemble weight was fitted and no member was refitted against 2020.\n"
            "- All five ensembles use exactly the same 105,402 forecast origins and six horizons."
        ),
        nbf.v4.new_markdown_cell("## Aggregate comparison\n\n" + aggregate_markdown),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) "
            "if (p / 'outputs' / '09_tcn_ensemble_comparison').is_dir())\n"
            "OUT = ROOT / 'outputs' / '09_tcn_ensemble_comparison'\n"
            "aggregate = pd.read_csv(OUT / 'ensemble_model_comparison_2020.csv')\n"
            "display(aggregate.round(4))"
        ),
        nbf.v4.new_markdown_cell(
            "### Ranking rule\n\n"
            "The point-performance order minimizes the mean ordinal rank across pooled "
            "six-horizon MAE, pooled six-horizon RMSE and 30-minute MAE. Pooled bias and "
            "MAPE remain visible guardrails. NLL and calibration diagnostics are shown "
            "only for TCN_starNLL because the deterministic models do not define predictive "
            "densities. This avoids inventing probabilistic scores for point models."
        ),
        nbf.v4.new_markdown_cell("## Six-horizon comparison"),
        nbf.v4.new_code_cell(
            "horizon = pd.read_csv(OUT / 'ensemble_horizon_comparison_2020.csv')\n"
            "horizon_mae = pd.read_csv(OUT / 'ensemble_horizon_mae_wide_2020.csv')\n"
            "display(horizon_mae.round(3))\n"
            "display(horizon.round(4))\n"
            "display(Image(filename=str(OUT / 'ensemble_mae_by_horizon_2020.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\n\n"
            f"Under the declared multi-metric point comparison, **{best['model']}** is "
            f"ranked first (pooled MAE {best['mae_mw']:.3f} MW, pooled RMSE "
            f"{best['rmse_mw']:.3f} MW and 30-minute MAE {best['mae_30m_mw']:.3f} MW). "
            "This is the frozen 2020 comparison of the five TCN ensemble families. "
            "Probabilistic quality is interpreted separately for TCN_starNLL from its "
            "mixture NLL, coverage, interval width and PIT diagnostics. Statistical and "
            "external reference models are not inserted here until their independently "
            "defined RR evaluations are complete."
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    NOTEBOOK.write_text(nbf.writes(notebook), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ENSEMBLES.mkdir(parents=True, exist_ok=True)
    origin = pd.read_parquet(MODEL_FRAME, columns=["datetime", "totaldemand_mw"])
    origin["datetime"] = pd.to_datetime(origin["datetime"])
    origin_levels = origin.drop_duplicates("datetime").set_index("datetime")["totaldemand_mw"]

    horizon_parts: list[pd.DataFrame] = []
    aggregate_parts: list[pd.DataFrame] = []
    for family in FAMILIES:
        print(f"Building {family} ensemble", flush=True)
        horizon, aggregate = build_family(family, origin_levels)
        horizon_parts.append(horizon)
        aggregate_parts.append(aggregate)

    horizon_table = pd.concat(horizon_parts, ignore_index=True)
    aggregate_table = pd.concat(aggregate_parts, ignore_index=True)
    for metric in ("mae_mw", "rmse_mw", "mae_30m_mw"):
        aggregate_table[f"{metric}_rank"] = aggregate_table[metric].rank(
            method="min", ascending=True
        )
    aggregate_table["multi_metric_mean_rank"] = aggregate_table[
        ["mae_mw_rank", "rmse_mw_rank", "mae_30m_mw_rank"]
    ].mean(axis=1)
    aggregate_table = aggregate_table.sort_values(
        ["multi_metric_mean_rank", "mae_mw", "rmse_mw"], kind="stable"
    ).reset_index(drop=True)
    aggregate_table.insert(0, "overall_point_rank", np.arange(1, len(aggregate_table) + 1))

    aggregate_table.to_csv(OUTPUT / "ensemble_model_comparison_2020.csv", index=False)
    horizon_table.to_csv(OUTPUT / "ensemble_horizon_comparison_2020.csv", index=False)
    horizon_mae_wide = (
        horizon_table.pivot(index="horizon_minutes", columns="model", values="mae_mw")
        .reindex(columns=FAMILIES)
        .reset_index()
    )
    horizon_mae_wide.to_csv(OUTPUT / "ensemble_horizon_mae_wide_2020.csv", index=False)
    (OUTPUT / "ensemble_model_comparison_2020.md").write_text(
        "# Frozen 2020 TCN ensemble comparison\n\n"
        + markdown_table(aggregate_table.round(4))
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for family in FAMILIES:
        subset = horizon_table[horizon_table.model == family]
        ax.plot(subset.horizon_minutes, subset.mae_mw, marker="o", linewidth=2, label=family)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Frozen 2020 three-seed ensemble MAE by horizon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "ensemble_mae_by_horizon_2020.png", dpi=180)
    plt.close(fig)

    write_json(
        OUTPUT / "comparison_manifest.json",
        {
            "status": "complete",
            "evaluation_year": 2020,
            "models": list(FAMILIES),
            "seeds": list(SEEDS),
            "forecast_origins": int(horizon_table.forecast_origins.min()),
            "horizons_minutes": list(HORIZONS),
            "point_ranking_rule": (
                "mean ordinal rank across pooled MAE, pooled RMSE and 30-minute MAE; "
                "tie-break pooled MAE then pooled RMSE"
            ),
            "probabilistic_comparison_scope": (
                "NLL, coverage, interval width and PIT are reported only for TCN_starNLL"
            ),
            "source": "RR frozen 2020 member predictions only",
        },
    )
    build_notebook(aggregate_table, horizon_table)
    print(aggregate_table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
