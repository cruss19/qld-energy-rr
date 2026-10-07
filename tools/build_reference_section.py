"""Maintainer-only builder for reference and final-comparison evidence.

This evaluation/reporting utility requires undistributed preserved row-level
predictions. It does not fit or train any model and is not a clean-clone path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rr_plot_style import apply_rr_plot_style


apply_rr_plot_style()
HORIZONS = (5, 10, 15, 20, 25, 30)
TCNS = ("TCN1", "TCN2", "TCN3", "TCN_star", "TCN_starNLL")
AEMO = "aemo_p5min_external_benchmark"
REFERENCE_LABELS = {
    "Persistence": "Persistence",
    "Weekly seasonal naive": "Weekly seasonal naive",
    "Comparable-history empirical change": "Comparable-history empirical change",
    "Bayesian random walk": "Bayesian random walk",
    "Ridge regression": "Ridge regression",
}
PROBABILISTIC_REFERENCES = ("Gaussian linear", "Student-t linear", "Empirical residual")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(map(str, view.columns)) + " |",
            "| " + " | ".join("---" for _ in view.columns) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def load_six_horizon(model: str) -> pd.DataFrame:
    if model in TCNS:
        path = ROOT / f"training_output/ensembles/{model}_ensemble_2020_seeds_42_142_242/ensemble_predictions.parquet"
        frame = pd.read_parquet(path)
    else:
        path = ROOT / "training_output/reference_models/aemo_p5min_external_benchmark/all_folds/predictions.parquet"
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["fold"].eq("evaluate_2020")].drop(columns="fold")
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    return frame.sort_values("forecast_origin").drop_duplicates("forecast_origin").set_index("forecast_origin")


def six_horizon_metrics(model: str, frame: pd.DataFrame) -> tuple[dict, list[dict]]:
    rows = []
    all_actual, all_predicted = [], []
    for horizon in HORIZONS:
        actual = frame[f"actual_change_{horizon}m_mw"].to_numpy(float)
        predicted = frame[f"predicted_change_{horizon}m_mw"].to_numpy(float)
        error = predicted - actual
        row = {
            "model": model,
            "horizon_minutes": horizon,
            "forecast_origins": len(frame),
            "mae_mw": np.mean(np.abs(error)),
            "rmse_mw": np.sqrt(np.mean(error**2)),
            "bias_mw": np.mean(error),
        }
        if f"log_density_{horizon}m" in frame:
            lower = frame[f"lower_95_{horizon}m_mw"].to_numpy(float)
            upper = frame[f"upper_95_{horizon}m_mw"].to_numpy(float)
            row.update(
                nll=-frame[f"log_density_{horizon}m"].mean(),
                coverage_95=np.mean((actual >= lower) & (actual <= upper)),
                mean_width_95_mw=np.mean(upper - lower),
                mean_pit=frame[f"pit_{horizon}m"].mean(),
            )
        rows.append(row)
        all_actual.append(actual)
        all_predicted.append(predicted)
    actual = np.concatenate(all_actual)
    predicted = np.concatenate(all_predicted)
    error = predicted - actual
    aggregate = {
        "model": model,
        "forecast_origins": len(frame),
        "origin_horizon_pairs": len(actual),
        "mae_mw": np.mean(np.abs(error)),
        "rmse_mw": np.sqrt(np.mean(error**2)),
        "bias_mw": np.mean(error),
    }
    if "nll" in rows[0]:
        for key in ("nll", "coverage_95", "mean_width_95_mw", "mean_pit"):
            aggregate[key] = np.mean([row[key] for row in rows])
    return aggregate, rows


def load_original_references() -> dict[str, pd.DataFrame]:
    path = ROOT / "outputs/reference_models/expanding_window_oof_predictions.parquet"
    source = pd.read_parquet(path)
    source = source.loc[source["evaluation_year"].eq(2020)].copy()
    source["forecast_origin"] = pd.to_datetime(source["forecast_origin"])
    result = {}
    for source_name, label in REFERENCE_LABELS.items():
        frame = source.loc[source["model"].eq(source_name)].copy()
        result[label] = frame.sort_values("forecast_origin").set_index("forecast_origin")
    return result


def load_probabilistic_references() -> dict[str, pd.DataFrame]:
    path = ROOT / "outputs/probabilistic_reference_models/probabilistic_oof_predictions.parquet"
    source = pd.read_parquet(path)
    source = source.loc[source["evaluation_year"].eq(2020)].copy()
    source["forecast_origin"] = pd.to_datetime(source["forecast_origin"])
    return {
        model: source.loc[source["model"].eq(model)].sort_values("forecast_origin").set_index("forecast_origin")
        for model in PROBABILISTIC_REFERENCES
    }


def endpoint_row(model: str, frame: pd.DataFrame, kind: str) -> dict:
    if kind == "six":
        actual = frame["actual_change_30m_mw"].to_numpy(float)
        predicted = frame["predicted_change_30m_mw"].to_numpy(float)
    elif kind == "original":
        actual = frame["actual_change_30min"].to_numpy(float)
        predicted = frame["predicted_change_30min"].to_numpy(float)
    else:
        actual = frame["actual_change_30min"].to_numpy(float)
        predicted = frame["location_change_30min"].to_numpy(float)
    error = predicted - actual
    row = {
        "model": model,
        "forecast_origins": len(frame),
        "mae_30m_mw": np.mean(np.abs(error)),
        "rmse_30m_mw": np.sqrt(np.mean(error**2)),
        "bias_30m_mw": np.mean(error),
        "mean_nll_30m": np.nan,
        "mean_crps_30m_mw": np.nan,
        "coverage_80": np.nan,
        "coverage_95": np.nan,
        "mean_width_95_mw": np.nan,
        "mean_pit": np.nan,
    }
    if kind == "probabilistic":
        row.update(
            mean_nll_30m=frame["negative_log_likelihood"].mean(),
            mean_crps_30m_mw=frame["crps_mw"].mean(),
            coverage_80=frame["covered_80"].mean(),
            coverage_95=frame["covered_95"].mean(),
            mean_width_95_mw=frame["width_95_mw"].mean(),
            mean_pit=frame["pit"].mean(),
        )
    elif model == "TCN_starNLL":
        row.update(
            mean_nll_30m=-frame["log_density_30m"].mean(),
            coverage_95=np.mean(
                (actual >= frame["lower_95_30m_mw"].to_numpy(float))
                & (actual <= frame["upper_95_30m_mw"].to_numpy(float))
            ),
            mean_width_95_mw=np.mean(
                frame["upper_95_30m_mw"].to_numpy(float)
                - frame["lower_95_30m_mw"].to_numpy(float)
            ),
            mean_pit=frame["pit_30m"].mean(),
        )
    elif model == "Bayesian random walk":
        location = frame["predictive_location_change_mw"].to_numpy(float)
        scale = frame["predictive_scale_mw"].to_numpy(float)
        degrees = frame["predictive_degrees_of_freedom"].to_numpy(float)
        valid = np.isfinite(location) & np.isfinite(scale) & np.isfinite(degrees)
        row["mean_nll_30m"] = -np.mean(student_t.logpdf(actual[valid], df=degrees[valid], loc=location[valid], scale=scale[valid]))
        row["coverage_80"] = np.mean(
            (frame["actual_demand_t_plus_30"] >= frame["predictive_80_lower_demand_mw"])
            & (frame["actual_demand_t_plus_30"] <= frame["predictive_80_upper_demand_mw"])
        )
        row["coverage_95"] = np.mean(
            (frame["actual_demand_t_plus_30"] >= frame["predictive_95_lower_demand_mw"])
            & (frame["actual_demand_t_plus_30"] <= frame["predictive_95_upper_demand_mw"])
        )
        row["mean_width_95_mw"] = np.mean(
            frame["predictive_95_upper_demand_mw"] - frame["predictive_95_lower_demand_mw"]
        )
        row["mean_pit"] = np.mean(student_t.cdf(actual[valid], df=degrees[valid], loc=location[valid], scale=scale[valid]))
    return row


def build_probabilistic_notebook() -> None:
    out = ROOT / "outputs" / "probabilistic_reference_models"
    by_fold = pd.read_csv(out / "probabilistic_metrics_by_fold.csv")
    pooled = pd.read_csv(out / "probabilistic_metrics_pooled_2017_2020.csv")
    parameters = pd.read_csv(out / "probabilistic_fold_parameters.csv")
    manifest = pd.read_json(out / "probabilistic_reference_assets_manifest.json", typ="series")
    cells = [
        nbf.v4.new_markdown_cell(
            "# 05b — Probabilistic linear reference models\n\n"
            "This RR notebook preserves DREAM's three established 30-minute probabilistic references: Gaussian linear, Student-t linear, and empirical residual. "
            "The retained evidence covers expanding-window evaluation years 2017–2020 only. No model was refitted during RR reconstruction."
        ),
        nbf.v4.new_markdown_cell(
            "## Fixed model contracts\n\n"
            "All three models use the same 55-feature, fold-fitted Ridge information contract as notebook 05 and the same 30-minute demand-change target. "
            "For each fold, the final 180 training days form a chronological calibration block; the location fit used for calibration sees only earlier training rows.\n\n"
            "- **Gaussian linear:** original Ridge location with Gaussian residual scale fitted on the calibration block.\n"
            "- **Student-t linear:** penalised linear location fitted with five Student-t IRLS updates; degrees of freedom and scale are calibrated on training-only residuals.\n"
            "- **Empirical residual:** original Ridge location plus the complete calibration-residual empirical distribution; interval/PIT calculations use its empirical CDF and log score uses a 30–200-bin Freedman–Diaconis histogram with a 0.5 pseudocount."
        ),
        nbf.v4.new_markdown_cell("## Fold results\n\n" + markdown_table(by_fold, ["evaluation_year", "model", "observations", "mean_nll", "mean_crps_mw", "coverage_80", "coverage_95"])),
        nbf.v4.new_markdown_cell("## Pooled 2017–2020 results\n\n" + markdown_table(pooled, ["model", "observations", "mean_nll", "mean_crps_mw", "coverage_80", "coverage_95", "mean_pit"])),
        nbf.v4.new_markdown_cell("## Fold-contained fitted parameters\n\n" + markdown_table(parameters, ["evaluation_year", "ridge_alpha", "gaussian_scale_mw", "student_degrees_freedom", "student_scale_mw", "empirical_histogram_bins"])),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\n"
            "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'outputs' / 'probabilistic_reference_models').is_dir())\n"
            "OUT = ROOT / 'outputs' / 'probabilistic_reference_models'\n"
            "display(pd.read_csv(OUT / 'probabilistic_metrics_by_fold.csv').round(4))\n"
            "display(pd.read_csv(OUT / 'probabilistic_metrics_pooled_2017_2020.csv').round(4))"
        ),
        nbf.v4.new_markdown_cell(
            "## Provenance and boundary\n\n"
            "The prediction evidence was copied from DREAM's preserved pre-v6 workspace and filtered before entering RR. "
            f"The retained years are {manifest['retained_evaluation_years']}; the latest target is {manifest['latest_target_timestamp']}. "
            "The source checksums and transformation are recorded in `probabilistic_reference_assets_manifest.json`."
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "qld-energy-rr", "language": "python", "name": "python3"}})
    nbf.write(notebook, ROOT / "notebooks" / "05b_probabilistic_reference_models.ipynb")


def render_comparison_figures(
    six_horizons: pd.DataFrame,
    endpoint: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Render public comparison figures from retained aggregate tables."""
    apply_rr_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, frame in six_horizons.groupby("model"):
        ax.plot(frame["horizon_minutes"], frame["mae_mw"], marker="o", label=model)
    ax.set(title="Frozen 2020 six-horizon comparison", xlabel="Horizon (minutes)", ylabel="MAE (MW)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(output_directory / "six_horizon_mae_2020.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = endpoint.sort_values("mae_30m_mw", ascending=True)
    ax.barh(ordered["model"], ordered["mae_30m_mw"])
    ax.set(title="Frozen 2020 30-minute endpoint comparison", xlabel="MAE (MW)", ylabel="")
    fig.tight_layout(); fig.savefig(output_directory / "endpoint_30m_mae_2020.png", dpi=160); plt.close(fig)


def main() -> None:
    build_probabilistic_notebook()
    out = ROOT / "outputs" / "10_final_2020_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    six = {model: load_six_horizon(model) for model in (*TCNS, AEMO)}
    six_common = next(iter(six.values())).index
    for frame in six.values():
        six_common = six_common.intersection(frame.index)
    six_common = six_common.sort_values()
    six = {model: frame.loc[six_common].copy() for model, frame in six.items()}

    six_aggregate, six_horizons = [], []
    for model, frame in six.items():
        aggregate, horizons = six_horizon_metrics(model, frame)
        six_aggregate.append(aggregate)
        six_horizons.extend(horizons)
    six_aggregate = pd.DataFrame(six_aggregate).sort_values(["mae_mw", "rmse_mw"]).reset_index(drop=True)
    six_aggregate.insert(0, "rank_by_mae", np.arange(1, len(six_aggregate) + 1))
    six_horizons = pd.DataFrame(six_horizons)
    six_aggregate.to_csv(out / "six_horizon_model_comparison_2020.csv", index=False)
    six_horizons.to_csv(out / "six_horizon_metrics_2020.csv", index=False)

    original = load_original_references()
    probabilistic = load_probabilistic_references()
    endpoint_frames = {**six, **original, **probabilistic}
    endpoint_common = next(iter(endpoint_frames.values())).index
    native_counts = {model: len(frame) for model, frame in endpoint_frames.items()}
    for frame in endpoint_frames.values():
        endpoint_common = endpoint_common.intersection(frame.index)
    endpoint_common = endpoint_common.sort_values()
    endpoint_frames = {model: frame.loc[endpoint_common].copy() for model, frame in endpoint_frames.items()}

    ridge = endpoint_frames["Ridge regression"]
    gaussian = endpoint_frames["Gaussian linear"]
    np.testing.assert_allclose(ridge["actual_change_30min"], gaussian["actual_change_30min"], rtol=0, atol=0)
    np.testing.assert_allclose(ridge["predicted_change_30min"], gaussian["location_change_30min"], rtol=0, atol=0)

    endpoint_rows = []
    for model, frame in endpoint_frames.items():
        kind = "six" if model in (*TCNS, AEMO) else "probabilistic" if model in PROBABILISTIC_REFERENCES else "original"
        endpoint_rows.append(endpoint_row(model, frame, kind))
    endpoint = pd.DataFrame(endpoint_rows).sort_values(["mae_30m_mw", "rmse_30m_mw"]).reset_index(drop=True)
    endpoint.insert(0, "rank_by_mae", np.arange(1, len(endpoint) + 1))
    endpoint.to_csv(out / "endpoint_30m_model_comparison_2020.csv", index=False)
    endpoint.loc[endpoint["mean_nll_30m"].notna()].sort_values("mean_nll_30m").to_csv(out / "probabilistic_30m_comparison_2020.csv", index=False)
    pd.DataFrame(
        [{"model": model, "native_2020_origins": count, "common_2020_origins": len(endpoint_common)} for model, count in native_counts.items()]
    ).to_csv(out / "common_sample_counts_2020.csv", index=False)

    render_comparison_figures(six_horizons, endpoint, out)

    cells = [
        nbf.v4.new_markdown_cell(
            "# 10 — Final frozen-2020 evaluation\n\n"
            f"The six-horizon table uses only the five TCN ensembles and AEMO on {len(six_common):,} common forecast origins. "
            f"The separate 30-minute endpoint table uses all fourteen eligible models on {len(endpoint_common):,} common origins. "
            "This separation prevents the original endpoint-only references from being misrepresented as six-horizon models."
        ),
        nbf.v4.new_markdown_cell("## Six-horizon comparison\n\n" + markdown_table(six_aggregate, ["rank_by_mae", "model", "forecast_origins", "mae_mw", "rmse_mw", "bias_mw"])),
        nbf.v4.new_markdown_cell("## 30-minute endpoint comparison\n\n" + markdown_table(endpoint, ["rank_by_mae", "model", "forecast_origins", "mae_30m_mw", "rmse_30m_mw", "bias_30m_mw"])),
        nbf.v4.new_markdown_cell(
            "## Probabilistic metrics\n\nNLL is comparable only among models with an explicit predictive density. CRPS is retained where the historical reference workflow recorded it. "
            "Missing metrics are not ranked or treated as zero.\n\n"
            + markdown_table(endpoint.loc[endpoint["mean_nll_30m"].notna()].sort_values("mean_nll_30m"), ["model", "mean_nll_30m", "mean_crps_30m_mw", "coverage_80", "coverage_95", "mean_width_95_mw", "mean_pit"])
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
            "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'outputs' / '10_final_2020_evaluation').is_dir())\n"
            "OUT = ROOT / 'outputs' / '10_final_2020_evaluation'\n"
            "display(pd.read_csv(OUT / 'six_horizon_model_comparison_2020.csv').round(4))\n"
            "display(Image(filename=str(OUT / 'six_horizon_mae_2020.png')))\n"
            "display(pd.read_csv(OUT / 'endpoint_30m_model_comparison_2020.csv').round(4))\n"
            "display(Image(filename=str(OUT / 'endpoint_30m_mae_2020.png')))"
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "qld-energy-rr", "language": "python", "name": "python3"}})
    nbf.write(notebook, ROOT / "notebooks" / "10_final_2020_evaluation.ipynb")
    print(six_aggregate[["rank_by_mae", "model", "mae_mw", "rmse_mw"]].to_string(index=False))
    print(endpoint[["rank_by_mae", "model", "mae_30m_mw", "rmse_30m_mw"]].to_string(index=False))


if __name__ == "__main__":
    main()
