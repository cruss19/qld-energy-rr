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
FINAL_CHAMPION = "TCN_starNLL_noSD"
FINAL_LABELS = {
    FINAL_CHAMPION: "TCN_starNLL_noSD three-seed ensemble",
    AEMO: "AEMO P5MIN external benchmark",
}
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


def load_final_30m_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the preselected champion and external AEMO benchmark."""
    champion_path = (
        ROOT
        / "training_output/ensembles/TCN_starNLL_noSD_ensemble_2020_seeds_42_142_242/ensemble_predictions.parquet"
    )
    aemo_path = (
        ROOT
        / "training_output/reference_models/aemo_p5min_external_benchmark/all_folds/predictions.parquet"
    )
    champion = pd.read_parquet(champion_path)
    aemo = pd.read_parquet(aemo_path)
    aemo = aemo.loc[aemo["fold"].eq("evaluate_2020")].drop(columns="fold")
    for frame in (champion, aemo):
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    champion = champion.sort_values("forecast_origin").drop_duplicates("forecast_origin").set_index("forecast_origin")
    aemo = aemo.sort_values("forecast_origin").drop_duplicates("forecast_origin").set_index("forecast_origin")
    return champion, aemo


def final_30m_row(model: str, frame: pd.DataFrame, actual: np.ndarray) -> dict:
    predicted = frame["predicted_change_30m_mw"].to_numpy(float)
    error = predicted - actual
    return {
        "model": FINAL_LABELS[model],
        "role": "2019-selected champion" if model == FINAL_CHAMPION else "external benchmark",
        "forecast_origins": len(frame),
        "mae_30m_mw": np.mean(np.abs(error)),
        "rmse_30m_mw": np.sqrt(np.mean(error**2)),
        "bias_30m_mw": np.mean(error),
    }


def render_final_30m_figure(comparison: pd.DataFrame, output_directory: Path) -> None:
    apply_rr_plot_style()
    ordered = comparison.sort_values("mae_30m_mw", ascending=False)
    gain = comparison.loc[
        comparison["role"].eq("2019-selected champion"),
        "mae_gain_vs_aemo_percent",
    ].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.cm.plasma(np.linspace(0.20, 0.75, len(ordered)))
    bars = ax.barh(ordered["model"], ordered["mae_30m_mw"], color=colors)
    ax.set(
        title=(
            f"Headline result: {gain:.2f}% lower 30-minute MAE than AEMO P5MIN\n"
            "Final held-out 2020 comparison"
        ),
        xlabel="MAE (MW)",
        ylabel="",
    )
    ax.bar_label(bars, fmt="%.2f MW", padding=4)
    ax.set_xlim(0, ordered["mae_30m_mw"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(output_directory / "endpoint_30m_mae_2020.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out = ROOT / "outputs" / "10_final_2020_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    champion, aemo = load_final_30m_comparison()
    champion_native = len(champion)
    aemo_native = len(aemo)
    common = champion.index.intersection(aemo.index).sort_values()
    champion = champion.loc[common].copy()
    aemo = aemo.loc[common].copy()
    actual = champion["actual_change_30m_mw"].to_numpy(float)
    np.testing.assert_allclose(
        actual,
        aemo["actual_change_30m_mw"].to_numpy(float),
        rtol=0,
        atol=1e-4,
    )

    comparison = pd.DataFrame(
        [
            final_30m_row(FINAL_CHAMPION, champion, actual),
            final_30m_row(AEMO, aemo, actual),
        ]
    ).sort_values(["mae_30m_mw", "rmse_30m_mw"]).reset_index(drop=True)
    comparison.insert(0, "rank_by_mae", np.arange(1, len(comparison) + 1))
    aemo_mae = comparison.loc[comparison["role"].eq("external benchmark"), "mae_30m_mw"].iloc[0]
    comparison["mae_gain_vs_aemo_percent"] = (aemo_mae - comparison["mae_30m_mw"]) / aemo_mae * 100
    comparison.loc[comparison["role"].eq("external benchmark"), "mae_gain_vs_aemo_percent"] = 0.0
    comparison.to_csv(out / "endpoint_30m_model_comparison_2020.csv", index=False)
    pd.DataFrame(
        [
            {"model": FINAL_LABELS[FINAL_CHAMPION], "native_2020_origins": champion_native, "common_2020_origins": len(common)},
            {"model": FINAL_LABELS[AEMO], "native_2020_origins": aemo_native, "common_2020_origins": len(common)},
        ]
    ).to_csv(out / "common_sample_counts_2020.csv", index=False)
    render_final_30m_figure(comparison, out)

    for stale_name in (
        "six_horizon_model_comparison_2020.csv",
        "six_horizon_metrics_2020.csv",
        "six_horizon_mae_2020.png",
        "probabilistic_30m_comparison_2020.csv",
        "reference_execution_status.csv",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()

    gain = comparison.loc[comparison["role"].eq("2019-selected champion"), "mae_gain_vs_aemo_percent"].iloc[0]

    cells = [
        nbf.v4.new_markdown_cell(
            f"# 10 — Final held-out 2020 evaluation\n\n"
            f"## Headline result: {gain:.2f}% lower 30-minute MAE than AEMO P5MIN\n\n"
            "The final comparison contains only the model selected from 2019 development evidence and the external AEMO P5MIN benchmark. "
            f"Both are scored at the equivalent 30-minute horizon on {len(common):,} common 2020 forecast origins."
        ),
        nbf.v4.new_markdown_cell(
            "## Frozen evaluation protocol\n\n"
            "`TCN_starNLL_noSD` was selected using 2019 validation only. Its seed 42, 142, and 242 members were then refitted on 2015–2019 for exactly 10 epochs, frozen, and combined as an equal-weight Student-t mixture before 2020 was scored. "
            "No other development candidate is ranked on the final period. AEMO is retained only as the independently published operational benchmark."
        ),
        nbf.v4.new_markdown_cell(
            "## Headline 30-minute comparison\n\n"
            + markdown_table(
                comparison,
                ["rank_by_mae", "model", "role", "forecast_origins", "mae_30m_mw", "rmse_30m_mw", "bias_30m_mw", "mae_gain_vs_aemo_percent"],
            )
        ),
        nbf.v4.new_markdown_cell(
            f"The extension ensemble reduces 30-minute MAE from {aemo_mae:.3f} MW to "
            f"{comparison.iloc[0]['mae_30m_mw']:.3f} MW: an absolute reduction of "
            f"{aemo_mae - comparison.iloc[0]['mae_30m_mw']:.3f} MW and a relative gain of **{gain:.2f}%**. "
            "The percentage is `(AEMO MAE - model MAE) / AEMO MAE × 100`, evaluated only on the common study-period origins."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
            "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'outputs' / '10_final_2020_evaluation').is_dir())\n"
            "OUT = ROOT / 'outputs' / '10_final_2020_evaluation'\n"
            "display(pd.read_csv(OUT / 'endpoint_30m_model_comparison_2020.csv').round(4))\n"
            "display(Image(filename=str(OUT / 'endpoint_30m_mae_2020.png')))"
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "qld-energy-rr", "language": "python", "name": "python3"}})
    nbf.write(notebook, ROOT / "notebooks" / "10_final_2020_evaluation.ipynb")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
