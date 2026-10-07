"""Maintainer-only reconstruction of TCN-development publication evidence.

This utility is deliberately results-only.  It reads completed RR development
runs from undistributed local artifacts, writes compact public evidence
tables/figures, and creates explanatory notebooks. It never imports a training
loop, fits a model, or scores 2020, and it is not a clean-clone command.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd


ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rr_plot_style import apply_rr_plot_style


RUNS = ROOT / "training_output" / "runs"
OUT = ROOT / "outputs" / "06_tcn_development"
NOTEBOOKS = ROOT / "notebooks"
MODELS = ("TCN1", "TCN2", "TCN3", "TCN_star", "TCN_starNLL")
YEARS = (2017, 2018, 2019)
HORIZONS = (5, 10, 15, 20, 25, 30)

ARCHITECTURE = [
    {
        "model": "TCN1",
        "temporal_design": "single consolidated 75-channel backbone",
        "sequence_steps": 13,
        "history": "forecast origin plus previous 60 minutes",
        "kernel_size": 3,
        "dilations": "1, 2",
        "branches": 1,
        "regional_treatment": "five regions flattened into 50 channels",
        "output": "six deterministic changes",
        "parameters": 72902,
    },
    {
        "model": "TCN2",
        "temporal_design": "symmetrical multi-branch equalizer",
        "sequence_steps": 13,
        "history": "forecast origin plus previous 60 minutes",
        "kernel_size": 3,
        "dilations": "1, 2",
        "branches": 3,
        "regional_treatment": "50-channel regional branch",
        "output": "six deterministic changes",
        "parameters": 96646,
    },
    {
        "model": "TCN3",
        "temporal_design": "hierarchical spatiotemporal learner",
        "sequence_steps": 13,
        "history": "forecast origin plus previous 60 minutes",
        "kernel_size": 3,
        "dilations": "1, 2",
        "branches": 3,
        "regional_treatment": "shared 10-channel encoder, then mean over regions",
        "output": "six deterministic changes",
        "parameters": 53222,
    },
    {
        "model": "TCN_star",
        "temporal_design": "deep-lookback TCN2 branch design",
        "sequence_steps": 505,
        "history": "forecast origin plus previous 42 hours",
        "kernel_size": 5,
        "dilations": "1, 2, 4, 8, 16, 32 repeated twice",
        "branches": 3,
        "regional_treatment": "50-channel regional branch",
        "output": "six deterministic changes",
        "parameters": 758470,
    },
    {
        "model": "TCN_starNLL",
        "temporal_design": "TCN_star backbone with Student-t heads",
        "sequence_steps": 505,
        "history": "forecast origin plus previous 42 hours",
        "kernel_size": 5,
        "dilations": "1, 2, 4, 8, 16, 32 repeated twice",
        "branches": 3,
        "regional_treatment": "50-channel regional branch",
        "output": "location, scale and df for each of six horizons",
        "parameters": 760018,
    },
]


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    summary_rows: list[dict] = []
    horizon_rows: list[dict] = []
    epoch_parts: list[pd.DataFrame] = []
    identities: dict[str, dict] = {}
    for model in MODELS:
        for year in YEARS:
            run = RUNS / model / "development" / f"validation_{year}" / "seed_42"
            completed_path = run / "completed_run.json"
            identity_path = run / "run_identity.json"
            epoch_path = run / "epoch_log.csv"
            for required in (completed_path, identity_path, epoch_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if completed.get("status") != "complete":
                raise RuntimeError(f"Development run is not complete: {run}")
            if model not in identities:
                identities[model] = identity
            metrics = pd.DataFrame(completed["metrics"])
            summary_rows.append(
                {
                    "model": model,
                    "validation_year": year,
                    "train_period": f"2015-{year - 1}",
                    "seed": 42,
                    "objective": identity["objective"],
                    "best_epoch": int(completed["best_epoch"]),
                    "completed_epochs": int(completed["completed_epochs"]),
                    "best_validation_objective": float(completed["best_validation_objective"]),
                    "mean_horizon_mae_mw": float(metrics["mae_mw"].mean()),
                    "combined_horizon_rmse_mw": float(math.sqrt((metrics["rmse_mw"] ** 2).mean())),
                    "mean_horizon_bias_mw": float(metrics["bias_mw"].mean()),
                    "mae_30m_mw": float(metrics.loc[metrics.horizon_minutes == 30, "mae_mw"].iloc[0]),
                    "rmse_30m_mw": float(metrics.loc[metrics.horizon_minutes == 30, "rmse_mw"].iloc[0]),
                }
            )
            for row in completed["metrics"]:
                horizon_rows.append({"model": model, "validation_year": year, "seed": 42, **row})
            epochs = pd.read_csv(epoch_path)
            epochs.insert(0, "model", model)
            epochs.insert(1, "validation_year", year)
            epochs.insert(2, "seed", 42)
            epoch_parts.append(epochs)
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(horizon_rows),
        pd.concat(epoch_parts, ignore_index=True),
        identities,
    )


def training_table(identities: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, identity in identities.items():
        hp = identity["training_hyperparameters"]
        rows.append(
            {
                "model": model,
                "objective": identity["objective"],
                "optimizer": identity["optimizer"],
                "learning_rate": hp["learning_rate"],
                "weight_decay": hp["weight_decay"],
                "dropout": hp["spatial_dropout"],
                "micro_batch_size": hp["batch_size"],
                "gradient_accumulation_steps": hp["gradient_accumulation_steps"],
                "gradient_clip_max_norm": hp["gradient_clip_max_norm"],
                "min_epochs": hp["min_epochs"],
                "max_epochs": hp["max_epochs"],
                "lr_patience": hp["lr_patience"],
                "early_stopping_patience": hp["es_patience"],
                "checkpoint_rule": identity["checkpoint_rule"],
            }
        )
    return pd.DataFrame(rows)


def save_evidence(summary: pd.DataFrame, horizon: pd.DataFrame, epochs: pd.DataFrame, identities: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "development_run_summary.csv", index=False)
    horizon.to_csv(OUT / "development_horizon_metrics.csv", index=False)
    epochs.to_csv(OUT / "development_epoch_logs.csv", index=False)
    pd.DataFrame(ARCHITECTURE).to_csv(OUT / "architecture_comparison.csv", index=False)
    training_table(identities).to_csv(OUT / "training_contract_comparison.csv", index=False)
    fold_mean = (
        summary.groupby("model", as_index=False)
        .agg(
            mean_horizon_mae_mw=("mean_horizon_mae_mw", "mean"),
            mean_combined_horizon_rmse_mw=("combined_horizon_rmse_mw", "mean"),
            mean_30m_mae_mw=("mae_30m_mw", "mean"),
            median_best_epoch=("best_epoch", "median"),
        )
        .sort_values("mean_horizon_mae_mw")
    )
    fold_mean.to_csv(OUT / "development_model_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    apply_rr_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for model in MODELS:
        part = summary[summary.model == model]
        ax.plot(part.validation_year, part.mean_horizon_mae_mw, marker="o", linewidth=2, label=model)
    ax.set(title="Development-fold six-horizon MAE", xlabel="Validation year", ylabel="Mean horizon MAE (MW)")
    ax.set_xticks(YEARS)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "development_fold_mae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    mean_horizon = horizon.groupby(["model", "horizon_minutes"], as_index=False).mae_mw.mean()
    for model in MODELS:
        part = mean_horizon[mean_horizon.model == model]
        ax.plot(part.horizon_minutes, part.mae_mw, marker="o", linewidth=2, label=model)
    ax.set(title="Mean development-fold MAE by forecast horizon", xlabel="Forecast horizon (minutes)", ylabel="MAE (MW)")
    ax.set_xticks(HORIZONS)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "development_horizon_mae.png", dpi=180)
    plt.close(fig)


def metadata() -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }


def setup_cell() -> str:
    return (
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) "
        "if (p / 'outputs' / '06_tcn_development').is_dir())\n"
        "OUT = ROOT / 'outputs' / '06_tcn_development'\n"
        "runs = pd.read_csv(OUT / 'development_run_summary.csv')\n"
        "horizons = pd.read_csv(OUT / 'development_horizon_metrics.csv')\n"
        "architecture = pd.read_csv(OUT / 'architecture_comparison.csv')\n"
        "training = pd.read_csv(OUT / 'training_contract_comparison.csv')\n"
    )


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata())
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{Path(name).stem[:20]}-{index:02d}"
    nbf.write(notebook, NOTEBOOKS / name)


def model_notebook(model: str, number: str, title: str, purpose: str, design: str, interpretation: str) -> None:
    write_notebook(
        f"{number}_{model.lower()}_development.ipynb",
        [
            nbf.v4.new_markdown_cell(
                f"# {number} — {title}\n\n{purpose}\n\n"
                "This is a results-only reconstruction from the completed RR development artifacts. "
                "It does not train, refit, or evaluate the frozen 2020 gate."
            ),
            nbf.v4.new_markdown_cell(
                "## Controlled comparison contract\n\n"
                "The prototype uses the same public feature groups, six targets, chronological folds, "
                "seed 42, optimizer family, preprocessing rules, and checkpoint-selection rule as its "
                "peers. The intervention here is architectural. The seventh demand channel is an "
                "invariant compatibility channel in 2015–2020; it contains no outage-event information."
            ),
            nbf.v4.new_code_cell(setup_cell()),
            nbf.v4.new_markdown_cell(f"## Architecture\n\n{design}"),
            nbf.v4.new_code_cell(
                f"display(architecture.loc[architecture.model.eq('{model}')].T)\n"
                f"display(training.loc[training.model.eq('{model}')].T)"
            ),
            nbf.v4.new_markdown_cell(
                "## Expanding-window development evidence\n\n"
                "The validation years are 2017, 2018, and 2019. Each row uses all public years from "
                "2015 through the year immediately before validation. Metrics are checkpoint outputs, "
                "not values copied from the historical source notebooks."
            ),
            nbf.v4.new_code_cell(
                f"display(runs.loc[runs.model.eq('{model}')].round(4))\n"
                f"display(horizons.loc[horizons.model.eq('{model}')].round(4))"
            ),
            nbf.v4.new_markdown_cell(f"## What this stage establishes\n\n{interpretation}"),
        ],
    )


def build_notebooks() -> None:
    model_notebook(
        "TCN1",
        "06a",
        "TCN1: consolidated temporal backbone",
        "TCN1 is the first neural reference in the public model lineage. It asks whether a compact "
        "causal convolutional network can learn the complete six-step demand-change path from the "
        "engineered public information set.",
        "All 75 temporal channels are concatenated before two residual causal blocks with 64 hidden "
        "channels. The last encoded state is fused with 55 forecast-origin calendar encodings and one "
        "population scalar. This is deliberately simple: it establishes a common-backbone baseline.",
        "TCN1 demonstrates that the engineered feature contract can be consumed by a causal six-output "
        "network. Its limitation is representational: demand, system, climate, and regional signals must "
        "share the same early filters. TCN2 tests whether preserving those roles until late fusion helps.",
    )
    model_notebook(
        "TCN2",
        "06b",
        "TCN2: symmetrical multi-branch equalizer",
        "TCN2 keeps the short 13-state input and deterministic six-horizon objective while changing "
        "only how feature families are encoded.",
        "Demand (7 channels), system plus climate (18 channels), and flattened regional context "
        "(50 channels) receive separate causal residual encoders. Their final states are peers at late "
        "fusion, alongside calendar and population context. The branch widths are 32, 32, and 64.",
        "This stage makes feature-family identity explicit and provides the branch structure later used "
        "by TCN_star. The move to TCN_star should not be read as a claim that TCN2 won every fold; it "
        "was the selected structural base for the controlled long-history intervention.",
    )
    model_notebook(
        "TCN3",
        "06c",
        "TCN3: hierarchical spatiotemporal learner",
        "TCN3 tests a different inductive bias for the five Queensland regions while preserving the "
        "same short history, targets, feature content, seed, and development folds.",
        "Demand and system/climate retain separate 32-channel encoders. Each region passes through the "
        "same 10-to-32-channel regional encoder; the five embeddings are then averaged. Weight sharing "
        "reduces the model to 53,222 parameters and treats the regions symmetrically.",
        "TCN3 is the strongest of the three compact prototypes on mean development-fold six-horizon "
        "MAE, but its regional mean pooling discards region identity after encoding. The lineage keeps "
        "this result visible while TCN_star follows TCN2's non-pooled regional representation.",
    )

    write_notebook(
        "06d_tcn_prototype_comparison.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 06d — Compact TCN prototype comparison\n\n"
                "This notebook brings TCN1, TCN2, and TCN3 together before the long-context extension. "
                "It uses only completed 2017–2019 RR development folds and never reads 2020 outcomes."
            ),
            nbf.v4.new_code_cell(setup_cell() + "model_summary = pd.read_csv(OUT / 'development_model_summary.csv')\n"),
            nbf.v4.new_markdown_cell(
                "## What differs\n\n"
                "The three compact models have the same 13 five-minute states, the same 75 temporal "
                "inputs, the same late calendar/population context, and the same deterministic six-output "
                "objective. They differ in when feature families are combined and how regions are encoded."
            ),
            nbf.v4.new_code_cell("display(architecture.loc[architecture.model.isin(['TCN1','TCN2','TCN3'])].T)"),
            nbf.v4.new_markdown_cell("## Development-fold comparison"),
            nbf.v4.new_code_cell(
                "display(model_summary.loc[model_summary.model.isin(['TCN1','TCN2','TCN3'])].round(4))\n"
                "display(runs.loc[runs.model.isin(['TCN1','TCN2','TCN3'])].round(4))\n"
                "display(Image(filename=str(OUT / 'development_fold_mae.png')))\n"
                "display(Image(filename=str(OUT / 'development_horizon_mae.png')))"
            ),
            nbf.v4.new_markdown_cell(
                "## Decision boundary\n\n"
                "TCN3 records the lowest mean development-fold six-horizon MAE among the compact "
                "prototypes. The next model nevertheless uses TCN2's branch topology as an explicit "
                "architectural choice: it retains a joint, non-pooled regional representation and changes "
                "the temporal history in isolation. This distinction avoids rewriting a structural design "
                "decision as a universal performance win."
            ),
        ],
    )

    write_notebook(
        "07_tcn_star_development.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 07 — TCN_star: long-context deterministic model\n\n"
                "TCN_star is the controlled deep-lookback continuation of TCN2. It retains the same "
                "feature groups, late-fusion design, six demand-change targets, and public development "
                "boundary; it expands the temporal context and convolutional stack."
            ),
            nbf.v4.new_code_cell(setup_cell()),
            nbf.v4.new_markdown_cell(
                "## Intervention and justification\n\n"
                "The input grows from 13 states to 505 states: the forecast origin plus 42 hours of "
                "five-minute history. Kernel size grows from 3 to 5, and dilations "
                "`1, 2, 4, 8, 16, 32` are repeated twice. With two causal convolutions per residual "
                "block, the theoretical receptive field is 1,009 states; the available 505-state window "
                "therefore lets every output use the complete supplied history. The aim is to expose "
                "intra-day shape and the preceding-day transition, not to add a new data source."
            ),
            nbf.v4.new_code_cell(
                "display(architecture.loc[architecture.model.isin(['TCN2','TCN_star'])].T)\n"
                "display(training.loc[training.model.isin(['TCN2','TCN_star'])].T)"
            ),
            nbf.v4.new_markdown_cell("## Development evidence"),
            nbf.v4.new_code_cell(
                "display(runs.loc[runs.model.isin(['TCN2','TCN_star'])].round(4))\n"
                "display(horizons.loc[horizons.model.isin(['TCN2','TCN_star'])].round(4))\n"
                "display(Image(filename=str(OUT / 'development_fold_mae.png')))"
            ),
            nbf.v4.new_markdown_cell(
                "## Finding\n\n"
                "Across each of the three public development folds, TCN_star lowers mean six-horizon "
                "MAE relative to TCN2. That repeated directional result supports carrying the long-context "
                "backbone forward. The next question is not another point-forecast architecture change, "
                "but whether uncertainty should vary by case and horizon."
            ),
        ],
    )

    write_notebook(
        "08_tcn_star_nll_development.ipynb",
        [
            nbf.v4.new_markdown_cell(
                "# 08 — TCN_starNLL: probabilistic Student-t extension\n\n"
                "TCN_starNLL preserves TCN_star's inputs and causal backbone. It replaces the deterministic "
                "six-value head with location, positive scale, and bounded degrees-of-freedom outputs at "
                "each horizon and trains by Student-t negative log likelihood."
            ),
            nbf.v4.new_code_cell(setup_cell()),
            nbf.v4.new_markdown_cell(
                "## Distribution contract\n\n"
                "For every origin and horizon, `mu` is both the Student-t location and the reported point "
                "forecast in standardized target space. `scale = softplus(raw_scale) + 1e-4`. Degrees of "
                "freedom are bounded strictly between 2.1 and 20, initialized near 4, so variance remains "
                "finite while heavy tails remain available. All three heads are dynamic; this is not a "
                "fixed residual distribution attached after deterministic training."
            ),
            nbf.v4.new_code_cell(
                "display(architecture.loc[architecture.model.isin(['TCN_star','TCN_starNLL'])].T)\n"
                "display(training.loc[training.model.isin(['TCN_star','TCN_starNLL'])].T)"
            ),
            nbf.v4.new_markdown_cell(
                "## Development evidence and comparison boundary\n\n"
                "The deterministic model is selected by standardized MSE; the probabilistic model is "
                "selected by NLL. Those objective values are not placed on a common ranking scale. Point "
                "MAE/RMSE remain comparable because both models report the Student-t location or direct "
                "output as the point forecast. Calibration is evaluated later for the frozen ensemble."
            ),
            nbf.v4.new_code_cell(
                "display(runs.loc[runs.model.isin(['TCN_star','TCN_starNLL'])].round(4))\n"
                "display(horizons.loc[horizons.model.isin(['TCN_star','TCN_starNLL'])].round(4))\n"
                "display(Image(filename=str(OUT / 'development_horizon_mae.png')))"
            ),
            nbf.v4.new_markdown_cell(
                "## What is carried to the frozen evaluation\n\n"
                "The public development evidence shows that probabilistic training preserves essentially "
                "the same point-forecast accuracy as TCN_star while defining case-specific predictive "
                "distributions. The official final comparison therefore evaluates both families as "
                "predeclared three-seed ensembles and reports NLL/calibration only where a predictive "
                "distribution genuinely exists."
            ),
        ],
    )


def provenance_document() -> str:
    return """# Notebook reconstruction provenance

## Purpose

RR originally moved directly from reference models to the frozen ensemble comparison.  This reconstruction inserts the missing model-development layer without importing later-period results or rerunning a model.

## Historical evidence consulted

Historical source repositories and reconstruction artifacts were consulted
read-only. The review was limited to architecture definitions, the intended
model-development sequence, and presentation structure relevant to the RR's
2015–2020 boundary. Private workspace names, commits and internal paths are not
needed to establish the public result and are deliberately omitted.

## Accepted and adapted

- The progression from consolidated TCN1, through branched TCN2 and shared-regional TCN3, to deep-lookback TCN_star and probabilistic TCN_starNLL.
- Architecture explanations verified against preserved source evidence and RR run identities.
- Results-analysis ideas limited to fold comparison, horizon profiles, convergence metadata, and an explicit decision boundary.
- Separation between compact prototypes, TCN_star, and TCN_starNLL.

## Rejected or superseded

- Material outside the RR's declared 2015–2020 public boundary.
- Alternative model lineages that do not belong to the RR reconstruction.
- Source-notebook outputs tied to historical local paths or different datasets.
- Training-launch and optional execution cells; the RR notebooks are results-only.

## Newly written for RR

- `06a`–`06d`, `07`, and `08` notebooks that read only curated RR evidence under `outputs/06_tcn_development/`.
- Development tables generated from completed seed-42 folds for validation years 2017, 2018, and 2019.
- Compact cross-model figures and explicit explanations of why objective values cannot be compared between MSE and NLL models.

Only evidence relevant to the 2015–2020 RR boundary was retained. Numerical
claims were checked against retained RR evidence, and no result outside that
boundary entered the public reconstruction.
"""


def main() -> int:
    summary, horizon, epochs, identities = load_runs()
    save_evidence(summary, horizon, epochs, identities)
    build_notebooks()
    (ROOT / "docs" / "notebook_reconstruction_provenance.md").write_text(provenance_document(), encoding="utf-8")
    print(f"Wrote development evidence to {OUT}")
    print("Wrote notebooks 06a-06d, 07 and 08")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
