"""Build the small public final-run identity manifest from local run records.

This is a maintainer/publication-build utility. The large historical training
directories it reads are deliberately not distributed. Its output contains no
checkpoints, predictions, metrics, or machine-specific paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODELS = ("TCN1", "TCN2", "TCN3", "TCN_star", "TCN_starNLL")
SEEDS = (42, 142, 242)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            run_dir = root / f"training_output/runs/{model}/final/train_2015_2019/seed_{seed}"
            completed = json.loads((run_dir / "completed_run.json").read_text(encoding="utf-8"))
            identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
            if completed.get("status") != "complete":
                raise RuntimeError(f"Incomplete retained final run: {model} seed {seed}")
            if identity["exact_model"] != model or int(identity["seed"]) != seed:
                raise RuntimeError(f"Run identity mismatch: {model} seed {seed}")
            if identity["checkpoint_rule"] != "final fixed epoch selected before 2020 scoring":
                raise RuntimeError(f"Unexpected final-state rule: {model} seed {seed}")
            rows.append(
                {
                    "model_family": model,
                    "variant": identity["variant"],
                    "seed": seed,
                    "training_start": identity["train_start"],
                    "training_end_exclusive": identity["train_end_exclusive"],
                    "evaluation_start": identity["evaluation_start"],
                    "evaluation_end_exclusive": identity["evaluation_end_exclusive"],
                    "fixed_epochs": int(identity["fixed_epochs"]),
                    "stopping_rule": identity["stopping_rule"],
                    "checkpoint_rule": identity["checkpoint_rule"],
                    "experiment_identity": f"{model}:original_public_lineage:final_fixed_duration",
                    "ensemble_identity": f"{model}_ensemble_2020_seeds_42_142_242",
                    "member_status": "complete",
                }
            )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 15:
        raise RuntimeError("Expected exactly 15 final TCN member identities")
    output = root / "outputs/model_artifacts/final_tcn_member_identities.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
