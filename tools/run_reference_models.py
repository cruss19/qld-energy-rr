"""Maintainer-only status report for preserved RR reference evidence.

The reference runs are historical and complete. This command requires
undistributed row-level prediction artifacts, validates and reports them, and
deliberately does not fit or train models. It is not a clean-clone command.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MODELS = (
    "Persistence",
    "Weekly seasonal naive",
    "Comparable-history empirical change",
    "Bayesian random walk",
    "Ridge regression",
    "Gaussian linear",
    "Student-t linear",
    "Empirical residual",
    "AEMO P5MIN external benchmark",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    required = (
        root / "outputs/reference_models/expanding_window_oof_predictions.parquet",
        root / "outputs/probabilistic_reference_models/probabilistic_oof_predictions.parquet",
        root / "training_output/reference_models/aemo_p5min_external_benchmark/all_folds/predictions.parquet",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Preserved reference evidence is missing: {missing}")
    status = pd.DataFrame({"model": MODELS, "status": "COMPLETE", "reason": ""})
    output = root / "outputs/10_final_2020_evaluation/reference_execution_status.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(output, index=False)
    print(status.to_string(index=False))
    print("No model fitting or training was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
