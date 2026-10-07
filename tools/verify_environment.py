"""Smoke-test the public RR Python environment without changing project state."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path


EXPECTED_VERSIONS = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "pyarrow": "24.0.0",
    "PyYAML": "6.0.3",
    "python-dateutil": "2.9.0.post0",
    "nemosis": "3.8.1",
    "requests": "2.34.2",
    "tensorboard": "2.21.0",
    "onnx": "1.22.0",
    "jupyterlab": "4.6.3",
    "ipykernel": "7.3.0",
    "nbformat": "5.11.1",
    "nbclient": "0.11.0",
    "matplotlib": "3.10.9",
    "seaborn": "0.13.2",
    "pytest": "9.1.1",
}

IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "pyarrow",
    "yaml",
    "torch",
    "tensorboard",
    "onnx",
    "nemosis",
    "requests",
    "matplotlib",
    "seaborn",
    "pytest",
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-environment-name",
        default="qld-energy-rr",
        help="Expected final component of sys.prefix (default: qld-energy-rr)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("RR environment verification")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    check(platform.python_version() == "3.11.15", "Expected Python 3.11.15")
    check(
        Path(sys.prefix).name.casefold() == args.expected_environment_name.casefold(),
        f"Expected {args.expected_environment_name}, found environment at {sys.prefix}",
    )

    for module_name in IMPORTS:
        importlib.import_module(module_name)
    print("PASS: required imports")

    for package_name, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package_name)
        check(
            actual == expected,
            f"{package_name}: expected {expected}, found {actual}",
        )
    print("PASS: pinned package versions")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import onnx
    import pandas as pd
    import pyarrow
    import sklearn
    import torch
    import yaml
    from sklearn.preprocessing import StandardScaler
    from torch.utils.tensorboard import SummaryWriter

    check(torch.__version__ == "2.13.0+cpu", f"Unexpected torch: {torch.__version__}")
    check(torch.version.cuda is None, "The RR reference environment must be CPU-only")
    check(not torch.cuda.is_available(), "CUDA unexpectedly available in CPU environment")
    print("PASS: CPU PyTorch identity")

    with tempfile.TemporaryDirectory(prefix="qld_energy_rr_verify_") as temp_dir:
        root = Path(temp_dir)

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="30min"),
                "value": np.array([1.0, 2.0, 3.0, 4.0]),
            }
        )
        parquet_path = root / "roundtrip.parquet"
        frame.to_parquet(parquet_path, index=False, engine="pyarrow")
        restored = pd.read_parquet(parquet_path, engine="pyarrow")
        pd.testing.assert_frame_equal(frame, restored)
        print("PASS: Pandas/PyArrow Parquet round trip")

        scaled = StandardScaler().fit_transform(frame[["value"]])
        check(np.isclose(float(scaled.mean()), 0.0), "StandardScaler mean check failed")
        print("PASS: scikit-learn transform")

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 1),
        )
        inputs = torch.randn(5, 3)
        loss = model(inputs).square().mean()
        loss.backward()
        check(all(p.grad is not None for p in model.parameters()), "Backward pass failed")
        print("PASS: PyTorch forward/backward")

        onnx_path = root / "smoke.onnx"
        torch.onnx.export(
            model.eval(),
            inputs,
            onnx_path,
            input_names=["features"],
            output_names=["prediction"],
            opset_version=18,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path))
        print("PASS: ONNX opset-18 export and checker")

        event_dir = root / "tensorboard"
        writer = SummaryWriter(log_dir=str(event_dir))
        writer.add_scalar("smoke/value", 1.0, 0)
        writer.close()
        check(any(event_dir.glob("events.out.tfevents.*")), "No TensorBoard event written")
        print("PASS: TensorBoard event creation")

        figure_path = root / "smoke.png"
        fig, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        fig.savefig(figure_path)
        plt.close(fig)
        check(figure_path.stat().st_size > 0, "Matplotlib output is empty")
        print("PASS: Matplotlib headless rendering")

        parsed = yaml.safe_load("project: qld-energy-rr\nverified: true\n")
        check(parsed == {"project": "qld-energy-rr", "verified": True}, "YAML failed")
        print("PASS: YAML parsing")

        report = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "onnx_opset": 18,
        }
        print(json.dumps(report, indent=2, sort_keys=True))

    print("SUCCESS: qld-energy-rr passed all environment checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
