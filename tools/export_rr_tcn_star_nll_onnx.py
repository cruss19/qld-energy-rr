"""Export and structurally verify the frozen RR TCN_starNLL seed-42 model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import onnx
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_figure(path: Path):
    from src.rr_plot_style import apply_rr_plot_style, plasma_tints

    colors = apply_rr_plot_style()
    fills = plasma_tints()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    style = dict(boxstyle="round,pad=0.35", linewidth=1.5)
    boxes = [
        (0.4, 6.4, 2.8, 0.9, "Demand history\n7 × 505", fills[0]),
        (0.4, 4.9, 2.8, 0.9, "System + climate\n(16 + 2) × 505", fills[2]),
        (0.4, 3.4, 2.8, 0.9, "Five regional histories\n5 × 10 × 505", fills[4]),
        (0.4, 1.9, 2.8, 0.9, "Calendar at origin\n55 encoded values", fills[6]),
        (0.4, 0.4, 2.8, 0.9, "Annual QLD population\n1 standardized value", fills[8]),
        (4.2, 6.25, 3.2, 1.2, "Demand causal TCN\n12 residual blocks; 32 channels", fills[0]),
        (4.2, 4.75, 3.2, 1.2, "Context causal TCN\n12 residual blocks; 32 channels", fills[2]),
        (4.2, 3.25, 3.2, 1.2, "Regional causal TCN\n12 residual blocks; 64 channels", fills[4]),
        (8.2, 3.5, 2.4, 1.4, "Late fusion\n184 → 128\nGELU + LayerNorm", fills[5]),
        (11.3, 5.5, 2.2, 0.9, "Location μ\n6 horizons", fills[3]),
        (11.3, 3.75, 2.2, 0.9, "Scale σ > 0\n6 horizons", fills[6]),
        (11.3, 2.0, 2.2, 0.9, "Degrees ν ∈ [2.1,20]\n6 horizons", fills[9]),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor=colors[0], **style))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [
        ((3.2, 6.85), (4.2, 6.85)), ((3.2, 5.35), (4.2, 5.35)), ((3.2, 3.85), (4.2, 3.85)),
        ((7.4, 6.85), (8.2, 4.55)), ((7.4, 5.35), (8.2, 4.35)), ((7.4, 3.85), (8.2, 4.05)),
        ((3.2, 2.35), (8.2, 3.85)), ((3.2, 0.85), (8.2, 3.65)),
        ((10.6, 4.2), (11.3, 5.95)), ((10.6, 4.2), (11.3, 4.2)), ((10.6, 4.2), (11.3, 2.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=colors[1], lw=1.5))
    ax.set_title("RR TCN_starNLL — frozen public architecture", fontsize=16, weight="bold")
    ax.text(
        7,
        0.18,
        "Causal convolutions: kernel 5; dilations 1,2,4,8,16,32 repeated twice; "
        "theoretical receptive field 1,009 positions; supplied history 505 states (42 hours)",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.tcn_star_nll_model import TCNStarNLL

    checkpoint = root / "training_output/runs/TCN_starNLL/final/train_2015_2019/seed_42/frozen_model.pt"
    output = root / "outputs/model_artifacts"
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "TCN_starNLL_2020_seed_42.onnx"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    calendar_dim = int(state["fusion_trunk.0.weight"].shape[1] - 129)
    model = TCNStarNLL(calendar_dim=calendar_dim).eval()
    model.load_state_dict(state, strict=True)
    inputs = (
        torch.zeros(1, 7, 505), torch.zeros(1, 16, 505), torch.zeros(1, 2, 505),
        torch.zeros(1, 5, 10, 505), torch.zeros(1, calendar_dim), torch.zeros(1, 1),
    )
    names = ("demand_history", "system_history", "climate_history", "regional_history", "calendar", "population")
    torch.onnx.export(
        model, inputs, onnx_path, export_params=True, opset_version=18,
        do_constant_folding=True, input_names=names, output_names=("mu", "scale", "df"),
        dynamic_axes={**{name: {0: "batch"} for name in names}, "mu": {0: "batch"}, "scale": {0: "batch"}, "df": {0: "batch"}},
        dynamo=False,
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph, full_check=True)
    properties = {
        "model_name": "TCN_starNLL", "variant": "original_public_lineage",
        "training_period": "2015-01-01 through 2019-12-31", "evaluation_period": "2020",
        "seed": "42", "opset": "18", "calendar_encoded_width": str(calendar_dim),
        "population_contract": "original annual total_qld_population; Population_2 excluded",
        "selection": "fixed epoch chosen by median best development epoch before 2020 scoring",
    }
    del graph.metadata_props[:]
    for key, value in properties.items():
        item = graph.metadata_props.add(); item.key = key; item.value = value
    onnx.save(graph, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path), full_check=True)
    architecture_figure(output / "TCN_starNLL_architecture.png")
    manifest = {
        "status": "PASS", "onnx_checker": "onnx.checker.check_model(full_check=True)",
        "opset": 18, "onnx_path": str(onnx_path.relative_to(root)), "onnx_sha256": sha256(onnx_path),
        "checkpoint_path": str(checkpoint.relative_to(root)), "checkpoint_sha256": sha256(checkpoint),
        "calendar_encoded_width": calendar_dim,
        "input_shapes": {"demand_history": ["batch", 7, 505], "system_history": ["batch", 16, 505],
                         "climate_history": ["batch", 2, 505], "regional_history": ["batch", 5, 10, 505],
                         "calendar": ["batch", calendar_dim], "population": ["batch", 1]},
        "output_shapes": {"mu": ["batch", 6], "scale": ["batch", 6], "df": ["batch", 6]},
    }
    (output / "TCN_starNLL_2020_seed_42_onnx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
