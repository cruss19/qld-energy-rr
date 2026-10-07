"""Render the frozen RR TCN_starNLL architecture without loading PyTorch."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs/model_artifacts/TCN_starNLL_architecture.png"
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    style = dict(boxstyle="round,pad=0.35", linewidth=1.5)
    boxes = [
        (0.4, 6.4, 2.8, 0.9, "Demand history\n7 × 505", "#d9edf7"),
        (0.4, 4.9, 2.8, 0.9, "System + climate\n(16 + 2) × 505", "#dff0d8"),
        (0.4, 3.4, 2.8, 0.9, "Five regional histories\n5 × 10 × 505", "#fcf8e3"),
        (0.4, 1.9, 2.8, 0.9, "Calendar at origin\n55 encoded values", "#f2dede"),
        (0.4, 0.4, 2.8, 0.9, "Annual QLD population\n1 standardized value", "#e8ddf5"),
        (4.2, 6.25, 3.2, 1.2, "Demand causal TCN\n12 residual blocks; 32 channels", "#d9edf7"),
        (4.2, 4.75, 3.2, 1.2, "Context causal TCN\n12 residual blocks; 32 channels", "#dff0d8"),
        (4.2, 3.25, 3.2, 1.2, "Regional causal TCN\n12 residual blocks; 64 channels", "#fcf8e3"),
        (8.2, 3.5, 2.4, 1.4, "Late fusion\n184 → 128\nGELU + LayerNorm", "#e5e5e5"),
        (11.3, 5.5, 2.2, 0.9, "Location μ\n6 horizons", "#cce5ff"),
        (11.3, 3.75, 2.2, 0.9, "Scale σ > 0\n6 horizons", "#d4edda"),
        (11.3, 2.0, 2.2, 0.9, "Degrees ν ∈ [2.1,20]\n6 horizons", "#fff3cd"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor="#333", **style))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [
        ((3.2, 6.85), (4.2, 6.85)), ((3.2, 5.35), (4.2, 5.35)), ((3.2, 3.85), (4.2, 3.85)),
        ((7.4, 6.85), (8.2, 4.55)), ((7.4, 5.35), (8.2, 4.35)), ((7.4, 3.85), (8.2, 4.05)),
        ((3.2, 2.35), (8.2, 3.85)), ((3.2, 0.85), (8.2, 3.65)),
        ((10.6, 4.2), (11.3, 5.95)), ((10.6, 4.2), (11.3, 4.2)), ((10.6, 4.2), (11.3, 2.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax.set_title("RR TCN_starNLL — frozen public architecture", fontsize=16, weight="bold")
    ax.text(
        7, 0.18,
        "Causal convolutions: kernel 5; dilations 1,2,4,8,16,32 repeated twice; "
        "theoretical receptive field 1,009 positions; supplied history 505 states (42 hours)",
        ha="center", fontsize=9,
    )
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
