#!/usr/bin/env python3
"""Build the ls20 marginal-complexity trace used in the manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_artifact_docs import load_artifact


OUT = Path(__file__).parent / "manuscript/figures/ls20_sawtooth.png"
def main() -> None:
    artifact = load_artifact("ls20")
    if not artifact.complete_ledger:
        raise ValueError("ls20 figure requires a complete manuscript ledger")
    levels = [level for level, _ in artifact.records]
    marginal_complexity = [cost for _, cost in artifact.records]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(6.6, 3.3), constrained_layout=True)

    axis.plot(
        levels,
        marginal_complexity,
        color="#263238",
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=2,
    )
    axis.scatter([2, 4], [2, 3], s=55, color="#00897b", zorder=3)
    axis.scatter([6], [130], s=55, color="#c43e2f", zorder=3)

    axis.set_xlabel("Level")
    axis.set_ylabel(r"Marginal complexity $C$")
    axis.set_xticks(levels)
    axis.set_ylim(0, 145)
    axis.set_yticks([0, 25, 50, 75, 100, 125])
    axis.grid(axis="y", color="#cfd8dc", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
