#!/usr/bin/env python3
"""Build the manuscript's GKM/OPINE source-growth comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_artifact_docs import load_artifact


ROOT = Path(__file__).parent
OUT = ROOT / "manuscript/figures/gkm_opine_marginal_complexity.png"

# OPINE values come from its published audit; GKM values come from the manuscript
# history sidecar, whose ledgers include every replay-validated level.
OPINE_ADDED = [3243, 3978, 1712, 2053, 1246, 1136, 266, 3310, 1444]
OPINE_LEVEL_END_REVISIONS = [3, 10, 15, 21, 29, 34, 38, 55, 65]
OPINE_RETAINED_AT_LEVEL_END = [2578, 5212, 6614, 8029, 8836, 9913, 10177, 13022, 14161]


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#cfd8dc", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def main() -> None:
    ls20 = load_artifact("ls20")
    wa30 = load_artifact("wa30")
    if not ls20.complete_ledger or not wa30.complete_ledger:
        raise ValueError("comparison figure requires complete manuscript ledgers")
    ls20_costs = [cost for _, cost in ls20.records]
    wa30_costs = [cost for _, cost in wa30.records]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    axis = axes[0]
    ls20_levels = [level for level, _ in ls20.records]
    wa30_levels = [level for level, _ in wa30.records]
    axis.plot(ls20_levels, ls20_costs, marker="o", linewidth=2,
              color="#007c91", label="ls20")
    axis.plot(wa30_levels, wa30_costs, marker="s", linewidth=2,
              color="#c23b22", label="wa30")
    axis.set_title("(a) Per-level marginal complexity")
    axis.set_xlabel("Level")
    axis.set_ylabel("Ledger units")
    axis.set_xticks(range(1, 10))
    axis.legend(frameon=False)
    style_axis(axis)

    axis = axes[1]
    levels = list(range(1, 10))
    axis.bar(levels, OPINE_ADDED, color="#b45f3c", width=0.72)
    axis.set_title("(b) OPINE source added")
    axis.set_xlabel("Level")
    axis.set_ylabel("Normalized Python tokens")
    axis.set_xticks(levels)
    style_axis(axis)

    axis = axes[2]
    axis.plot(
        OPINE_LEVEL_END_REVISIONS,
        OPINE_RETAINED_AT_LEVEL_END,
        marker="o",
        linewidth=2,
        color="#5f4b8b",
    )
    axis.set_title("(c) OPINE retained source")
    axis.set_xlabel("Synthesis revision")
    axis.set_ylabel("Normalized Python tokens")
    axis.set_xticks([1, 10, 20, 30, 40, 50, 60, 65])
    style_axis(axis)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
