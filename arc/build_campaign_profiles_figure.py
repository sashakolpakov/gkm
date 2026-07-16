#!/usr/bin/env python3
"""Build bounded-campaign marginal profiles from canonical checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
SOLUTIONS = ROOT / "crack_lab/agent_solutions"
OUT = ROOT / "manuscript/figures/bounded_campaign_profiles.png"
GAMES = ("ft09", "g50t", "r11l", "sp80", "tr87")


def load_profile(game: str) -> tuple[list[int], list[int]]:
    checkpoint = json.loads(
        (SOLUTIONS / f"{game}_legs/checkpoint.json").read_text()
    )
    records = checkpoint["records"]
    levels = [int(record["level"]) for record in records]
    costs = [int(record["marginal_C"]) for record in records]
    if not checkpoint["validated"] or checkpoint["reached"] != 4:
        raise ValueError(f"{game} is not a validated level-4 artifact")
    if levels != [1, 2, 3, 4] or len(levels) != len(set(levels)):
        raise ValueError(f"{game} does not have one canonical row per level")
    if sum(costs) != checkpoint["total_marginal_C"]:
        raise ValueError(f"{game} checkpoint total is inconsistent")
    return levels, costs


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(9.2, 5.1), sharex=True, sharey=True,
        constrained_layout=True,
    )
    colors = ("#007c91", "#c23b22", "#5f4b8b", "#2e7d32", "#9a6700")

    for axis, game, color in zip(axes.flat, GAMES, colors):
        levels, costs = load_profile(game)
        axis.plot(
            levels, costs, color=color, linewidth=2, marker="o",
            markersize=5, markerfacecolor="white", markeredgewidth=1.4,
        )
        for level, cost in zip(levels, costs):
            axis.annotate(
                str(cost), (level, cost), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8,
            )
        axis.set_title(game)
        axis.set_xticks(levels)
        axis.set_ylim(0, 500)
        axis.grid(axis="y", color="#cfd8dc", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    axes.flat[-1].axis("off")
    figure.supxlabel("Level")
    figure.supylabel(r"Marginal complexity $C_k$")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
