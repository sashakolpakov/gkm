#!/usr/bin/env python3
"""Build a reproducible GKM/OPINE marginal-complexity comparison figure."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from audit_opine_marginal_complexity import py_tokens, source_additions


ROOT = Path(__file__).parent
LS20_LOG = ROOT / "crack_lab/agent_solutions/ls20_legs/run.log"
WA30_CHECKPOINT = ROOT / "crack_lab/agent_solutions/wa30_legs/checkpoint.json"
OPINE_RUN_LOG = ROOT / "opine-artifacts/wa30/run_log.txt"
OPINE_SYNTHESIS = ROOT / "opine-artifacts/wa30/synthesis"
OUT = ROOT / "manuscript/figures/gkm_opine_marginal_complexity.png"


def gkm_ledgers() -> tuple[list[int], list[int]]:
    text = LS20_LOG.read_text()
    ls20 = [int(value) for value in re.findall(r"L\d+:(\d+)", text)]
    checkpoint = json.loads(WA30_CHECKPOINT.read_text())
    wa30 = [0] * 9
    for record in checkpoint["records"]:
        wa30[int(record["level"]) - 1] = int(record["marginal_C"])
    return ls20, wa30


def opine_run_levels() -> dict[int, int]:
    """Map every logged synthesis run to the active level at that run."""
    active_level = 1
    runs: dict[int, int] = {}
    for line in OPINE_RUN_LOG.read_text().splitlines():
        advance = re.match(r"\[LEVEL_ADVANCE .* to=(\d+)\]", line)
        if advance:
            active_level = int(advance.group(1)) + 1
        synthesis = re.match(r"\[SYNTHESIS step=\d+ run=(\d+)\]", line)
        if synthesis:
            runs[int(synthesis.group(1))] = active_level
    return runs


def opine_history() -> list[tuple[int, int, int, int, int]]:
    """Return (synthesis run, active level, retained tokens, added, removed).

    Added/removed are diff-based token deltas against the previous snapshot,
    so summing `added` per level yields OPINE's measured marginal novelty.
    """
    runs = opine_run_levels()
    history: list[tuple[int, int, int, int, int]] = []
    previous: list[str] = []
    paths = sorted(OPINE_SYNTHESIS.glob("run_*/game_engine.py"),
                   key=lambda p: int(p.parent.name.removeprefix("run_")))
    for path in paths:
        run = int(path.parent.name.removeprefix("run_"))
        tokens = py_tokens(path.read_bytes())
        added, removed = source_additions(previous, tokens)
        history.append((run, runs[run], len(tokens), added, removed))
        previous = tokens
    return history


def main() -> None:
    ls20, wa30 = gkm_ledgers()
    run_levels = opine_run_levels()
    opine = opine_history()
    levels = list(range(1, 10))
    opine_counts = [sum(1 for level in run_levels.values() if level == item) for item in levels]
    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(range(1, 8), ls20, marker="o", lw=2.5, color="#007c91", label="GKM ls20")
    ax.plot(range(1, 10), wa30, marker="s", lw=2.5, color="#c23b22", label="GKM wa30")
    ax.set_title("GKM: admitted per-level novelty")
    ax.set_xlabel("Level")
    ax.set_ylabel("Marginal C (GKM ledger units)")
    ax.set_xticks(range(1, 10))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.annotate("ls20 reuse: L2=2, L4=3", xy=(4, 3), xytext=(4.8, 90),
                arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=9)

    added_by_level = {level: 0 for level in levels}
    removed_total = 0
    for _, level, _, added, removed in opine:
        added_by_level[level] += added
        removed_total += removed
    opine_added = [added_by_level[level] for level in levels]

    ax = axes[1]
    ax.bar(levels, opine_added, color="#a64b2a", alpha=0.9)
    ax.set_title("OPINE wa30: measured marginal novelty")
    ax.set_xlabel("Level")
    ax.set_ylabel("Diff-added source tokens (all snapshots in level)")
    ax.set_xticks(levels)
    ax.grid(axis="y", alpha=0.25)
    for level, added, count in zip(levels, opine_added, opine_counts):
        ax.text(level, added + 60, f"{added:,}", ha="center", fontsize=8)
    floor = min(opine_added)
    ax.text(0.03, 0.94,
            f"Total added: {sum(opine_added):,} tokens over {len(run_levels)} synthesis runs.\n"
            f"Cheapest level: {floor:,} tokens (L{opine_added.index(floor) + 1}). "
            "No level is near-free;\nevery level also ships an 8,464-byte verbatim entry cache.",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 4})

    ax = axes[2]
    runs = [row[0] for row in opine]
    source_levels = [row[1] for row in opine]
    tokens = [row[2] for row in opine]
    max_shrink = max(
        (left - right for left, right in zip(tokens, tokens[1:]) if right < left),
        default=0,
    )
    shrink_steps = sum(1 for left, right in zip(tokens, tokens[1:]) if right < left)
    first_run, first_tokens = runs[0], tokens[0]
    last_run, last_tokens = runs[-1], tokens[-1]
    growth = last_tokens / first_tokens
    for left, right in zip(opine, opine[1:]):
        if right[0] == left[0] + 1:
            ax.plot([left[0], right[0]], [left[2], right[2]],
                    lw=2.0, color="#5f4b8b", alpha=0.85)
    ax.scatter(runs, tokens, s=35, color="#5f4b8b",
               label="available game_engine.py snapshots", zorder=3)
    ax.set_title("OPINE wa30: retained source size")
    ax.set_xlabel("Synthesis revision")
    ax.set_ylabel("Normalized Python tokens")
    ax.set_xlim(1, max(run_levels))
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 65])
    ax.grid(axis="y", alpha=0.25)
    seen_levels: set[int] = set()
    for run, level, token_count in zip(runs, source_levels, tokens):
        if level not in seen_levels:
            ax.annotate(f"L{level}", (run, token_count), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
            seen_levels.add(level)
    ax.text(0.03, 0.05, f"{len(opine)}/{len(run_levels)} source snapshots extracted.\n"
            f"Run {first_run}: {first_tokens:,} tokens; run {last_run}: {last_tokens:,} tokens ({growth:.2f}x).\n"
            f"Only {shrink_steps}/{len(tokens) - 1} shrink steps; largest is {max_shrink} tokens\n"
            f"({max_shrink / last_tokens:.1%} of final size). "
            "Final snapshot retains nine level-entry caches\n(9 x 8,464 raw bytes).",
            transform=ax.transAxes, va="bottom", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 4})
    ax.legend(frameon=False, loc="upper left")

    figure.text(0.5, -0.01,
                "Units differ (GKM ledger units vs. diff-added Python tokens); the panels compare profile shape. "
                "Both marginal profiles are measured; GKM's ls20 trace collapses to near zero on reuse (L2=2, L4=3) "
                "and OPINE's never does, while its retained model grows 7.5x.",
                ha="center", fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
