#!/usr/bin/env python3
"""Regenerate figures used by the Gödel--Kolmogorov Machine manuscript.

Conditional marginals come from the exact-winning-source audit; supporting source
charges come from replay-validated checkpoints. Rendering parameters are pinned to
reproduce the delivered PNG geometry under Matplotlib 3.10.8:

* figures/ls20_sawtooth.png: 1728 x 912 pixels
* figures/bounded_campaign_profiles.png: 2034 x 1072 pixels
* figures/marginal_complexity_profiles.png: 2448 x 912 pixels
* figures/source_charge_comparison.png: 2448 x 912 pixels
* figures/marginal_complexity_all_games.png: 2688 x 2880 pixels

The script also emits PDF versions for vector reuse.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Final, Mapping, Sequence

import json

# Stabilize PDF CreationDate metadata unless the caller pins another epoch.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

# A non-interactive backend makes the script suitable for CI and headless builds.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DPI: Final[int] = 240

BOUNDED_GAMES: Final[tuple[str, ...]] = (
    "ft09", "g50t", "r11l", "sp80", "tr87",
)

EXPECTED_PNG_SIZES: Final[Mapping[str, tuple[int, int]]] = {
    "ls20_sawtooth.png": (1728, 912),
    "bounded_campaign_profiles.png": (2034, 1072),
    "marginal_complexity_profiles.png": (2448, 912),
    "source_charge_comparison.png": (2448, 912),
    "marginal_complexity_all_games.png": (2688, 2880),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all Matplotlib figures used by the "
            "Gödel--Kolmogorov Machine paper."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory receiving PNG and PDF outputs (default: figures).",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("../crack_lab/agent_solutions"),
        help=(
            "Directory containing <game>_legs/checkpoint.json files "
            "(default: ../crack_lab/agent_solutions from manuscript cwd)."
        ),
    )
    parser.add_argument(
        "--comparison-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generated/source_charge_comparison.json",
        help="Authenticated paired recorded-charge/promoted-growth audit JSON.",
    )
    parser.add_argument(
        "--marginal-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generated/conditional_marginal_profiles.json",
        help="Conditional AST novelty at exact winning-source boundaries.",
    )
    parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Do not verify the expected PNG pixel dimensions.",
    )
    return parser.parse_args()


def _load_profile(solutions_dir: Path, game: str) -> tuple[int, ...]:
    path = solutions_dir / f"{game}_legs" / "checkpoint.json"
    payload = json.loads(path.read_text())
    if payload.get("game") != game:
        raise ValueError(f"{path}: game field is not {game!r}")
    if not payload.get("validated"):
        raise ValueError(f"{path}: checkpoint is not replay validated")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: missing non-empty records list")
    levels = [int(record["level"]) for record in records]
    if levels != list(range(1, len(records) + 1)):
        raise ValueError(f"{path}: records are not consecutive from level 1")
    if any(not record.get("reached") for record in records):
        raise ValueError(f"{path}: contains a non-promoted record")
    charges = tuple(int(record["marginal_C"]) for record in records)
    if sum(charges) != int(payload["total_marginal_C"]):
        raise ValueError(f"{path}: marginal charge total does not match records")
    return charges


def _set_common_axes(ax: plt.Axes, levels: Sequence[int]) -> None:
    ax.set_xlabel("Promoted level")
    ax.set_ylabel(r"Recorded workspace charge $C_k$")
    ax.set_xticks(levels)
    ax.grid(True, alpha=0.25)


def _save_pair(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight_bbox: bool,
) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    save_kwargs = {"bbox_inches": "tight"} if tight_bbox else {}

    fig.savefig(png_path, dpi=DPI, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)


def make_ls20_sawtooth(
    output_dir: Path, profile: dict,
) -> None:
    """Show the conditional marginal, rather than its source-charge surrogate."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _plot_conditional_marginal(ax, profile)
    ax.set_xlabel("Solved level")
    ax.set_ylabel(r"Conditional AST novelty $M_L$ (compressed bytes)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_pair(fig, output_dir, "ls20_sawtooth", tight_bbox=False)
    plt.close(fig)


def make_bounded_campaign_profiles(
    output_dir: Path, profiles: Mapping[str, Sequence[int]],
) -> None:
    """Create the shared-scale profiles for the bounded campaign."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    for game, charges in profiles.items():
        levels = tuple(range(1, len(charges) + 1))
        ax.plot(
            levels,
            charges,
            marker="o",
            linewidth=1.5,
            label=game,
        )

    max_level = max(len(charges) for charges in profiles.values())
    _set_common_axes(ax, tuple(range(1, max_level + 1)))
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
    )
    fig.tight_layout()
    _save_pair(
        fig,
        output_dir,
        "bounded_campaign_profiles",
        tight_bbox=True,
    )
    plt.close(fig)


def _plot_conditional_marginal(ax: plt.Axes, profile: dict) -> None:
    levels, values = profile["levels"], profile["M_bytes"]
    ax.plot(levels, values, color="#245e91", marker="o", markersize=4,
            linewidth=1.6, label=r"Conditional novelty $M_L$")
    drops = [i for i in range(1, len(values)) if values[i] is not None
             and values[i - 1] is not None and values[i] < values[i - 1]]
    ax.scatter([levels[i] for i in drops], [values[i] for i in drops],
               facecolors="none", edgecolors="#d46a1f", s=67, linewidths=1.6,
               zorder=4, label="Marginal decrease")
    coupled = profile["sharp_drop_with_static_call_levels"]
    ax.scatter(coupled, [values[level - 1] for level in coupled],
               marker="D", color="#008060", s=29, zorder=5,
               label="Sharp drop + direct leg call site")
    ax.set_xticks(levels)
    ax.set_xlim(0.6, len(levels) + 0.4)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22)


def make_marginal_complexity_profiles(output_dir: Path, payload: dict) -> None:
    """Show the three declared cases; all other games appear in the suite figure."""
    order = ("dc22", "wa30", "ls20")
    profiles = {g["game"]: g for g in payload["games"]}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    subtitles = {
        "dc22": "conditional dips despite source growth",
        "wa30": "complete uniform history",
        "ls20": "unchanged-leg call site",
    }
    for ax, game in zip(axes, order):
        _plot_conditional_marginal(ax, profiles[game])
        ax.set_title(f"{game}: {subtitles[game]}", fontsize=9)
        ax.set_xlabel("Solved level")
    axes[0].set_ylabel(r"Conditional AST novelty $M_L$ (compressed bytes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save_pair(
        fig,
        output_dir,
        "marginal_complexity_profiles",
        tight_bbox=False,
    )
    plt.close(fig)


def make_all_marginal_profiles(output_dir: Path, payload: dict) -> None:
    """Publish every game, including missing and unsolved boundaries, without smoothing."""
    games = payload["games"]
    if len(games) != 25:
        raise ValueError("the frozen suite must contain 25 games")
    fig, axes = plt.subplots(5, 5, figsize=(11.2, 12.0))
    for index, (ax, profile) in enumerate(zip(axes.flat, games)):
        _plot_conditional_marginal(ax, profile)
        observed = sum(value is not None for value in profile["M_bytes"])
        ax.set_title(f"{profile['game']}  ({observed}/{profile['total_levels']} measured)", fontsize=10)
        ax.tick_params(labelsize=8)
        if not observed:
            ax.text(0.5, 0.5, "No exact-source\nmarginal available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, wrap=True)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        if index // 5 == 4:
            ax.set_xlabel("Solved level", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.supylabel(r"Conditional AST novelty $M_L$ (compressed bytes; separate linear scales)", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.965), h_pad=1.3)
    _save_pair(fig, output_dir, "marginal_complexity_all_games", tight_bbox=False)
    plt.close(fig)


def _read_png_size(path: Path) -> tuple[int, int]:
    """Read a PNG's width and height directly from its IHDR chunk."""
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG file: {path}")
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or length < 8:
            raise ValueError(f"missing PNG IHDR chunk: {path}")
        width, height = struct.unpack(">II", handle.read(8))
    return width, height


def _comparison_profile(payload: dict, game: str) -> tuple[tuple, tuple, tuple]:
    """Align both curves to authenticated pairs and retain gaps as missing data."""
    matches = [row for row in payload["games"] if row["game"] == game]
    if len(matches) != 1:
        raise ValueError(f"expected one paired source profile for {game}")
    row = matches[0]
    levels = tuple(row["levels"])
    recorded = tuple(row["recorded_C"])
    growth = tuple(row["primary_G"])
    if levels != tuple(range(1, len(levels) + 1)) or not (len(levels) == len(recorded) == len(growth)):
        raise ValueError(f"{game}: inconsistent paired source coordinates")
    if any(type(value) is not int or value < 0 for value in recorded):
        raise ValueError(f"{game}: invalid recorded source charge")
    if any(value is not None and (type(value) is not int or value < 0) for value in growth):
        raise ValueError(f"{game}: invalid adjacent promoted growth")
    if not any(value is not None for value in growth):
        raise ValueError(f"{game}: no authenticated source pairs")
    paired_recorded = tuple(c if g is not None else None for c, g in zip(recorded, growth))
    return levels, paired_recorded, growth


def make_source_charge_comparison(output_dir: Path, payload: dict) -> None:
    """Compare the two coordinates on the same authenticated level domains."""
    games = ("dc22", "wa30", "ls20")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    for ax, game in zip(axes, games):
        levels, recorded, growth = _comparison_profile(payload, game)
        valid_levels = [level for level, value in zip(levels, growth) if value is not None]
        ax.plot(levels, recorded, color="#8051a5", linestyle="--", marker="o",
                markerfacecolor="none", markersize=6, linewidth=2.4, alpha=0.8,
                label=r"Recorded workspace charge $C_k^{\mathrm{recorded}}$")
        ax.plot(levels, growth, color="#007c91", linestyle="-", marker="^",
                markersize=3.5, linewidth=1.4,
                label=r"Adjacent promoted growth $G_k$")
        ax.set_title(game, fontsize=10)
        ax.set_xlabel("Promoted level")
        ax.set_xticks(valid_levels)
        ax.set_xlim(min(valid_levels) - 0.25, max(valid_levels) + 0.25)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Source units (lines + literal elements)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_pair(fig, output_dir, "source_charge_comparison", tight_bbox=False)
    plt.close(fig)


def _verify_png_sizes(output_dir: Path) -> None:
    for filename, expected in EXPECTED_PNG_SIZES.items():
        path = output_dir / filename
        actual = _read_png_size(path)
        if actual != expected:
            raise RuntimeError(
                f"{filename} has size {actual}; expected {expected}. "
                "Use Matplotlib 3.10.8 and the default bundled DejaVu fonts."
            )


def main() -> int:
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir: Path = args.solutions_dir
    marginals = json.loads(args.marginal_data.read_text())
    bounded_profiles = {
        game: _load_profile(solutions_dir, game)
        for game in BOUNDED_GAMES
    }

    # Ignore user matplotlibrc files so the paper's defaults remain reproducible.
    plt.rcdefaults()
    make_ls20_sawtooth(output_dir, next(g for g in marginals["games"] if g["game"] == "ls20"))
    make_bounded_campaign_profiles(output_dir, bounded_profiles)
    make_marginal_complexity_profiles(output_dir, marginals)
    make_all_marginal_profiles(output_dir, marginals)
    make_source_charge_comparison(output_dir, json.loads(args.comparison_data.read_text()))

    if not args.skip_size_check:
        _verify_png_sizes(output_dir)

    for stem in (
        "ls20_sawtooth",
        "bounded_campaign_profiles",
        "marginal_complexity_profiles",
        "marginal_complexity_all_games",
        "source_charge_comparison",
    ):
        print(output_dir / f"{stem}.png")
        print(output_dir / f"{stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
