#!/usr/bin/env python3
"""Regenerate figures used by the Gödel--Kolmogorov Machine manuscript.

Numerical inputs are read from the replay-validated campaign checkpoints rather
than duplicated in this script.  The rendering parameters are pinned to
reproduce the delivered PNG geometry under Matplotlib 3.10.8:

* figures/ls20_sawtooth.png: 1728 x 912 pixels
* figures/bounded_campaign_profiles.png: 2034 x 1072 pixels
* figures/marginal_complexity_profiles.png: 2448 x 912 pixels

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
    ax.set_ylabel(r"Marginal description charge $C_k$")
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
    output_dir: Path, charges: Sequence[int],
) -> None:
    """Create the seven-level ls20 historical marginal-charge profile."""
    levels = tuple(range(1, len(charges) + 1))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(
        levels,
        charges,
        marker="o",
        linewidth=1.8,
    )
    _set_common_axes(ax, levels)
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


def make_marginal_complexity_profiles(
    output_dir: Path, profiles: Mapping[str, Sequence[int]],
) -> None:
    """Contrast the strongest raw sawtooth with the two uniform histories.

    ``su15`` is selected mechanically by direction reversals in the stored
    ledger.  It is shown as a scalar-shape comparison, not as a source-coupled
    reuse witness.  ``wa30`` and ``ls20`` are the complete uniform promotion
    histories used for the manuscript's source-level case studies.
    """
    order = ("su15", "wa30", "ls20")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    colors = {"su15": "#6a3d9a", "wa30": "#c23b22", "ls20": "#007c91"}
    subtitles = {
        "su15": "strongest raw oscillation",
        "wa30": "uniform sawtooth history",
        "ls20": "uniform coupled-reuse case",
    }
    for ax, game in zip(axes, order):
        charges = profiles[game]
        levels = tuple(range(1, len(charges) + 1))
        ax.plot(
            levels,
            charges,
            marker="o",
            linewidth=1.8,
            color=colors[game],
        )
        ax.set_title(f"{game}: {subtitles[game]}", fontsize=9)
        ax.set_xlabel("Promoted level")
        ax.set_xticks(levels)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"Marginal description charge $C_k$")
    fig.tight_layout()
    _save_pair(
        fig,
        output_dir,
        "marginal_complexity_profiles",
        tight_bbox=False,
    )
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
    ls20_charges = _load_profile(solutions_dir, "ls20")
    bounded_profiles = {
        game: _load_profile(solutions_dir, game)
        for game in BOUNDED_GAMES
    }
    comparison_profiles = {
        game: _load_profile(solutions_dir, game)
        for game in ("su15", "wa30", "ls20")
    }

    # Ignore user matplotlibrc files so the paper's defaults remain reproducible.
    plt.rcdefaults()
    make_ls20_sawtooth(output_dir, ls20_charges)
    make_bounded_campaign_profiles(output_dir, bounded_profiles)
    make_marginal_complexity_profiles(output_dir, comparison_profiles)

    if not args.skip_size_check:
        _verify_png_sizes(output_dir)

    for stem in (
        "ls20_sawtooth",
        "bounded_campaign_profiles",
        "marginal_complexity_profiles",
    ):
        print(output_dir / f"{stem}.png")
        print(output_dir / f"{stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
