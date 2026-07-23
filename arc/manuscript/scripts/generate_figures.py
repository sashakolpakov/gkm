#!/usr/bin/env python3
"""Regenerate figures used by the Gödel--Kolmogorov Machine manuscript.

The numerical inputs below are the values reported in the manuscript's empirical
ledger tables.  The rendering parameters are pinned to reproduce the delivered
PNG geometry under Matplotlib 3.10.8:

* figures/ls20_sawtooth.png: 1728 x 912 pixels
* figures/bounded_campaign_profiles.png: 2034 x 1072 pixels

The script also emits PDF versions for vector reuse.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Final, Mapping, Sequence

import matplotlib

# A non-interactive backend makes the script suitable for CI and headless builds.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DPI: Final[int] = 240

LS20_LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)
LS20_CHARGES: Final[tuple[int, ...]] = (43, 2, 45, 3, 72, 130, 67)

BOUNDED_LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
BOUNDED_PROFILES: Final[Mapping[str, tuple[int, ...]]] = {
    "ft09": (107, 2, 184, 132, 177, 2),
    "g50t": (244, 100, 2, 2, 134),
    "r11l": (103, 449, 30, 341, 362, 98),
    "sp80": (131, 40, 81, 56),
    "tr87": (156, 241, 183, 69, 249, 127),
}

EXPECTED_PNG_SIZES: Final[Mapping[str, tuple[int, int]]] = {
    "ls20_sawtooth.png": (1728, 912),
    "bounded_campaign_profiles.png": (2034, 1072),
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
        "--skip-size-check",
        action="store_true",
        help="Do not verify the expected PNG pixel dimensions.",
    )
    return parser.parse_args()


def _validate_data() -> None:
    if len(LS20_LEVELS) != len(LS20_CHARGES):
        raise ValueError("ls20 levels and charges must have equal length")
    if sum(LS20_CHARGES) != 362:
        raise ValueError("ls20 charge ledger must sum to 362")

    for game, profile in BOUNDED_PROFILES.items():
        if not profile or len(profile) > len(BOUNDED_LEVELS):
            raise ValueError(
                f"{game} has invalid profile length {len(profile)}"
            )


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


def make_ls20_sawtooth(output_dir: Path) -> None:
    """Create the seven-level ls20 historical marginal-charge profile."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(
        LS20_LEVELS,
        LS20_CHARGES,
        marker="o",
        linewidth=1.8,
    )
    _set_common_axes(ax, LS20_LEVELS)
    fig.tight_layout()
    _save_pair(fig, output_dir, "ls20_sawtooth", tight_bbox=False)
    plt.close(fig)


def make_bounded_campaign_profiles(output_dir: Path) -> None:
    """Create the shared-scale profiles for the bounded campaign."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    for game, charges in BOUNDED_PROFILES.items():
        ax.plot(
            BOUNDED_LEVELS[: len(charges)],
            charges,
            marker="o",
            linewidth=1.5,
            label=game,
        )

    _set_common_axes(ax, BOUNDED_LEVELS)
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

    # Ignore user matplotlibrc files so the paper's defaults remain reproducible.
    plt.rcdefaults()
    _validate_data()
    make_ls20_sawtooth(output_dir)
    make_bounded_campaign_profiles(output_dir)

    if not args.skip_size_check:
        _verify_png_sizes(output_dir)

    for stem in ("ls20_sawtooth", "bounded_campaign_profiles"):
        print(output_dir / f"{stem}.png")
        print(output_dir / f"{stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
