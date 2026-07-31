"""Canonical provenance for the manuscript's published GKM histories.

The live protocol remains defined by :mod:`gkm_legs`: clean state is promoted at
the artifact root and dirty continuation state is retained under ``wip_context``.
This manuscript sidecar is a read-only audit index over those existing files.  It
does not change checkpoint loading, artifact seeding, promotion, or WIP restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS = ROOT / "arc" / "crack_lab" / "agent_solutions"


@dataclass(frozen=True)
class LedgerEntry:
    level: int
    marginal_C: int
    evidence: str


@dataclass(frozen=True)
class PromotionSource:
    """A preserved clean source state verified through ``through_level``.

    ``kind`` is one of ``git``, ``wip``, ``promotion``, or ``current``.  A WIP
    source names a lifecycle snapshot whose metadata records that the target
    level was reached.  A promotion source names a uniform
    ``promotion_evidence/level_XX`` boundary.  Only promoted core files are
    copied, never probe files.
    """

    through_level: int
    kind: str
    source: str
    prefix: str = ""


@dataclass(frozen=True)
class ArtifactHistory:
    game: str
    max_level: int
    replay_actions: int
    final_public_commit: str | None
    ledger: tuple[LedgerEntry, ...]
    promotions: tuple[PromotionSource, ...]

    @property
    def total_marginal_C(self) -> int:
        return sum(entry.marginal_C for entry in self.ledger)


HISTORIES: Mapping[str, ArtifactHistory] = {
    "ls20": ArtifactHistory(
        game="ls20",
        max_level=7,
        replay_actions=365,
        final_public_commit=None,
        ledger=(
            LedgerEntry(1, 40, "fresh clean promotion L1"),
            LedgerEntry(2, 54, "fresh clean promotion L2"),
            LedgerEntry(3, 86, "fresh clean promotion L3"),
            LedgerEntry(4, 114, "fresh clean promotion L4"),
            LedgerEntry(5, 138, "fresh clean promotion L5"),
            LedgerEntry(6, 170, "fresh clean promotion L6"),
            LedgerEntry(7, 158, "fresh clean promotion L7"),
        ),
        promotions=(
            PromotionSource(1, "promotion", "level_01"),
            PromotionSource(2, "promotion", "level_02"),
            PromotionSource(3, "promotion", "level_03"),
            PromotionSource(4, "promotion", "level_04"),
            PromotionSource(5, "promotion", "level_05"),
            PromotionSource(6, "promotion", "level_06"),
            PromotionSource(7, "promotion", "level_07"),
        ),
    ),
    "wa30": ArtifactHistory(
        game="wa30",
        max_level=9,
        replay_actions=597,
        final_public_commit=None,
        ledger=(
            LedgerEntry(1, 43, "fresh clean promotion L1"),
            LedgerEntry(2, 20, "fresh clean promotion L2"),
            LedgerEntry(3, 32, "fresh clean promotion L3"),
            LedgerEntry(4, 50, "fresh clean promotion L4"),
            LedgerEntry(5, 39, "fresh clean promotion L5"),
            LedgerEntry(6, 23, "fresh clean promotion L6"),
            LedgerEntry(7, 28, "fresh clean promotion L7"),
            LedgerEntry(8, 34, "fresh clean promotion L8"),
            LedgerEntry(9, 49, "fresh clean promotion L9"),
        ),
        promotions=(
            PromotionSource(1, "promotion", "level_01"),
            PromotionSource(2, "promotion", "level_02"),
            PromotionSource(3, "promotion", "level_03"),
            PromotionSource(4, "promotion", "level_04"),
            PromotionSource(5, "promotion", "level_05"),
            PromotionSource(6, "promotion", "level_06"),
            PromotionSource(7, "promotion", "level_07"),
            PromotionSource(8, "promotion", "level_08"),
            PromotionSource(9, "promotion", "level_09"),
        ),
    ),
}


def get_history(game: str) -> ArtifactHistory:
    try:
        return HISTORIES[game]
    except KeyError as exc:
        raise ValueError(f"no canonical artifact history for {game!r}") from exc
