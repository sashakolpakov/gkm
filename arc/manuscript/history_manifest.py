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

    ``kind`` is one of ``git``, ``wip``, or ``current``.  A WIP source names a
    lifecycle snapshot whose metadata records that the target level was reached;
    only the promoted core files are copied from it, never its probe files.
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
    final_public_commit: str
    ledger: tuple[LedgerEntry, ...]
    promotions: tuple[PromotionSource, ...]

    @property
    def total_marginal_C(self) -> int:
        return sum(entry.marginal_C for entry in self.ledger)


HISTORIES: Mapping[str, ArtifactHistory] = {
    "ls20": ArtifactHistory(
        game="ls20",
        max_level=7,
        replay_actions=393,
        final_public_commit="875e5b868a6b0b6c230945704bb54ff1a0fb9d50",
        ledger=(
            LedgerEntry(1, 43, "final checkpoint record L1"),
            LedgerEntry(2, 2, "final checkpoint record L2"),
            LedgerEntry(3, 45, "final checkpoint record L3"),
            LedgerEntry(4, 3, "final checkpoint record L4"),
            LedgerEntry(5, 72, "final checkpoint record L5"),
            LedgerEntry(6, 130, "final checkpoint record L6"),
            LedgerEntry(7, 67, "final checkpoint record L7"),
        ),
        promotions=(
            PromotionSource(1, "wip", "level_01/after_debrief_b2872b134107"),
            PromotionSource(2, "wip", "level_03/not_reached_a547893d6480"),
            PromotionSource(3, "wip", "level_03/after_debrief_2b9bd3a82595"),
            PromotionSource(4, "wip", "level_04/after_debrief_7738f07791ee"),
            PromotionSource(5, "wip", "level_05/after_debrief_878509461d47"),
            PromotionSource(6, "wip", "level_06/recovered_after_credit_out_52480755b3fc"),
            PromotionSource(7, "current", "."),
        ),
    ),
    "wa30": ArtifactHistory(
        game="wa30",
        max_level=9,
        replay_actions=596,
        final_public_commit="07a188544d10d57850d81c6bcf65b82bdd12f52c",
        ledger=(
            LedgerEntry(1, 112, "15c0a049 FINDINGS and clean L1-L3 source"),
            LedgerEntry(2, 78, "15c0a049 FINDINGS and clean L1-L3 source"),
            LedgerEntry(3, 95, "15c0a049 FINDINGS and clean L1-L3 source"),
            LedgerEntry(4, 47, "0ccea257 FINDINGS and clean L1-L4 source"),
            LedgerEntry(5, 405, "level_05 after-debrief snapshot and later checkpoint"),
            LedgerEntry(6, 225, "level_06 after-debrief snapshot and later checkpoint"),
            LedgerEntry(7, 145, "level_07 after-debrief snapshot and later checkpoint"),
            LedgerEntry(8, 204, "level_08 after-debrief snapshot and later checkpoint"),
            LedgerEntry(9, 147, "07a1885 final promoted checkpoint"),
        ),
        promotions=(
            PromotionSource(
                3,
                "git",
                "15c0a04911bfdc6b03d4649a3a5b3b9a072ce0b4",
                "crack_lab/agent_solutions/wa30_legs",
            ),
            PromotionSource(
                4,
                "git",
                "0ccea2572ea7990f1573ac34bfbc5d83bd688bd7",
                "arc/crack_lab/agent_solutions/wa30_legs",
            ),
            PromotionSource(5, "wip", "level_05/after_debrief_353f8a9c7fa1"),
            PromotionSource(6, "wip", "level_06/after_debrief_2ebd8c25d36e"),
            PromotionSource(7, "wip", "level_07/after_debrief_d408ab2ee2f6"),
            PromotionSource(8, "wip", "level_08/after_debrief_54edb89a013b"),
            PromotionSource(9, "current", "."),
        ),
    ),
}


def get_history(game: str) -> ArtifactHistory:
    try:
        return HISTORIES[game]
    except KeyError as exc:
        raise ValueError(f"no canonical artifact history for {game!r}") from exc
