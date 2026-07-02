"""Object-relative MDL world models for compression-progress exploration.

The connector supplies the anchor and logical grid. This module only sees
relative object signatures and opaque actions. Competing programs differ in
which context fields they condition on; rewrites require an explicit MDL gain.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log2
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from logical_grid import objects
from universal_connector import BindingPacket

ALPHA = 0.5


def _bucket(value: int, limit: int = 3) -> int:
    return max(-limit, min(limit, value))


def _size_bucket(size: int) -> int:
    if size <= 4:
        return 0
    if size <= 16:
        return 1
    if size <= 64:
        return 2
    return 3


def _anchor_cell(arr: np.ndarray, packet: BindingPacket, previous=None):
    return packet.anchor.locate(arr, packet.grid, prev_cell=previous)


def _relative_objects(arr: np.ndarray, packet: BindingPacket, anchor_cell):
    if anchor_cell is None:
        return ()
    out = []
    skipped_anchor = False
    for obj in objects(arr, packet.grid):
        if (not skipped_anchor and obj.color == packet.anchor.color
                and obj.cell == anchor_cell):
            skipped_anchor = True
            continue
        out.append((
            obj.color,
            _size_bucket(obj.size),
            _bucket(obj.cell[0] - anchor_cell[0]),
            _bucket(obj.cell[1] - anchor_cell[1]),
        ))
    return tuple(sorted(out))


@dataclass(frozen=True)
class ObjectTransition:
    action: int
    anchor_delta: Tuple[int, int]
    near_before: Tuple[Tuple[int, int, int, int], ...]
    target_bucket: int
    outcome: Tuple


def state_context(arr: np.ndarray, packet: BindingPacket,
                  target_cells: Iterable[Tuple[int, int]] = ()) -> Tuple:
    """Guard vocabulary shared by world models and induced programs."""
    anchor = _anchor_cell(arr, packet)
    relative = _relative_objects(arr, packet, anchor)
    near = tuple(obj for obj in relative if abs(obj[2]) + abs(obj[3]) <= 2)
    target_cells = tuple(target_cells)
    if anchor is None or not target_cells:
        target_bucket = 4
    else:
        target_bucket = min(
            4,
            min(abs(x - anchor[0]) + abs(y - anchor[1])
                for x, y in target_cells),
        )
    return near, target_bucket


def encode_transition(before: np.ndarray, action: int, after: np.ndarray,
                      packet: BindingPacket,
                      target_cells: Iterable[Tuple[int, int]] = ()) -> ObjectTransition:
    before_anchor = _anchor_cell(before, packet)
    after_anchor = _anchor_cell(after, packet, before_anchor)
    if before_anchor is None or after_anchor is None:
        anchor_delta = (0, 0)
    else:
        anchor_delta = (
            after_anchor[0] - before_anchor[0],
            after_anchor[1] - before_anchor[1],
        )

    before_objects = _relative_objects(before, packet, before_anchor)
    after_objects = _relative_objects(after, packet, after_anchor)
    before_counts = Counter(before_objects)
    after_counts = Counter(after_objects)
    removed = tuple(sorted((before_counts - after_counts).elements()))
    added = tuple(sorted((after_counts - before_counts).elements()))
    near_before, target_bucket = state_context(
        before, packet, target_cells)
    outcome = (anchor_delta, removed, added)
    return ObjectTransition(
        action=action,
        anchor_delta=anchor_delta,
        near_before=near_before,
        target_bucket=target_bucket,
        outcome=outcome,
    )


CONTEXT_FIELDS = ("action", "near", "target")
CONTEXT_MASKS = (
    ("action",),
    ("action", "near"),
    ("action", "target"),
    ("action", "near", "target"),
)


@dataclass
class ObjectProgram:
    fields: Tuple[str, ...]
    counts: Dict[Tuple, Counter] = field(default_factory=dict)
    mdl_bits: float = float("inf")

    @property
    def syntax_bits(self) -> float:
        return log2(len(CONTEXT_MASKS) + 1) + len(self.fields)

    def context(self, transition: ObjectTransition) -> Tuple:
        values = {
            "action": transition.action,
            "near": transition.near_before,
            "target": transition.target_bucket,
        }
        return tuple(values[field] for field in self.fields)

    def fit(self, history: Sequence[ObjectTransition]) -> "ObjectProgram":
        counts: Dict[Tuple, Counter] = defaultdict(Counter)
        outcomes = sorted({transition.outcome for transition in history}, key=repr)
        q = max(2, len(outcomes))
        bits = self.syntax_bits
        for transition in history:
            context = self.context(transition)
            row = counts[context]
            total = sum(row.values())
            bits -= log2(
                (row[transition.outcome] + ALPHA)
                / (total + ALPHA * q)
            )
            row[transition.outcome] += 1
        self.counts = dict(counts)
        self.mdl_bits = bits
        return self

    def probability(self, transition: ObjectTransition) -> float:
        row = self.counts.get(self.context(transition), Counter())
        outcomes = set(row)
        outcomes.add(transition.outcome)
        q = max(2, len(outcomes))
        return (
            row[transition.outcome] + ALPHA
        ) / (sum(row.values()) + ALPHA * q)


@dataclass(frozen=True)
class ModelUpdate:
    compression_progress: float
    surprise_bits: float
    disagreement: float
    rewritten: bool
    active_fields: Tuple[str, ...]
    mdl_bits: float


class ObjectMDL:
    """Bounded Gödel-Kolmogorov learner over object-relative programs."""

    def __init__(self, memory: int = 96, rewrite_cost: float = 1.0,
                 delta: float = 0.05):
        self.memory = memory
        self.rewrite_cost = rewrite_cost
        self.delta = delta
        self.history: List[ObjectTransition] = []
        self.active = ObjectProgram(CONTEXT_MASKS[0]).fit([])
        self.ensemble = [self.active]

    @property
    def rewrite_threshold(self) -> float:
        return self.rewrite_cost + log2(1.0 / self.delta)

    def assess(self, transition: ObjectTransition, commit: bool = True) -> ModelUpdate:
        old_probability = self.active.probability(transition)
        extended = (self.history + [transition])[-self.memory:]
        candidates = sorted(
            (ObjectProgram(fields).fit(extended) for fields in CONTEXT_MASKS),
            key=lambda program: program.mdl_bits,
        )
        old_extended = ObjectProgram(self.active.fields).fit(extended)
        best = candidates[0]
        raw_gain = old_extended.mdl_bits - best.mdl_bits
        progress = max(0.0, raw_gain - self.rewrite_threshold)
        rewritten = best.fields != self.active.fields and progress > 0.0

        probabilities = [program.probability(transition)
                         for program in candidates[:3]]
        disagreement = max(probabilities) - min(probabilities)
        update = ModelUpdate(
            compression_progress=progress,
            surprise_bits=-log2(max(old_probability, 1e-12)),
            disagreement=disagreement,
            rewritten=rewritten,
            active_fields=best.fields if rewritten else self.active.fields,
            mdl_bits=best.mdl_bits if rewritten else old_extended.mdl_bits,
        )
        if commit:
            self.history = extended
            self.active = best if rewritten else old_extended
            self.ensemble = candidates[:3]
        return update
