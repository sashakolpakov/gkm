"""PowerPlay/OOPS induction of minimal guarded leg extensions.

Existing solvers remain immutable. Candidate programs are enumerated in
description-length order and admitted only when they reduce the selected
counterexample's failure while preserving matching solved-history checkpoints.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from math import log2
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from object_mdl import ObjectMDL, encode_transition, state_context
from universal_connector import BindingPacket, BoundObjective


@dataclass
class ReplayCase:
    game: object
    fd: object
    expected_level: int


@dataclass(frozen=True)
class PowerPlayLeg:
    actions: Tuple[int, ...]
    guard: Tuple
    failure: float
    replay_regressions: int
    description_length: float
    free_energy: float
    name: str

    def applicable(self, arr: np.ndarray, packet: BindingPacket,
                   objective: BoundObjective) -> bool:
        return state_context(arr, packet, objective.target_cells) == self.guard


def description_length(actions: Sequence[int], alphabet_size: int,
                       guard: Tuple) -> float:
    action_bits = len(actions) * log2(max(2, alphabet_size))
    guard_bits = 1.0 + log2(2 + len(repr(guard)))
    return action_bits + guard_bits


def execute_program(step, frame_of, level_of, terminal, game, fd,
                    actions: Sequence[int]):
    current_game, current_fd = game, fd
    for action in actions:
        current_game, current_fd = step(current_game, action)
        if terminal(current_fd):
            break
    return current_game, current_fd


def replay_regressions(leg: PowerPlayLeg, cases: Sequence[ReplayCase],
                       step, frame_of, level_of, terminal,
                       packet: BindingPacket, objective: BoundObjective) -> int:
    regressions = 0
    for case in cases:
        arr = frame_of(case.fd)
        if not leg.applicable(arr, packet, objective):
            continue
        _, end_fd = execute_program(
            step, frame_of, level_of, terminal,
            case.game, case.fd, leg.actions)
        if terminal(end_fd) or level_of(end_fd) < case.expected_level:
            regressions += 1
    return regressions


def induce_extension(
    step,
    frame_of,
    level_of,
    terminal,
    game,
    fd,
    packet: BindingPacket,
    objective: BoundObjective,
    world_model: ObjectMDL,
    replay_cases: Sequence[ReplayCase],
    lambda_description: float = 0.02,
    depth_cap: int = 12,
    node_cap: int = 5000,
    beam_cap: int = 1000,
    time_cap: float = 90.0,
) -> Optional[PowerPlayLeg]:
    """OOPS search: shortest programs first, free energy breaks ties."""
    baseline_level = level_of(fd)
    baseline_potential = objective.potential(frame_of(fd))
    guard = state_context(frame_of(fd), packet, objective.target_cells)
    seen = {frame_of(fd).tobytes()}
    heap = [(0, 0.0, 0, game, fd, ())]
    counter = 1
    nodes = 0
    started = time.time()
    shortlist = []
    while heap and nodes < node_cap and time.time() - started < time_cap:
        depth, _, _, current_game, current_fd, path = heapq.heappop(heap)
        if depth >= depth_cap:
            continue
        for action in packet.actions:
            next_game, next_fd = step(current_game, action)
            nodes += 1
            next_path = path + (action,)
            if not getattr(next_fd, "frame", None):
                continue
            before = frame_of(current_fd)
            after = frame_of(next_fd)
            update = world_model.assess(encode_transition(
                before, action, after, packet, objective.target_cells),
                commit=False,
            )
            won = level_of(next_fd) > baseline_level
            improved = objective.potential(after) < baseline_potential - 1e-9
            viable = won
            if improved and not won and not terminal(next_fd):
                viable = any(
                    not terminal(probe_fd)
                    and getattr(probe_fd, "frame", None)
                    for probe_action in packet.actions
                    for _, probe_fd in [step(next_game, probe_action)]
                )
            informative = (
                update.compression_progress > 0.0
                or update.surprise_bits >= 2.0
                or update.disagreement >= 0.2
            )
            failure = (
                0.0 if won
                else (0.2 if improved and viable else 1.0)
            )
            desc = description_length(next_path, len(packet.actions), guard)
            provisional = PowerPlayLeg(
                actions=next_path,
                guard=guard,
                failure=failure,
                replay_regressions=0,
                description_length=desc,
                free_energy=failure + lambda_description * desc,
                name=f"powerplay[{len(next_path)}]",
            )
            if provisional.free_energy < 1.0:
                shortlist.append(provisional)
            if terminal(next_fd):
                continue
            key = after.tobytes()
            if key in seen:
                continue
            seen.add(key)
            priority = (
                failure + lambda_description * desc
                - 0.1 * update.compression_progress
                - 0.02 * update.disagreement
                - 0.02 * float(informative)
            )
            heapq.heappush(heap, (
                depth + 1, priority, counter,
                next_game, next_fd, next_path))
            counter += 1
        if len(heap) > beam_cap:
            heap = heapq.nsmallest(beam_cap, heap)
            heapq.heapify(heap)
    shortlist.sort(key=lambda leg: (leg.free_energy, len(leg.actions)))
    for provisional in shortlist[:24]:
        regressions = replay_regressions(
            provisional, replay_cases, step, frame_of, level_of,
            terminal, packet, objective)
        candidate = PowerPlayLeg(
            actions=provisional.actions,
            guard=provisional.guard,
            failure=provisional.failure,
            replay_regressions=regressions,
            description_length=provisional.description_length,
            free_energy=provisional.failure + regressions
            + lambda_description * provisional.description_length,
            name=provisional.name,
        )
        if candidate.free_energy < 1.0:
            return candidate
    return None
