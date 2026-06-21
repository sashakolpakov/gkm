"""Universal staged cracker: primitives for L1, trace-induced legs for L2.

The engine never binds colours or action meanings. `universal_connector` owns
those game-specific choices and exposes only verified effects and potentials.
"""
from __future__ import annotations

import copy
import heapq
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import llm_binder
from arcengine import ActionInput, GameAction as EA
from lab import make_env
from logical_grid import objects
from object_mdl import ObjectMDL, encode_transition
from powerplay import (
    PowerPlayLeg,
    ReplayCase,
    execute_program,
    induce_extension,
)
from universal_connector import (
    BindingPacket,
    BoundObjective,
    bind_game,
    external_object_change,
)


def frame_of(fd) -> np.ndarray:
    return np.asarray(fd.frame[-1])


def terminal(fd) -> bool:
    return str(fd.state).endswith("GAME_OVER")


def apply_action(game, action: int):
    clone = copy.deepcopy(game)
    fd = clone.perform_action(
        ActionInput(id=EA[f"ACTION{action}"]), raw=True)
    return clone, fd


@dataclass(frozen=True)
class EvolvedEffectLeg:
    """An anti-unified L1 trace pattern, expressed only through observed effects."""

    trigger_action: int
    driver_actions: Tuple[int, ...]
    evidence: int
    reliability: float
    description_cost: float
    free_energy: float
    name: str


@dataclass(frozen=True)
class EvolvedAttachmentLeg:
    trigger_action: int
    driver_actions: Tuple[int, ...]
    source_color: int
    source_size: int
    status_color: int
    status_size: int
    evidence: int
    free_energy: float
    name: str


@dataclass(frozen=True)
class EvolvedTraceLeg:
    actions: Tuple[int, ...]
    description_cost: float
    evidence: int
    name: str


@dataclass
class CrackResult:
    path: List[int]
    game: object
    fd: object
    objective: BoundObjective
    evolved_legs_used: int = 0
    compression_progress: float = 0.0
    model_fields: Tuple[str, ...] = ()


@dataclass
class MacroSearchReport:
    crack: Optional[CrackResult]
    best_potential: float
    best_path: List[int]
    expanded: int
    states: int
    objective: Optional[BoundObjective] = None


@dataclass(frozen=True)
class FreeEnergyConfig:
    lambda_action: float = 0.02
    lambda_leg: float = 0.12
    lambda_schema: float = 0.02
    lambda_objective: float = 0.01
    beta_compression: float = 0.8
    beta_disagreement: float = 0.1


FREE_ENERGY = FreeEnergyConfig()


def _search_score(objective: BoundObjective, arr: np.ndarray, primitive_cost: int,
                  leg_cost: int = 0) -> float:
    """GKM target: empirical goal risk plus explicit description complexity."""
    return (
        objective.potential(arr)
        + FREE_ENERGY.lambda_action * primitive_cost
        + FREE_ENERGY.lambda_leg * leg_cost
        + FREE_ENERGY.lambda_objective * objective.description_cost
    )


def discover_level_one(game, fd, packet: BindingPacket, budget: int = 90000,
                       depth_cap: int = 90, time_cap: float = 240.0) -> Optional[CrackResult]:
    """Primitive-only discovery. No learned or authored macro is available."""
    level0 = fd.levels_completed
    started = time.time()
    for objective in packet.objectives:
        world_model = ObjectMDL()
        total_compression_progress = 0.0
        start = frame_of(fd)
        seen = [{start.tobytes()}, {start.tobytes()}]
        heaps = [
            [(_search_score(objective, start, 0), 0, 0, game, fd, ())],
            [(_search_score(objective, start, 0), 0, 0, game, fd, ())],
        ]
        counter = 1
        nodes = 0
        route_turn = 0
        while (any(heaps) and nodes < budget
               and time.time() - started < time_cap):
            route = route_turn % 2
            route_turn += 1
            if not heaps[route]:
                route = 1 - route
            _, _, depth, current_game, current_fd, path = heapq.heappop(
                heaps[route])
            if depth >= depth_cap:
                continue
            for action in packet.actions:
                next_game, next_fd = apply_action(current_game, action)
                nodes += 1
                next_path = path + (action,)
                model_update = None
                if route == 1 and getattr(next_fd, "frame", None):
                    model_update = world_model.assess(encode_transition(
                        frame_of(current_fd), action, frame_of(next_fd), packet,
                        objective.target_cells,
                    ))
                    total_compression_progress += (
                        model_update.compression_progress)
                if next_fd.levels_completed > level0:
                    return CrackResult(
                        list(next_path), next_game, next_fd, objective,
                        compression_progress=total_compression_progress,
                        model_fields=world_model.active.fields,
                    )
                if terminal(next_fd) or not getattr(next_fd, "frame", None):
                    continue
                arr = frame_of(next_fd)
                key = arr.tobytes()
                if key in seen[route]:
                    continue
                seen[route].add(key)
                priority = _search_score(objective, arr, depth + 1)
                if route == 1:
                    priority -= (
                        FREE_ENERGY.beta_compression
                        * (model_update.compression_progress
                           if model_update else 0.0)
                        + FREE_ENERGY.beta_disagreement
                        * (model_update.disagreement
                           if model_update else 0.0)
                    )
                heapq.heappush(heaps[route], (
                    priority,
                    counter, depth + 1, next_game, next_fd, next_path))
                counter += 1
            if len(heaps[route]) > 1500:
                heaps[route] = heapq.nsmallest(1500, heaps[route])
                heapq.heapify(heaps[route])
    return None


def evolve_legs(root_game, root_fd, path: Sequence[int],
                packet: BindingPacket) -> List[EvolvedEffectLeg]:
    """Induce repeated driver* -> external-effect segments from the L1 trace."""
    effect_actions = set(packet.effect_candidates)
    evidence = {action: 0 for action in effect_actions}
    calls = {action: 0 for action in effect_actions}
    observed_driver_actions = set()
    game = copy.deepcopy(root_game)
    fd = copy.deepcopy(root_fd)
    for action in path:
        if action in effect_actions:
            calls[action] += 1
        before = frame_of(fd)
        game, next_fd = apply_action(game, action)
        after = frame_of(next_fd)
        external = external_object_change(before, after, packet)
        if action in packet.movement and external == 0:
            observed_driver_actions.add(action)
        if action in effect_actions and (
            next_fd.levels_completed > fd.levels_completed
            or external > 0
        ):
            evidence[action] += 1
        fd = next_fd

    drivers = tuple(sorted(packet.movement))
    candidates = []
    for action, count in evidence.items():
        if count < 2:
            continue
        reliability = count / max(1, calls[action])
        description_cost = 1.0 + len(drivers)
        free_energy = (
            1.0 - reliability
            + FREE_ENERGY.lambda_schema * description_cost
        )
        candidates.append(EvolvedEffectLeg(
            trigger_action=action,
            driver_actions=drivers,
            evidence=count,
            reliability=reliability,
            description_cost=description_cost,
            free_energy=free_energy,
            name=f"drivers_then_effect[{action}]",
        ))
    primitive_baseline = 1.0
    return sorted(
        (leg for leg in candidates if leg.free_energy < primitive_baseline),
        key=lambda leg: (leg.free_energy, -leg.evidence, leg.trigger_action),
    )


def train_world_model(root_game, root_fd, path: Sequence[int],
                      packet: BindingPacket, objective: BoundObjective):
    model = ObjectMDL()
    replay_cases = []
    game = copy.deepcopy(root_game)
    fd = copy.deepcopy(root_fd)
    for index, action in enumerate(path):
        if index % 4 == 0:
            replay_cases.append(ReplayCase(
                copy.deepcopy(game), copy.deepcopy(fd),
                fd.levels_completed,
            ))
        before = frame_of(fd)
        game, next_fd = apply_action(game, action)
        if getattr(next_fd, "frame", None):
            model.assess(encode_transition(
                before, action, frame_of(next_fd), packet,
                objective.target_cells,
            ))
        fd = next_fd
    replay_cases.append(ReplayCase(
        copy.deepcopy(game), copy.deepcopy(fd), fd.levels_completed))
    return model, replay_cases


def evolve_trace_legs(path: Sequence[int],
                      packet: BindingPacket) -> List[EvolvedTraceLeg]:
    """Compile exact L1-observed effect-delimited programs."""
    effect_actions = set(packet.effect_candidates)
    raw_segments = []
    segment = []
    for action in path:
        segment.append(action)
        if action in effect_actions:
            raw_segments.append(tuple(segment))
            segment = []
    if segment:
        raw_segments.append(tuple(segment))

    counts = Counter(raw_segments)
    return sorted((
        EvolvedTraceLeg(
            actions=actions,
            description_cost=1.0 + len(actions),
            evidence=evidence,
            name=f"trace_leg[{index}]",
        )
        for index, (actions, evidence) in enumerate(counts.items())
        if len(actions) > 1
    ), key=lambda leg: (
        FREE_ENERGY.lambda_schema * leg.description_cost
        - 0.05 * leg.evidence,
        len(leg.actions),
        leg.actions,
    ))


def apply_program(game, actions: Sequence[int]):
    clone = copy.deepcopy(game)
    fd = None
    for action in actions:
        fd = clone.perform_action(
            ActionInput(id=EA[f"ACTION{action}"]), raw=True)
        if terminal(fd):
            break
    return clone, fd


def search_with_trace_legs(game, fd, packet: BindingPacket,
                           legs: Sequence[EvolvedTraceLeg],
                           time_cap: float = 480.0,
                           node_cap: int = 120000,
                           action_cap: int = 70):
    """OOPS ordering: learned programs first, primitives only as extensions."""
    level0 = fd.levels_completed
    started = time.time()
    for objective in packet.objectives:
        start = frame_of(fd)
        model = ObjectMDL()
        tokens = [
            ("leg", leg.actions, leg.description_cost, leg.name)
            for leg in legs
        ] + [
            ("primitive", (action,), 1.0, f"action[{action}]")
            for action in packet.actions
        ]
        heap = [(_search_score(objective, start, 0), 0, 0, game, fd, (), ())]
        seen = {start.tobytes()}
        counter = 1
        nodes = 0
        best = (objective.potential(start), ())
        while (heap and nodes < node_cap
               and time.time() - started < time_cap):
            _, _, action_depth, current_game, current_fd, path, calls = heapq.heappop(heap)
            for kind, actions, token_cost, name in tokens:
                if action_depth + len(actions) > action_cap:
                    continue
                next_game, next_fd = apply_program(current_game, actions)
                nodes += 1
                if next_fd is None:
                    continue
                next_path = path + actions
                next_calls = calls + (name,)
                if next_fd.levels_completed > level0:
                    return CrackResult(
                        list(next_path), next_game, next_fd, objective,
                        evolved_legs_used=sum(
                            call.startswith("trace_leg") for call in next_calls),
                        model_fields=model.active.fields,
                    )
                if terminal(next_fd) or not getattr(next_fd, "frame", None):
                    continue
                arr = frame_of(next_fd)
                key = arr.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                potential = objective.potential(arr)
                if potential < best[0]:
                    best = (potential, next_path)
                update = model.assess(encode_transition(
                    frame_of(current_fd),
                    actions[0] if kind == "primitive" else actions,
                    arr, packet,
                    objective.target_cells,
                ))
                leg_calls = sum(
                    call.startswith("trace_leg") for call in next_calls)
                priority = (
                    _search_score(
                        objective, arr, len(next_path), leg_calls)
                    + FREE_ENERGY.lambda_schema * token_cost
                    - FREE_ENERGY.beta_compression
                    * update.compression_progress
                    - FREE_ENERGY.beta_disagreement
                    * update.disagreement
                )
                heapq.heappush(heap, (
                    priority, counter, len(next_path),
                    next_game, next_fd, next_path, next_calls))
                counter += 1
            if len(heap) > 5000:
                heap = heapq.nsmallest(5000, heap)
                heapq.heapify(heap)
        return MacroSearchReport(
            None, best[0], list(best[1]), nodes, len(seen), objective)
    return MacroSearchReport(None, float("inf"), [], 0, 0, None)


def _near_anchor_objects(arr, packet: BindingPacket, previous=None):
    anchor_cell = packet.anchor.locate(arr, packet.grid, prev_cell=previous)
    if anchor_cell is None:
        return None, []
    nearby = [
        obj for obj in objects(arr, packet.grid)
        if obj.color != packet.anchor.color
        and abs(obj.cell[0] - anchor_cell[0]) + abs(obj.cell[1] - anchor_cell[1]) <= 1
    ]
    return anchor_cell, nearby


def evolve_attachment_legs(root_game, root_fd, path: Sequence[int],
                           packet: BindingPacket) -> List[EvolvedAttachmentLeg]:
    """Induce reversible nearby-component disappearance/appearance."""
    trigger_vanished = Counter()
    trigger_appeared = Counter()
    replacements = Counter()
    game = copy.deepcopy(root_game)
    fd = copy.deepcopy(root_fd)
    previous_anchor = None
    for action in path:
        before = frame_of(fd)
        before_anchor, before_near = _near_anchor_objects(
            before, packet, previous_anchor)
        before_global = {}
        for obj in objects(before, packet.grid):
            if obj.color != packet.anchor.color:
                before_global.setdefault(obj.cell, Counter())[
                    (obj.color, obj.size)] += 1
        game, next_fd = apply_action(game, action)
        after = frame_of(next_fd)
        after_anchor, after_near = _near_anchor_objects(
            after, packet, before_anchor)
        after_global = {}
        for obj in objects(after, packet.grid):
            if obj.color != packet.anchor.color:
                after_global.setdefault(obj.cell, Counter())[
                    (obj.color, obj.size)] += 1
        previous_anchor = after_anchor
        if before_anchor is None or after_anchor is None:
            fd = next_fd
            continue
        before_counts = Counter(
            (obj.color, obj.size) for obj in before_near)
        after_counts = Counter(
            (obj.color, obj.size) for obj in after_near)
        removed = before_counts - after_counts
        added = after_counts - before_counts
        for signature, count in removed.items():
            trigger_vanished[(action, *signature)] += count
        for signature, count in added.items():
            trigger_appeared[(action, *signature)] += count
        for cell in set(before_global) | set(after_global):
            global_removed = (
                before_global.get(cell, Counter())
                - after_global.get(cell, Counter()))
            global_added = (
                after_global.get(cell, Counter())
                - before_global.get(cell, Counter()))
            for source, source_count in global_removed.items():
                for status, status_count in global_added.items():
                    replacements[(*source, *status)] += min(
                        source_count, status_count)
        fd = next_fd

    legs = []
    for key, vanish_count in trigger_vanished.items():
        action, status_color, status_size = key
        appear_count = trigger_appeared.get(key, 0)
        if vanish_count < 2 or appear_count < 2:
            continue
        latent = [
            (count, source_color, source_size)
            for (source_color, source_size, target_color, target_size), count
            in replacements.items()
            if target_color == status_color and target_size == status_size
            and count >= 2
        ]
        if not latent:
            continue
        _, source_color, source_size = max(latent)
        evidence = min(vanish_count, appear_count)
        description_cost = 6.0
        free_energy = 1.0 / (1.0 + evidence) + (
            FREE_ENERGY.lambda_schema * description_cost)
        legs.append(EvolvedAttachmentLeg(
            trigger_action=action,
            driver_actions=tuple(sorted(packet.movement)),
            source_color=source_color,
            source_size=source_size,
            status_color=status_color,
            status_size=status_size,
            evidence=evidence,
            free_energy=free_energy,
            name=f"acquire_relocate_release[{action}]",
        ))
    return sorted(legs, key=lambda leg: (leg.free_energy, -leg.evidence))


def _anchor_cell(arr, packet: BindingPacket):
    return packet.anchor.locate(arr, packet.grid)


def _matching_objects(arr, packet: BindingPacket, color: int, size: int):
    return [
        obj for obj in objects(arr, packet.grid, [color])
        if abs(obj.size - size) <= max(1, size // 4)
    ]


def induce_attachment_objective(
    level_frame: np.ndarray,
    packet: BindingPacket,
    base: BoundObjective,
    leg: EvolvedAttachmentLeg,
) -> Optional[BoundObjective]:
    """Compose an L1-learned object role with a connector-bound target relation."""
    if not base.target_cells:
        return None

    def role_objects(arr):
        source = _matching_objects(
            arr, packet, leg.source_color, leg.source_size)
        status = _matching_objects(
            arr, packet, leg.status_color, leg.status_size)
        return source + status

    expected_count = len(role_objects(level_frame))
    if expected_count == 0:
        return None
    relation_cost = max(1, len(leg.driver_actions))

    def distance_to_target(cell):
        return min(
            abs(cell[0] - target[0]) + abs(cell[1] - target[1])
            for target in base.target_cells
        )

    def potential(arr):
        visible = role_objects(arr)
        distances = [distance_to_target(obj.cell) for obj in visible]
        unresolved = sum(distance > 0 for distance in distances)
        missing = max(0, expected_count - len(visible))
        anchor = _anchor_cell(arr, packet)
        hidden_distance = (
            distance_to_target(anchor)
            if anchor is not None else relation_cost
        )
        hidden_cost = (
            0 if hidden_distance == 0
            else relation_cost + hidden_distance
        )
        return float(
            sum(distances)
            + relation_cost * unresolved
            + missing * hidden_cost
        )

    def rebind(next_frame):
        rebound_base = base.rebind(next_frame)
        return induce_attachment_objective(
            np.asarray(next_frame), packet, rebound_base, leg)

    return BoundObjective(
        name=f"{leg.name}->{base.name}",
        verb="relational_transport",
        potential=potential,
        rebind=rebind,
        source="l1-induced+connector-verified",
        description_cost=base.description_cost + 6.0,
        target_cells=base.target_cells,
        metadata={
            "expected_role_count": expected_count,
            "attachment_leg": leg.name,
        },
    )


def augment_with_learned_objectives(
    packet: BindingPacket,
    level_frame: np.ndarray,
    attachment_legs: Sequence[EvolvedAttachmentLeg],
) -> BindingPacket:
    """Add only objectives expressible as learned-role × verified-relation."""
    learned = []
    for leg in attachment_legs:
        for base in packet.objectives:
            objective = induce_attachment_objective(
                level_frame, packet, base, leg)
            if objective is not None:
                learned.append(objective)
    return BindingPacket(
        actions=packet.actions,
        anchor=packet.anchor,
        grid=packet.grid,
        movement=packet.movement,
        effect_candidates=packet.effect_candidates,
        objectives=learned + packet.objectives,
        source=packet.source,
    )


def _signature_count(arr, packet: BindingPacket,
                     leg: EvolvedAttachmentLeg) -> int:
    return len(_matching_objects(
        arr, packet, leg.status_color, leg.status_size))


def _near_source(arr, packet: BindingPacket,
                 leg: EvolvedAttachmentLeg,
                 target_cells=()) -> bool:
    anchor = _anchor_cell(arr, packet)
    if anchor is None:
        return False
    target_cells = set(target_cells)
    return any(
        abs(obj.cell[0] - anchor[0]) + abs(obj.cell[1] - anchor[1]) <= 1
        for obj in _matching_objects(
            arr, packet, leg.status_color, leg.status_size)
        if obj.cell not in target_cells
    )


def _distance_to_sources(arr, packet: BindingPacket,
                         leg: EvolvedAttachmentLeg,
                         target_cells=()) -> int:
    anchor = _anchor_cell(arr, packet)
    target_cells = set(target_cells)
    sources = _matching_objects(
        arr, packet, leg.source_color, leg.source_size)
    sources = [obj for obj in sources if obj.cell not in target_cells]
    if anchor is None or not sources:
        return 10**6
    return min(
        abs(obj.cell[0] - anchor[0]) + abs(obj.cell[1] - anchor[1])
        for obj in sources
    )


def _distance_to_target(arr, packet: BindingPacket, objective: BoundObjective) -> int:
    anchor = _anchor_cell(arr, packet)
    if anchor is None or not objective.target_cells:
        return 10**6
    return min(
        abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
        for cell in objective.target_cells
    )


def _drive_attachment_phase(game, fd, leg: EvolvedAttachmentLeg,
                            packet: BindingPacket, objective: BoundObjective,
                            level0: int, acquiring: bool,
                            node_cap: int = 10000, depth_cap: int = 36,
                            outcome_cap: int = 4,
                            deadline: Optional[float] = None):
    start = frame_of(fd)
    seen = {start.tobytes()}
    distance = _distance_to_sources if acquiring else _distance_to_target
    start_distance = (
        distance(start, packet, leg, objective.target_cells)
        if acquiring else distance(start, packet, objective)
    )
    heap = [(start_distance, 0, game, fd, ())]
    counter = 1
    nodes = 0
    outcomes = []
    outcome_keys = set()
    while (
        heap
        and nodes < node_cap
        and (deadline is None or time.time() < deadline)
    ):
        _, _, current_game, current_fd, prefix = heapq.heappop(heap)
        before = frame_of(current_fd)
        before_count = _signature_count(before, packet, leg)
        trigger_ready = (
            _near_source(
                before, packet, leg, objective.target_cells)
            if acquiring else True)
        if trigger_ready:
            trigger_game, trigger_fd = apply_action(
                current_game, leg.trigger_action)
            nodes += 1
            if trigger_fd.levels_completed > level0:
                return [("win", trigger_game, trigger_fd,
                         list(prefix) + [leg.trigger_action])]
            if not terminal(trigger_fd) and getattr(trigger_fd, "frame", None):
                after = frame_of(trigger_fd)
                after_count = _signature_count(after, packet, leg)
                changed = (
                    after_count < before_count
                    if acquiring else (
                        after_count > before_count
                        or objective.potential(after)
                        < objective.potential(before)
                    )
                )
                if changed:
                    if acquiring:
                        anchor = _anchor_cell(before, packet)
                        status_cells = tuple(sorted(
                            (obj.cell, (
                                obj.cell[0] - anchor[0],
                                obj.cell[1] - anchor[1],
                            ))
                            for obj in _matching_objects(
                                before, packet,
                                leg.status_color, leg.status_size)
                            if anchor is not None
                            and abs(obj.cell[0] - anchor[0])
                            + abs(obj.cell[1] - anchor[1]) <= 1
                        ))
                        outcome_key = status_cells
                    else:
                        outcome_key = frame_of(trigger_fd).tobytes()
                    if outcome_key in outcome_keys:
                        continue
                    outcome_keys.add(outcome_key)
                    outcomes.append((
                        "done", trigger_game, trigger_fd,
                        list(prefix) + [leg.trigger_action],
                    ))
                    if acquiring and len(outcomes) >= outcome_cap:
                        return outcomes
        if len(prefix) >= depth_cap:
            continue
        for action in leg.driver_actions:
            next_game, next_fd = apply_action(current_game, action)
            nodes += 1
            if terminal(next_fd) or not getattr(next_fd, "frame", None):
                continue
            arr = frame_of(next_fd)
            key = arr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            next_prefix = prefix + (action,)
            d = (
                _distance_to_sources(
                    arr, packet, leg, objective.target_cells)
                if acquiring else _distance_to_target(arr, packet, objective)
            )
            heapq.heappush(heap, (
                d + FREE_ENERGY.lambda_action * len(next_prefix),
                counter, next_game, next_fd, next_prefix))
            counter += 1
    if not acquiring:
        outcomes.sort(key=lambda item: (
            _distance_to_target(frame_of(item[2]), packet, objective),
            len(item[3]),
        ))
    return outcomes[:outcome_cap]


def enumerate_attachment_outcomes(game, fd, legs: Sequence[EvolvedAttachmentLeg],
                                  packet: BindingPacket, objective: BoundObjective,
                                  level0: int,
                                  deadline: Optional[float] = None):
    outcomes = []
    for leg in legs:
        expected_count = objective.metadata.get("expected_role_count")
        if (
            objective.metadata.get("attachment_leg") == leg.name
            and expected_count is not None
        ):
            arr = frame_of(fd)
            visible_count = len(_matching_objects(
                arr, packet, leg.source_color, leg.source_size))
            visible_count += len(_matching_objects(
                arr, packet, leg.status_color, leg.status_size))
            if visible_count < expected_count:
                resumed = _drive_attachment_phase(
                    game, fd, leg, packet, objective, level0, False,
                    node_cap=12000, depth_cap=45, outcome_cap=4,
                    deadline=deadline)
                for status, end_game, end_fd, end_path in resumed:
                    outcomes.append((
                        status, end_game, end_fd, end_path, leg))
                if deadline is not None and time.time() >= deadline:
                    break
        first = _drive_attachment_phase(
            game, fd, leg, packet, objective, level0, True,
            node_cap=18000, depth_cap=50, outcome_cap=6,
            deadline=deadline)
        for status, phase_game, phase_fd, phase_path in first:
            if deadline is not None and time.time() >= deadline:
                break
            if status == "win":
                outcomes.append((
                    status, phase_game, phase_fd, phase_path, leg))
                continue
            second = _drive_attachment_phase(
                phase_game, phase_fd, leg, packet, objective, level0, False,
                node_cap=12000, depth_cap=45, outcome_cap=4,
                deadline=deadline)
            for status2, end_game, end_fd, end_path in second:
                outcomes.append((
                    status2, end_game, end_fd,
                    phase_path + end_path, leg))
    outcomes.sort(key=lambda item: (
        _search_score(objective, frame_of(item[2]), len(item[3]), 1)
        + item[4].free_energy
    ))
    return outcomes[:6]


def enumerate_powerplay_outcomes(game, fd, legs: Sequence[PowerPlayLeg],
                                 packet: BindingPacket,
                                 objective: BoundObjective, level0: int):
    outcomes = []
    arr = frame_of(fd)
    for leg in legs:
        if not leg.applicable(arr, packet, objective):
            continue
        end_game, end_fd = execute_program(
            apply_action, frame_of, lambda item: item.levels_completed,
            terminal, game, fd, leg.actions)
        status = "win" if end_fd.levels_completed > level0 else "done"
        if terminal(end_fd) and status != "win":
            continue
        outcomes.append((
            status, end_game, end_fd, list(leg.actions), leg))
    return outcomes


def enumerate_leg_outcomes(game, fd, leg: EvolvedEffectLeg,
                           packet: BindingPacket, level0: int,
                           objective: BoundObjective,
                           depth_cap: int = 18, node_cap: int = 1500,
                           outcome_cap: int = 4):
    """Rebind one evolved leg to productive sites in the current real state."""
    start = frame_of(fd)
    seen = {start.tobytes()}
    outcomes = []
    outcome_keys = set()
    queue = [(_search_score(objective, start, 0), 0, game, fd, ())]
    counter = 1
    nodes = 0
    while queue and nodes < node_cap:
        _, _, current_game, current_fd, prefix = heapq.heappop(queue)
        before = frame_of(current_fd)

        effect_game, effect_fd = apply_action(current_game, leg.trigger_action)
        nodes += 1
        if effect_fd.levels_completed > level0:
            outcomes.append(("win", effect_game, effect_fd,
                             list(prefix) + [leg.trigger_action]))
            return outcomes
        if (not terminal(effect_fd) and getattr(effect_fd, "frame", None)
                and external_object_change(
                    before, frame_of(effect_fd), packet) > 0):
            key = frame_of(effect_fd).tobytes()
            if key not in outcome_keys:
                outcome_keys.add(key)
                outcomes.append(("done", effect_game, effect_fd,
                                 list(prefix) + [leg.trigger_action]))

        if len(prefix) >= depth_cap:
            continue
        for action in leg.driver_actions:
            next_game, next_fd = apply_action(current_game, action)
            nodes += 1
            if terminal(next_fd) or not getattr(next_fd, "frame", None):
                continue
            arr = frame_of(next_fd)
            key = arr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            next_prefix = prefix + (action,)
            heapq.heappush(queue, (
                _search_score(objective, arr, len(next_prefix)),
                counter, next_game, next_fd, next_prefix))
            counter += 1
        if len(queue) > 1800:
            queue = heapq.nsmallest(1800, queue)
            heapq.heapify(queue)
    outcomes.sort(key=lambda item: _search_score(
        objective, frame_of(item[2]), len(item[3]), 1))
    return outcomes[:outcome_cap]


def enumerate_library_outcomes(game, fd, legs: Sequence[EvolvedEffectLeg],
                               packet: BindingPacket, level0: int,
                               objective: BoundObjective, depth_cap: int = 18,
                               node_cap: int = 1000, outcome_cap: int = 12,
                               deadline: Optional[float] = None):
    """Search shared drivers once, testing every induced effect at each state."""
    start = frame_of(fd)
    drivers = tuple(sorted({action for leg in legs for action in leg.driver_actions}))
    seen = [{start.tobytes()}, {start.tobytes()}]
    queues = [
        [(_search_score(objective, start, 0), 0, game, fd, ())],
        [(-objective.potential(start), 0, game, fd, ())],
    ]
    counter = 1
    nodes = 0
    outcomes = []
    outcome_keys = set()
    route_turn = 0
    while (any(queues) and nodes < node_cap
           and (deadline is None or time.time() < deadline)):
        route = route_turn % 2
        route_turn += 1
        if not queues[route]:
            route = 1 - route
        _, _, current_game, current_fd, prefix = heapq.heappop(queues[route])
        before = frame_of(current_fd)
        for leg in legs:
            effect_game, effect_fd = apply_action(current_game, leg.trigger_action)
            nodes += 1
            suffix = list(prefix) + [leg.trigger_action]
            if effect_fd.levels_completed > level0:
                return [("win", effect_game, effect_fd, suffix, leg)]
            if (terminal(effect_fd) or not getattr(effect_fd, "frame", None)
                    or external_object_change(
                        before, frame_of(effect_fd), packet) == 0):
                continue
            key = frame_of(effect_fd).tobytes()
            if key not in outcome_keys:
                outcome_keys.add(key)
                outcomes.append(("done", effect_game, effect_fd, suffix, leg))

        if len(prefix) >= depth_cap:
            continue
        for action in drivers:
            next_game, next_fd = apply_action(current_game, action)
            nodes += 1
            if terminal(next_fd) or not getattr(next_fd, "frame", None):
                continue
            arr = frame_of(next_fd)
            key = arr.tobytes()
            if key in seen[route]:
                continue
            seen[route].add(key)
            next_prefix = prefix + (action,)
            priority = (
                _search_score(objective, arr, len(next_prefix))
                if route == 0
                else -objective.potential(arr)
                + FREE_ENERGY.lambda_action * len(next_prefix)
            )
            heapq.heappush(queues[route], (
                priority,
                counter, next_game, next_fd, next_prefix))
            counter += 1
        if len(queues[route]) > 900:
            queues[route] = heapq.nsmallest(900, queues[route])
            heapq.heapify(queues[route])
    def rank(item):
        return (
            _search_score(objective, frame_of(item[2]), len(item[3]), 1)
            + item[4].free_energy
        )

    # Routed selection: retain the low-free-energy execution and one explicit
    # counterfactual per learned effect. The outer search still prices the
    # counterfactual; it is merely not erased at the inner beam boundary.
    routed = []
    by_trigger = {}
    for item in outcomes:
        by_trigger.setdefault(item[4].trigger_action, []).append(item)
    for trigger in sorted(by_trigger):
        group = sorted(by_trigger[trigger], key=rank)
        routed.append(group[0])
        if len(group) > 1:
            routed.append(max(
                group,
                key=lambda item: (
                    objective.potential(frame_of(item[2])),
                    len(item[3]),
                )))
    routed.extend(sorted(outcomes, key=rank))
    selected = []
    selected_keys = set()
    for item in routed:
        key = frame_of(item[2]).tobytes()
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(item)
        if len(selected) >= outcome_cap:
            break
    return selected


def crack_with_evolved_legs(game, fd, packet: BindingPacket,
                            legs: Sequence[EvolvedEffectLeg],
                            attachment_legs: Sequence[EvolvedAttachmentLeg] = (),
                            powerplay_legs: Sequence[PowerPlayLeg] = (),
                            macro_cap: int = 300, time_cap: float = 600.0) -> MacroSearchReport:
    """L2 search. Every expansion is an evolved leg rebound to the current state."""
    if not legs and not attachment_legs and not powerplay_legs:
        return MacroSearchReport(None, float("inf"), [], 0, 0, None)
    level0 = fd.levels_completed
    started = time.time()
    deadline = started + time_cap
    overall_best = float("inf")
    overall_path: List[int] = []
    overall_expanded = 0
    overall_states = 0
    overall_progress = float("-inf")
    overall_objective = None
    for objective in packet.objectives:
        start = frame_of(fd)
        start_potential = objective.potential(start)
        best_potential = start_potential
        best_path: List[int] = []
        fallback_path: List[int] = []
        fallback_score = float("inf")
        heap = [(_search_score(objective, start, 0), 0, game, fd, (), 0)]
        seen = {start.tobytes()}
        counter = 1
        expanded = 0
        while heap and expanded < macro_cap and time.time() - started < time_cap:
            _, _, current_game, current_fd, path, used = heapq.heappop(heap)
            expanded += 1
            current_potential = objective.potential(frame_of(current_fd))
            if current_potential < best_potential:
                best_potential = current_potential
                best_path = list(path)
            outcomes = enumerate_powerplay_outcomes(
                current_game, current_fd, powerplay_legs,
                packet, objective, level0)
            if not outcomes and time.time() < deadline:
                outcomes = enumerate_attachment_outcomes(
                    current_game, current_fd, attachment_legs,
                    packet, objective, level0, deadline=deadline)
            if not outcomes and time.time() < deadline:
                outcomes = enumerate_library_outcomes(
                    current_game, current_fd, legs, packet, level0, objective,
                    deadline=deadline)
            for status, next_game, next_fd, suffix, leg in outcomes:
                next_path = path + tuple(suffix)
                if status == "win":
                    crack = CrackResult(
                        list(next_path), next_game, next_fd, objective, used + 1)
                    return MacroSearchReport(
                        crack, objective.potential(frame_of(next_fd)),
                        list(next_path), expanded, len(seen), objective)
                arr = frame_of(next_fd)
                key = arr.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidate_score = (
                    _search_score(
                        objective, arr, len(next_path), used + 1)
                    + leg.free_energy
                )
                if candidate_score < fallback_score:
                    fallback_score = candidate_score
                    fallback_path = list(next_path)
                heapq.heappush(heap, (
                    candidate_score,
                    counter, next_game, next_fd, next_path, used + 1))
                counter += 1
        if not best_path:
            best_path = fallback_path
        progress = start_potential - best_potential
        if (
            progress > overall_progress
            or (
                progress == overall_progress
                and bool(best_path) and not overall_path
            )
        ):
            overall_progress = progress
            overall_best = best_potential
            overall_path = best_path
            overall_objective = objective
        overall_expanded += expanded
        overall_states += len(seen)
    return MacroSearchReport(
        None, overall_best, overall_path, overall_expanded, overall_states,
        overall_objective)


def validate(game_name: str, path: Sequence[int], expected_level: int) -> bool:
    env = make_env(game_name)()
    env.reset()
    game = copy.deepcopy(env._env._game)
    fd = game.perform_action(ActionInput(id=EA.RESET), raw=True)
    for action in path:
        fd = game.perform_action(
            ActionInput(id=EA[f"ACTION{action}"]), raw=True)
    return fd.levels_completed >= expected_level


def main():
    game_name = sys.argv[1] if len(sys.argv) > 1 else "wa30"
    model = sys.argv[2] if len(sys.argv) > 2 else llm_binder.DEFAULT_MODEL
    use_llm = "--no-llm" not in sys.argv
    l1_only = "--l1-only" in sys.argv
    trace_diagnostic = "--trace-diagnostic" in sys.argv
    factory = make_env(game_name)

    packet = bind_game(factory, model=model, use_llm=use_llm)
    print(f"binding source={packet.source}")
    print(f"actions={packet.actions} movement={packet.movement} "
          f"effect_candidates={packet.effect_candidates}")
    print(f"objectives={[(o.verb, o.name) for o in packet.objectives]}")

    env = factory()
    env.reset()
    root_game = copy.deepcopy(env._env._game)
    root_fd = root_game.perform_action(ActionInput(id=EA.RESET), raw=True)

    l1 = discover_level_one(root_game, root_fd, packet)
    if l1 is None:
        print("L1 not cracked by primitive-only discovery")
        return
    print(f"L1 cracked: path_len={len(l1.path)} objective={l1.objective.verb} "
          f"validated={validate(game_name, l1.path, 1)} "
          f"compression_progress={l1.compression_progress:.3f} "
          f"model={l1.model_fields}")
    if l1_only:
        print(f"PATH={l1.path}")
        return

    legs = evolve_legs(root_game, root_fd, l1.path, packet)
    trace_legs = evolve_trace_legs(l1.path, packet)
    attachment_legs = evolve_attachment_legs(
        root_game, root_fd, l1.path, packet)
    print("evolved legs=" + repr([
        (leg.name, leg.evidence, round(leg.reliability, 3),
         round(leg.free_energy, 3))
        for leg in legs
    ]))
    print("evolved attachment legs=" + repr([
        (leg.name, (leg.source_color, leg.source_size),
         (leg.status_color, leg.status_size), leg.evidence,
         round(leg.free_energy, 3))
        for leg in attachment_legs
    ]))
    print("evolved trace legs=" + repr([
        (leg.name, leg.actions, leg.evidence)
        for leg in trace_legs
    ]))
    if not legs:
        print("L2 not attempted: L1 yielded no replay-verified transferable leg")
        return

    l2_packet = packet.at_level(frame_of(l1.fd))
    l2_packet = augment_with_learned_objectives(
        l2_packet, frame_of(l1.fd), attachment_legs)
    print("L2 objectives=" + repr([
        (objective.verb, objective.name, objective.source)
        for objective in l2_packet.objectives
    ]))
    if trace_diagnostic:
        trace_result = search_with_trace_legs(
            l1.game, l1.fd, l2_packet, trace_legs)
        if trace_result.crack is not None:
            l2 = trace_result.crack
            full_path = l1.path + l2.path
            print(f"L2 cracked by evolved trace legs: suffix_len={len(l2.path)} "
                  f"evolved_legs_used={l2.evolved_legs_used} "
                  f"validated={validate(game_name, full_path, 2)}")
            print(f"PATH={full_path}")
            return
        print("trace-leg diagnostic plateau: "
              f"best_potential={trace_result.best_potential} "
              f"best_path_len={len(trace_result.best_path)} "
              f"expanded={trace_result.expanded} states={trace_result.states}")

    world_model, replay_cases = train_world_model(
        root_game, root_fd, l1.path, packet, l1.objective)
    l2_report = crack_with_evolved_legs(
        l1.game, l1.fd, l2_packet, legs, attachment_legs,
        macro_cap=60, time_cap=150.0)
    powerplay_legs = []
    counterexample_path = list(l2_report.best_path)
    counterexample_objective = (
        l2_report.objective or l2_packet.objectives[0])
    best_counterexample_path = list(counterexample_path)
    best_counterexample_potential = l2_report.best_potential
    counterexample_game = counterexample_fd = None
    if counterexample_path:
        counterexample_game, counterexample_fd = execute_program(
            apply_action, frame_of, lambda item: item.levels_completed,
            terminal, l1.game, l1.fd, counterexample_path)
    counterexample_states = (
        {frame_of(counterexample_fd).tobytes()}
        if counterexample_fd is not None else set()
    )
    for round_index in range(12):
        if (
            l2_report.crack is not None
            or counterexample_fd is None
            or terminal(counterexample_fd)
        ):
            break
        extension = induce_extension(
            apply_action,
            frame_of,
            lambda item: item.levels_completed,
            terminal,
            counterexample_game,
            counterexample_fd,
            l2_packet,
            counterexample_objective,
            world_model,
            replay_cases,
        )
        if extension is not None and (extension.actions, extension.guard) in {
                (leg.actions, leg.guard) for leg in powerplay_legs}:
            extension = None
        if extension is None:
            reuse = crack_with_evolved_legs(
                counterexample_game,
                counterexample_fd,
                l2_packet,
                legs,
                attachment_legs,
                macro_cap=6,
                time_cap=90.0,
            )
            if reuse.crack is not None:
                counterexample_path.extend(reuse.crack.path)
                l2_report = MacroSearchReport(
                    CrackResult(
                        counterexample_path,
                        reuse.crack.game,
                        reuse.crack.fd,
                        counterexample_objective,
                        evolved_legs_used=2 + len(powerplay_legs),
                    ),
                    reuse.best_potential,
                    counterexample_path,
                    l2_report.expanded + reuse.expanded,
                    l2_report.states + reuse.states,
                    counterexample_objective,
                )
                break
            if not reuse.best_path:
                break
            next_game, next_fd = execute_program(
                apply_action, frame_of, lambda item: item.levels_completed,
                terminal, counterexample_game, counterexample_fd,
                reuse.best_path)
            state_key = frame_of(next_fd).tobytes()
            if terminal(next_fd) or state_key in counterexample_states:
                break
            counterexample_states.add(state_key)
            counterexample_game, counterexample_fd = next_game, next_fd
            counterexample_path.extend(reuse.best_path)
            potential = counterexample_objective.potential(
                frame_of(counterexample_fd))
            if potential < best_counterexample_potential:
                best_counterexample_potential = potential
                best_counterexample_path = list(counterexample_path)
            print(
                f"reuse round {round_index + 1}: "
                f"suffix_len={len(reuse.best_path)} "
                f"potential={potential:.3f}")
            l2_report = MacroSearchReport(
                None,
                best_counterexample_potential,
                best_counterexample_path,
                l2_report.expanded + reuse.expanded,
                l2_report.states + reuse.states,
                counterexample_objective,
            )
            continue
        powerplay_legs.append(extension)
        counterexample_game, counterexample_fd = execute_program(
            apply_action, frame_of, lambda item: item.levels_completed,
            terminal, counterexample_game, counterexample_fd,
            extension.actions)
        counterexample_path.extend(extension.actions)
        counterexample_states.add(frame_of(counterexample_fd).tobytes())
        potential = (
            counterexample_objective.potential(frame_of(counterexample_fd))
            if getattr(counterexample_fd, "frame", None)
            else float("inf")
        )
        if potential < best_counterexample_potential:
            best_counterexample_potential = potential
            best_counterexample_path = list(counterexample_path)
        print(
            f"PowerPlay round {round_index + 1}: actions={extension.actions} "
            f"failure={extension.failure:.3f} "
            f"regressions={extension.replay_regressions} "
            f"description={extension.description_length:.2f} "
            f"F={extension.free_energy:.3f} potential={potential:.3f}")
        if counterexample_fd.levels_completed > l1.fd.levels_completed:
            l2_report = MacroSearchReport(
                CrackResult(
                    counterexample_path,
                    counterexample_game,
                    counterexample_fd,
                    counterexample_objective,
                    evolved_legs_used=1 + len(powerplay_legs),
                ),
                potential,
                counterexample_path,
                l2_report.expanded,
                l2_report.states,
                counterexample_objective,
            )
            break
        l2_report = MacroSearchReport(
            None,
            best_counterexample_potential,
            best_counterexample_path,
            l2_report.expanded,
            l2_report.states,
            counterexample_objective,
        )
    l2 = l2_report.crack
    if l2 is None:
        print("L2 not cracked by evolved-leg-only search; "
              f"best_potential={l2_report.best_potential} "
              f"best_path_len={len(l2_report.best_path)} "
              f"expanded={l2_report.expanded} states={l2_report.states}")
        return
    full_path = l1.path + l2.path
    print(f"L2 cracked: suffix_len={len(l2.path)} evolved_legs_used="
          f"{l2.evolved_legs_used} validated={validate(game_name, full_path, 2)}")
    print(f"PATH={full_path}")


if __name__ == "__main__":
    main()
