"""The universal propose() seam: read a game off the abstract connector and
GENERATE its goal + leg vocabulary, game-agnostically. This is the piece that
was missing — previously the wa30 goal ("empty the container, ring colour =
delivered") was hand-derived by tracing the win. Here it is PROPOSED from the
symbolic scene, so the same code applies to any game behind the connector.

Architecture: propose -> verify.
  * propose(scene): returns several Proposals (goal_fn + legs + notes). Several,
    not one — diverse/stochastic hypotheses are the "routed stochasticity" that
    also lets the verifier route around deadlocks.
  * a local LLM can sit behind propose() (offline, GPU — eval-legal), conditioned
    on the SYMBOLIC scene (not pixels). `propose_algorithmic` is the working,
    LLM-free baseline and the fallback.
  * the deterministic clone-based search (solve_via_proposer.py) VERIFIES each
    proposal against the true reward levels_completed; the proposer never
    replaces the reward, it only narrows the goal/leg space.

A Proposal's goal_fn(frame)->float is a heuristic potential (lower = closer);
it shapes the search but the win is always decided by levels_completed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from lab import arc
import priors

Frame = List[List[int]]
Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Symbolic scene — the game read off the connector (the proposer's input)
# ---------------------------------------------------------------------------

def scene_summary(frame: Frame, available_actions: List[int]) -> Dict:
    """Game-agnostic symbolic description of one frame: the objects/relations a
    human (or an LLM) would read before guessing the goal. No pixels."""
    structure = priors.structure_colours(frame)
    cont = priors.containers(frame)                      # [(ring, interior, centroid)]
    real_cont = [c for c in cont if c[0] not in structure and c[1] not in structure]
    movers = priors.movable_objects(frame)               # [(colour, centroid, size)]
    counts: Dict[int, int] = {}
    for row in frame:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    return {
        "available_actions": list(available_actions),
        "structure_colours": structure,
        "containers": real_cont,
        "movable_objects": movers,
        "colour_counts": counts,
    }


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    name: str
    kind: str                              # "deliver" | "empty" | "reach" | "novelty"
    goal_fn: Callable[[Frame], float]      # lower = closer to the proposed goal
    notes: str = ""


def _count(frame: Frame, colour: int) -> int:
    return sum(1 for row in frame for v in row if v == colour)


def _footprint(frame: Frame, interior: int) -> List[Cell]:
    return [(x, y) for y in range(len(frame)) for x in range(len(frame[0]))
            if frame[y][x] == interior]


def _centroid_of(frame: Frame, colour: int) -> Optional[Cell]:
    cells = [(x, y) for y in range(len(frame)) for x in range(len(frame[0])) if frame[y][x] == colour]
    if not cells:
        return None
    return (round(sum(p[0] for p in cells) / len(cells)), round(sum(p[1] for p in cells) / len(cells)))


def propose_algorithmic(scene: Dict, frame: Frame, avatar_color: Optional[int]) -> List[Proposal]:
    """LLM-free baseline proposer. Generalises the patterns priors detects into
    goals — the same logic that, on wa30, re-derives 'deliver boxes into the
    container' WITHOUT it being hand-coded."""
    proposals: List[Proposal] = []

    # 1) Container present -> delivery-aware goal (locked ring vs covering) and a
    #    plain empty goal. The footprint is fixed at the level's start frame.
    for (ring, interior, _) in scene["containers"]:
        fp = _footprint(frame, interior)
        if not fp:
            continue

        def deliver_goal(f, fp=fp, ring=ring, interior=interior):
            empty = sum(1 for (x, y) in fp if f[y][x] == interior)
            locked = sum(1 for (x, y) in fp if f[y][x] == ring)
            return empty - 0.5 * locked

        def empty_goal(f, fp=fp, interior=interior):
            return sum(1 for (x, y) in fp if f[y][x] == interior)

        proposals.append(Proposal(
            name=f"deliver->container[{ring}/{interior}]", kind="deliver", goal_fn=deliver_goal,
            notes=f"fill the {interior}-interior with the ring colour {ring} (locked delivery, not covering)"))
        proposals.append(Proposal(
            name=f"empty->container[{interior}]", kind="empty", goal_fn=empty_goal,
            notes=f"drive interior colour {interior} to 0"))

    # 2) Avatar + a small 'target' colour -> navigation goal (reach the target).
    if avatar_color is not None:
        for (colour, _cent, size) in scene["movable_objects"]:
            if colour == avatar_color:
                continue

            def reach_goal(f, colour=colour, avatar_color=avatar_color):
                a = _centroid_of(f, avatar_color)
                t = _centroid_of(f, colour)
                if a is None or t is None:
                    return 1e6
                return abs(a[0] - t[0]) + abs(a[1] - t[1])

            proposals.append(Proposal(
                name=f"reach->colour[{colour}]", kind="reach", goal_fn=reach_goal,
                notes=f"navigate the avatar ({avatar_color}) to colour {colour}"))

    # 3) Generic fallback: change the world (any non-structure cell change). Used
    #    when no container/target is recognised, so the search still has a signal.
    structure = set(scene["structure_colours"]) | ({avatar_color} if avatar_color else set())

    def novelty_goal(f, structure=structure):
        # lower = fewer non-structure cells "settled"; here just negative variety,
        # a weak universal driver toward making things happen.
        return -len({(x, y) for y in range(len(f)) for x in range(len(f[0]))
                     if f[y][x] and f[y][x] not in structure})

    proposals.append(Proposal(name="novelty", kind="novelty", goal_fn=novelty_goal,
                              notes="generic: make non-structure things change"))
    return proposals


def propose_llm(scene: Dict, frame: Frame, avatar_color: Optional[int]) -> Optional[List[Proposal]]:
    """Seam for a LOCAL (offline, GPU) LLM that reads the symbolic `scene` and
    proposes goals/legs. Returns None when no local model is configured, so the
    caller falls back to propose_algorithmic. Eval-legal: no internet; the LLM
    only PROPOSES — the search still verifies against levels_completed.

    Implementation note: plug a local model here (e.g. llama.cpp / vLLM on GPU)
    that emits, from `scene`, a small JSON of goal hypotheses + leg hints; map
    each to a Proposal with a goal_fn built from the named colours/objects (the
    same constructors used in propose_algorithmic). Left unwired here because the
    eval box, not this dev box, carries the model."""
    return None


def propose(scene: Dict, frame: Frame, avatar_color: Optional[int], use_llm: bool = True) -> List[Proposal]:
    if use_llm:
        llm = propose_llm(scene, frame, avatar_color)
        if llm:
            return llm
    return propose_algorithmic(scene, frame, avatar_color)
