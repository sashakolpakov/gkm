"""
Effect-control leg discovery: LEARN the legs from a game, don't hand-build them.

Background (the gap this closes). The colimit-cone program has so far learned
the CONE over a FIXED, hand-built leg library (cone_foraging_bound.evolve_bound_task),
the GOAL from reward (cone_goal_induction / arc_goal_induction), and classifier
predicate MACROS over given atoms (the Bongard / abstraction-emergence line).
What it has never done is learn the LEG BODIES — the action-policy fragments —
from interaction. On ARC-AGI-3 you cannot hand-build them (the right primitives
are game-specific: push, slide, sidestep) and you cannot evolve them against the
extrinsic reward (levels_completed is never hit by a random policy: no gradient).

The missing ingredient is an INTRINSIC learning signal that does not require
winning: a fragment is good if, from a recognizable state, it RELIABLY produces
a predictable, REPLICABLE change in the scene (a controllable effect). The
perception for that already exists in arc_agi3_adapter (the scene functor and
SceneDelta) and the avatar is found by action-response, not hardcoded.

This module mines such fragments and validates them with two controls, in the
project's honesty style:
  * held-out replication: an effect must reproduce on states NOT used to propose
    it, or it is pruned;
  * a random-action control: a learned directional leg must beat the directional
    consistency a random option achieves by chance.

A discovered leg is channel-blind (it names a displacement, not a colour), so
naturality — the load-bearing invariant of the cone program — is preserved: the
same leg can later be CALLed bound to any colour slot. cone_leg_composition.py
glues the discovered legs into a seek cone.

Everything here is environment-agnostic: it drives any object exposing
reset()/step()->Snapshot (the StubNavigationGame, ArcEnv, or LocalArcEnv), so it
is tested hermetically on the stub and runs unchanged on real local frames.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import arc_agi3_adapter as arc

Vector = Tuple[int, int]
MakeEnv = Callable[[], object]  # () -> env with reset()/step()->Snapshot

CARDINALS = {  # ARC simple action -> (dx, dy) and a readable name
    1: ((0, -1), "UP"),
    2: ((1, 0), "RIGHT"),
    3: ((0, 1), "DOWN"),
    4: ((-1, 0), "LEFT"),
}


# ---------------------------------------------------------------------------
# Perception helpers (whole-colour centroids; avatar by action response)
# ---------------------------------------------------------------------------

def whole_color_centroids(frame: arc.Frame) -> Dict[int, Vector]:
    """Centroid of ALL cells of each colour (a coarse rigid-translation probe).
    Unlike scene_delta this does not split components, so it is robust for the
    single-object colours the avatar/blocks usually are."""
    sums: Dict[int, List[int]] = {}
    h = len(frame)
    w = len(frame[0]) if h else 0
    for y in range(h):
        row = frame[y]
        for x in range(w):
            c = row[x]
            if c == arc.BACKGROUND:
                continue
            acc = sums.setdefault(c, [0, 0, 0])
            acc[0] += x
            acc[1] += y
            acc[2] += 1
    return {c: (round(sx / n), round(sy / n)) for c, (sx, sy, n) in sums.items()}


def detect_avatar(make_env: MakeEnv, move_actions: List[int], *, max_step: int = 20) -> Optional[int]:
    """Avatar = the colour whose entire cell-set translates by k*delta under some
    directional action (k >= 1), probed from a fresh reset per action. Returns
    None when no colour cleanly translates (no free-moving avatar)."""
    def cells_by_color(frame: arc.Frame) -> Dict[int, set]:
        out: Dict[int, set] = {}
        for y in range(len(frame)):
            for x in range(len(frame[0])):
                c = frame[y][x]
                if c != arc.BACKGROUND:
                    out.setdefault(c, set()).add((x, y))
        return out

    for action in move_actions:
        delta, _ = CARDINALS[action]
        env = make_env()
        snap = env.reset()
        if action not in (getattr(env, "available_actions", None) or move_actions):
            continue
        before = cells_by_color(snap.frame)
        after = cells_by_color(env.step(arc.GameAction(action)).frame)
        for color, b in before.items():
            a = after.get(color, set())
            if not b or len(a) != len(b):
                continue
            for k in range(1, max_step + 1):
                if a == {(px + k * delta[0], py + k * delta[1]) for px, py in b}:
                    return color
    return None


# ---------------------------------------------------------------------------
# Options (action macros) and the effects they produce
# ---------------------------------------------------------------------------

def _walk_to_state(make_env: MakeEnv, seed: int, steps: int, move_actions: List[int]):
    """Reset and take `steps` seeded random moves, returning (env, snapshot). The
    seed makes the pre-state reproducible, so every option is probed from the
    SAME state and the comparison is fair."""
    env = make_env()
    snap = env.reset()
    rng = random.Random(seed)
    for _ in range(steps):
        if snap.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            break
        snap = env.step(arc.GameAction(rng.choice(move_actions)))
    return env, snap


def _apply_saturating(env, snap, action: int, avatar_color: int, cap: int):
    """Repeat `action` until the avatar stops moving (blocked) or `cap`/terminal.
    This is the 'move in a direction until it can't' option — robust to being
    blocked (then it is a no-op). Returns (snapshot, avatar_displacement)."""
    cents = whole_color_centroids(snap.frame)
    start = cents.get(avatar_color)
    cur = snap
    prev = start
    for _ in range(cap):
        if cur.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            break
        cur = env.step(arc.GameAction(action))
        here = whole_color_centroids(cur.frame).get(avatar_color)
        if here is None or here == prev:  # avatar didn't move => blocked / no-op
            break
        prev = here
    end = whole_color_centroids(cur.frame).get(avatar_color)
    if start is None or end is None:
        return cur, (0, 0)
    return cur, (end[0] - start[0], end[1] - start[1])


def _aligned_or_zero(disp: Vector, delta: Vector) -> bool:
    """disp is consistent with delta if it is zero, or moves the right way along
    delta's axis and not at all on the other axis (cardinal moves only)."""
    dx, dy = disp
    ex, ey = delta
    if (dx, dy) == (0, 0):
        return True
    if ex != 0:
        return dy == 0 and dx * ex > 0
    return dx == 0 and dy * ey > 0


def _nonzero_aligned(disp: Vector, delta: Vector) -> bool:
    return disp != (0, 0) and _aligned_or_zero(disp, delta)


# ---------------------------------------------------------------------------
# Discovered legs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveredLeg:
    """A learned, channel-blind effect primitive: 'drive the avatar in
    `direction` until blocked' via repeating `action`. Validated on held-out
    states."""

    name: str
    action: int
    direction: Vector
    consistency: float  # frac of holdout states with aligned-or-zero displacement
    efficacy: float     # frac of holdout states with nonzero aligned displacement
    mean_step: float    # mean |displacement| over states where it moved
    support: int        # number of holdout states


@dataclass
class DiscoveryResult:
    avatar_color: Optional[int]
    legs: List[DiscoveredLeg]                       # survivors (the learned library)
    rejected: List[DiscoveredLeg]                   # failed a control (kept for honesty)
    random_consistency: float                       # best-direction consistency of a random option (the floor)
    cooccurring: Dict[int, Dict[int, float]]        # action -> {other_color: frac it co-moved with avatar} (push candidates)


def discover_effect_legs(
    make_env: MakeEnv,
    avatar_color: Optional[int] = None,
    *,
    trials: int = 24,
    holdout_frac: float = 0.5,
    walk_max: int = 8,
    cap: int = 20,
    seed: int = 0,
    consistency_min: float = 0.85,
    efficacy_min: float = 0.30,
) -> DiscoveryResult:
    """Learn directional effect-legs for `make_env`'s avatar from interaction.

    For each directional action, the saturating option is probed from `trials`
    seeded pre-states; the predicted direction is the action's delta (the leg
    names a displacement, not a colour). A leg survives if its HELD-OUT
    consistency and efficacy clear the thresholds AND it beats the random
    control's best-direction consistency."""
    env0 = make_env()
    env0.reset()
    avail = getattr(env0, "available_actions", None) or [1, 2, 3, 4]
    move_actions = [a for a in avail if a in CARDINALS]
    if not move_actions:
        return DiscoveryResult(None, [], [], 0.0, {})

    if avatar_color is None:
        avatar_color = detect_avatar(make_env, move_actions)
    if avatar_color is None:
        return DiscoveryResult(None, [], [], 0.0, {})

    split = max(1, int(trials * (1 - holdout_frac)))  # train indices [0:split], holdout [split:]

    # Per directional option: collect displacement + colour co-movements per probe.
    disp_by_action: Dict[int, List[Vector]] = {a: [] for a in move_actions}
    cooccur: Dict[int, Dict[int, int]] = {a: {} for a in move_actions}
    for action in move_actions:
        delta, _ = CARDINALS[action]
        for i in range(trials):
            env, snap = _walk_to_state(make_env, seed + i, i % (walk_max + 1), move_actions)
            before_cents = whole_color_centroids(snap.frame)
            snap2, disp = _apply_saturating(env, snap, action, avatar_color, cap)
            disp_by_action[action].append(disp)
            # which other colours translated together with a moving avatar? (push)
            if disp != (0, 0):
                after_cents = whole_color_centroids(snap2.frame)
                for c, c0 in before_cents.items():
                    if c == avatar_color:
                        continue
                    c1 = after_cents.get(c)
                    if c1 is not None and (c1[0] - c0[0], c1[1] - c0[1]) == disp:
                        cooccur[action][c] = cooccur[action].get(c, 0) + 1

    # Random control option: K random moves, net avatar displacement per probe.
    rnd_disps: List[Vector] = []
    for i in range(trials):
        env, snap = _walk_to_state(make_env, seed + i, i % (walk_max + 1), move_actions)
        cents = whole_color_centroids(snap.frame)
        start = cents.get(avatar_color)
        rng = random.Random(seed * 131 + i)
        cur = snap
        for _ in range(cap):
            if cur.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
                break
            cur = env.step(arc.GameAction(rng.choice(move_actions)))
        end = whole_color_centroids(cur.frame).get(avatar_color)
        rnd_disps.append((end[0] - start[0], end[1] - start[1]) if start and end else (0, 0))
    random_consistency = max(
        sum(_aligned_or_zero(d, CARDINALS[a][0]) for d in rnd_disps[split:]) / max(1, len(rnd_disps[split:]))
        for a in move_actions
    )

    legs: List[DiscoveredLeg] = []
    rejected: List[DiscoveredLeg] = []
    for action in move_actions:
        delta, dname = CARDINALS[action]
        holdout = disp_by_action[action][split:]
        n = max(1, len(holdout))
        consistency = sum(_aligned_or_zero(d, delta) for d in holdout) / n
        efficacy = sum(_nonzero_aligned(d, delta) for d in holdout) / n
        moved = [d for d in holdout if d != (0, 0)]
        mean_step = (sum(abs(d[0]) + abs(d[1]) for d in moved) / len(moved)) if moved else 0.0
        leg = DiscoveredLeg(
            name=f"move_{dname.lower()}_until_blocked", action=action, direction=delta,
            consistency=round(consistency, 3), efficacy=round(efficacy, 3),
            mean_step=round(mean_step, 2), support=len(holdout),
        )
        if consistency >= consistency_min and efficacy >= efficacy_min and consistency > random_consistency:
            legs.append(leg)
        else:
            rejected.append(leg)

    cooccur_frac = {
        a: {c: round(cnt / max(1, sum(1 for d in disp_by_action[a] if d != (0, 0))), 3)
            for c, cnt in cs.items()}
        for a, cs in cooccur.items() if cs
    }
    return DiscoveryResult(avatar_color, legs, rejected, round(random_consistency, 3), cooccur_frac)


# ---------------------------------------------------------------------------
# Interaction legs: non-move actions (ACTION5 interact / ACTION7 undo / etc.)
# whose effect is a REPLICABLE non-avatar scene change. These are the legs pure
# navigation misses — on wa30 the level-up is an ACTION5 interaction near a box,
# not a move. Same intrinsic signal (the world reacts) + control (the move
# actions, which only move the avatar, are the floor) as the directional legs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionLeg:
    name: str
    action: int
    efficacy: float   # frac of holdout states where the action causes a NON-avatar scene change
    sites: int        # distinct avatar positions from which it had such an effect
    support: int


def _nonavatar_changed(before: "arc.Frame", after: "arc.Frame", avatar_color: Optional[int]) -> int:
    cnt = 0
    for y in range(len(before)):
        rb, ra = before[y], after[y]
        for x in range(len(rb)):
            b, a = rb[x], ra[x]
            if b != a and b != avatar_color and a != avatar_color:
                cnt += 1
    return cnt


def discover_interaction_legs(
    make_env: MakeEnv,
    avatar_color: Optional[int] = None,
    *,
    trials: int = 24,
    holdout_frac: float = 0.5,
    walk_max: int = 10,
    seed: int = 0,
    efficacy_min: float = 0.10,
) -> Tuple[List[InteractionLeg], float]:
    """Learn interaction legs: non-move actions whose effect is a replicable
    NON-avatar scene change, probed from held-out walked states. Returns
    (legs, control_floor) where control_floor is the best non-avatar-change
    efficacy among the MOVE actions (≈0 — moves just move the avatar). An
    interaction leg survives if its efficacy clears efficacy_min AND beats the
    move-control floor."""
    env0 = make_env()
    env0.reset()
    avail = getattr(env0, "available_actions", None) or [1, 2, 3, 4, 5]
    move_actions = [a for a in avail if a in CARDINALS]
    interact_actions = [a for a in avail if a not in CARDINALS]
    if avatar_color is None and move_actions:
        avatar_color = detect_avatar(make_env, move_actions)
    split = max(1, int(trials * (1 - holdout_frac)))

    def measure(action: int) -> Tuple[float, int, int]:
        eff = 0
        sites = set()
        idxs = list(range(split, trials))
        for i in idxs:
            env, snap = _walk_to_state(make_env, seed + i, i % (walk_max + 1), move_actions or [action])
            before = snap.frame
            av = whole_color_centroids(before).get(avatar_color) if avatar_color else None
            after = env.step(arc.GameAction(action)).frame
            if _nonavatar_changed(before, after, avatar_color) > 0:
                eff += 1
                if av is not None:
                    sites.add(av)
        n = max(1, len(idxs))
        return eff / n, len(sites), n

    control_floor = 0.0
    for a in move_actions:
        e, _, _ = measure(a)
        control_floor = max(control_floor, e)

    legs: List[InteractionLeg] = []
    for a in interact_actions:
        e, sites, n = measure(a)
        if e >= efficacy_min and e > control_floor:
            legs.append(InteractionLeg(name=f"interact_action{a}", action=a,
                                       efficacy=round(e, 3), sites=sites, support=n))
    return legs, round(control_floor, 3)
