"""
Compose discovered legs into a cone.

cone_leg_discovery learns a library of channel-blind directional effect-legs
("drive the avatar this way until blocked"). This module glues them into a
controller for a goal — the cone over the LEARNED fragments. It is the ARC
analogue of cone_method_foraging.cone_for_features, except the legs are no
longer hand-built witnesses: they were learned from the game.

The composed seeker is channel-blind in the same sense as witness_seek_leg: it
reads only the azimuth from the avatar to the nearest object of the bound goal
colour (the substrate's per-channel observation) and CALLs the discovered leg
whose learned direction best matches that azimuth. The crucial difference from
the hand-built witness is that it can only use directions that were DISCOVERED
to be controllable — so on a game where a direction does nothing, the learned
seeker never relies on it.

This faithfully demonstrates the full pipeline end to end — learn legs from the
game, then form the cone over them — and, on a navigation-shaped environment,
the learned seeker reaches the goal exactly like the hand-built one.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import arc_agi3_adapter as arc
import cone_leg_discovery as cld

Vector = Tuple[int, int]


def _nearest_goal(scene: arc.Scene, goal_color: int, here: Vector) -> Optional[Vector]:
    targets = [o.centroid for o in scene.objects_of_color(goal_color)
               if o is not scene.avatar]
    if not targets:
        return None
    return min(targets, key=lambda c: abs(c[0] - here[0]) + abs(c[1] - here[1]))


def choose_leg(legs: List[cld.DiscoveredLeg], here: Vector, target: Vector) -> Optional[cld.DiscoveredLeg]:
    """Pick the learned leg whose direction reduces the larger remaining axis gap
    to the target. Returns None when no learned direction makes progress (the
    honest stuck signal — e.g. the only useful direction was never learned)."""
    dx, dy = target[0] - here[0], target[1] - here[1]
    best = None
    best_gain = 0
    for leg in legs:
        ex, ey = leg.direction
        gain = (ex * dx if ex else 0) + (ey * dy if ey else 0)  # positive if it closes a gap
        if gain > best_gain:
            best_gain = gain
            best = leg
    return best


def run_composed_seek(
    env,
    legs: List[cld.DiscoveredLeg],
    goal_color: int,
    avatar_color: Optional[int] = None,
    max_steps: int = 64,
) -> Tuple[object, int, bool]:
    """Drive `env` toward the nearest object of `goal_color` using only the
    discovered legs. Returns (final_snapshot, steps_taken, reached_goal)."""
    snap = env.reset()
    steps = 0
    for _ in range(max_steps):
        if snap.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            break
        scene = arc.extract_scene(snap.frame, avatar_color=avatar_color)
        if scene.avatar is None:
            break
        here = scene.avatar.centroid
        target = _nearest_goal(scene, goal_color, here)
        if target is None or target == here:
            return snap, steps, True
        leg = choose_leg(legs, here, target)
        if leg is None:
            break  # no learned direction makes progress: honest stuck-halt
        before = here
        snap = env.step(arc.GameAction(leg.action))
        steps += 1
        moved = arc.extract_scene(snap.frame, avatar_color=avatar_color).avatar
        if moved is not None and moved.centroid == before:
            # the chosen leg was blocked here; drop it for this state and retry
            legs_wo = [l for l in legs if l is not leg]
            alt = choose_leg(legs_wo, before, target)
            if alt is None:
                break
            snap = env.step(arc.GameAction(alt.action))
            steps += 1
    if snap.state == arc.GameState.WIN:
        return snap, steps, True
    scene = arc.extract_scene(snap.frame, avatar_color=avatar_color)
    reached = (scene.avatar is not None
               and _nearest_goal(scene, goal_color, scene.avatar.centroid) == scene.avatar.centroid)
    return snap, steps, reached
