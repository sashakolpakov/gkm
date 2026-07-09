"""
Scene-atom discovery from raw frames.

Goal induction (arc_goal_induction.py) previously ran over a HAND-GIVEN atom
vocabulary: the candidate colours and the clear/avoid evaluators were supplied.
This module discovers the atom vocabulary from raw frames instead:

1. AVATAR DISCOVERY — find which object the agent's actions move. Issue each
   simple action from a fresh frame and detect the colour whose object
   translates by the action's delta. This is the first cone any ARC agent must
   learn ("which object do my actions control?").
2. OBJECT-COLOUR DISCOVERY — the non-background, non-avatar colours present.
3. TEMPLATE INSTANTIATION — apply relation schemas {clear, reach, avoid} to
   each discovered colour, yielding candidate atoms with frame-computable
   evaluators.
4. VARIANCE PRUNING — run exploration probes and keep only atoms whose value
   actually varies; constant atoms (e.g. clear@c for a colour nothing
   collects, avoid@c for a colour always far) carry no information and are
   dropped. This is the susceptibility / informativeness idiom used elsewhere
   in the repository, now applied to discover which (colour, relation) pairs
   are live.

The only remaining hand-bias is the template SCHEMA library (clear/reach/avoid)
— the legitimate inductive bias of the morphism vocabulary (COLIMIT_CONE_APPROACH.md
Section 0, consequence 1). Discovering the schemas themselves is the next
frontier; discovering which colours and which schemas are live, and the avatar,
from raw frames, is done here.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cone_foraging as cf
import arc_agi3_adapter as arc

Frame = arc.Frame
Color = int
Cell = Tuple[int, int]

TEMPLATES = ("clear", "reach", "avoid")
TEMPLATE_BEHAVIOR = {"clear": "seek", "reach": "seek", "avoid": "flee"}
SAFE_RADIUS = cf.SAFE_RADIUS


# ---------------------------------------------------------------------------
# Frame-level measurements (all computable by the agent from raw frames)
# ---------------------------------------------------------------------------

def _centroid(cells: Sequence[Cell]) -> Cell:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))


def colors_present(frame: Frame) -> List[Color]:
    present = set()
    for row in frame:
        present.update(row)
    present.discard(arc.BACKGROUND)
    return sorted(present)


def count_color(frame: Frame, color: Color) -> int:
    return len(arc.connected_components(frame, color))


def color_centroids(frame: Frame, color: Color) -> List[Cell]:
    return [_centroid(cells) for cells in arc.connected_components(frame, color)]


def avatar_cell(frame: Frame, avatar_color: Color) -> Optional[Cell]:
    comps = arc.connected_components(frame, avatar_color)
    return _centroid(comps[0]) if comps else None


def nearest_distance(frame: Frame, avatar_color: Color, color: Color) -> Optional[int]:
    here = avatar_cell(frame, avatar_color)
    if here is None:
        return None
    cents = [c for c in color_centroids(frame, color) if c != here]
    if not cents:
        return None
    return min(abs(cx - here[0]) + abs(cy - here[1]) for cx, cy in cents)


# ---------------------------------------------------------------------------
# Avatar discovery: which object do my actions move?
# ---------------------------------------------------------------------------

def discover_avatar_color(game_factory: Callable[[int], object], seed: int = 0) -> Optional[Color]:
    """Find the colour whose object translates by the action delta when the
    agent acts. Votes across the four directional actions; the avatar wins."""
    base = game_factory(seed)
    frame0 = base.render()
    candidates = colors_present(frame0)
    votes: Dict[Color, int] = {c: 0 for c in candidates}
    actions = [
        (arc.GameAction.ACTION1, (0, -1)),
        (arc.GameAction.ACTION2, (1, 0)),
        (arc.GameAction.ACTION3, (0, 1)),
        (arc.GameAction.ACTION4, (-1, 0)),
    ]
    for action, (dx, dy) in actions:
        game = game_factory(seed)
        before = {c: set(color_centroids(frame0, c)) for c in candidates}
        game.step(action)
        after_frame = game.render()
        for color in candidates:
            after = set(color_centroids(after_frame, color))
            shifted = {(x + dx, y + dy) for (x, y) in before[color]}
            # The avatar's object set translates wholesale by the action delta
            # (counts unchanged). A blocked move (wall) leaves it in place;
            # that action simply does not vote.
            if after and after == shifted and len(after) == len(before[color]):
                votes[color] += 1
    if not votes or max(votes.values()) == 0:
        return None
    return max(votes, key=lambda c: votes[c])


def discover_object_colors(frame: Frame, avatar_color: Color) -> List[Color]:
    return [c for c in colors_present(frame) if c != avatar_color]


# ---------------------------------------------------------------------------
# Atoms with frame-computable evaluators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    template: str
    color: Color

    @property
    def name(self) -> str:
        return f"{self.template}@{self.color}"

    @property
    def behavior(self) -> str:
        return TEMPLATE_BEHAVIOR[self.template]

    def evaluate(self, frame0: Frame, frame1: Frame, avatar_color: Color) -> float:
        """Satisfaction in [0,1] from the reset frame (frame0) and the current
        frame (frame1). All inputs are raw frames the agent observes."""
        if self.template == "clear":
            initial = count_color(frame0, self.color)
            now = count_color(frame1, self.color)
            return 1.0 - (now / initial if initial else 0.0)
        distance = nearest_distance(frame1, avatar_color, self.color)
        if self.template == "reach":
            if distance is None:
                return 1.0
            max_dist = max(1, len(frame1[0]) + len(frame1) - 2)
            return 1.0 - distance / max_dist
        # avoid
        if distance is None:
            return 1.0
        return min(1.0, distance / SAFE_RADIUS)


def evaluate_atom(name: str, frame0: Frame, frame1: Frame, avatar_color: Color) -> float:
    template, color = name.split("@")
    return Atom(template, int(color)).evaluate(frame0, frame1, avatar_color)


def instantiate_atoms(colors: Sequence[Color]) -> List[Atom]:
    return [Atom(t, c) for c in colors for t in TEMPLATES]


# ---------------------------------------------------------------------------
# Discovery: avatar + colours + variance-pruned atom vocabulary
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredVocabulary:
    avatar_color: Color
    colors: List[Color]
    atoms: List[Atom]
    pruned: List[Atom] = field(default_factory=list)
    variances: Dict[str, float] = field(default_factory=dict)

    @property
    def atom_names(self) -> List[str]:
        return [a.name for a in self.atoms]


def discover_vocabulary(
    game_factory: Callable[[int], object],
    run_cone: Callable[[object, Sequence], None],
    explore_phase_sets: Sequence[Sequence],
    seeds: Sequence[int],
    avatar_color: Optional[Color] = None,
    variance_eps: float = 0.02,
) -> DiscoveredVocabulary:
    """Discover the atom vocabulary from raw frames. Avatar by action response,
    colours by frame content, atoms by template instantiation, then prune to
    the atoms that vary under the exploration probes."""
    if avatar_color is None:
        avatar_color = discover_avatar_color(game_factory, seeds[0])
    if avatar_color is None:
        raise RuntimeError("could not discover an avatar from action responses")

    frame0_seed0 = game_factory(seeds[0]).render()
    colors = discover_object_colors(frame0_seed0, avatar_color)
    atoms = instantiate_atoms(colors)

    # Exploration: run each probe on several instances, record atom values.
    values: Dict[str, List[float]] = {a.name: [] for a in atoms}
    for phases in explore_phase_sets:
        for seed in seeds:
            game = game_factory(seed)
            frame0 = game.render()
            run_cone(game, phases)
            frame1 = game.render()
            for atom in atoms:
                values[atom.name].append(atom.evaluate(frame0, frame1, avatar_color))

    variances = {
        name: (statistics.pvariance(vals) if len(vals) > 1 else 0.0)
        for name, vals in values.items()
    }
    kept = [a for a in atoms if variances[a.name] >= variance_eps]
    pruned = [a for a in atoms if variances[a.name] < variance_eps]
    return DiscoveredVocabulary(
        avatar_color=avatar_color, colors=colors, atoms=kept, pruned=pruned, variances=variances
    )
