"""LLM-proposed, interaction-verified bindings for the abstract crack engine.

This is the only layer allowed to name rendered object attributes. The engine
receives opaque actions, verified action effects, and callable goal potentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import llm_binder
import priors
import proposer
from anchor_connector import AnchorConnector
from cofibrant import Anchor
from llm_binder import BoundVerb, attach_goal_fn
from logical_grid import Grid, objects

Frame = np.ndarray
Vector = Tuple[int, int]
_OBJECT_COUNTER_CACHE = {}


@dataclass
class BoundObjective:
    name: str
    verb: str
    potential: Callable[[Frame], float] = field(repr=False)
    rebind: Callable[[Frame], "BoundObjective"] = field(repr=False)
    source: str = ""
    description_cost: float = 1.0
    target_cells: frozenset = frozenset()
    metadata: dict = field(default_factory=dict)


@dataclass
class BindingPacket:
    actions: Tuple[int, ...]
    anchor: Anchor
    grid: Grid
    movement: Dict[int, Vector]
    effect_candidates: Tuple[int, ...]
    objectives: List[BoundObjective]
    source: str
    manipulation: str = "unknown"   # grounded action SEMANTICS: how the agent
                                    # moves objects (push | pick_up_and_carry | ...)

    def at_level(self, frame: Frame) -> "BindingPacket":
        """Rebind level-relative relations such as footprints to a new frame."""
        return BindingPacket(
            actions=self.actions,
            anchor=self.anchor,
            grid=self.grid,
            movement=self.movement,
            effect_candidates=self.effect_candidates,
            objectives=[objective.rebind(frame) for objective in self.objectives],
            source=self.source,
            manipulation=self.manipulation,
        )


def _expand_goal_variants(bound: Sequence[BoundVerb], scene: dict) -> List[BoundVerb]:
    """Turn imperfect LLM bindings into verifier candidates without game constants."""
    interiors = {interior for _, interior, _ in scene["containers"]}
    nonstructure = set(scene["colour_counts"]) - set(scene["structure_colours"])
    out: List[BoundVerb] = []
    seen = set()
    for proposal in bound:
        candidates = [proposal]
        if proposal.verb == "transport":
            carrier = proposal.params.get("carrier")
            region = proposal.params.get("region")
            candidates = []
            for c, r in ((carrier, region), (region, carrier)):
                if r in interiors and c in nonstructure and c != r:
                    candidates.append(BoundVerb(
                        name=f"transport({c}->{r})",
                        verb="transport",
                        params={"carrier": c, "region": r},
                        rationale=proposal.rationale,
                    ))
        for candidate in candidates:
            key = (candidate.verb, tuple(sorted(candidate.params.items())))
            if key not in seen:
                seen.add(key)
                out.append(candidate)
    return out


def _algorithmic_goal_candidates(scene: dict, frame: list, anchor_color: int) -> List[BoundVerb]:
    """Fallback proposals use generic scene relations, then face the same verifier."""
    out = []
    for ring, interior, _ in scene["containers"]:
        out.append(BoundVerb(
            name="container transport",
            verb="transport",
            params={"carrier": ring, "region": interior},
            rationale="generic detected container relation",
        ))
    if not out:
        for proposal in proposer.propose_algorithmic(scene, frame, anchor_color):
            if proposal.kind in ("empty", "reach"):
                out.append(BoundVerb(proposal.name, proposal.kind, {}, proposal.notes))
    return out


def bind_game(make_env, model: str = llm_binder.DEFAULT_MODEL,
              use_llm: bool = True) -> BindingPacket:
    """Propose bindings with the local LLM and retain only measurable contracts."""
    env = make_env()
    snapshot = env.reset()
    actions = tuple(env.available_actions or ())
    anchor_result = AnchorConnector(model=model, use_llm=use_llm).identify(
        make_env, actions=actions)
    if anchor_result.anchor is None:
        raise RuntimeError("connector could not verify a controllable anchor")
    anchor = anchor_result.anchor
    movement = {
        action: vector for action, vector in anchor.vectors.items()
        if action in actions and vector != (0, 0)
    }
    effect_candidates = tuple(action for action in actions if action not in movement)

    frame = snapshot.frame
    scene = proposer.scene_summary(frame, list(actions))
    proposals = []
    if use_llm:
        try:
            proposals = llm_binder.bind(
                scene, anchor.color, snapshot.win_levels, model=model)
        except Exception:
            proposals = []
    proposals = _expand_goal_variants(proposals, scene)
    if not proposals:
        proposals = _algorithmic_goal_candidates(scene, frame, anchor.color)

    source = "llm-proposed" if use_llm else "algorithmic-proposed"

    def make_objective(proposal: BoundVerb, level_frame) -> Optional[BoundObjective]:
        rebound = BoundVerb(
            proposal.name, proposal.verb, dict(proposal.params),
            proposal.rationale)
        raw = level_frame.tolist() if isinstance(level_frame, np.ndarray) else level_frame
        attach_goal_fn(rebound, raw, anchor.color)
        if rebound.goal_fn is None:
            return None
        level_grid = Grid.infer(np.asarray(raw))
        target_cells = frozenset()
        if rebound.verb == "transport" and "region" in rebound.params:
            region = rebound.params["region"]
            carrier = rebound.params.get("carrier")
            footprint = tuple(
                (x, y)
                for y, row in enumerate(raw)
                for x, value in enumerate(row)
                if value == region
            )
            if footprint:
                xs = [x for x, _ in footprint]
                ys = [y for _, y in footprint]
                x0, x1 = max(0, min(xs) - 1), min(
                    len(raw[0]) - 1, max(xs) + 1)
                y0, y1 = max(0, min(ys) - 1), min(
                    len(raw) - 1, max(ys) + 1)
                target_cells = frozenset(
                    level_grid.cell_of_pixel(x, y)
                    for y in range(y0, y1 + 1)
                    for x in range(x0, x1 + 1)
                )

            def potential(arr, footprint=footprint, carrier=carrier):
                return -float(sum(
                    arr[y, x] == carrier for x, y in footprint))
        else:
            potential = lambda arr, fn=rebound.goal_fn: float(fn(arr.tolist()))

        def rebind(next_frame, proposal=proposal):
            return make_objective(proposal, next_frame)

        return BoundObjective(
            name=rebound.name,
            verb=rebound.verb,
            potential=potential,
            rebind=rebind,
            source=source,
            target_cells=target_cells,
        )

    objectives: List[BoundObjective] = []
    for proposal in proposals:
        objective = make_objective(proposal, frame)
        if objective is not None:
            objectives.append(objective)
    if not objectives:
        raise RuntimeError("connector produced no executable goal hypotheses")

    grid = Grid.infer(np.asarray(frame))
    try:
        manipulation, manip_src = ground_manipulation(
            make_env, anchor, movement, effect_candidates, grid,
            model=model, use_llm=use_llm)
    except Exception:
        manipulation, manip_src = "unknown", "probe-failed"
    source = (f"anchor:{anchor_result.source}; goals:{objectives[0].source}; "
              f"manipulation:{manipulation}({manip_src})")
    return BindingPacket(
        actions=actions,
        anchor=anchor,
        grid=grid,
        movement=movement,
        effect_candidates=effect_candidates,
        objectives=objectives,
        source=source,
        manipulation=manipulation,
    )


def ground_manipulation(make_env, anchor: Anchor, movement: Dict[int, Vector],
                        effect_candidates: Tuple[int, ...], grid: Grid,
                        model: str = llm_binder.DEFAULT_MODEL,
                        use_llm: bool = True) -> Tuple[str, str]:
    """Ground the SEMANTICS of the actions by interaction: does the anchor PUSH a
    movable object (it slides when walked into) or PICK IT UP AND CARRY it (walking
    into it is blocked, but an effect action attaches it so it co-moves)? A human
    forms exactly this concept by watching; without it the search is blind.

    Probe -> (optional) local-LLM naming from legible trials -> interaction verifier
    is ground truth. Returns (verb, source). Anchor-relative, no game constants.
    """
    import copy
    from arcengine import ActionInput, GameAction as EA

    name = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}

    def fr(fd):
        return np.asarray(fd.frame[-1])

    def stp(g, a):
        gc = copy.deepcopy(g)
        return gc, gc.perform_action(ActionInput(id=EA[name[a]]), raw=True)

    def over(fd):
        return str(fd.state).endswith("GAME_OVER")

    env = make_env(); env.reset()
    g = copy.deepcopy(env._env._game)
    fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)

    def movables(arr, acell):
        out = []
        for o in objects(arr, grid):
            if o.color == anchor.color or o.size > 8:
                continue
            out.append(o)
        return out

    arr = fr(fd)
    acell = anchor.locate(arr, grid)
    movs = movables(arr, acell) if acell else []
    if not acell or not movs:
        return ("unknown", "no movable object to probe")
    target = min(movs, key=lambda o: abs(o.cell[0] - acell[0]) + abs(o.cell[1] - acell[1]))
    tcell0 = target.cell

    # navigate the anchor to a cell orthogonally adjacent to the target (clones)
    import heapq
    adj = {(tcell0[0]-1, tcell0[1]), (tcell0[0]+1, tcell0[1]),
           (tcell0[0], tcell0[1]-1), (tcell0[0], tcell0[1]+1)}
    seen = {acell}
    heap = [(0, 0, g, fd, [])]
    ctr = 1
    reached = None
    while heap:
        _, n, cg, cfd, path = heapq.heappop(heap)
        c = anchor.locate(fr(cfd), grid)
        if c in adj:
            reached = (cg, cfd, c)
            break
        if n > 3000 or len(path) > 30:
            continue
        for a in movement:
            ng, nfd = stp(cg, a)
            if over(nfd) or not getattr(nfd, "frame", None):
                continue
            c2 = anchor.locate(fr(nfd), grid)
            if c2 is None or c2 in seen:
                continue
            seen.add(c2)
            h = min(abs(c2[0]-p[0])+abs(c2[1]-p[1]) for p in adj)
            heapq.heappush(heap, (h, ctr, ng, nfd, path)); ctr += 1
    if reached is None:
        return ("unknown", "could not reach an object to probe")
    ag, afd, ac = reached
    into = next((a for a, v in movement.items()
                 if (ac[0]+ (1 if v[0]>0 else -1 if v[0]<0 else 0)) == tcell0[0]
                 and (ac[1]+(1 if v[1]>0 else -1 if v[1]<0 else 0)) == tcell0[1]), None)
    if into is None:
        return ("unknown", "no movement faces the object")

    def track(arr, prev):
        cands = [o for o in objects(arr, grid) if o.color != anchor.color and o.size <= 8]
        if not cands:
            return None
        return min(cands, key=lambda o: abs(o.cell[0]-prev[0])+abs(o.cell[1]-prev[1])).cell

    # Experiment A: walk straight into the object.
    pg, pfd = stp(ag, into)
    moved = track(fr(pfd), tcell0)
    objA = "object_displaced" if (moved and moved != tcell0) else "object_static_agent_blocked"

    table = {"move_into_object": objA}
    # Experiment B: face + each effect action; then move and test co-movement.
    verb_truth = "push" if objA == "object_displaced" else "no_effect"
    for eff in effect_candidates:
        fg, ffd = stp(ag, into)        # face
        fg, ffd = stp(fg, eff)         # effect
        tc = track(fr(ffd), tcell0)
        if tc is None:
            continue
        ac2 = anchor.locate(fr(ffd), grid)
        comove = False
        for mv in movement:
            mg, mfd = stp(fg, mv)
            ac3 = anchor.locate(fr(mfd), grid)
            tc3 = track(fr(mfd), tc)
            if ac3 is None or tc3 is None or ac3 == ac2:
                continue
            d_a = (ac3[0]-ac2[0], ac3[1]-ac2[1]); d_t = (tc3[0]-tc[0], tc3[1]-tc[1])
            if d_a == d_t and d_t != (0, 0):
                comove = True; break
        table["effect_action"] = "object_attaches" if comove else table.get("effect_action", "no_attach")
        if comove:
            verb_truth = "pick_up_and_carry"
            break

    # propose -> verify: LLM names it from legible trials, verifier decides.
    source = "interaction-verifier"
    if use_llm:
        try:
            trials = (
                f"Trial 1 (agent walks into the object): "
                f"{'the object slid away' if objA=='object_displaced' else 'the object did NOT move; the agent was blocked'}.\n"
                f"Trial 2 (agent faces it, presses a special action, then walks): "
                f"{'the object moved together with the agent' if verb_truth=='pick_up_and_carry' else 'the object stayed where it was'}.")
            prompt = ("Reverse-engineer a grid game. Observed trials:\n\n" + trials +
                      "\n\nHow does the agent move objects? Choose ONE: \"push\" "
                      "(slides when walked into) / \"pick_up_and_carry\" (attaches and "
                      "travels with the agent) / \"no_effect\". "
                      "Reply JSON: {\"verb\": <one>, \"why\": <one sentence>}.")
            out = llm_binder.ollama_json(prompt, model=model)
            verb = out.get("verb") if out else None
            if verb in ("push", "pick_up_and_carry", "no_effect"):
                agree = verb == verb_truth
                source = f"llm:{model}{'' if agree else ' [OVERRIDDEN]'} ({out.get('why','')[:60]})"
        except Exception:
            pass
    return (verb_truth, source)


def non_anchor_change(before: Frame, after: Frame, anchor_color: int) -> int:
    """Count changed pixels not attributable solely to the controlled object."""
    changed = before != after
    mask = changed & (before != anchor_color) & (after != anchor_color)
    return int(mask.sum())


def external_object_change(before: Frame, after: Frame, packet: BindingPacket) -> int:
    """Logical object delta after removing only the tracked anchor component."""
    before_anchor = packet.anchor.locate(before, packet.grid)
    after_anchor = packet.anchor.locate(
        after, packet.grid, prev_cell=before_anchor)
    def object_counter(arr):
        key = arr.tobytes()
        cached = _OBJECT_COUNTER_CACHE.get(key)
        if cached is None:
            cached = Counter(
                (obj.color, obj.cell) for obj in objects(arr, packet.grid))
            if len(_OBJECT_COUNTER_CACHE) >= 12000:
                _OBJECT_COUNTER_CACHE.clear()
            _OBJECT_COUNTER_CACHE[key] = cached
        return cached.copy()

    before_objects = object_counter(before)
    after_objects = object_counter(after)
    if before_anchor is not None:
        before_objects[(packet.anchor.color, before_anchor)] -= 1
    if after_anchor is not None:
        after_objects[(packet.anchor.color, after_anchor)] -= 1
    before_objects += Counter()
    after_objects += Counter()
    return sum((before_objects - after_objects).values()) + sum(
        (after_objects - before_objects).values())
