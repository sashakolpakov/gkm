"""GKM cracker: an ABSTRACT engine + a game-specific CONNECTOR built by the local
LLM and interaction-learning. One self-contained module.

Separation of concerns (the whole point):
  * The CONNECTOR owns every game-specific fact -- which colour is the avatar, what
    a carrier/target is, what "attached" vs "resting" means, how a box is delivered.
    It is BUILT FROM SCRATCH at game start by the local LLM (anchor + manipulation
    verb) plus interaction-learning (carrier/region colours, the carried/rest
    border colours, the toggle action). It exposes only an abstract surface:
        level(fd)            current level index (the reward)
        objective(fd)        a potential to MINIMISE (0 = current level solvable)
        propose_legs(g, fd)  grounded macros to try, cheapest first
        solved(g)            whether the level's win predicate holds
  * The ENGINE is abstract GKM and knows NOTHING about boxes/borders/colours. It
    composes the connector's legs into a cone, committing a leg only when it lowers
    the connector's objective or advances the level (the lambda*C move-budget
    pressure), and chains across levels. The full action path is replay-validated.

Run:  python gkm_crack.py wa30            # LLM-built connector, L1->L2->L3
      python gkm_crack.py wa30 --no-llm   # algorithmic connector (offline/CI)
"""
from __future__ import annotations
import copy
import heapq
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import priors
from lab import make_env
from logical_grid import components, BACKGROUND
from anchor_connector import AnchorConnector
from arcengine import ActionInput, GameAction as EA

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}
MOVES = (1, 2, 3, 4)
PITCH = 4


# ---------------------------------------------------------------------------
# clone-simulation primitives (shared; the game is the ground-truth verifier)
# ---------------------------------------------------------------------------
def arr_of(fd) -> np.ndarray:
    return np.asarray(fd.frame[-1])


def step(g, a: int):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def terminal(fd) -> bool:
    return str(fd.state).endswith("GAME_OVER")


def _centroid(pixels):
    return (sum(p[0] for p in pixels) / len(pixels),
            sum(p[1] for p in pixels) / len(pixels))


# ===========================================================================
# ABSTRACT GKM ENGINE  --  no game-specific knowledge below this line
# ===========================================================================
def gkm_cone(connector, g, fd, level0, deadline, max_rounds=16, verbose=True):
    """Cone over the connector's grounded legs: try legs cheapest-first; commit one
    only if it advances the level or strictly lowers the objective; repeat until the
    level is solved or no leg makes progress. Returns (g, fd, path)."""
    path: List[int] = []
    for rnd in range(max_rounds):
        if connector.level(fd) > level0:
            break
        before = connector.objective(fd)
        progressed = False
        for leg in connector.propose_legs(g, fd):
            status, ng, nfd, lp = leg(deadline)
            if status == "lost":
                continue
            if connector.level(nfd) > level0:
                return ng, nfd, path + lp
            if status == "done" and connector.objective(nfd) < before - 1e-9:
                g, fd, path = ng, nfd, path + lp
                progressed = True
                if verbose:
                    print(f"    round {rnd}: objective {before:.0f}->"
                          f"{connector.objective(fd):.0f} (+{len(lp)} steps)")
                break
        if not progressed:
            if verbose:
                print(f"    round {rnd}: plateau (objective={before:.0f})")
            break
    return g, fd, path


def gkm_sequential(connector, max_level=3, deadline_s=2400, verbose=True):
    """Crack levels in sequence with the abstract cone; replay-validate the path."""
    g, fd = connector.fresh()
    path: List[int] = []
    started = time.time()
    reached = connector.level(fd)
    while reached < max_level:
        level0 = connector.level(fd)
        if verbose:
            print(f"\n[level {level0 + 1}] {connector.describe(fd)}")
        g, fd, lp = gkm_cone(connector, g, fd, level0,
                             deadline=started + deadline_s, verbose=verbose)
        path += lp
        if connector.level(fd) <= level0:
            if verbose:
                print(f"[level {level0 + 1}] NOT cracked (plateau); stopping")
            break
        reached = connector.level(fd)
        if verbose:
            print(f"[level {level0 + 1}] CRACKED -> level {reached} (cum path={len(path)})")
    ok = connector.validate(path, reached) if path else False
    print(f"\n=== {connector.game}: reached level {reached}/{max_level} | "
          f"path_len={len(path)} | replay-validated={ok} ===")
    if path:
        print(f"PATH={path}")
    return {"reached": reached, "path": path, "validated": ok, "connector": connector}


# ===========================================================================
# GAME-SPECIFIC CONNECTOR  --  built by the local LLM + interaction-learning.
# Everything the engine must NOT know lives here.
# ===========================================================================
@dataclass
class Binding:
    avatar: int
    carrier: int                 # movable box colour (== container-ring colour)
    region: int                  # target interior colour
    mechanic: str = "unknown"
    toggle: int = 5              # the effect action that attaches/releases (learned)
    rest_border: int = 4         # box border at rest (learned)
    carried_border: int = 0      # box border while carried by avatar (learned)
    background: int = 1          # the frame background colour (learned)
    sources: dict = field(default_factory=dict)


@dataclass
class Box:
    tl: Tuple[int, int]          # exact sprite top-left (integer pixels)
    center: Tuple[float, float]
    border: int


@dataclass
class Scene:
    avatar: Optional[Tuple[float, float]]
    boxes: List[Box]
    ring_bbox: Optional[Tuple[int, int, int, int]]
    targets: List[Tuple[int, int]]


class CarryConnector:
    """A pick-up-and-carry connector. Owns all wa30-shaped semantics; the engine
    only sees level/objective/propose_legs. Built by `build` (LLM + learning)."""

    def __init__(self, game: str, factory, B: Binding):
        self.game = game
        self.factory = factory
        self.B = B
        # the target region is a STATIC level landmark; carriers share its colour,
        # so a delivered box overwrites the ring and corrupts a live re-detection.
        # Perceive it ONCE per level (from the clean entry frame) and cache it.
        self._region_cache: Dict[int, Optional[Tuple[int, int, int, int]]] = {}
        self._frontier_cache: Dict[int, Tuple[float, bool]] = {}

    def region_bbox(self, fd) -> Optional[Tuple[int, int, int, int]]:
        lvl = fd.levels_completed
        if lvl not in self._region_cache:
            carr = components(arr_of(fd), self.B.carrier)
            ring = max(carr, key=len) if carr else None
            if ring is not None and len(ring) > 8:
                xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
                self._region_cache[lvl] = (min(xs), min(ys), max(xs), max(ys))
            else:
                self._region_cache[lvl] = None
        return self._region_cache[lvl]

    # ---- construction: the LLM + interaction-learning build the binding --------
    @classmethod
    def build(cls, game: str, model=None, use_llm=True, verbose=True) -> "CarryConnector":
        factory = make_env(game)
        env = factory(); snap = env.reset()
        actions = tuple(env.available_actions or ())
        frame = np.asarray(snap.frame)

        kw = {} if model is None else {"model": model}
        ar = AnchorConnector(use_llm=use_llm, **kw).identify(factory, actions=actions)
        if ar.anchor is None:
            raise RuntimeError("connector could not verify a controllable anchor")
        avatar = ar.anchor.color
        movement = {a: v for a, v in ar.anchor.vectors.items()
                    if a in actions and v != (0, 0)}
        effects = tuple(a for a in actions if a not in movement)

        struct = set(priors.structure_colours(frame.tolist()))
        conts = [(r, i) for (r, i, _c) in priors.containers(frame.tolist())
                 if r not in struct and i not in struct]
        if not conts:
            raise RuntimeError("connector found no container/target region")
        conts.sort(key=lambda t: -sum(1 for row in frame for v in row if v == t[1]))
        carrier, region = conts[0]

        background = int(np.bincount(frame.flatten()).argmax())   # learned: modal colour
        B = Binding(avatar=avatar, carrier=carrier, region=region,
                    background=background, sources={"anchor": ar.source})
        self = cls(game, factory, B)

        g = copy.deepcopy(env._env._game)
        fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
        table, learned = self._probe(g, fd, effects)
        B.rest_border = learned.get("rest_border", B.rest_border)
        B.carried_border = learned.get("carried_border", B.carried_border)
        B.toggle = learned.get("toggle", B.toggle)
        B.mechanic, msrc = self._name_mechanic(table, model=model, use_llm=use_llm)
        B.sources["mechanic"] = msrc
        if verbose:
            print("CONNECTOR (built from scratch by LLM + interaction-learning):")
            print(f"  anchor/avatar = colour {avatar}   [{ar.source}]")
            print(f"  carrier(box)  = colour {carrier}   region = colour {region}   [container percept]")
            print(f"  borders: rest={B.rest_border} carried={B.carried_border}   toggle=ACTION{B.toggle}   [learned by probe]")
            print(f"  manipulation  = {B.mechanic}   [{msrc}]")
            print(f"  probe trials  = {table}")
        return self

    # ---- perception (game-specific) -------------------------------------------
    def _border_of(self, arr, comp):
        xs = [p[0] for p in comp]; ys = [p[1] for p in comp]
        vals = []
        for y in range(max(0, min(ys) - 1), min(arr.shape[0], max(ys) + 2)):
            for x in range(max(0, min(xs) - 1), min(arr.shape[1], max(xs) + 2)):
                v = int(arr[y, x])
                if v != self.B.carrier:
                    vals.append(v)
        return max(set(vals), key=vals.count) if vals else self.B.carrier

    def perceive(self, arr, region_bbox=None) -> Scene:
        B = self.B
        av = components(arr, B.avatar)
        avatar = _centroid(max(av, key=len)) if av else None
        carr = components(arr, B.carrier)
        if region_bbox is None:
            ring = max(carr, key=len) if carr else None
            ring_bbox = None
            if ring is not None and len(ring) > 8:
                xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
                ring_bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            ring_bbox = region_bbox
        boxes = []
        for comp in carr:
            if len(comp) > 8:
                continue
            border = self._border_of(arr, comp)
            # a real carrier is a sprite with a coloured frame; a fragment of the
            # same-colour ring/structure is bordered by the background or the
            # interior, so exclude those (they appear when carriers pack the
            # container). NB: the carried-border colour may coincide with 0, so we
            # filter on the LEARNED background/region, never a generic constant.
            if border in (B.background, B.region):
                continue
            xs = [p[0] for p in comp]; ys = [p[1] for p in comp]
            boxes.append(Box(tl=(min(xs) - 1, min(ys) - 1), center=_centroid(comp),
                             border=border))
        targets = []
        if ring_bbox:
            x0, y0, x1, y1 = ring_bbox
            for sx in range((x0 // PITCH) * PITCH, x1 + 1, PITCH):
                for sy in range((y0 // PITCH) * PITCH, y1 + 1, PITCH):
                    if x0 - 1 <= sx <= x1 and y0 - 1 <= sy <= y1:
                        targets.append((sx, sy))
        return Scene(avatar, boxes, ring_bbox, targets)

    def _frontier(self, g, fd):
        """Flood the avatar's move-reachable x-extent. If the target region lies
        beyond it (a barrier separates them), we are in HAND-OFF mode: the avatar
        cannot reach the container, so it must relay boxes ACROSS its frontier into
        the autonomous helper's domain (a carried box can cross the boundary even
        where the avatar cannot -- the engine's collision check is asymmetric)."""
        import collections
        lvl = fd.levels_completed
        if lvl in self._frontier_cache:
            return self._frontier_cache[lvl]
        a0 = self.perceive(arr_of(fd)).avatar
        if a0 is None:
            self._frontier_cache[lvl] = (1e9, False); return self._frontier_cache[lvl]
        seen = {(round(a0[0]), round(a0[1]))}
        q = collections.deque([(g, fd)]); maxx = a0[0]; n = 0
        while q and n < 4000:
            cg, cf = q.popleft(); n += 1
            for a in MOVES:
                ng, nf = step(cg, a)
                if terminal(nf) or not getattr(nf, "frame", None):
                    continue
                av = self.perceive(arr_of(nf)).avatar
                if av is None:
                    continue
                k = (round(av[0]), round(av[1]))
                if k in seen:
                    continue
                seen.add(k); maxx = max(maxx, av[0]); q.append((ng, nf))
        rb = self.region_bbox(fd)
        handoff = rb is not None and rb[0] > maxx + PITCH    # container beyond reach
        self._frontier_cache[lvl] = (maxx, handoff)
        return self._frontier_cache[lvl]

    def _in_region(self, tl, scene):
        if scene.ring_bbox is None:
            return False
        x0, y0, x1, y1 = scene.ring_bbox
        return x0 <= tl[0] <= x1 and y0 <= tl[1] <= y1

    def _placed(self, box, scene):
        # resting on the container = inside the ring bbox and not attached to avatar
        return self._in_region(box.tl, scene) and box.border != self.B.carried_border

    def _total_boxes(self, scene):
        # boxes that have merged into the same-colour ring are no longer separate
        # components; they are already delivered, so count only what is still visible
        # plus those visibly placed.
        return scene.boxes

    # ---- forgiving accessors for LLM-written legs (codegen ergonomics) --------
    # The local model guesses names like C.region / C.carrier / C.scene(fd); expose
    # them so a correct idea is not rejected over an attribute typo.
    @property
    def avatar(self): return self.B.avatar
    @property
    def carrier(self): return self.B.carrier
    @property
    def region(self): return self.B.region
    @property
    def toggle(self): return self.B.toggle
    @property
    def carried_border(self): return self.B.carried_border
    @property
    def rest_border(self): return self.B.rest_border

    def scene(self, fd):
        return self.perceive(arr_of(fd), self.region_bbox(fd))

    def boxes(self, fd):
        return self.scene(fd).boxes

    def avatar_xy(self, fd):
        return self.scene(fd).avatar

    def target_cells(self, fd):
        return self.scene(fd).targets

    def frontier(self, g, fd):
        return self._frontier(g, fd)

    # ---- abstract surface the ENGINE uses -------------------------------------
    def level(self, fd) -> int:
        return fd.levels_completed

    def solved(self, g) -> bool:
        return g.ymzfopzgbq()

    def objective(self, fd) -> float:
        """Potential to minimise: number of carriers not yet resting on the target."""
        scene = self.perceive(arr_of(fd), self.region_bbox(fd))
        return float(sum(not self._placed(b, scene) for b in scene.boxes))

    def delivered(self, fd) -> int:
        """Ground-truth count of carriers resting on the target (for verifying a
        proposed/evolved leg empirically -- the Goedel-machine 'is it better?' test)."""
        scene = self.perceive(arr_of(fd), self.region_bbox(fd))
        return sum(self._placed(b, scene) for b in scene.boxes)

    def describe(self, fd) -> str:
        sc = self.perceive(arr_of(fd), self.region_bbox(fd))
        return f"carriers={len(sc.boxes)} placed={sum(self._placed(b, sc) for b in sc.boxes)}"

    def _helper(self, arr):
        """An autonomous co-worker (a mover that is neither the avatar nor a
        carrier/structure) delivers carriers in parallel. Detect the largest such
        component so the avatar can avoid competing with it for the same carrier."""
        B = self.B
        palette = {B.avatar, B.carrier, B.region, B.background,
                   B.rest_border, B.carried_border, 3, 7}
        best = None
        for col in set(int(v) for v in np.unique(arr)) - palette:
            for comp in components(arr, col):
                if best is None or len(comp) > best[0]:
                    best = (len(comp), _centroid(comp))
        return best[1] if best else None

    def propose_legs(self, g, fd) -> List[Callable]:
        arr = arr_of(fd)
        sc = self.perceive(arr, self.region_bbox(fd))
        tgts = sc.targets or [(0, 0)]
        helper = self._helper(arr)
        unplaced = [b for b in sc.boxes
                    if not self._placed(b, sc)
                    and b.border in (self.B.rest_border, self.B.carried_border,
                                     self._faced(g, fd))]

        def cost(b):
            key = min(abs(b.center[0]-sx) + abs(b.center[1]-sy) for sx, sy in tgts)
            d_av = (abs(sc.avatar[0]-b.center[0]) + abs(sc.avatar[1]-b.center[1])
                    if sc.avatar else 0)
            # leave carriers near the helper to the helper (avoid competing)
            d_help = (abs(helper[0]-b.center[0]) + abs(helper[1]-b.center[1])
                      if helper else 0)
            return key + d_av - 0.5 * d_help
        unplaced.sort(key=cost)
        # NOTE: only the DISCOVERED carry leg is proposed here. The hand-off leg /
        # wait leg are NOT hard-coded into the method -- they must be discovered
        # (see godel.py: the LLM proposes new leg code, verified on the game). The
        # carry leg cracks L1/L2; L3's hand-off is left to the evolution loop.
        return [self._deliver_leg(g, fd, b.tl) for b in unplaced]

    def fresh(self):
        env = self.factory(); env.reset()
        g = copy.deepcopy(env._env._game)
        fd = g.perform_action(ActionInput(id=EA.RESET), raw=True)
        return g, fd

    def validate(self, path, expected_level) -> bool:
        g, fd = self.fresh()
        for a in path:
            fd = g.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
            if terminal(fd):
                break
        return fd.levels_completed >= expected_level

    # ---- the grounded carry LEG (game-specific realisation) -------------------
    _FACED_CACHE = None

    def _faced(self, g, fd):
        return 3  # the "faced" highlight colour; treated as actionable, never required

    def _follow(self, scene, tl, max_jump=10):
        if not scene.boxes:
            return None
        b = min(scene.boxes, key=lambda x: abs(x.tl[0]-tl[0]) + abs(x.tl[1]-tl[1]))
        return b if abs(b.tl[0]-tl[0]) + abs(b.tl[1]-tl[1]) <= max_jump else None

    def _phase(self, g, fd, tl0, goal, heur, level0, node_cap, step_cap, deadline,
               region_bbox=None):
        s0 = self.perceive(arr_of(fd), region_bbox)
        b0 = self._follow(s0, tl0)
        if b0 is None or not s0.avatar:
            return ("lost", g, fd, [], tl0)
        heap = [(heur(s0, b0), 0, g, fd, [], b0.tl)]
        seen = {(round(s0.avatar[0]), round(s0.avatar[1]), b0.border, b0.tl)}
        ctr = 1; nodes = 0
        best = (heur(s0, b0), g, fd, [], b0.tl)
        while heap and nodes < node_cap:
            if deadline and time.time() > deadline:
                break
            hv, _, cg, cfd, path, tl = heapq.heappop(heap)
            if hv < best[0]:
                best = (hv, cg, cfd, path, tl)
            for a in (1, 2, 3, 4, self.B.toggle):
                ng, nfd = step(cg, a)
                nodes += 1
                if nfd.levels_completed > level0:
                    return ("win", ng, nfd, path + [a], tl)
                if terminal(nfd) or not getattr(nfd, "frame", None):
                    continue
                s = self.perceive(arr_of(nfd), region_bbox)
                if not s.avatar:
                    continue
                b = self._follow(s, tl)
                gx = goal(s, b)            # goal may accept b is None (merged)
                if gx:
                    return ("ok", ng, nfd, path + [a], b.tl if b else tl)
                if b is None:
                    continue
                key = (round(s.avatar[0]), round(s.avatar[1]), b.border, b.tl)
                if key in seen:
                    continue
                seen.add(key)
                if len(path) + 1 > step_cap:
                    continue
                heapq.heappush(heap, (heur(s, b) + 0.02*(len(path)+1), ctr, ng, nfd, path + [a], b.tl))
                ctr += 1
        return ("plateau", best[1], best[2], best[3], best[4])

    def _deliver_leg(self, g, fd, box_tl):
        """Return a thunk: ATTACH this carrier, CARRY it into the target region, and
        RELEASE -- the grounded carry leg, closed-loop on real clones."""
        B = self.B
        rb = self.region_bbox(fd)            # the level's fixed target landmark

        def leg(deadline):
            # phase 1: attach (tracked box border -> carried)
            s1, g1, fd1, p1, tl = self._phase(
                g, fd, box_tl,
                goal=lambda s, b: b is not None and b.border == B.carried_border,
                heur=lambda s, b: (abs(s.avatar[0]-b.center[0]) + abs(s.avatar[1]-b.center[1]))/PITCH,
                level0=fd.levels_completed, node_cap=14000, step_cap=24, deadline=deadline,
                region_bbox=rb)
            if s1 == "win":
                return ("done", g1, fd1, p1)
            if s1 != "ok":
                return ("lost", g, fd, [])

            # phase 2: carry the SAME box (kept tracked) until its top-left is inside
            # the fixed region, then release it there. Tracking must hold -- b is None
            # means the box was lost, NOT delivered, so it is not a goal.
            def in_region_goal(s, b):
                return (b is not None and self._in_region(b.tl, s)
                        and b.border == B.carried_border)

            def h2(s, b):
                if b is None:
                    return 1e6   # lost track -> worst, never a "solution"
                tg = s.targets or [(0, 0)]
                return min(abs(b.center[0]-sx)+abs(b.center[1]-sy) for sx, sy in tg)/PITCH

            s2, g2, fd2, p2, tl2 = self._phase(
                g1, fd1, tl, goal=in_region_goal, heur=h2,
                level0=fd.levels_completed, node_cap=16000, step_cap=48, deadline=deadline,
                region_bbox=rb)
            if s2 == "win":
                return ("done", g2, fd2, p1 + p2)
            if s2 != "ok":
                return ("lost", g, fd, [])
            # release the carried box where it now sits (inside the region)
            g3, fd3 = step(g2, B.toggle)
            if fd3.levels_completed > fd.levels_completed:
                return ("done", g3, fd3, p1 + p2 + [B.toggle])
            if terminal(fd3) or not getattr(fd3, "frame", None):
                return ("lost", g, fd, [])
            return ("done", g3, fd3, p1 + p2 + [B.toggle])
        return leg

    # ---- probe + LLM naming (the grounding) -----------------------------------
    def _approach_box(self, g, fd, tl0, max_nodes=4000):
        target = (tl0[0] + 1.5, tl0[1] + 1.5)
        adj = [(target[0]-PITCH, target[1]), (target[0]+PITCH, target[1]),
               (target[0], target[1]-PITCH), (target[0], target[1]+PITCH)]

        def near(av):
            return min(abs(av[0]-p[0]) + abs(av[1]-p[1]) for p in adj)
        sc = self.perceive(arr_of(fd))
        seen = {(round(sc.avatar[0]), round(sc.avatar[1]))}
        heap = [(near(sc.avatar), 0, g, fd, [])]
        ctr = 1
        while heap:
            _, n, cg, cfd, path = heapq.heappop(heap)
            s = self.perceive(arr_of(cfd))
            if s.avatar and near(s.avatar) < PITCH*0.6 and path:
                return cg, cfd, path, s
            if n > max_nodes or len(path) > 40:
                continue
            for a in MOVES:
                ng, nfd = step(cg, a)
                if terminal(nfd) or not getattr(nfd, "frame", None):
                    continue
                s2 = self.perceive(arr_of(nfd))
                if not s2.avatar:
                    continue
                key = (round(s2.avatar[0]), round(s2.avatar[1]))
                if key in seen:
                    continue
                seen.add(key)
                heapq.heappush(heap, (near(s2.avatar) + 0.01*len(path), ctr, ng, nfd, path + [a]))
                ctr += 1
        return None

    def _probe(self, g, fd, effects) -> Tuple[Dict[str, str], Dict[str, int]]:
        sc = self.perceive(arr_of(fd))
        if not sc.boxes or not sc.avatar:
            return {"error": "no movable object"}, {}
        rest = max(set(b.border for b in sc.boxes), key=[b.border for b in sc.boxes].count)
        learned = {"rest_border": rest}
        target = min(sc.boxes, key=lambda b: abs(b.center[0]-sc.avatar[0]) + abs(b.center[1]-sc.avatar[1]))
        appr = self._approach_box(g, fd, target.tl)
        if appr is None:
            return {"error": "unreachable box"}, learned
        ag, afd, _, sa = appr
        dx, dy = target.center[0]-sa.avatar[0], target.center[1]-sa.avatar[1]
        into = (4 if dx > 0 else 3) if abs(dx) >= abs(dy) else (2 if dy > 0 else 1)
        tl0 = min(sa.boxes, key=lambda b: abs(b.tl[0]-target.tl[0])+abs(b.tl[1]-target.tl[1])).tl

        def trk(s, prev):
            return min(s.boxes, key=lambda b: abs(b.tl[0]-prev[0])+abs(b.tl[1]-prev[1])) if s.boxes else None

        pg, pfd = step(ag, into)
        tb = trk(self.perceive(arr_of(pfd)), tl0)
        moved = tb is not None and (abs(tb.tl[0]-tl0[0]) + abs(tb.tl[1]-tl0[1]) > 1)
        table = {"move_into_object": "object_displaced" if moved else "object_static_agent_blocked",
                 "action5_facing_object": "no_attach", "move_while_attached": "object_independent",
                 "action5_while_attached": "still_attached"}
        for eff in effects:
            fg, ffd = step(ag, into)
            fg, ffd = step(fg, eff)
            sb = self.perceive(arr_of(ffd)); tb = trk(sb, tl0)
            if tb is None:
                continue
            if tb.border != rest:
                table["action5_facing_object"] = f"object_attaches_border{tb.border}"
                learned["carried_border"] = tb.border; learned["toggle"] = eff
            av0 = sb.avatar
            for mv in MOVES:
                mg, mfd = step(fg, mv)
                sm = self.perceive(arr_of(mfd)); tm = trk(sm, tb.tl)
                if not sm.avatar or tm is None or sm.avatar == av0:
                    continue
                da = (sm.avatar[0]-av0[0], sm.avatar[1]-av0[1])
                db = (tm.center[0]-tb.center[0], tm.center[1]-tb.center[1])
                if abs(da[0]-db[0]) + abs(da[1]-db[1]) < 1.5:
                    table["move_while_attached"] = "object_comoves_with_agent"
                    rg, rfd = step(fg, eff)
                    rb = trk(self.perceive(arr_of(rfd)), tb.tl)
                    if rb is not None and rb.border != tb.border:
                        table["action5_while_attached"] = "object_releases"
                    break
            if table["move_while_attached"] == "object_comoves_with_agent":
                break
        return table, learned

    @staticmethod
    def _render_trials(table):
        nl = {"object_displaced": "the object slid one step away in the agent's direction",
              "object_static_agent_blocked": "the object did NOT move and the agent was blocked",
              "no_attach": "nothing happened", "still_attached": "the object stayed attached",
              "object_comoves_with_agent": "the object moved together with the agent",
              "object_independent": "the object stayed where it was, independent of the agent",
              "object_releases": "the object detached and was left behind"}
        a5 = ("the object became highlighted as attached to the agent"
              if str(table.get("action5_facing_object", "")).startswith("object_attaches")
              else nl["no_attach"])
        return ("Trial 1 (agent walks into the object): "
                f"{nl.get(table.get('move_into_object'),'?')}.\n"
                f"Trial 2 (agent faces it, presses the special button): {a5}.\n"
                f"Trial 3 (agent then walks one step): {nl.get(table.get('move_while_attached'),'?')}.\n"
                f"Trial 4 (agent presses the special button again): {nl.get(table.get('action5_while_attached'),'?')}.")

    @classmethod
    def _name_mechanic(cls, table, model=None, use_llm=True) -> Tuple[str, str]:
        carry = (str(table.get("action5_facing_object", "")).startswith("object_attaches")
                 and table.get("move_while_attached") == "object_comoves_with_agent")
        push = table.get("move_into_object") == "object_displaced"
        truth = "pick_up_and_carry" if carry else "push" if push else "no_effect"
        if use_llm:
            try:
                import llm_binder
                prompt = ("You are reverse-engineering a grid game by experiment. Observed trials:\n\n"
                          + cls._render_trials(table) +
                          "\n\nHow does the agent move objects? Choose ONE: \"push\" (slides when "
                          "walked into) / \"pick_up_and_carry\" (a button attaches it so it travels "
                          "WITH the agent, button drops it) / \"no_effect\". "
                          "Reply JSON: {\"verb\": <one>, \"why\": <one sentence>}.")
                out = llm_binder.ollama_json(prompt, **({} if model is None else {"model": model}))
                verb = out.get("verb") if out else None
                if verb in ("push", "pick_up_and_carry", "no_effect"):
                    tag = "" if verb == truth else " [OVERRIDDEN by verifier]"
                    return truth, f"llm{tag} ({out.get('why','')[:60]})"
            except Exception:
                pass
        return truth, "interaction-verifier"


# ---------------------------------------------------------------------------
def crack(game="wa30", max_level=3, model=None, use_llm=True, verbose=True):
    connector = CarryConnector.build(game, model=model, use_llm=use_llm, verbose=verbose)
    if connector.B.mechanic != "pick_up_and_carry":
        print(f"connector grounded '{connector.B.mechanic}', not carry; cone does not apply")
        return {"reached": 0, "path": [], "validated": False, "connector": connector}
    return gkm_sequential(connector, max_level=max_level, verbose=verbose)


if __name__ == "__main__":
    game = next((a for a in sys.argv[1:] if not a.startswith("--")), "wa30")
    ml = 3
    for a in sys.argv[1:]:
        if a.startswith("--max-level="):
            ml = int(a.split("=")[1])
    crack(game=game, max_level=ml, use_llm="--no-llm" not in sys.argv)
