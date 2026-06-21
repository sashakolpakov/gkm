"""Game-agnostic colimit/cofibration engine. NO game semantics here — no colours,
no 'box', no 'Sokoban', no push rule. The ONLY dynamics is the real game's
step(); the ONLY reward is levels_completed. Everything game-specific is supplied
by a Connector, which the local LLM provides at runtime.

Categorical backbone:
  * PARTIAL  = a reachable game state (an action prefix's result).
  * LEG      = an action macro driving the state to a proposed sub-goal.
  * COFIBRANT EXTENSION = gluing a leg onto a partial while PRESERVING the
    sub-goals already satisfied (no undoing progress). A partial with no
    admissible extension toward the goal is non-cofibrant = a deadlock.
  * COLIMIT  = the goal: levels_completed rises. A solve is the colimit of a
    chain of cofibrant extensions.
  * SEARCH   = find that chain; every leg is verified by the REAL step;
    deadlocked partials are memoised and backtracked.

A Connector (game-specific, LLM-supplied) implements:
  actions                          : list[int]            available actions
  read(frame)            -> state  : hashable typed state (the connector's view)
  is_terminal(fd)        -> bool   : game over?
  propose_subgoals(state)-> [sg]   : the soft cofibration topology (admissible next legs)
  reached(state, sg)     -> bool   : sub-goal satisfied?
  progress(state, sg)    -> float  : lower = closer (the leg-search gradient)
  preserves(prev,new,satisfied)->bool : is the extension cofibrant? (keeps satisfied sgs)
The engine calls only these plus the env adapters; it never inspects a frame itself.
"""
from __future__ import annotations
import heapq, time
from dataclasses import dataclass


def _frame_key(frame):
    return hash(tuple(tuple(r) for r in frame))


@dataclass
class EngineConfig:
    leg_budget: int = 8000     # real-steps to satisfy one sub-goal (one cofibration)
    leg_depth: int = 50        # max actions per leg
    max_partials: int = 200    # cofibration-chain nodes explored per level
    time_cap: float = 600.0
    maxheap: int = 2500


def drive_to_subgoal(step, frame_of, level_of, conn, game, fd0, subgoal, cfg, level0):
    """Find a leg from `game` that satisfies `subgoal`, verified by the real step.
    Best-first on conn.progress (game-agnostic gradient).
    Returns ('win', game, fd) | ('done', game, fd, state) | None."""
    f0 = frame_of(fd0)
    seen = {_frame_key(f0)}                      # dedup on the REAL frame, not the coarse view
    heap = [(conn.progress(conn.read(f0), subgoal), 0, 0, game, fd0)]
    ctr = 1; nodes = 0
    while heap and nodes < cfg.leg_budget:
        _, _, d, g, _ = heapq.heappop(heap)
        if d >= cfg.leg_depth:
            continue
        for a in conn.actions:
            gc, fd = step(g, a); nodes += 1
            if level_of(fd) > level0:
                return ("win", gc, fd)
            if conn.is_terminal(fd):
                continue
            frame = frame_of(fd)
            st = conn.read(frame)
            if conn.reached(st, subgoal):
                return ("done", gc, fd, st)
            k = _frame_key(frame)
            if k in seen:
                continue
            seen.add(k)
            heapq.heappush(heap, (conn.progress(st, subgoal) + 0.02 * (d + 1), ctr, d + 1, gc, fd)); ctr += 1
        if len(heap) > cfg.maxheap:
            heap = heapq.nsmallest(cfg.maxheap, heap); heapq.heapify(heap)
    return None


def solve_level(step, frame_of, level_of, conn, game, fd, level0, cfg=EngineConfig()):
    """DFS a cofibration chain to the colimit of ONE level (level_of rises above
    level0), backtracking over the LLM-proposed sub-goal extensions. Returns
    ('win', game, fd) | ('stuck',). The caller rebuilds the connector per level
    (its footprint/roles are read from each level's start frame)."""
    t0 = time.time()
    visited = set(); explored = [0]

    def dfs(game, fd, satisfied):
        if time.time() - t0 > cfg.time_cap or explored[0] >= cfg.max_partials:
            return None
        state = conn.read(frame_of(fd))
        key = repr(state)
        if key in visited:                                          # cycle-safe: each partial once
            return None
        visited.add(key)
        explored[0] += 1
        for sg in conn.propose_subgoals(state):                    # cofibration topology (LLM):
            # legs may DELIVER or DISPLACE (un-deliver to clear a path). No
            # monotone pressure — the cofibration ALLOWS the non-monotone
            # "complicate then unknot" detour; cycles are cut by `visited`.
            r = drive_to_subgoal(step, frame_of, level_of, conn, game, fd, sg, cfg, level0)
            if r is None:
                continue
            if r[0] == "win":
                return ("win", r[1], r[2])
            _, ng, nfd, nstate = r
            res = dfs(ng, nfd, satisfied + [sg])
            if res:
                return res
        return None

    res = dfs(game, fd, [])
    return res if res else ("stuck",)
