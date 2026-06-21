"""A Connector for cone_engine, parametrised by LLM-supplied roles. The schema
(deliver carrier-pieces into a region) is game-agnostic; the ROLES (which colour
is carrier / region / avatar) are supplied by the LLM reading the game off. The
engine explores the delivery ORDER (the soft cofibration topology) and never
undoes a delivery (preserves => cofibrant)."""
from __future__ import annotations
from lab import arc
import priors


class DeliveryConnector:
    def __init__(self, carrier, region, avatar, footprint, actions):
        self.carrier = carrier            # colour that fills a slot on a genuine delivery
        self.region = region              # container interior colour
        self.avatar = avatar
        self.actions = list(actions)
        self.footprint = frozenset(footprint)   # region cells at this level's start
        if footprint:
            self.rc = (sum(x for x, y in footprint) / len(footprint),
                       sum(y for x, y in footprint) / len(footprint))
        else:
            self.rc = (0.0, 0.0)

    def is_terminal(self, fd):
        return str(fd.state).endswith("GAME_OVER")

    def read(self, frame):
        av = priors.avatar_centroid(frame, self.avatar)
        delivered = sum(1 for (x, y) in self.footprint if frame[y][x] == self.carrier)
        empty = sum(1 for (x, y) in self.footprint if frame[y][x] == self.region)
        und = tuple(sorted(priors._cen(c) for c in arc.connected_components(frame, self.carrier)
                           if priors._cen(c) not in self.footprint))   # carrier pieces outside the region
        return (av, delivered, empty, und)

    def propose_subgoals(self, state):
        # cofibration topology: one "deliver next" leg per undelivered piece (the
        # engine backtracks over WHICH piece next = the soft delivery order). When
        # all pieces are delivered, propose a FINISH leg that BFS-searches for the
        # level-up trigger (e.g. a final interact) — the win is often a step AFTER
        # the last delivery, not the delivery itself.
        av, delivered, empty, und = state
        if und:
            return [("deliver", b, delivered) for b in und]
        return [("finish",)]

    def reached(self, state, sg):
        if sg[0] == "finish":
            return False                   # only the engine's level-up check ends a finish leg
        return state[1] > sg[2]            # a NEW slot locked (delivered count rose)

    def progress(self, state, sg):
        av, delivered, empty, und = state
        if sg[0] == "finish":
            return 0.0                     # uniform -> BFS toward the level-up trigger
        if not und or av is None:
            return -10.0 * delivered
        # per-box PUSH heuristic so different sub-goals deliver DIFFERENT boxes
        # (the delivery-order diversity the cofibration backtracking needs): track
        # the targeted box, reward it approaching the region, and reward the
        # avatar getting BEHIND it (opposite the region) so a push moves it in.
        b = sg[1]
        bn = min(und, key=lambda u: abs(u[0] - b[0]) + abs(u[1] - b[1]))
        rx, ry = self.rc
        sx = 1 if rx > bn[0] else (-1 if rx < bn[0] else 0)
        sy = 1 if ry > bn[1] else (-1 if ry < bn[1] else 0)
        push = (bn[0] - sx * 8, bn[1] - sy * 8)
        d_box = abs(bn[0] - rx) + abs(bn[1] - ry)
        d_push = abs(av[0] - push[0]) + abs(av[1] - push[1])
        return -10.0 * delivered + d_box + 0.3 * d_push

    def preserves(self, prev, new, satisfied):
        return new[1] >= prev[1]           # never undo a delivery -> cofibrant extension
