"""Summarize source/target peg coverage for relay routes."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

from players import play_level_1
from probe_level2_collect import covered, peg_centers
from probe_level2_relay import A_ROUTES, B_ROUTES, C_ROUTE


GROUP = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11),
         "D": (9, 10)}


def select_color(node, color):
    ys, xs = np.where(np.asarray(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def summarize(node, pegs, source, target, route):
    seen_source = set()
    seen_target = set()
    events = []
    for index, action in enumerate(route, 1):
        node.step(action)
        new_source = {
            i for i in GROUP[source] if covered(node, pegs[i])
        } - seen_source
        new_target = {
            i for i in GROUP[target] if covered(node, pegs[i])
        } - seen_target
        seen_source |= new_source
        seen_target |= new_target
        if new_source or new_target or action == 5:
            events.append((index, action, tuple(sorted(new_source)),
                           tuple(sorted(new_target))))
    return sorted(seen_source), sorted(seen_target), events


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for pair, route in A_ROUTES.items():
        print("A", pair, summarize(env.clone(), pegs, "A", "B", route))
    node = env.clone()
    select_color(node, 14)
    for pair, route in B_ROUTES.items():
        print("B", pair, summarize(node.clone(), pegs, "B", "C", route))
    node = env.clone()
    select_color(node, 11)
    print("C", summarize(node, pegs, "C", "D", C_ROUTE))


arena.run_program("cn04", probe)
