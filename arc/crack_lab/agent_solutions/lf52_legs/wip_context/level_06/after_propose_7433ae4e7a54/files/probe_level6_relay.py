import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_level6_lower import brief, click, pieces
from probe_level6_lower_route import stage_remote


def perform(env, source, destination):
    before = brief(env)
    click(env, source)
    click(env, destination)
    after = brief(env)
    print("macro", source, destination, "changed", after != before, after)


def probe(env):
    stage_remote(env)
    for action in (1, 1, 3, 3, 3, 3, 2, 2):
        env.step(action)
    for _ in range(9):
        env.step(3)
    print("aligned", brief(env))
    for source, destination in (
        ((42, 34), (42, 22)),
        ((42, 28), (42, 16)),
        ((42, 22), (42, 10)),
        ((42, 16), (42, 4)),
    ):
        perform(env, source, destination)
    for source, destination in (
        ((42, 4), (42, 16)),
        ((42, 10), (42, 22)),
        ((42, 16), (42, 28)),
        ((42, 22), (42, 34)),
    ):
        perform(env, source, destination)
    print("relayed", brief(env))
    stop = env.clone()
    for _ in range(7):
        stop.step(4)
    print("upper stop", brief(stop))
    for action in (1, 2, 3, 4):
        node = stop.clone()
        stop_trace = []
        for count in range(8):
            current = brief(node)
            if count == 0 or current != stop_trace[-1][1]:
                stop_trace.append((count, current))
            node.step(action)
        print("stop key", action, stop_trace)
    lifted = stop.clone()
    lifted.step(1)
    lifted.step(1)
    print("peg lifted", brief(lifted))
    for action in (3, 4, 2):
        node = lifted.clone()
        lifted_trace = []
        for count in range(9):
            current = brief(node)
            if count == 0 or current != lifted_trace[-1][1]:
                lifted_trace.append((count, current))
            node.step(action)
        print("lifted key", action, lifted_trace)
    for action in (1, 2, 3, 4):
        node = env.clone()
        trace = []
        for count in range(30):
            current = brief(node)
            if count == 0 or not trace or current != trace[-1][1]:
                trace.append((count, current))
            node.step(action)
        print("key", action, trace)


A.run_program("lf52", probe)
