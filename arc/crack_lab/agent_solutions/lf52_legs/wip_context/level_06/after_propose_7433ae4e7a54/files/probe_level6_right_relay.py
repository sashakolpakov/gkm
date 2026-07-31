import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import brief, click
from probe_level6_lower_route import stage_remote


def perform(env, source, destination):
    before = brief(env)
    click(env, source)
    click(env, destination)
    after = brief(env)
    print("macro", source, destination, after != before, after)


def ascii_map(frame):
    symbols = {
        0: "0", 1: ".", 5: "#", 8: "B", 9: "+", 10: " ",
        11: "c", 12: "C", 14: "P", 15: "=",
    }
    data = P.arr(frame)
    return "\n".join(
        "".join(
            symbols.get(int(data[row, col]), "?")
            for col in range(0, 64, 2)
        )
        for row in range(0, 64, 2)
    )


def probe(env):
    stage_remote(env)
    for action in (1, 1, 3, 3, 1, 1):
        env.step(action)
    for source, destination in (
        ((30, 28), (18, 28)),
        ((18, 28), (18, 40)),
        ((30, 46), (18, 46)),
        ((18, 40), (18, 52)),
        ((18, 46), (18, 58)),
    ):
        perform(env, source, destination)
    print("right edge", brief(env))
    print("right map\n" + ascii_map(env.frame()))
    right_bridge = env.clone()
    for source, destination in (
        ((18, 2), (18, 14)),
        ((18, 8), (18, 20)),
        ((18, 14), (18, 26)),
        ((18, 20), (18, 32)),
        ((18, 26), (18, 38)),
        ((18, 38), (30, 38)),
    ):
        perform(right_bridge, source, destination)
    print("right bridge loaded", brief(right_bridge))
    right_bridge.step(2)
    right_bridge.step(2)
    print("right bridge descended", brief(right_bridge))
    for action in (1, 2, 3, 4):
        node = right_bridge.clone()
        right_trace = []
        previous = None
        for count in range(12):
            current = brief(node)
            if count == 0 or current != previous:
                right_trace.append((count, current))
            previous = current
            node.step(action)
        print("right bridge key", action, right_trace)
    for horizontal in range(3):
        node = right_bridge.clone()
        for _ in range(horizontal):
            node.step(4)
        shaft_trace = []
        previous = None
        for count in range(7):
            current = brief(node)
            if count == 0 or current != previous:
                shaft_trace.append((count, current))
            previous = current
            node.step(2)
        print("right shaft", horizontal, shaft_trace)
    loaded = env.clone()
    perform(loaded, (18, 2), (30, 2))
    print("wrapped loaded", brief(loaded))
    for action in (1, 2, 3, 4):
        node = loaded.clone()
        loaded_trace = []
        previous = None
        for count in range(10):
            current = brief(node)
            if count == 0 or current != previous:
                loaded_trace.append((count, current))
            previous = current
            node.step(action)
        print("loaded key", action, loaded_trace)
    descended = loaded.clone()
    descended.step(2)
    descended.step(2)
    print("wrapped descended", brief(descended))
    for action in (3, 4):
        node = descended.clone()
        descended_trace = []
        previous = None
        for count in range(14):
            current = brief(node)
            if count == 0 or current != previous:
                descended_trace.append((count, current))
            previous = current
            node.step(action)
        print("descended key", action, descended_trace)
    for action in (1, 2, 3, 4):
        node = env.clone()
        trace = []
        previous = None
        for count in range(8):
            current = brief(node)
            if count == 0 or current != previous:
                trace.append((count, current))
            previous = current
            node.step(action)
        print("key", action, trace)


A.run_program("lf52", probe)
