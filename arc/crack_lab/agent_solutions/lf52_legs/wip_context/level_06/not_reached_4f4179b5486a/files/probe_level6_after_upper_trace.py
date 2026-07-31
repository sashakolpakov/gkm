import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_level6_lower import brief, stage_upper


def trace_keys(root, prefix, action, count=14):
    node = root.clone()
    for item in prefix:
        node.step(item)
    out = []
    last = None
    for index in range(count):
        current = brief(node)
        if index == 0 or current != last:
            out.append((index, current))
        last = current
        node.step(action)
    print("trace", prefix, action, out)


def probe(env):
    stage_upper(env)
    print("exit", brief(env))
    for prefix in ((), (2,), (2, 2)):
        for action in (1, 2, 3, 4):
            trace_keys(env, prefix, action)
    for horizontal_action, maximum in ((3, 6), (4, 3)):
        for count in range(maximum):
            prefix = (2, 2) + (horizontal_action,) * count
            trace_keys(env, prefix, 2, count=7)
    aligned = env.clone()
    for action in (2, 2, 3, 3, 1, 1):
        aligned.step(action)
    print("candidate reload", brief(aligned))
    offset = env.clone()
    for action in (2, 2, 4, 4, 1, 1):
        offset.step(action)
    print("offset reload", brief(offset))
    for action in (3, 4):
        trace_keys(offset, (), action, count=10)
    wrapped = env.clone()
    for action in (2, 2) + (3,) * 9 + (1, 1):
        wrapped.step(action)
    print("wrapped reload", brief(wrapped))
    trace_keys(wrapped, (), 4, count=14)
    boundary = env.clone()
    for action in (
        (2, 2) + (3,) * 9 + (4,) * 6 + (1, 1) + (4, 4)
    ):
        boundary.step(action)
    print("boundary reload", brief(boundary))
    peg_only = env.clone()
    for action in (
        (2, 2) + (4, 4) + (1, 1) + (3,) * 5 + (1, 1)
    ):
        peg_only.step(action)
    print("peg only", brief(peg_only))
    for action in (3, 4, 2):
        trace_keys(peg_only, (), action, count=12)
    clear_bridge = env.clone()
    for action in (2, 2) + (3,) * 5 + (1, 1):
        clear_bridge.step(action)
    print("clear bridge", brief(clear_bridge))
    for action in (3, 4, 2):
        trace_keys(clear_bridge, (), action, count=12)


A.run_program("lf52", probe)
