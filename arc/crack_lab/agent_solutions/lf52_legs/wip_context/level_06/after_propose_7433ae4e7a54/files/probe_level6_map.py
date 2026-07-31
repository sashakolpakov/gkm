import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import brief, click, pieces
from probe_level6_lower_route import stage_remote


def describe(label, env):
    holes, carriers, movable, pegs, static = pieces(env.frame())
    print(
        label,
        "H", tuple(sorted(holes)),
        "C", tuple(sorted(carriers)),
        "B", tuple(sorted(movable)),
        "P", tuple(sorted(pegs)),
        "S", tuple(sorted(static)),
        "M", brief(env)[3],
    )


def probe(env):
    stage_remote(env)
    for action in (1, 1, 3, 3, 1, 1):
        env.step(action)
    click(env, (30, 28))
    click(env, (18, 28))
    click(env, (18, 28))
    click(env, (18, 40))
    click(env, (30, 46))
    click(env, (18, 46))
    click(env, (18, 40))
    click(env, (18, 52))
    describe("remote", env)
    for source, destination in (
        ((18, 2), (18, 14)),
        ((18, 8), (18, 20)),
        ((18, 14), (18, 26)),
        ((18, 20), (18, 32)),
        ((18, 26), (18, 38)),
        ((18, 32), (18, 44)),
        ((18, 38), (30, 38)),
    ):
        click(env, source)
        click(env, destination)
    env.step(2)
    env.step(2)
    for horizontal in range(3):
        if horizontal:
            env.step(4)
        describe(f"lower-{horizontal}", env)
        for vertical in (1, 2):
            node = env.clone()
            trace = []
            previous = brief(node)
            for _ in range(4):
                node.step(vertical)
                current = brief(node)
                if current == previous:
                    break
                trace.append(current)
                previous = current
            print("vertical", horizontal, vertical, tuple(trace))
    for _ in range(4):
        env.step(1)
    describe("upper-50", env)
    for source, destination in (
        ((18, 44), (18, 56)),
        ((18, 56), (30, 56)),
    ):
        before = brief(env)
        click(env, source)
        click(env, destination)
        print("next", source, destination, before != brief(env), brief(env))
    describe("capture-2", env)
    env.step(2)
    env.step(2)
    describe("middle-bridge", env)
    for source, destination in (
        ((30, 56), (30, 44)),
    ):
        before = brief(env)
        click(env, source)
        click(env, destination)
        print("middle", source, destination, before != brief(env), brief(env))
    for action in (2, 2, 3, 3, 1, 1):
        env.step(action)
    describe("middle-relay", env)
    for source, destination in (
        ((30, 44), (30, 32)),
        ((30, 32), (42, 32)),
    ):
        before = brief(env)
        click(env, source)
        click(env, destination)
        print("middle", source, destination, before != brief(env), brief(env))
    describe("capture-3", env)
    env.step(2)
    env.step(2)
    describe("final-bridge", env)
    base_level = env.levels_completed
    for source, destination in (
        ((42, 38), (42, 26)),
        ((42, 32), (42, 20)),
        ((42, 20), (54, 20)),
    ):
        before = brief(env)
        click(env, source)
        click(env, destination)
        print("final", source, destination, before != brief(env), brief(env))
    print("reward", env.levels_completed - base_level)


A.run_program("lf52", probe)
