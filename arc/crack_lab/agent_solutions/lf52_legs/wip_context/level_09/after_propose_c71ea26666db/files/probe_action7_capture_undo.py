"""Verify whether action 7 fully restores a peg capture."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
)


def key(node):
    return arr(node.frame())[1:, :].tobytes()


def pegs(node):
    return tuple(sorted(
        blob.top_left
        for blob in connected_components(node.frame(), colors=(14,))
        if blob.size == (4, 4)
    ))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)
    for source, destination in OPENING:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    pre_key = key(env)
    pre_pegs = pegs(env)
    safe_step(env, (6, 31, 49))
    selected_key = key(env)
    selected_pegs = pegs(env)
    safe_step(env, (6, 31, 37))
    captured_pegs = pegs(env)
    safe_step(env, 7)
    once = (key(env) == selected_key, pegs(env))
    safe_step(env, 7)
    twice = (key(env) == pre_key, pegs(env))
    print("capture_undo", pre_pegs, selected_pegs, captured_pegs,
          once, twice, flush=True)


arena.run_program("lf52", probe)
