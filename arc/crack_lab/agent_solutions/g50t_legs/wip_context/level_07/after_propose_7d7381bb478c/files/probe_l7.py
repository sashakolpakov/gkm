import json
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import (
    action_deltas,
    arr,
    bounded_bfs,
    color_counts,
    connected_components,
    replay,
)
from legs import (
    _avatar_pos,
    _use_macro,
    fast_reach,
    _special_frontier,
    plan_cooperative_gate_chain,
    plan_frontier_unlock,
)


def movers(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, colors=(8, 9, 11, 14, 15), min_area=4)
    ]


def tile_map(frame):
    chars = {0: ".", 1: "1", 5: "#", 8: "G", 9: "A",
             11: "S", 14: "B", 15: "X"}
    return "/".join(
        "".join(chars.get(int(frame[r + 2][c + 2]), "?")
                for c in range(2, 62, 6))
        for r in range(2, 62, 6)
    )


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)

    print("level", int(env.levels_completed), "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("objects", movers(env.frame()))
    print("map", tile_map(env.frame()))
    print("deltas", {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env).items()
    })
    for action in env.actions:
        child = replay(env, [action])
        print("after", action, movers(child.frame()))
    reward_path, reach = fast_reach(env)
    frontier = _special_frontier(reach, env.frame())
    print("reach", len(reach), "win", reward_path,
          "frontier", [(p, len(path)) for p, path in frontier])
    outcomes = defaultdict(list)
    for first_pos, first_walk in sorted(reach.items()):
        _, first_macro, first_child, first_reward, first_reach = _use_macro(
            env, first_pos, first_walk, fast_reach)
        second_choices = _special_frontier(first_reach, first_child.frame())
        if first_reward is not None or not second_choices:
            outcome = ("first", first_reward is not None, len(first_reach))
        else:
            second_pos, second_walk = second_choices[-1]
            _, second_macro, second_child, second_reward, second_reach = _use_macro(
                first_child, second_pos, second_walk, fast_reach)
            f = arr(second_child.frame())
            signature = np.where(np.isin(f[7:62], (9, 11, 14, 15)),
                                 f[7:62], 0).tobytes()
            outcome = (
                signature,
                second_reward is not None,
                len(second_reach),
                tuple(second_macro),
            )
        outcomes[outcome].append((first_pos, tuple(first_macro)))
    print("two_stage_groups", len(outcomes))
    for index, (outcome, firsts) in enumerate(outcomes.items()):
        print("two_stage", index, "summary", outcome[1:],
              "firsts", firsts)
    special_pos, special_walk = frontier[-1]
    _, open_macro, opened, _, _ = _use_macro(
        env, special_pos, special_walk, fast_reach)
    for label, start in (("closed", env.clone()), ("opened", opened)):
        node = start
        last = None
        for tick in range(1, 31):
            action = 2 if tick % 2 else 1
            node.step(action)
            state = (
                int(node.levels_completed),
                tuple((b.color, b.bbox, b.area) for b in connected_components(
                    node.frame(), colors=(9, 11, 14), min_area=4)),
            )
            if state != last:
                print("pump", label, tick, state)
                last = state
            if int(node.levels_completed) > 6:
                break
    node = env.clone()
    strategy = open_macro + [2, 1, 2, 5]
    prefix = []
    for cycle in range(1, 13):
        for action in strategy:
            node.step(action)
            prefix.append(action)
            if int(node.levels_completed) > 6:
                break
        specials = [(b.bbox, b.area) for b in connected_components(
            node.frame(), colors=(11,), min_area=1)]
        print("progress_cycle", cycle, "level", int(node.levels_completed),
              "special", specials, "avatar", _avatar_pos(node.frame()),
              "objects", movers(node.frame()))
        if int(node.levels_completed) > 6:
            print("progress_path", prefix)
            break
    for padding in range(10):
        timed = [2, 1] * padding + [2, 2, 3, 5]
        node = replay(env, timed)
        rp, timed_reach = fast_reach(node)
        print("timed_commit", padding, "length", len(timed),
              "level", int(node.levels_completed), "reach", len(timed_reach),
              "win", rp, "objects", movers(node.frame()))
    for pos, path in frontier:
        _, macro, child, child_reward, child_reach = _use_macro(
            env, pos, path, fast_reach)
        print("commit", pos, macro, "reach", len(child_reach),
              "avatar", _avatar_pos(child.frame()),
              "frontier", [(p, len(pth)) for p, pth
                           in _special_frontier(child_reach, child.frame())],
              "objects", movers(child.frame()),
              "map", tile_map(child.frame()))
    node = env.clone()
    prefix = []
    for cycle in range(1, 13):
        rp, cycle_reach = fast_reach(node)
        choices = _special_frontier(cycle_reach, node.frame())
        if rp is not None or not choices:
            print("cycle_stop", cycle, "win", rp, "choices", len(choices))
            break
        pos, walk = choices[-1]
        _, macro, node, child_reward, next_reach = _use_macro(
            node, pos, walk, fast_reach)
        prefix += macro
        specials = [(b.bbox, b.area) for b in connected_components(
            node.frame(), colors=(11,), min_area=1)]
        print("cycle", cycle, "macro", macro, "special", specials,
              "reach", len(next_reach), "reward", int(node.levels_completed),
              "direct_win", child_reward)
    for name, planner, kwargs in (
        ("frontier", plan_frontier_unlock, {"max_stages": 12, "max_stalls": 4}),
        ("coop", plan_cooperative_gate_chain, {"max_stages": 10, "max_expand": 300}),
    ):
        started = time.time()
        plan = planner(env.clone(), **kwargs)
        child = replay(env, plan or [])
        print("plan", name, "seconds", round(time.time() - started, 3),
              "length", None if plan is None else len(plan),
              "reward", int(child.levels_completed),
              "path", plan)
    started = time.time()
    bfs_plan = bounded_bfs(
        env,
        lambda node, path: int(node.levels_completed) > 6,
        key_fn=lambda node: (
            int(node.levels_completed),
            bytes(
                int(v) if int(v) in (9, 11, 14, 15) else 0
                for v in arr(node.frame())[7:62].flat
            ),
        ),
        max_states=50000,
        max_depth=120,
    )
    print("action_bfs", round(time.time() - started, 3), bfs_plan)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
