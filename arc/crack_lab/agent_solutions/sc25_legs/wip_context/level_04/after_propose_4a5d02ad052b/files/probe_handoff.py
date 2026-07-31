import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
import players


PLUS = [(6, 30, 50), (6, 25, 55), (6, 35, 55), (6, 30, 60)]
VERTICAL = [(6, 30, 50), (6, 30, 55), (6, 30, 60)]


def reach_level_4(env):
    for k in (1, 2, 3):
        getattr(players, f"play_level_{k}")(env)


def agents(node):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(6, 9, 10, 13, 14), min_area=1
        )
        if b.color in (9, 10) or b.bbox[1] < 60
    )


def apply(node, actions):
    for action in actions:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    reach_level_4(env)
    common = [2, 2, 4, 4] + PLUS + [2] + [3] * 16 + VERTICAL
    opened = apply(env.clone(), common)
    contexts = {
        "gate_edge": [],
        "inside_left": [3, 3],
    }
    clicks = {
        "plain": (6,),
        "main": (6, 8, 29),
        "left_dock": (6, 8, 29),
        "remote": (6, 53, 37),
        "right_dock": (6, 48, 37),
        "panel_center": (6, 30, 55),
    }
    for context, moves in contexts.items():
        base = apply(opened.clone(), moves)
        print("BASE", context, base.levels_completed, agents(base))
        for click, action in clicks.items():
            node = base.clone()
            before = node.frame()
            node.step(*action)
            click_delta = P.frame_delta(before, node.frame())
            after_click = agents(node)
            outcomes = []
            for direction in (1, 2, 3, 4):
                test = node.clone()
                before_move = test.frame()
                test.step(direction)
                outcomes.append(
                    (
                        direction,
                        P.frame_delta(before_move, test.frame())["bbox"],
                        agents(test),
                        test.levels_completed,
                    )
                )
            print(
                "TRY", context, click, click_delta["count"],
                click_delta["bbox"], after_click, outcomes
            )
    for name, route in {
        "right_gate": [4] * 16 + [2] * 3 + [4] * 6,
        "right_gate_low": [2] * 3 + [4] * 22,
        "timer": [1] * 40,
    }.items():
        node = opened.clone()
        print("ROUTE_START", name, agents(node))
        for index, action in enumerate(route, 1):
            before = agents(node)
            node.step(action)
            after = agents(node)
            if before != after or node.levels_completed > env.levels_completed:
                print(
                    "ROUTE_STEP", name, index, action,
                    node.levels_completed, node.terminal(), after
                )
            if node.terminal() or node.levels_completed > env.levels_completed:
                break
    for name, sequence in {
        "dock_then_cross_no_marker": (
            PLUS + [2] * 5 + [3] * 12 + VERTICAL + [3, 3] + PLUS
        ),
        "dock_then_cross_marker": (
            [2, 2, 4, 4] + PLUS + [2] + [3] * 16
            + VERTICAL + [3, 3] + PLUS
        ),
        "edge_then_cross_no_marker": (
            PLUS + [2] * 5 + [3] * 12 + VERTICAL + PLUS
        ),
    }.items():
        node = env.clone()
        print("COMMAND_ROUTE_START", name)
        for index, action in enumerate(sequence, 1):
            before = agents(node)
            node.step(*action) if isinstance(action, tuple) else node.step(action)
            after = agents(node)
            if index >= len(sequence) - 5 or node.levels_completed > env.levels_completed:
                print(
                    "COMMAND_ROUTE_STEP", name, index, action,
                    node.levels_completed, node.terminal(), before, after
                )
            if node.terminal() or node.levels_completed > env.levels_completed:
                break


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)
