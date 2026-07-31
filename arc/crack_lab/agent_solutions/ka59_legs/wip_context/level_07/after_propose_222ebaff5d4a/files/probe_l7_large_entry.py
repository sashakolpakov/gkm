import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


PREFIX = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
    + [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * 6
    + [(6, 34, 47)] + [1] * 2 + [4] * 4 + [1] * 6
    + [3] * 4 + [1] * 4 + [4]
    + [(6, 53, 10)] + [4] * 2
)


def state(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    apply(node, PREFIX)
    print("ready", state(node))
    previous = state(node)
    for index in range(1, 19):
        node.step(3)
        current = state(node)
        old_movable = tuple(x for x in previous if x[0] in (11, 14))
        new_movable = tuple(x for x in current if x[0] in (11, 14))
        if new_movable != old_movable:
            print("left", index, new_movable)
        previous = current

    relayed = env.clone()
    apply(relayed, PREFIX + [3] * 3)
    for name, path in (
        ("retrigger", [4, 3] * 6),
        (
            "advance_v_retrigger",
            [(6, 28, 11), 3, (6, 53, 10)] + [4, 3] * 6,
        ),
        (
            "advance_v_twice",
            [(6, 28, 11)] + [3] * 2
            + [(6, 53, 10)] + [4, 3] * 6,
        ),
        ("push_agent_right", [(6, 28, 11)] + [4] * 8),
        (
            "push_agent_then_relay",
            [(6, 28, 11)] + [4] * 2
            + [(6, 53, 10)] + [3] * 8,
        ),
        (
            "ratchet_once",
            [(6, 28, 11), 4, (6, 53, 10)] + [3] * 8,
        ),
        ("shift_v_down", [(6, 28, 11)] + [2] * 8),
        ("shift_v_up", [(6, 28, 11)] + [1] * 8),
        (
            "circle_agent_left",
            [(6, 28, 11), 2] + [4] * 3 + [1] + [3] * 8,
        ),
    ):
        child = relayed.clone()
        previous = state(child)
        print(name, "base", previous)
        for index, action in enumerate(path, 1):
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            current = state(child)
            old_movable = tuple(x for x in previous if x[0] in (11, 14))
            new_movable = tuple(x for x in current if x[0] in (11, 14))
            if new_movable != old_movable:
                print(name, index, action, new_movable)
            previous = current

    inspect = relayed.clone()
    for action in ((6, 28, 11), 4, (6, 53, 10), 3):
        inspect.step(*action) if isinstance(action, tuple) else inspect.step(action)
        print("ratchet_state", action, state(inspect))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
