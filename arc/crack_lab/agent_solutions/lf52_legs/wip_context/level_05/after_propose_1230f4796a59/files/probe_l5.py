import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import action_deltas, connected_components, frame_delta


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)
PRE_CAPTURE = PREFIX[:-2]
AFTER_SECOND = PREFIX[:9]


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)["final_path"]
    previous = env.levels_completed
    for index, action in enumerate(checkpoint):
        act(env, tuple(action) if isinstance(action, list) else action)
        if env.levels_completed != previous:
            print("TRANSITION", index + 1, env.levels_completed)
            previous = env.levels_completed
    for action in AFTER_SECOND:
        act(env, action)
    print("AFTER_SECOND", _bridge_carrier_state(env.frame()))
    print("AFTER_SECOND_MOVES", _bridge_carrier_moves(env.frame()))
    for path in (
        (4, 4, 4, 4, 4, 4, 2, 2, 2),
        (4, 4, 4, 4, 4, 4, 4, 2, 2, 2),
        (4, 4, 4, 4, 4, 2, 2, 2),
        (3, 3, 3, 2, 2, 2),
        (4, 4, 4, 2, 2, 2, 4, 4, 4),
        (4, 4, 4, 2, 2, 2, 3, 3, 3),
    ):
        child = env.clone()
        for action in path:
            child.step(action)
        print("SECOND_TEST", path, _bridge_carrier_state(child.frame()), _bridge_carrier_moves(child.frame()))
    for action in PRE_CAPTURE[len(AFTER_SECOND):]:
        act(env, action)
    print("PRE_CAPTURE", _bridge_carrier_state(env.frame()))
    print("PRE_CAPTURE_MOVES", _bridge_carrier_moves(env.frame()))
    for action in PREFIX[-2:]:
        act(env, action)
    print("POST", _bridge_carrier_state(env.frame()))
    print(
        "SPECIAL",
        [(b.color, b.bbox, b.area) for b in connected_components(
            env.frame(), colors=(7, 11, 12, 15)
        )],
    )
    print("MOVES", _bridge_carrier_moves(env.frame()))
    print(
        "DELTAS",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    tests = (
        (3,), (4,), (1,), (2,),
        (3, 3), (4, 4), (1, 1), (2, 2),
        (3, 3, 3), (4, 4, 4), (1, 1, 1), (2, 2, 2),
        (3, 3, 2, 2, 2), (4, 4, 2, 2, 2),
        (3, 3, 3, 2, 2, 2), (4, 4, 4, 2, 2, 2),
        (3, 3, 1, 1, 1), (4, 4, 1, 1, 1),
    )
    for path in tests:
        node = env.clone()
        for action in path:
            node.step(action)
        state = _bridge_carrier_state(node.frame())
        print(
            "TEST", path, "level", node.levels_completed,
            "peg", tuple(state[1]), "carrier", tuple(state[2]),
            "moves", _bridge_carrier_moves(node.frame()),
        )
    aligned_path = (3, 3, 2, 2, 2)
    aligned = env.clone()
    for action in aligned_path:
        aligned.step(action)
    print("ALIGNED", _bridge_carrier_state(aligned.frame()))
    for click in ((31, 43), (31, 25), (19, 25), (25, 25), (37, 25)):
        child = aligned.clone()
        before = child.frame()
        child.step(6, *click)
        print(
            "CLICK", click, frame_delta(before, child.frame())["count"],
            child.levels_completed, _bridge_carrier_state(child.frame()),
        )


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)
