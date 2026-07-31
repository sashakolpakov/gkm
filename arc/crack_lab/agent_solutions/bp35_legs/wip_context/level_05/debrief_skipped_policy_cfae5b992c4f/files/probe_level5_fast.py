import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
TO_FINAL = (
    ((4,),) * 4
    + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57))
    + ((3,), (3,), (6, 27, 33), (3,), (6, 21, 33), (3,), (3,))
    + ((4,),) * 4
    + ((6, 39, 33),)
)
CORRIDOR = (
    (4,),
    (4,),
    (4,),
    (6, 57, 33),
    (6, 57, 27),
    (6, 57, 33),
    (3,),
    (3,),
    (3,),
    (3,),
    (3,),
    (3,),
)


def avatar(node):
    frame = node.frame()
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def specials(node):
    frame = node.frame()
    return tuple(
        (i, j, int(frame[y][x]))
        for i, y in enumerate(ROWS)
        for j, x in enumerate(COLS)
        if int(frame[y][x]) not in (3, 5, 9, 10, 11)
    )


def final_actions(node):
    frame = node.frame()
    position = avatar(node)
    out = [(3,), (4,)]
    if position is None:
        return out
    ai, aj = position
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            color = int(frame[y][x])
            if color == 14 and abs(j - aj) <= 1 and abs(i - ai) <= 1:
                out.append((6, x, y))
            elif color == 15 and abs(j - aj) <= 2 and 0 < ai - i <= 2:
                out.append((6, x, y))
    return out


def state_key(node):
    frame = perception.arr(node.frame())
    return frame[:63].tobytes(), legs.moves_used(frame) % 2


def probe(env):
    with open("checkpoint.json") as source:
        prefix = json.load(source)["final_path"]
    for action in prefix:
        env.step(*action) if isinstance(action, list) else env.step(action)

    objects = perception.object_candidates(env.frame(), cell=6, min_area=4)
    summary = tuple(
        (item["color"], item["area"], item["bbox"]) for item in objects[:12]
    )
    print("L5", env.levels_completed, avatar(env), summary)
    print("KEY_DELTAS", perception.action_deltas(env, actions=(3, 4)))
    try:
        clone = env.clone()
        clone.step(7)
        print("ACTION_7", perception.frame_delta(env.frame(), clone.frame()))
    except Exception as error:
        print("ACTION_7", type(error).__name__, str(error))
    base = env.frame()
    for label, action in (
        ("CLICK_8", (6, 51, 39)),
        ("CLICK_14", (6, 45, 15)),
    ):
        clone = env.clone()
        clone.step(*action)
        print(label, perception.frame_delta(base, clone.frame()))

    node = env.clone()
    for label, path in (("ASCENT", TO_FINAL),):
        for index, action in enumerate(path, 1):
            before = node.frame()
            node.step(*action)
            shift = legs.band_shift(before, node.frame())
            if shift or action[0] == 6 or node.terminal():
                print(
                    label,
                    index,
                    action,
                    "gain",
                    shift,
                    "level",
                    node.levels_completed,
                    "dead",
                    node.terminal(),
                    "avatar",
                    avatar(node),
                    "specials",
                    specials(node),
                )
            if node.terminal() or node.levels_completed > 4:
                break
    final_root = node.clone()
    for index, action in enumerate(CORRIDOR, 1):
        before = node.frame()
        node.step(*action)
        shift = legs.band_shift(before, node.frame())
        if shift or action[0] == 6 or node.terminal():
            print(
                "CORRIDOR",
                index,
                action,
                "gain",
                shift,
                "level",
                node.levels_completed,
                "dead",
                node.terminal(),
                "avatar",
                avatar(node),
                "specials",
                specials(node),
            )
        if node.terminal() or node.levels_completed > 4:
            break

    solution = perception.bounded_replay_bfs(
        final_root,
        perception.level_goal(4),
        final_actions,
        key_fn=state_key,
        max_states=2500,
        max_depth=20,
    )
    print("FINAL_SEARCH", solution)
    if solution:
        solved = final_root.clone()
        for index, action in enumerate(solution, 1):
            before = solved.frame()
            solved.step(*action)
            print(
                "SOLUTION",
                index,
                action,
                "gain",
                legs.band_shift(before, solved.frame()),
                "level",
                solved.levels_completed,
                "dead",
                solved.terminal(),
                "avatar",
                avatar(solved),
            )


if __name__ == "__main__":
    levels, _, error = A.run_program("bp35", probe)
    print("PROBE_RESULT", levels, error)
