"""Compact verification of the level-9 route and final placement."""

import gkm_try

from perception import arr, connected_components


class ReachedLevel9(Exception):
    pass


class StopAtLevel9:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 8:
            raise ReachedLevel9
        return result


TARGET = {
    (3, 5), (3, 6), (3, 7),
    (4, 5),         (4, 7),
    (5, 5), (5, 6), (5, 7),
}


def boxes(frame, color):
    return tuple(
        blob.bbox
        for blob in connected_components(frame, colors=(color,), min_area=4)
    )


def target_state(frame):
    grid = arr(frame)
    empty = []
    filled = []
    signatures = {}
    for row, col in sorted(TARGET):
        colors = tuple(sorted(set(
            int(value)
            for value in grid[
                row * 4:row * 4 + 4,
                col * 4:col * 4 + 4,
            ].flat
        )))
        signatures[(row, col)] = colors
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        if 4 in colors and 9 in colors:
            filled.append((row, col))
    return {
        "empty": tuple(empty),
        "filled": tuple(filled),
        "signatures": signatures,
    }


def glyph(cell):
    colors = set(int(value) for value in cell.flat)
    for color, symbol in ((14, "A"), (15, "X"), (12, "C"), (4, "o")):
        if color in colors:
            return symbol
    if colors == {5}:
        return "#"
    if 5 in colors:
        return "+"
    if 2 in colors and 9 in colors:
        return "T"
    if 2 in colors:
        return "="
    if 7 in colors:
        return "|"
    return "."


def tile_map(frame):
    grid = arr(frame)
    return tuple(
        "".join(
            glyph(grid[row:row + 4, col:col + 4])
            for col in range(0, 64, 4)
        )
        for row in range(0, 64, 4)
    )


def summary(env, turn):
    return {
        "turn": turn,
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "courier_12": boxes(env.frame(), 12),
        "courier_15": boxes(env.frame(), 15),
        "target": target_state(env.frame()),
    }


def layered_suffix_search(env, max_depth=6, max_transitions=20000):
    base_level = env.levels_completed
    frontier = [(env.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        best = None
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                if child.levels_completed > base_level:
                    print(
                        "L9_SEARCH",
                        {
                            "depth": depth,
                            "states": len(next_states),
                            "transitions": transitions,
                            "win": child_path,
                        },
                        flush=True,
                    )
                    return child_path
                target = target_state(child.frame())
                score = (len(target["filled"]), -len(target["empty"]))
                if best is None or score > best[0]:
                    best = (score, child_path, target)
                key = arr(child.frame()).tobytes()
                next_states.setdefault(key, (child, child_path))
                if transitions >= max_transitions:
                    print("L9_SEARCH_LIMIT", transitions, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "L9_SEARCH",
            {
                "depth": depth,
                "states": len(frontier),
                "transitions": transitions,
                "best": best,
            },
            flush=True,
        )
    return None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    initial_avatar = boxes(env.frame(), 14)
    action_poses = {}
    for action in env.actions:
        child = env.clone()
        child.step(action)
        action_poses[action] = boxes(child.frame(), 14)
    print("L9_ACTIONS", initial_avatar, action_poses, flush=True)
    print("L9_INITIAL", summary(env, 0), *tile_map(env.frame()), sep="\n",
          flush=True)

    remote_pick = [2] + [4] * 6 + [1, 5, 2]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = first_delivery + [2] * 4 + [4] * 5 + [1] * 2 + [5]
    second_delivery = (
        second_pick + [2] + [3] * 4 + [1] * 2 + [3, 1, 5]
    )
    dismiss_courier = second_delivery + [1, 3, 5]
    align_local = (
        dismiss_courier
        + [2] + [3] * 5 + [5, 4]
        + [1] * 6 + [4] * 4
    )

    checkpoint = env.clone()
    previous = 0
    states = {}
    for name, actions in (
        ("first_delivery", first_delivery),
        ("second_pick", second_pick),
        ("second_delivery", second_delivery),
        ("dismiss_courier", dismiss_courier),
        ("align_local", align_local),
    ):
        for action in actions[previous:]:
            checkpoint.step(action)
        previous = len(actions)
        states[name] = checkpoint.clone()
        print("L9_PHASE", name, summary(checkpoint, previous), flush=True)

    contact_wins = []
    contact_frontier = [(states["second_delivery"].clone(), [])]
    for depth in range(1, 4):
        next_frontier = []
        for node, path in contact_frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    contact_wins.append(
                        (child_path, boxes(child.frame(), 14))
                    )
                else:
                    next_frontier.append((child, child_path))
        if contact_wins:
            print("L9_CONTACT_WINS", depth, contact_wins, flush=True)
            break
        contact_frontier = next_frontier

    idle = states["dismiss_courier"].clone()
    idle_turn = len(dismiss_courier)
    prior_target = None
    while not idle.terminal():
        current_target = target_state(idle.frame())
        condensed = (
            current_target["empty"],
            current_target["filled"],
        )
        if condensed != prior_target or idle_turn % 4 == 2:
            print(
                "L9_IDLE",
                idle_turn,
                condensed,
                boxes(idle.frame(), 12),
                boxes(idle.frame(), 4),
                flush=True,
            )
        prior_target = condensed
        idle.step(5)
        idle_turn += 1
    print("L9_IDLE_TERMINAL", idle_turn, summary(idle, idle_turn),
          flush=True)

    print("L9_CHECKPOINT_MAP", *tile_map(checkpoint.frame()), sep="\n",
          flush=True)
    for action in env.actions:
        child = checkpoint.clone()
        child.step(action)
        print(
            "L9_CHECKPOINT_ACTION",
            action,
            summary(child, len(align_local) + 1),
            flush=True,
        )
    suffix = layered_suffix_search(checkpoint)
    print("L9_SUFFIX", suffix, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
