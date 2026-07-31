"""Small level-9 probes over the documented arena/environment surface."""

from itertools import product

import gkm_try

from perception import arr, connected_components


def compact(frame):
    return {
        color: [
            (blob.bbox, blob.area)
            for blob in connected_components(frame, colors=(color,), min_area=4)
        ]
        for color in (1, 2, 4, 7, 9, 12, 14, 15)
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
    return [
        "".join(glyph(grid[row:row + 4, col:col + 4])
                for col in range(0, 64, 4))
        for row in range(0, 64, 4)
    ]


TARGET = {
    (3, 5), (3, 6), (3, 7),
    (4, 5), (4, 7),
    (5, 5), (5, 6), (5, 7),
}


def metric(frame):
    grid = arr(frame)
    empty = []
    cargo = []
    for row, col in TARGET:
        colors = set(int(value) for value in
                     grid[row * 4:row * 4 + 4,
                          col * 4:col * 4 + 4].flat)
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        if 4 in colors and 9 in colors:
            cargo.append((row, col))
    return {"empty": sorted(empty), "filled": sorted(cargo)}


def outcome(env, name, actions):
    clone = env.clone()
    before = clone.levels_completed
    used = 0
    for action in actions:
        if clone.terminal():
            break
        clone.step(action)
        used += 1
    print("PATH", name, {
        "requested": len(actions),
        "used": used,
        "reward": clone.levels_completed - before,
        "terminal": clone.terminal(),
        "target": metric(clone.frame()),
        "actors": {
            color: [blob.bbox for blob in connected_components(
                clone.frame(), colors=(color,), min_area=4)]
            for color in (4, 12, 14, 15)
        },
    })


def avatar_cell(frame):
    blobs = connected_components(frame, colors=(14,), min_area=4)
    if not blobs:
        return (99, 99)
    row, col = blobs[0].centroid
    return (round(row / 4), round(col / 4))


def replay_from(env, actions):
    node = env.clone()
    for action in actions:
        if node.terminal():
            break
        node.step(action)
    return node


def beam_suffix(env, prefix, width=10, depth=28):
    base_level = env.levels_completed
    frontier = [[]]
    for step in range(1, depth + 1):
        candidates = {}
        for path in frontier:
            for action in (1, 2, 3, 4, 5):
                child_path = path + [action]
                child = replay_from(env, prefix + child_path)
                if child.levels_completed > base_level:
                    print("BEAM_WIN", child_path, flush=True)
                    return child_path
                frame = child.frame()
                state_metric = metric(frame)
                avatar = avatar_cell(frame)
                cargo = tuple(blob.bbox for blob in connected_components(
                    frame, colors=(4,), min_area=4))
                helper = tuple(blob.bbox for blob in connected_components(
                    frame, colors=(12,), min_area=4))
                key = (
                    tuple(state_metric["empty"]),
                    tuple(state_metric["filled"]),
                    avatar,
                    cargo,
                    helper,
                )
                occupied = len(TARGET) - len(state_metric["empty"])
                settled = len(state_metric["filled"])
                target_distance = min(
                    (abs(avatar[0] - row) + abs(avatar[1] - col)
                     for row, col in state_metric["empty"]),
                    default=0,
                )
                score = (occupied * 100 + settled * 20
                         - target_distance)
                old = candidates.get(key)
                if old is None or score > old[0]:
                    candidates[key] = (score, child_path, state_metric, avatar)
        ranked = sorted(
            candidates.values(), key=lambda item: item[0], reverse=True)
        frontier = [path for _, path, _, _ in ranked[:width]]
        best_score, best_path, best_metric, best_avatar = ranked[0]
        if step % 4 == 0 or step == depth:
            print("BEAM_STEP", step, {
                "score": best_score,
                "target": best_metric,
                "avatar": best_avatar,
                "path": best_path,
                "states": len(candidates),
            }, flush=True)
    best_node = replay_from(env, prefix + frontier[0])
    print("BEAM_BEST", frontier[0], {
        "target": metric(best_node.frame()),
        "avatar": avatar_cell(best_node.frame()),
    }, flush=True)
    return None


def staged_interception(env, prefix, max_depth=10):
    frontier = [[]]
    for depth in range(1, max_depth + 1):
        next_states = {}
        for path in frontier:
            for action in (1, 2, 3, 4, 5):
                child_path = path + [action]
                child = replay_from(env, prefix + child_path)
                thief = connected_components(
                    child.frame(), colors=(15,), min_area=4)
                if not thief:
                    print("STAGED_SEARCH_WIN", child_path, {
                        "depth": depth,
                        "avatar": avatar_cell(child.frame()),
                        "target": metric(child.frame()),
                    }, flush=True)
                    return child_path
                grid = arr(child.frame())
                avatars = connected_components(
                    child.frame(), colors=(14,), min_area=4)
                if avatars:
                    r0, c0, r1, c1 = avatars[0].bbox
                    pose = (
                        avatars[0].bbox,
                        grid[r0:r1 + 1, c0:c1 + 1].tobytes(),
                    )
                else:
                    pose = ()
                cargo = tuple(blob.bbox for blob in connected_components(
                    child.frame(), colors=(4,), min_area=4))
                next_states.setdefault((pose, cargo), child_path)
        frontier = list(next_states.values())
        print("STAGED_SEARCH_STEP", depth, len(frontier), flush=True)
    return None


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


def inspect_handoff(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass
    print("HANDOFF", {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "actions": env.actions,
        "objects": compact(env.frame()),
    })
    print("MAP", *tile_map(env.frame()), sep="\n")
    budget = env.clone()
    turns = 0
    while not budget.terminal():
        budget.step(5)
        turns += 1
    print("BUDGET", turns)
    for action in env.actions:
        clone = env.clone()
        level = clone.levels_completed
        clone.step(action)
        print("ACTION", action, {
            "reward": clone.levels_completed - level,
            "actors": {
                color: [blob.bbox for blob in connected_components(
                    clone.frame(), colors=(color,), min_area=4)]
                for color in (12, 14, 15)
            },
        })
    first_alt = [1, 3, 5] + [3] * 7 + [1] * 3 + [5]
    second_alt = (
        [2] * 3 + [4] * 4 + [1, 5, 2]
        + [3] * 6 + [1] * 3 + [5]
    )
    alt_two = first_alt + second_alt
    alt_events = env.clone()
    for turn, action in enumerate(alt_two, 1):
        alt_events.step(action)
        if action == 5:
            print("ALT_EVENT", turn, {
                "target": metric(alt_events.frame()),
                "avatar": [blob.bbox for blob in connected_components(
                    alt_events.frame(), colors=(14,), min_area=4)],
                "helper": [blob.bbox for blob in connected_components(
                    alt_events.frame(), colors=(12,), min_area=4)],
                "cargo": [blob.bbox for blob in connected_components(
                    alt_events.frame(), colors=(4,), min_area=4)],
            })
    contacts = {
        "cut5": [2, 3, 3, 3, 5],
        "cut6": [2, 3, 3, 3, 3, 5],
        "upcut5": [3, 3, 3, 2, 5],
    }
    stage_lower = [2] + [3] * 2 + [5] + [4] * 2 + [1, 5]
    fill_center = [3] * 3 + [5] + [4] * 4 + [1] * 2 + [5]
    stage_below = [2, 3, 3, 5, 2, 5, 4]
    fill_below = [1] + [3] * 2 + [1, 5] + [4] * 4 + [1] * 2 + [5]
    remote_pick = [2] + [4] * 6 + [1, 5, 2]
    remote_stages = {
        "gap_left": remote_pick + [3] * 6 + [5],
        "right_port": remote_pick + [3] * 4 + [1, 3, 5],
        "left_port": remote_pick + [3] * 6 + [1, 4, 5],
    }
    remote_direct = (
        remote_pick + [3] * 6 + [1] * 4 + [3, 5, 2, 1]
    )
    remote_place = remote_pick + [3] * 6 + [1] * 4 + [3, 5]
    stage_last_left = [2] * 2 + [3] * 6 + [5, 1, 5]
    outcome(env, "stage_last_left", remote_place + stage_last_left)
    outcome(
        env, "stage_last_left_wait10",
        remote_place + stage_last_left + [5] * 10,
    )
    last_left_pick = remote_place + [2] * 3 + [3] * 6 + [5]
    for name, placement in {
        "right_up": [4, 1, 5],
        "right_up_left": [4, 1, 3, 5],
        "right_up_up": [4, 1, 1, 5],
        "right_up_left_hold": [4, 1, 3],
        "right_up_hold": [4, 1],
        "right2_up_hold": [4, 4, 1],
        "right3_up_hold": [4, 4, 4, 1],
        "right2_up_left_hold": [4, 4, 1, 3],
    }.items():
        outcome(env, "left_" + name, last_left_pick + placement)
    staged_left = last_left_pick + [4, 1]
    staged_contacts = {
        "r": [4, 5],
        "rd": [4, 2, 5],
        "ru": [4, 1, 5],
        "dr": [2, 4, 5],
        "ur": [1, 4, 5],
        "rr": [4, 4, 5],
        "rrd": [4, 4, 2, 5],
        "drr": [2, 4, 4, 5],
        "urr": [1, 4, 4, 5],
        "rur": [4, 1, 4, 5],
        "rrr": [4, 4, 4, 5],
    }
    for wait in range(5, 12):
        for name, contact in staged_contacts.items():
            result = replay_from(
                env, staged_left + [5] * wait + contact)
            if not connected_components(
                    result.frame(), colors=(15,), min_area=4):
                print("STAGED_DISMISS", wait, name, {
                    "length": len(staged_left) + wait + len(contact),
                    "target": metric(result.frame()),
                    "avatar": avatar_cell(result.frame()),
                    "cargo": [blob.bbox for blob in connected_components(
                        result.frame(), colors=(4,), min_area=4)],
                })
    staged_watch = replay_from(env, staged_left)
    for wait in range(17):
        if wait % 2 == 0:
            print("STAGED_TRACE", 34 + wait, {
                "avatar": avatar_cell(staged_watch.frame()),
                "thief": [blob.bbox for blob in connected_components(
                    staged_watch.frame(), colors=(15,), min_area=4)],
                "helper": [blob.bbox for blob in connected_components(
                    staged_watch.frame(), colors=(12,), min_area=4)],
                "cargo": [blob.bbox for blob in connected_components(
                    staged_watch.frame(), colors=(4,), min_area=4)],
            })
        staged_watch.step(5)
    staged_interception(env, staged_left)
    pick_watch = replay_from(env, remote_place)
    print("PICK_EVENT", 22, avatar_cell(pick_watch.frame()))
    for segment in ([2] * 2, [3] * 6, [5]):
        for action in segment:
            pick_watch.step(action)
        print("PICK_EVENT", 22 + sum(
            len(part) for part in (
                ([2] * 2,) if segment == [2] * 2 else
                (([2] * 2, [3] * 6) if segment == [3] * 6 else
                 ([2] * 2, [3] * 6, [5]))
            )
        ), {
            "avatar": avatar_cell(pick_watch.frame()),
            "cargo": [blob.bbox for blob in connected_components(
                pick_watch.frame(), colors=(4,), min_area=4)],
        })
    for name, stage in remote_stages.items():
        outcome(env, "remote_" + name, stage)
        outcome(env, "remote_" + name + "_wait20", stage + [5] * 20)
    outcome(env, "remote_direct", remote_direct)
    remote_watch = env.clone()
    for action in remote_direct:
        remote_watch.step(action)
    for wait in range(27):
        thief = connected_components(
            remote_watch.frame(), colors=(15,), min_area=4)
        if wait % 2 == 0 or not thief:
            print("REMOTE_TRACE", len(remote_direct) + wait, {
                "target": metric(remote_watch.frame()),
                "avatar": avatar_cell(remote_watch.frame()),
                "helper": [blob.bbox for blob in connected_components(
                    remote_watch.frame(), colors=(12,), min_area=4)],
                "thief": [blob.bbox for blob in thief],
            })
        if remote_watch.terminal():
            break
        remote_watch.step(5)
    remote_contacts = {
        "dl": [2, 3, 5],
        "dll": [2, 3, 3, 5],
        "ddl": [2, 2, 3, 5],
        "ld": [3, 2, 5],
    }
    for wait in range(12, 21):
        for name, contact in remote_contacts.items():
            result = replay_from(
                env, remote_direct + [5] * wait + contact)
            thief = connected_components(
                result.frame(), colors=(15,), min_area=4)
            if not thief:
                print("REMOTE_DISMISS", wait, name, {
                    "length": len(remote_direct) + wait + len(contact),
                    "target": metric(result.frame()),
                    "avatar": avatar_cell(result.frame()),
                })
    earliest = None
    for wait in range(14, 20):
        for moves_count in range(2, 6):
            for moves in product((2, 3), repeat=moves_count):
                candidate = [5] * wait + list(moves) + [5]
                total = len(remote_direct) + len(candidate)
                if earliest is not None and total > earliest:
                    continue
                result = replay_from(env, remote_direct + candidate)
                if not connected_components(
                        result.frame(), colors=(15,), min_area=4):
                    earliest = total
                    print("EARLY_REMOTE_DISMISS", wait, list(moves), {
                        "length": total,
                        "target": metric(result.frame()),
                        "avatar": avatar_cell(result.frame()),
                    })
    remote_dismiss = remote_direct + [5] * 18 + [2, 2, 3, 5]
    second_remote = (
        [2] * 2 + [4] * 6 + [1] * 2 + [5, 2]
        + [3] * 4 + [1] * 4 + [3, 5]
    )
    outcome(
        env, "two_remote_finish",
        remote_dismiss + second_remote + [2, 1],
    )
    early_remote_dismiss = (
        remote_direct + [5] * 14 + [2, 2, 2, 2, 3, 5]
    )
    early_second_remote = (
        [4] * 6 + [1] * 2 + [5, 2]
        + [3] * 4 + [1] * 4 + [3, 5]
    )
    outcome(
        env, "early_two_remote_finish",
        early_remote_dismiss + early_second_remote + [2, 1] + [5] * 4,
    )
    second_watch = replay_from(env, early_remote_dismiss)
    print("SECOND_EVENT", 44, {
        "avatar": avatar_cell(second_watch.frame()),
        "helper": [blob.bbox for blob in connected_components(
            second_watch.frame(), colors=(12,), min_area=4)],
        "cargo": [blob.bbox for blob in connected_components(
            second_watch.frame(), colors=(4,), min_area=4)],
    })
    for offset, action in enumerate(early_second_remote, 1):
        second_watch.step(action)
        if action == 5:
            print("SECOND_EVENT", 44 + offset, {
                "avatar": avatar_cell(second_watch.frame()),
                "target": metric(second_watch.frame()),
                "helper": [blob.bbox for blob in connected_components(
                    second_watch.frame(), colors=(12,), min_area=4)],
                "cargo": [blob.bbox for blob in connected_components(
                    second_watch.frame(), colors=(4,), min_area=4)],
            })
    outcome(env, "alt_two", alt_two)
    outcome(env, "idle11", [5] * 11)
    outcome(env, "alt_two_wait11", alt_two + [5] * 11)
    for name, contact in contacts.items():
        outcome(env, name, alt_two + contact)
        outcome(env, name + "_after_wait11", alt_two + [5] * 11 + contact)
    outcome(
        env, "final_coop",
        alt_two + contacts["cut5"] + stage_lower + fill_center + [5],
    )
    outcome(
        env, "final_coop_after_wait11",
        alt_two + [5] * 11 + contacts["cut5"]
        + stage_lower + fill_center + [5],
    )
    outcome(
        env, "below_coop",
        alt_two + contacts["cut5"] + stage_below + fill_below,
    )
    trace = env.clone()
    for action in alt_two:
        trace.step(action)
    for wait in range(37):
        if wait % 2 == 0 or not connected_components(
                trace.frame(), colors=(15,), min_area=4):
            print("TRACE", 34 + wait, {
                "target": metric(trace.frame()),
                "avatar": [blob.bbox for blob in connected_components(
                    trace.frame(), colors=(14,), min_area=4)],
                "helper": [blob.bbox for blob in connected_components(
                    trace.frame(), colors=(12,), min_area=4)],
                "thief": [blob.bbox for blob in connected_components(
                    trace.frame(), colors=(15,), min_area=4)],
            })
        if trace.terminal():
            break
        trace.step(5)
    dismiss_state = env.clone()
    for action in alt_two:
        dismiss_state.step(action)
    dismissal = [1, 2, 2, 2, 2, 4, 2, 5]
    print("DISMISSAL_REPLAY", dismissal)
    if dismissal:
        outcome(env, "dismissal_bfs", alt_two + dismissal)
        outcome(
            env, "dismiss_stage_lower",
            alt_two + dismissal + stage_lower + fill_center + [5] * 8,
        )
        outcome(
            env, "dismiss_stage_below",
            alt_two + dismissal + stage_below + fill_below + [5] * 10,
        )
        after_dismissal = env.clone()
        for action in alt_two + dismissal:
            after_dismissal.step(action)
        print("SAFE_MAP", *tile_map(after_dismissal.frame()), sep="\n")
        for wait in range(29):
            if wait % 4 == 0:
                print("SAFE_TRACE", 42 + wait, {
                    "target": metric(after_dismissal.frame()),
                    "avatar": [blob.bbox for blob in connected_components(
                        after_dismissal.frame(), colors=(14,), min_area=4)],
                    "helper": [blob.bbox for blob in connected_components(
                        after_dismissal.frame(), colors=(12,), min_area=4)],
                    "cargo": [blob.bbox for blob in connected_components(
                        after_dismissal.frame(), colors=(4,), min_area=4)],
                })
            if after_dismissal.terminal():
                break
            after_dismissal.step(5)
        # beam_suffix(env, alt_two + dismissal)


gkm_try.A.run_program("wa30", inspect_handoff)
