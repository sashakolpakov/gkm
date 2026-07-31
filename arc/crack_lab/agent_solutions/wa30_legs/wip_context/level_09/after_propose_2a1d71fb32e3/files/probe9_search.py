"""Bounded interception search from the verified level-9 staged state."""

from itertools import product

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


def replay(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def boxes(frame, color):
    return tuple(blob.bbox for blob in connected_components(
        frame, colors=(color,), min_area=4))


def avatar_pose(frame):
    blobs = connected_components(frame, colors=(14,), min_area=4)
    if not blobs:
        return ()
    grid = arr(frame)
    r0, c0, r1, c1 = blobs[0].bbox
    return (blobs[0].bbox, grid[r0:r1 + 1, c0:c1 + 1].tobytes())


TARGET = {
    (3, 5), (3, 6), (3, 7),
    (4, 5), (4, 7),
    (5, 5), (5, 6), (5, 7),
}


def target_metric(frame):
    grid = arr(frame)
    result = {}
    for row, col in sorted(TARGET):
        colors = tuple(sorted(set(int(value) for value in
                                  grid[row * 4:row * 4 + 4,
                                       col * 4:col * 4 + 4].flat)))
        result[(row, col)] = colors
    return result


def search(env, prefix, max_depth=10):
    frontier = [[]]
    for depth in range(1, max_depth + 1):
        next_states = {}
        for path in frontier:
            for action in (1, 2, 3, 4, 5):
                child_path = path + [action]
                child = replay(env, prefix + child_path)
                if not boxes(child.frame(), 15):
                    print("WIN", child_path, "depth", depth,
                          "avatar", boxes(child.frame(), 14))
                    return child_path
                key = avatar_pose(child.frame())
                next_states.setdefault(key, child_path)
        frontier = list(next_states.values())
        print("STEP", depth, "states", len(frontier), flush=True)
    print("NO_WIN")
    return None


def reward_search(env, prefix, max_depth=5):
    frontier = [[]]
    base_level = env.levels_completed
    for depth in range(1, max_depth + 1):
        next_states = {}
        for path in frontier:
            for action in (1, 2, 3, 4, 5):
                child_path = path + [action]
                child = replay(env, prefix + child_path)
                if child.levels_completed > base_level:
                    print("REWARD_WIN", child_path, "depth", depth,
                          flush=True)
                    return child_path
                next_states.setdefault(
                    arr(child.frame()).tobytes(), child_path)
        frontier = list(next_states.values())
        print("REWARD_STEP", depth, len(frontier), flush=True)
    print("REWARD_NO_WIN", flush=True)
    return None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    remote_pick = [2] + [4] * 6 + [1, 5, 2]
    remote_place = remote_pick + [3] * 6 + [1] * 4 + [3, 5]
    last_left_pick = remote_place + [2] * 3 + [3] * 6 + [5]
    staged_left = last_left_pick + [4, 1]
    staged = replay(env, staged_left)
    print("STAGED", len(staged_left), boxes(staged.frame(), 14),
          boxes(staged.frame(), 4), flush=True)
    result = [1, 1, 1, 2, 5, 2, 2, 5]
    dismissed = replay(env, staged_left + result)
    print("VERIFIED", not boxes(dismissed.frame(), 15),
          boxes(dismissed.frame(), 14))

    travel = (
        [2] + [4] * 9 + [1] * 2 + [5, 2] + [3] * 4
    )
    candidates = {
        "blocked_then_place": (
            travel + [1] * 3 + [3, 5] + [2, 3, 1, 1, 5]
        ),
        "zigzag_place": (
            travel + [1, 1, 3, 1, 5] + [5] * 5
        ),
    }
    for name, suffix in candidates.items():
        final = replay(env, staged_left + result + suffix)
        print("FINAL", name, {
            "length": len(staged_left + result + suffix),
            "reward": final.levels_completed - env.levels_completed,
            "terminal": final.terminal(),
            "avatar": boxes(final.frame(), 14),
            "cargo": boxes(final.frame(), 4),
            "helper": boxes(final.frame(), 12),
        })

    two_remote_pick = (
        remote_place + [2] * 4 + [4] * 5 + [1] * 2 + [5]
    )
    picked = replay(env, two_remote_pick)
    print("TWO_REMOTE_PICK", len(two_remote_pick),
          boxes(picked.frame(), 14), boxes(picked.frame(), 4), flush=True)
    carrying_dismiss = two_remote_pick + [2] + [3] * 7 + [5]
    carrying_state = replay(env, carrying_dismiss)
    print("CARRYING_DISMISS", len(carrying_dismiss),
          not boxes(carrying_state.frame(), 15),
          boxes(carrying_state.frame(), 14))
    carry_contacts = {
        "left7_use2": [2] + [3] * 7 + [5, 5],
        "left7_up": [2] + [3] * 7 + [1, 5],
        "left7_down": [2] + [3] * 7 + [2, 5],
        "left6_up_left": [2] + [3] * 6 + [1, 3, 5],
        "left6_down_left": [2] + [3] * 6 + [2, 3, 5],
    }
    for name, contact in carry_contacts.items():
        state = replay(env, two_remote_pick + contact)
        print("CARRY_CONTACT", name, len(two_remote_pick + contact),
              not boxes(state.frame(), 15), boxes(state.frame(), 14))
    carry_tail = None
    base_contact = [2] + [3] * 5
    for moves_count in range(2, 2):
        for moves in product((1, 2, 3, 4), repeat=moves_count):
            contact = base_contact + list(moves) + [5]
            state = replay(env, two_remote_pick + contact)
            if not boxes(state.frame(), 15):
                carry_tail = contact
                print("CARRY_TAIL_WIN", contact,
                      len(two_remote_pick + contact),
                      boxes(state.frame(), 14), flush=True)
                break
        if carry_tail:
            break
    second_place = (
        two_remote_pick
        + [2] + [3] * 4 + [1] * 2 + [3, 1, 5]
    )
    second_state = replay(env, second_place)
    print("SECOND_PLACE", len(second_place),
          boxes(second_state.frame(), 14), boxes(second_state.frame(), 4),
          target_metric(second_state.frame()))
    place_contacts = {
        "dl": [2, 3, 5],
        "ddl": [2, 2, 3, 5],
        "ld": [3, 2, 5],
        "dll": [2, 3, 3, 5],
        "dld": [2, 3, 2, 5],
        "dlll": [2, 3, 3, 3, 5],
        "dllu": [2, 3, 3, 1, 5],
        "dlul": [2, 3, 1, 3, 5],
        "dlllu": [2, 3, 3, 3, 1, 5],
    }
    for name, contact in place_contacts.items():
        prefix = second_place + contact
        state = replay(env, prefix)
        print("PLACE_CONTACT", name, len(prefix),
              not boxes(state.frame(), 15), boxes(state.frame(), 14))
        if not boxes(state.frame(), 15):
            final_path = prefix + [5] * (70 - len(prefix))
            final = replay(env, final_path)
            print("PLACE_FINAL", name, {
                "reward": final.levels_completed - env.levels_completed,
                "terminal": final.terminal(),
                "cargo": boxes(final.frame(), 4),
                "helper": boxes(final.frame(), 12),
            })
    search(env, second_place, max_depth=8)
    exact_dismiss = second_place + [1, 3, 5]
    for name, tail in {
        "idle": [5] * 23,
        "down_idle": [2] + [5] * 22,
        "right_idle": [4] + [5] * 22,
    }.items():
        final_path = exact_dismiss + tail
        final = replay(env, final_path)
        print("EXACT_FINAL", name, {
            "length": len(final_path),
            "reward": final.levels_completed - env.levels_completed,
            "terminal": final.terminal(),
            "cargo": boxes(final.frame(), 4),
            "helper": boxes(final.frame(), 12),
        })
    local_pick = [2] * 2 + [3] * 5 + [5] + [1] * 6
    for column, right_steps in ((6, 4), (7, 5)):
        local_finish = (
            local_pick + [4] * right_steps + [2, 5, 1, 5]
        )
        final_path = exact_dismiss + local_finish
        final = replay(env, final_path)
        print("LOCAL_FINAL", column, {
            "length": len(final_path),
            "reward": final.levels_completed - env.levels_completed,
            "terminal": final.terminal(),
            "cargo": boxes(final.frame(), 4),
            "helper": boxes(final.frame(), 12),
            "avatar": boxes(final.frame(), 14),
        })
        if column == 6:
            pushed = replay(env, final_path + [2])
            print("LOCAL_PUSH", {
                "length": len(final_path) + 1,
                "reward": pushed.levels_completed - env.levels_completed,
                "terminal": pushed.terminal(),
                "cargo": boxes(pushed.frame(), 4),
                "avatar": boxes(pushed.frame(), 14),
            })
    local_align = exact_dismiss + local_pick + [4] * 4
    aligned = replay(env, local_align)
    print("LOCAL_ALIGN", len(local_align), boxes(aligned.frame(), 14),
          boxes(aligned.frame(), 4), target_metric(aligned.frame()))
    for action in (1, 2, 3, 4, 5):
        one = replay(env, local_align + [action])
        print("ALIGN_ACTION", action, boxes(one.frame(), 14),
              boxes(one.frame(), 4))
    # reward_search(env, local_align, max_depth=5)
    local_align7 = exact_dismiss + local_pick + [4] * 5
    aligned7 = replay(env, local_align7)
    print("LOCAL_ALIGN7", len(local_align7), boxes(aligned7.frame(), 14),
          target_metric(aligned7.frame()), flush=True)
    # reward_search(env, local_align7, max_depth=4)
    local_pick_short = [2] * 2 + [3] * 4 + [5] + [1] * 6
    local_align7_short = exact_dismiss + local_pick_short + [4] * 5
    aligned7_short = replay(env, local_align7_short)
    print("LOCAL_ALIGN7_SHORT", len(local_align7_short),
          boxes(aligned7_short.frame(), 14),
          boxes(aligned7_short.frame(), 4),
          target_metric(aligned7_short.frame()), flush=True)
    # reward_search(env, local_align7_short, max_depth=5)
    local_pick_up5 = [2] * 2 + [3] * 5 + [5] + [1] * 5
    local_align_up5 = exact_dismiss + local_pick_up5 + [4] * 5
    aligned_up5 = replay(env, local_align_up5)
    print("LOCAL_ALIGN_UP5", len(local_align_up5),
          boxes(aligned_up5.frame(), 14),
          boxes(aligned_up5.frame(), 4),
          target_metric(aligned_up5.frame()), flush=True)
    # reward_search(env, local_align_up5, max_depth=5)
    remote_place_short = (
        remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    )
    two_remote_pick_short = (
        remote_place_short + [2] * 4 + [4] * 5 + [1] * 2 + [5]
    )
    second_place_short = (
        two_remote_pick_short
        + [2] + [3] * 4 + [1] * 2 + [3, 1, 5]
    )
    short_state = replay(env, second_place_short)
    print("SECOND_PLACE_SHORT", len(second_place_short),
          boxes(short_state.frame(), 14),
          boxes(short_state.frame(), 4),
          target_metric(short_state.frame()))
    for name, contact in {
        "ul": [1, 3, 5],
        "uul": [1, 1, 3, 5],
        "dul": [2, 1, 3, 5],
        "ull": [1, 3, 3, 5],
    }.items():
        state = replay(env, second_place_short + contact)
        print("SHORT_CONTACT", name,
              len(second_place_short + contact),
              not boxes(state.frame(), 15),
              boxes(state.frame(), 14))
    exact_short = second_place_short + [1, 3, 5]
    short_align7 = exact_short + local_pick + [4] * 5
    short_aligned = replay(env, short_align7)
    print("SHORT_ALIGN7", len(short_align7),
          boxes(short_aligned.frame(), 14),
          boxes(short_aligned.frame(), 4),
          target_metric(short_aligned.frame()), flush=True)
    # reward_search(env, short_align7, max_depth=5)
    short_local_pick = [2] + [3] * 5 + [5] + [1] * 6
    short_align7_fixed = exact_short + short_local_pick + [4] * 5
    short_fixed = replay(env, short_align7_fixed)
    print("SHORT_ALIGN7_FIXED", len(short_align7_fixed),
          boxes(short_fixed.frame(), 14),
          boxes(short_fixed.frame(), 4),
          target_metric(short_fixed.frame()), flush=True)
    # reward_search(env, short_align7_fixed, max_depth=6)
    short_local_clear = (
        [2] + [3] * 5 + [5, 4] + [1] * 6 + [4] * 4
    )
    short_clear_align = exact_short + short_local_clear
    clear_state = replay(env, short_clear_align)
    print("SHORT_CLEAR_ALIGN", len(short_clear_align),
          boxes(clear_state.frame(), 14),
          boxes(clear_state.frame(), 4),
          target_metric(clear_state.frame()), flush=True)
    reward_search(env, short_clear_align, max_depth=6)
    placement_tails = {
        "ud_use": [1, 2, 5, 1, 5],
        "ud_use_down": [1, 2, 5, 2, 5],
        "du_du": [2, 5, 1, 2, 5],
        "du_dd": [2, 5, 1, 2, 2],
        "d_duu": [2, 5, 2, 1, 1],
        "u_dd": [1, 2, 2, 5, 1],
        "dd_u": [2, 2, 5, 1, 5],
        "dud": [2, 1, 2, 5, 1],
    }
    for name, tail in placement_tails.items():
        candidate = local_align + tail
        final = replay(env, candidate)
        print("PLACE_TAIL", name, {
            "reward": final.levels_completed - env.levels_completed,
            "cargo": boxes(final.frame(), 4),
            "avatar": boxes(final.frame(), 14),
        })
    carrying_place = [4] * 3 + [1] * 2 + [3, 1, 5]
    final_path = carrying_dismiss + carrying_place + [2] + [5] * 18
    final = replay(env, final_path)
    print("CARRYING_FINAL", {
        "length": len(final_path),
        "reward": final.levels_completed - env.levels_completed,
        "terminal": final.terminal(),
        "avatar": boxes(final.frame(), 14),
        "cargo": boxes(final.frame(), 4),
        "helper": boxes(final.frame(), 12),
    })


gkm_try.A.run_program("wa30", inspect)
