"""Verify the compact interval-network model for sp80 level 6."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def moves(start, target, negative, positive):
    return [negative if target < start else positive] * (abs(target - start) // 3)


def path_for(a, b, c, d):
    return (
        [(6, 30, 19)]
        + moves(17, a[0], 1, 2) + moves(29, a[1], 3, 4)
        + [(6, 45, 18)]
        + moves(14, b[0], 1, 2) + moves(44, b[1], 3, 4)
        + [(6, 25, 33)]
        + moves(32, c[0], 1, 2) + moves(23, c[1], 3, 4)
        + [(6, 31, 46)]
        + moves(44, d[0], 1, 2) + moves(29, d[1], 3, 4)
        + [5]
    )


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)

    # A and D turn the central vertical axis onto the upper/lower side rows.
    # B spans all three side rows; C retains its central marker alignment.
    cases = {
        f"model_b{b_left}": ((20, 29), (26, b_left), (32, 23), (35, 26))
        for b_left in (14, 29, 44)
    }
    cases.update({
        "a_above": ((17, 29), (26, 29), (32, 23), (35, 26)),
        "a_below": ((23, 29), (26, 29), (32, 23), (35, 26)),
        "d_above": ((20, 29), (26, 29), (32, 23), (32, 26)),
        "d_below": ((20, 29), (26, 29), (32, 23), (38, 26)),
        "d_unaligned": ((20, 29), (26, 29), (32, 23), (35, 29)),
    })
    for c_left in (17, 20, 23, 26):
        for b_left in (26, 29, 32):
            cases[f"directed_c{c_left}_b{b_left}"] = (
                (29, 35), (26, b_left), (26, c_left), (20, 14)
            )
    for b_left in (14, 29, 44):
        cases[f"socket_stack_b{b_left}"] = (
            (35, 29), (26, b_left), (32, 23), (20, 26)
        )
    for b_left in (11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44):
        cases[f"tiled_chain_b{b_left}"] = (
            (29, 38), (26, b_left), (26, 23), (20, 17)
        )
    for b_top in (23, 26, 29, 32, 35, 38):
        cases[f"tiled_chain_bt{b_top}"] = (
            (29, 38), (b_top, 29), (26, 23), (20, 17)
        )
    for a_left in (29, 32, 35):
        for b_top in (23, 26, 29):
            cases[f"central_branch_a{a_left}_bt{b_top}"] = (
                (29, a_left), (b_top, 29), (38, 23), (20, 26)
            )
    for c_top in (26, 32, 38):
        for b_left in (11, 29, 47):
            cases[f"socket_edges_ct{c_top}_b{b_left}"] = (
                (29, 47), (26, b_left), (c_top, 23), (20, 11)
            )
    for c_top in (41, 44, 47, 50):
        for b_top in (26, 29, 32):
            for a_left in (32, 35):
                cases[f"lower_bus_c{c_top}_b{b_top}_a{a_left}"] = (
                    (29, a_left), (b_top, 29), (c_top, 23), (20, 26)
                )
    for c_top in (14, 17):
        for d_left in (20, 23, 26):
            for a_left in (32, 35):
                for b_top in (26, 29, 32):
                    cases[
                        f"source_bus_c{c_top}_d{d_left}_a{a_left}_b{b_top}"
                    ] = (
                        (29, a_left), (b_top, 29), (c_top, 23),
                        (20, d_left),
                    )
    for d_left in (20, 23, 26, 29, 32):
        for a_left in (29, 32, 35):
            for b_left in (26, 29, 32):
                cases[f"top_bus_d{d_left}_a{a_left}_b{b_left}"] = (
                    (29, a_left), (26, b_left), (23, 23), (35, d_left)
                )
    for b_top in range(14, 42, 3):
        cases[f"axis_roles_upper_d_b{b_top}"] = (
            (29, 38), (b_top, 29), (38, 23), (20, 17)
        )
        cases[f"axis_roles_upper_c_b{b_top}"] = (
            (29, 38), (b_top, 29), (23, 23), (35, 17)
        )
    for d_left in (14, 17, 20):
        for a_left in (35, 38, 41):
            for b_left in (26, 29, 32):
                for b_top in range(14, 42, 3):
                    cases[
                        f"axis_grid_d{d_left}_a{a_left}_"
                        f"bl{b_left}_bt{b_top}_upper_d"
                    ] = (
                        (29, a_left), (b_top, b_left), (38, 23),
                        (20, d_left),
                    )
                    cases[
                        f"axis_grid_d{d_left}_a{a_left}_"
                        f"bl{b_left}_bt{b_top}_upper_c"
                    ] = (
                        (29, a_left), (b_top, b_left), (23, 23),
                        (35, d_left),
                    )
    for d_left in (14, 17, 20):
        for a_left in (35, 38, 41):
            for b_top in (23, 26, 29, 32):
                for c_top in (35, 38, 41):
                    cases[
                        f"complement_upper_d_dl{d_left}_al{a_left}_"
                        f"bt{b_top}_ct{c_top}"
                    ] = (
                        (32, a_left), (b_top, 29), (c_top, 23),
                        (23, d_left),
                    )
                for c_top in (20, 23, 26):
                    cases[
                        f"complement_lower_d_dl{d_left}_al{a_left}_"
                        f"bt{b_top}_ct{c_top}"
                    ] = (
                        (32, a_left), (b_top, 29), (c_top, 23),
                        (38, d_left),
                    )
    for c_top in (14, 17, 20):
        for d_left in (20, 23, 26, 29, 32):
            for a_left in (23, 26, 29, 32, 35):
                for b_left in (5, 8, 11, 14, 17, 20, 23, 26, 29):
                    cases[
                        f"source_split_c{c_top}_d{d_left}_"
                        f"a{a_left}_b{b_left}"
                    ] = (
                        (32, a_left), (26, b_left), (c_top, 23),
                        (23, d_left),
                    )
                    cases[
                        f"beam_split_c{c_top}_d{d_left}_"
                        f"a{a_left}_b{b_left}"
                    ] = (
                        (29, a_left), (26, b_left), (c_top, 23),
                        (20, d_left),
                    )
    for d_left in (14, 17, 20, 23, 26, 29, 32):
        for b_left in (38, 41, 44, 47):
            cases[f"projection_cover_d{d_left}_b{b_left}"] = (
                (17, 29), (26, b_left), (32, 23), (35, d_left)
            )
    for d_left in (14, 17, 20):
        for c_left in (20, 23, 26):
            for a_left in (35, 38, 41):
                for b_left in (14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44):
                    cases[
                        f"shifted_chain_d{d_left}_c{c_left}_"
                        f"a{a_left}_b{b_left}"
                    ] = (
                        (32, a_left), (29, b_left), (29, c_left),
                        (23, d_left),
                    )
    for c_top in range(14, 57, 3):
        for b_left in (14, 29):
            for b_top in (29, 32, 35, 38):
                cases[f"bar_lower_c{c_top}_bl{b_left}_bt{b_top}"] = (
                    (32, 38), (b_top, b_left), (c_top, 23), (23, 17)
                )
            for b_top in (14, 17, 20, 23):
                cases[f"bar_upper_c{c_top}_bl{b_left}_bt{b_top}"] = (
                    (32, 38), (b_top, b_left), (c_top, 23), (38, 17)
                )
    for d_left in (14, 17, 20):
        for c_left in (20, 23, 26):
            for a_left in (35, 38, 41):
                for b_left in (14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44):
                    cases[
                        f"corrected_chain_d{d_left}_c{c_left}_"
                        f"a{a_left}_b{b_left}"
                    ] = (
                        (32, a_left), (26, b_left), (29, c_left),
                        (23, d_left),
                    )
    results = {}
    for name, targets in cases.items():
        if len(sys.argv) > 1 and not name.startswith(sys.argv[1]):
            continue
        node = replay(env, path_for(*targets))
        results[name] = int(node.levels_completed)
        if node.levels_completed > env.levels_completed:
            print("INTERVAL_WIN", name, targets, "PATH", path_for(*targets))
            return
    print("INTERVAL_CASES", results)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
