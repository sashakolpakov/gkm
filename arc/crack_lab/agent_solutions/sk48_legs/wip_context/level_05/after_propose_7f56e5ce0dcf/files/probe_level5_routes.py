"""Compact route probes for the level-5 anchor transfer."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def pieces(env):
    return tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def runs(spec):
    path = []
    for action, count in spec:
        path.extend([action] * count)
    return path


def carried(env):
    before = set(pieces(env))
    result = []
    for action in (1, 2):
        branch = p.replay(env, [action])
        after = set(pieces(branch))
        moved = tuple(sorted(before - after))
        if moved:
            result.append((action, moved))
    return tuple(result)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    routes = {
        "above_cross": ((1, 1), (4, 4), (2, 1)),
        "above_cross_pull": ((1, 1), (4, 4), (2, 1), (3, 4)),
        "below_cross": ((2, 1), (4, 4), (1, 1)),
        "peel1": ((1, 1), (3, 1), (2, 1)),
        "peel2": ((1, 1), (3, 1), (2, 1)) * 2,
    }
    for name, spec in routes.items():
        branch = p.replay(env, runs(spec))
        print(name, pieces(branch), "carry", carried(branch))

    for shift in range(1, 7):
        branch = p.replay(env, [1] + [4] * shift + [2])
        print("DROP", shift, pieces(branch), "carry", carried(branch))

    anchor_path = [1, 3, 2] * 2
    anchor = p.replay(env, anchor_path)
    initial_eights = tuple(piece for piece in pieces(anchor) if piece[0] == 8)
    print("ANCHOR", pieces(anchor), "carry", carried(anchor))
    for lane_path in ([], [1], [2], [1, 1], [2, 2]):
        for extension in range(0, 8):
            for retraction in range(0, 8):
                path = lane_path + [4] * extension + [3] * retraction
                branch = p.replay(anchor, path)
                current_eights = tuple(
                    piece for piece in pieces(branch) if piece[0] == 8
                )
                if current_eights != initial_eights:
                    print("EIGHT_MOVED", path, pieces(branch), "carry", carried(branch))
                    return
    print("ANCHOR_HORIZONTAL_NONE")

    legs = (
        ("reverse", lambda branch: players.reverse_row_train(
            branch,
            approach_lanes=4,
            stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
            final_extension=4,
        )),
        ("weave", lambda branch: players.weave_vertical_four_train(branch)),
        ("unweave", lambda branch: (
            players.unweave_horizontal_pairs_to_vertical_heads(branch)
        )),
    )
    for first_name, first in legs:
        branch = env.clone()
        first(branch)
        print("LEG", first_name, branch.levels_completed, pieces(branch))
        for second_name, second in legs:
            pair = branch.clone()
            second(pair)
            print(
                "LEGS",
                first_name,
                second_name,
                pair.levels_completed,
                pieces(pair),
            )

    initial_eights = tuple(piece for piece in pieces(env) if piece[0] == 8)
    found = 0
    for approach in (1,):
        for span in range(2, 6):
            for reach in range(4, 8):
                for compact in range(3, 8):
                    path = (
                        [1] * approach
                        + [4] * span
                        + [2, 1]
                        + [3] * span
                        + [2]
                        + [4] * reach
                        + [3] * compact
                    )
                    branch = p.replay(env, path)
                    current_eights = tuple(
                        piece for piece in pieces(branch) if piece[0] == 8
                    )
                    if current_eights != initial_eights:
                        print(
                            "STAGE_MOVED",
                            (approach, span, reach, compact),
                            branch.levels_completed,
                            pieces(branch),
                        )
                        found += 1
                        if found == 12:
                            return
    print("ONE_STAGE_NONE")

    stage_sets = (
        ((5, 6, 6),),
        ((5, 6, 6), (4, 6, 5)),
        ((5, 6, 6), (4, 6, 5), (2, 5, 4)),
        ((4, 6, 6), (3, 6, 5)),
        ((3, 6, 6), (2, 6, 5)),
        ((2, 6, 6), (2, 6, 5)),
        ((4, 7, 6), (3, 7, 5), (2, 7, 4)),
    )
    for stages in stage_sets:
        branch = env.clone()
        players.reverse_row_train(
            branch,
            approach_lanes=1,
            stages=stages,
            final_extension=0,
        )
        print("STAGES", stages, branch.levels_completed, pieces(branch))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
