"""Test reset/extend/retract collection from an empty level-5 train."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def pieces(env):
    return tuple(
        (blob.color, *blob.bbox[:2])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    empty_path = [1, 3, 2] * 3
    base = p.replay(env, empty_path)
    initial_eights = tuple(piece for piece in pieces(base) if piece[0] == 8)
    print("EMPTY", pieces(base))
    found = 0
    for reset in range(7):
        for extension in range(7):
            for retraction in range(1, 8):
                path = (
                    empty_path
                    + [3] * reset
                    + [4] * extension
                    + [3] * retraction
                )
                branch = p.replay(env, path)
                current_eights = tuple(
                    piece for piece in pieces(branch) if piece[0] == 8
                )
                if current_eights != initial_eights:
                    print(
                        "FOUND",
                        reset,
                        extension,
                        retraction,
                        pieces(branch),
                        branch.levels_completed,
                    )
                    found += 1
                    if found == 12:
                        return
    print("SAME_ROW_NONE")
    for reset in range(7):
        for extension in range(9):
            for retraction in range(7):
                for direction in (1, 2):
                    path = (
                        empty_path
                        + [3] * reset
                        + [2]
                        + [4] * extension
                        + [1]
                        + [3] * retraction
                        + [direction]
                    )
                    branch = p.replay(env, path)
                    current_eights = tuple(
                        piece for piece in pieces(branch) if piece[0] == 8
                    )
                    if current_eights != initial_eights:
                        print(
                            "CROSS_FOUND",
                            reset,
                            extension,
                            retraction,
                            direction,
                            pieces(branch),
                            branch.levels_completed,
                        )
                        found += 1
                        if found == 12:
                            return
    print("CROSS_NONE")


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
