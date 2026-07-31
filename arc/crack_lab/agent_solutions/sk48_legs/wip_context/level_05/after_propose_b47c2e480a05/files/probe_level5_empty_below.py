"""Probe an empty tether routed above the level-5 center block."""
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


def rope(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in p.connected_components(
            env.frame(), colors=(0,), min_area=1
        )
        if blob.bbox[0] < 53
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    release = [2, 3, 1] * 3
    empty = p.replay(env, release)
    initial_eights = tuple(piece for piece in pieces(empty) if piece[0] == 8)
    print("EMPTY_BELOW", pieces(empty), "rope", rope(empty))
    for path in (
        [3],
        [4],
        [1, 4, 4, 4, 4],
        [1, 4, 4, 4, 4, 2],
        [1, 4, 4, 4, 4, 2, 3],
    ):
        branch = p.replay(empty, path)
        print("ROPE", path, rope(branch), p.frame_delta(empty.frame(), branch.frame()))
    anchor = p.replay(env, [2, 3, 1] * 2)
    initial_eights = tuple(piece for piece in pieces(anchor) if piece[0] == 8)
    for contact_extension in range(3, 7):
        contact = [1] + [4] * contact_extension + [2]
        for first_action in (3, 4):
            for first_count in range(1, 9):
                for second_action in (3, 4):
                    for second_count in range(0, 9):
                        path = (
                            contact
                            + [first_action] * first_count
                            + [second_action] * second_count
                        )
                        branch = p.replay(anchor, path)
                        current_eights = tuple(
                            piece for piece in pieces(branch) if piece[0] == 8
                        )
                        if (
                            current_eights != initial_eights
                            or branch.levels_completed > env.levels_completed
                        ):
                            print(
                                "CONTACT_HORIZONTAL_FOUND",
                                path,
                                branch.levels_completed,
                                pieces(branch),
                            )
                            return
    print("CONTACT_HORIZONTAL_NONE")
    return
    for crossing in range(3, 9):
        for far_extension in range(0, 6):
            for retraction in range(1, 9):
                path = (
                    [1]
                    + [4] * crossing
                    + [2]
                    + [4] * far_extension
                    + [3] * retraction
                )
                branch = p.replay(empty, path)
                current_eights = tuple(
                    piece for piece in pieces(branch) if piece[0] == 8
                )
                if (
                    current_eights != initial_eights
                    or branch.levels_completed > env.levels_completed
                ):
                    print(
                        "BEND_FOUND",
                        crossing,
                        far_extension,
                        retraction,
                        branch.levels_completed,
                        pieces(branch),
                    )
                    return
    print("BEND_NONE")
    for reset in ():
        for extension in range(1, 9):
            for retraction in range(1, 9):
                path = (
                    [3] * reset
                    + [1]
                    + [4] * extension
                    + [2]
                    + [3] * retraction
                )
                branch = p.replay(empty, path)
                current_eights = tuple(
                    piece for piece in pieces(branch) if piece[0] == 8
                )
                if (
                    current_eights != initial_eights
                    or branch.levels_completed > env.levels_completed
                ):
                    print(
                        "FOUND",
                        reset,
                        extension,
                        retraction,
                        branch.levels_completed,
                        pieces(branch),
                    )
                    return
    print("NONE")

    anchor = p.replay(env, [2, 3, 1] * 2)
    initial_eights = tuple(piece for piece in pieces(anchor) if piece[0] == 8)
    print("ANCHOR_BELOW", pieces(anchor))
    for extension in range(1, 9):
        for retraction in range(0, 9):
            for suffix in ([], [1], [2], [4], [3]):
                path = (
                    [1]
                    + [4] * extension
                    + [2]
                    + [3] * retraction
                    + suffix
                )
                branch = p.replay(anchor, path)
                current_eights = tuple(
                    piece for piece in pieces(branch) if piece[0] == 8
                )
                if (
                    current_eights != initial_eights
                    or branch.levels_completed > env.levels_completed
                ):
                    print(
                        "ANCHOR_FOUND",
                        path,
                        branch.levels_completed,
                        pieces(branch),
                    )
                    return
    print("ANCHOR_NONE")


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
