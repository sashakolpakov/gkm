"""Adapt the verified three-token ladder to a controlled 8-9 pair."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


TRANSFER = (
    4, 4, 4, 4, 4, 1, 1, 1, 2, 3, 3, 2, 6,
    1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 6, 1, 1,
    4, 4, 4, 4, 4, 4, 2, 2,
)
CANDIDATE = (
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3,
    1, 2, 2, 2, 2, 6, 3, 3, 3, 3, 4, 4, 4, 6, 4, 4, 6, 1,
)
CONTROLLED_PAIR = (
    4, 4, 3, 3, 3, 3, 1, 1, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 2,
)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def controlled(env):
    before = set(pieces(env))
    out = []
    for action in (3, 4):
        branch = env.clone()
        branch.step(action)
        removed = before - set(pieces(branch))
        for row in {item[1] for item in removed}:
            sequence = tuple(
                color
                for color, _row, _col in sorted(
                    (item for item in removed if item[1] == row),
                    key=lambda item: item[2],
                )
            )
            if sequence:
                out.append((action, sequence))
    return tuple(out)


def ladder(first_retract, middle_extend, far_extend):
    return (
        (4, 5), (1, 3), (2, 1), (3, first_retract), (2, 1),
        (1, 6), (2, 1), (4, middle_extend), (1, 2),
        (4, far_extend), (2, 2),
    )


def apply_runs(env, spec):
    for action, count in spec:
        for _ in range(count):
            env.step(action)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    root = p.replay(env, TRANSFER + CANDIDATE + CONTROLLED_PAIR)
    print("ROOT", pieces(root), controlled(root))
    transfer_ladder = (
        (4, 5), (1, 3), (2, 1), (3, 2), (2, 1),
        (1, 6), (2, 1), (4, 3), (1, 2), (4, 6), (2, 2),
    )
    for shift in range(4):
        joined = root.clone()
        apply_runs(joined, ((4, shift), (1, 2)))
        print("JOIN", shift, pieces(joined), controlled(joined))
        for descent in range(3):
            branch = joined.clone()
            apply_runs(branch, ((2, descent),))
            apply_runs(branch, transfer_ladder)
            print(
                "JOIN_LADDER", shift, descent, branch.levels_completed,
                pieces(branch), controlled(branch),
            )
            if shift == 0 and descent == 2:
                repeated = p.replay(branch, CANDIDATE)
                print(
                    "REPEATED_CANDIDATE",
                    repeated.levels_completed,
                    pieces(repeated),
                    controlled(repeated),
                )
                for approach in range(7):
                    for thread in range(7):
                        configured = repeated.clone()
                        try:
                            players.weave_vertical_four_train(
                                configured,
                                approach_lanes=approach,
                                thread_steps=thread,
                            )
                            configured_control = controlled(configured)
                        except IndexError:
                            continue
                        if (
                            configured.levels_completed
                            > repeated.levels_completed
                            or any(
                                sequence[:1] == (8,)
                                for _action, sequence in configured_control
                            )
                        ):
                            print(
                                "CONFIGURED_WEAVE",
                                approach,
                                thread,
                                configured.levels_completed,
                                pieces(configured),
                                configured_control,
                            )
                for leg_name, leg in (
                    (
                        "reverse",
                        lambda node: players.reverse_row_train(
                            node,
                            approach_lanes=4,
                            stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
                            final_extension=4,
                        ),
                    ),
                    ("weave", players.weave_vertical_four_train),
                    (
                        "unweave",
                        players.unweave_horizontal_pairs_to_vertical_heads,
                    ),
                ):
                    leg_node = repeated.clone()
                    try:
                        leg(leg_node)
                        fingerprint = controlled(leg_node)
                    except IndexError:
                        print("REPEATED_LEG_INVALID", leg_name)
                        continue
                    print(
                        "REPEATED_LEG", leg_name,
                        leg_node.levels_completed,
                        pieces(leg_node),
                        fingerprint,
                    )
                for repeat in range(2, 5):
                    repeated = p.replay(repeated, CANDIDATE)
                    print(
                        "REPEATED_CANDIDATE",
                        repeat,
                        repeated.levels_completed,
                        pieces(repeated),
                        controlled(repeated),
                    )
    existing = (
        (
            "reverse",
            lambda node: players.reverse_row_train(
                node,
                approach_lanes=4,
                stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
                final_extension=4,
            ),
        ),
        ("weave", players.weave_vertical_four_train),
        ("unweave", players.unweave_horizontal_pairs_to_vertical_heads),
    )
    for first_name, first in existing:
        first_node = root.clone()
        try:
            first(first_node)
        except IndexError:
            print("LEG_INVALID", first_name)
            continue
        print(
            "LEG", first_name, first_node.levels_completed,
            pieces(first_node), controlled(first_node),
        )
        for second_name, second in existing:
            second_node = first_node.clone()
            try:
                second(second_node)
                second_controlled = controlled(second_node)
            except IndexError:
                print("LEGS_INVALID", first_name, second_name)
                continue
            print(
                "LEGS", first_name, second_name, second_node.levels_completed,
                pieces(second_node), second_controlled,
            )
    best = []
    for first_retract in range(0, 4):
        for middle_extend in range(1, 6):
            for far_extend in range(3, 8):
                branch = root.clone()
                apply_runs(
                    branch,
                    ladder(first_retract, middle_extend, far_extend),
                )
                if branch.levels_completed > root.levels_completed:
                    print(
                        "FOUND",
                        first_retract, middle_extend, far_extend,
                        branch.levels_completed,
                    )
                    return
                fingerprint = controlled(branch)
                value = max(
                    (len(sequence) for _action, sequence in fingerprint),
                    default=0,
                )
                best.append(
                    (
                        value,
                        first_retract,
                        middle_extend,
                        far_extend,
                        pieces(branch),
                        fingerprint,
                    )
                )
    for result in sorted(best, reverse=True)[:12]:
        print("BEST", result)


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
