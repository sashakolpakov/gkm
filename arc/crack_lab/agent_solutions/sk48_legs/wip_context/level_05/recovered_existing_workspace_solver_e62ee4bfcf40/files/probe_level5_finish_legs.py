"""Try existing general train legs from verified level-5 frontiers."""
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


def state(env):
    data = p.arr(env.frame())
    avatar = int(np.argwhere(data[:53, :11] == 6)[:, 0].min()) + 1
    items = tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )
    lane = tuple(
        color
        for color, _row, _col in sorted(
            (item for item in items if item[1] == avatar),
            key=lambda item: item[2],
        )
    )
    return avatar, lane, items


LEGS = (
    (
        "reverse",
        lambda env: players.reverse_row_train(
            env,
            approach_lanes=4,
            stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
            final_extension=4,
        ),
    ),
    ("weave", players.weave_vertical_four_train),
    ("unweave", players.unweave_horizontal_pairs_to_vertical_heads),
)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    roots = (
        ("start", env.clone()),
        ("transfer", p.replay(env, TRANSFER)),
        ("candidate", p.replay(env, TRANSFER + CANDIDATE)),
    )
    for root_name, root in roots:
        print("ROOT", root_name, state(root))
        for first_name, first in LEGS:
            first_node = root.clone()
            first(first_node)
            print(
                "ONE", root_name, first_name,
                first_node.levels_completed, state(first_node),
            )
            for second_name, second in LEGS:
                second_node = first_node.clone()
                second(second_node)
                print(
                    "TWO", root_name, first_name, second_name,
                    second_node.levels_completed, state(second_node),
                )
                if (
                    root_name == "candidate"
                    and first_name == "weave"
                    and second_name == "reverse"
                ):
                    for extension in range(7):
                        ladder = second_node.clone()
                        ladder.step(1)
                        for _ in range(extension):
                            ladder.step(4)
                        ladder.step(2)
                        print(
                            "LADDER", extension,
                            ladder.levels_completed, state(ladder),
                        )
                if root_name == "candidate" and first_name == "weave":
                    for third_name, third in LEGS:
                        third_node = second_node.clone()
                        third(third_node)
                        print(
                            "THREE", first_name, second_name, third_name,
                            third_node.levels_completed, state(third_node),
                        )


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
