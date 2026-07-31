import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


STAGE_HORIZONTAL = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
)


def movable(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 14), min_area=2
        )
    )


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    base = env.clone()
    apply(base, STAGE_HORIZONTAL)
    hits = []
    for horizontal_up in (4, 5, 6, 7):
        horizontal_y = 27 - 3 * horizontal_up
        with_horizontal = base.clone()
        apply(
            with_horizontal,
            [(6, 35, 34)]
            + [1] * 2 + [4] * 6 + [1] * horizontal_up,
        )
        for vertical_up in (4, 5, 6):
            vertical_y = 24 - 3 * vertical_up
            aligned = with_horizontal.clone()
            apply(
                aligned,
                [(6, 34, 47)]
                + [1] * 2 + [4] * 4 + [1] * 6
                + [3] * 4 + [1] * vertical_up + [4],
            )
            before = movable(aligned)
            print("aligned", horizontal_y, vertical_y, before)
            for delay in range(6):
                contact = aligned.clone()
                apply(
                    contact,
                    [(6, 53, horizontal_y + 1)]
                    + [4] * 2 + [4] * delay + [3] * 5,
                )
                after_contact = movable(contact)
                if after_contact[:2] != before[:2]:
                    hits.append(
                        (horizontal_y, vertical_y, delay, "push",
                         after_contact)
                    )
                for name, path in (
                    ("pull_v", [(6, 37, vertical_y + 2)] + [3] * 6),
                    ("push_v", [(6, 37, vertical_y + 2)] + [4] * 6),
                ):
                    child = contact.clone()
                    apply(child, path)
                    after = movable(child)
                    if after[0] != before[0]:
                        hits.append(
                            (horizontal_y, vertical_y, delay, name, after)
                        )
    print("hits", hits)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
