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
    node = env.clone()
    apply(node, STAGE_HORIZONTAL)
    print("horizontal_central", movable(node))

    # The horizontal ring fits through the central room's right opening.
    # Keep it just to the right of the inaccessible upper cycling agent.
    apply(node, [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * 5)
    print("horizontal_upper_right", movable(node))

    # Bring the vertical ring under the lower cycling agent's relay.
    apply(node, [(6, 34, 47)] + [1] * 2 + [4] * 4)
    print("vertical_waiting", movable(node))
    previous = movable(node)
    for index in range(1, 7):
        node.step(1)
        current = movable(node)
        if current != previous:
            print("vertical_up", index, current)
        previous = current

    # Finish aligning the vertical ring with the large piece and the waiting
    # horizontal ring.  Then approach the upper agent at every phase.
    apply(node, [3] * 4 + [1] * 4 + [4])
    print("vertical_middle", movable(node))
    for delay in range(6):
        child = node.clone()
        apply(
            child,
            [(6, 53, 13)]
            + [4] * 2
            + [4] * delay
            + [3] * 2,
        )
        print("phase", delay, "safe", movable(child))
        previous = movable(child)
        for index in range(1, 9):
            child.step(3)
            current = movable(child)
            if current != previous:
                print("relay_left", delay, index, current)
            previous = current

        # Recreate the first wall contact, then move the vertical ring away
        # from or toward it to test pulling as well as pushing.
        contact = node.clone()
        apply(
            contact,
            [(6, 53, 13)]
            + [4] * 2
            + [4] * delay
            + [3] * 5,
        )
        for name, path in (
            ("pull_left", [(6, 37, 14)] + [3] * 6),
            ("push_right", [(6, 37, 14)] + [4] * 6),
            ("align_down_pull", [(6, 37, 14), 2] + [3] * 6),
            ("align_up_pull", [(6, 37, 14), 1] + [3] * 6),
        ):
            test = contact.clone()
            before = movable(test)
            apply(test, path)
            after = movable(test)
            print("linked", delay, name, before, after)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
