import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


LARGE_DOWN = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
    + [(6, 35, 34)] + [1] * 2 + [4] * 6 + [1] * 6
    + [(6, 34, 47)] + [1] * 2 + [4] * 4 + [1] * 6
    + [3] * 4 + [1] * 4 + [4]
    + [(6, 53, 10)] + [4] * 2 + [3] * 3
    + [(6, 28, 11), 2] + [4] * 3 + [1] + [3] * 2
)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            arr(env.frame())[:63], colors=(11, 12, 13, 14), min_area=2
        )
    )


def large(env):
    return tuple(item for item in pieces(env) if item[0] == 11)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    apply(root, LARGE_DOWN)
    print("large_down", pieces(root))

    # Bring the horizontal ring through the right and left three-pixel
    # openings, immediately below the displaced large piece.
    apply(
        root,
        [(6, 53, 10)] + [2] * 6 + [3] * 8 + [2] + [3] * 5,
    )
    print("horizontal_below", pieces(root))
    initial = large(root)
    hits = []
    down_hits = []
    for horizontal_shift in (-1, 0):
        aligned = root.clone()
        shift = [3] if horizontal_shift < 0 else []
        apply(aligned, shift)
        for delay in range(6):
            for name, path in (
                ("up", [1] * 8),
                ("up_down", [1] + [2] * 8),
                ("up2_down", [1] * 2 + [2] * 8),
                ("up_left_down", [1, 3] + [2] * 8),
                ("up_right_down", [1, 4] + [2] * 8),
            ):
                child = aligned.clone()
                apply(child, [4, 3] * delay + path)
                if large(child) != initial:
                    hits.append(
                        (horizontal_shift, delay, name, large(child),
                         pieces(child))
                    )
            for up_count in range(1, 9):
                child = aligned.clone()
                apply(child, [4, 3] * delay + [1] * up_count)
                for down_count in range(1, 13):
                    child.step(2)
                    current = large(child)
                    if current and current[0][1][0] > initial[0][1][0]:
                        down_hits.append(
                            (horizontal_shift, delay, up_count, down_count,
                             current, pieces(child))
                        )
                        break
    print("hits", hits)
    print("down_hits", down_hits)

    interlocked = root.clone()
    apply(interlocked, [3] + [1] * 8)
    print("large_with_agent", pieces(interlocked))
    for name, path in (
        ("wait_h_sideways", [4, 3] * 8),
        ("v_left", [(6, 34, 11)] + [3] * 12),
        ("v_right", [(6, 34, 11)] + [4] * 12),
        ("v_down", [(6, 34, 11)] + [2] * 12),
        ("v_up", [(6, 34, 11)] + [1] * 12),
        ("v_touch_agent", [(6, 34, 11)] + [3] * 4 + [4, 3] * 6),
    ):
        child = interlocked.clone()
        before = large(child)
        changed = []
        for index, action in enumerate(path, 1):
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            current = large(child)
            if current != before:
                changed.append((index, action, current))
            before = current
        print("agent_test", name, changed, pieces(child))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
