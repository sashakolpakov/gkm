"""Probe coordinate-bearing actions on visible level-5 objects."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def summary(env):
    return (
        env.levels_completed,
        tuple(
            (blob.color, blob.bbox[0], blob.bbox[1])
            for blob in p.connected_components(
                env.frame(), colors=(8, 9), min_area=4
            )
            if blob.bbox[0] < 53
        ),
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    base = env.frame()
    points = (
        (7, 26),
        (13, 26),
        (25, 26),
        (31, 26),
        (37, 26),
        (43, 26),
        (49, 26),
        (29, 59),
        (35, 59),
        (41, 59),
    )
    for action in (6, 7):
        branch = env.clone()
        try:
            branch.step(action)
            print("KEY", action, p.frame_delta(base, branch.frame()), summary(branch))
        except Exception as exc:
            print("KEY_ERROR", action, type(exc).__name__)
        for x, y in points:
            branch = env.clone()
            try:
                branch.step(action, x, y)
                delta = p.frame_delta(base, branch.frame())
                if delta["count"] or branch.levels_completed != env.levels_completed:
                    print("COORD", action, x, y, delta, summary(branch))
            except Exception as exc:
                print("COORD_ERROR", action, x, y, type(exc).__name__)
                break

    sequences = (
        ((37, 26), (13, 26), (43, 26)),
        ((49, 26), (25, 26), (43, 26)),
        ((29, 59), (35, 59), (41, 59)),
        ((37, 26), (13, 26), (49, 26)),
    )
    for sequence in sequences:
        branch = env.clone()
        for x, y in sequence:
            branch.step(6, x, y)
        print("CLICKS", sequence, p.frame_delta(base, branch.frame()), summary(branch))

    contexts = {
        "empty_below": [2, 3, 1] * 3 + [3] * 6,
        "two_contacts": [3, 1] + [4] * 5 + [2],
    }
    for name, path in contexts.items():
        context = p.replay(env, path)
        switched = p.replay(context, [6])
        before = {
            action: p.frame_delta(
                context.frame(), p.replay(context, [action]).frame()
            )
            for action in (1, 2, 3, 4)
        }
        after = {
            action: p.frame_delta(
                switched.frame(), p.replay(switched, [action]).frame()
            )
            for action in (1, 2, 3, 4)
        }
        print("MODE", name, before == after, summary(context), summary(switched))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
