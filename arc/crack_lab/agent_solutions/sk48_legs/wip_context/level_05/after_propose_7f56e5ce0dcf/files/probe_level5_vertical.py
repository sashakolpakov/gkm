"""Compact vertical-contact probes for sk48 level 5."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def state(env):
    objects = p.object_candidates(env.frame())
    avatar = next(
        o["bbox"][:2]
        for o in objects
        if o["color"] == 6 and o["area"] == 18 and o["bbox"][0] < 53
    )
    pieces = tuple(
        (o["color"], *o["bbox"][:2])
        for o in objects
        if o["color"] in (8, 9) and o["bbox"][0] < 53
    )
    return avatar, pieces


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    for path in (
        [1],
        [1, 3],
        [1, 3, 2],
        [4],
        [4, 3],
        [4, 3, 3],
        [4, 4, 3],
        [4, 3, 1],
        [4, 1],
        [4, 2],
    ):
        print("TRACE", path, state(p.replay(env, path)))
    one_nine = p.replay(env, [1, 3, 2, 1, 3, 2])
    print("BASE", state(one_nine))
    for extension in range(1, 6):
        for retraction in range(1, 6):
            branch = p.replay(
                one_nine, [4] * extension + [3] * retraction
            )
            if state(branch) != state(one_nine):
                print("HORIZONTAL", extension, retraction, state(branch))
    for path in (
        [4, 4, 4, 1],
        [4, 4, 4, 2],
        [4, 4, 4, 3, 1],
        [4, 4, 4, 3, 2],
    ):
        print("CONTACT_MOVE", path, state(p.replay(one_nine, path)))
    for extension in range(7):
        contact = p.replay(one_nine, [2] + [4] * extension + [1])
        probes = []
        for action in env.actions:
            after = p.replay(contact, [action])
            probes.append((action, state(after)))
        print("BELOW", extension, state(contact), "NEXT", probes)
    for extension in (4, 5, 6):
        contact_path = [2] + [4] * extension + [1]
        for suffix in (
            [6, 1],
            [6, 2],
            [6, 3, 1],
            [6, 4, 1],
            [6, 6, 1],
        ):
            branch = p.replay(one_nine, contact_path + suffix)
            print("VERTICAL_USE", extension, suffix, state(branch))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
