"""Bounded deterministic macro rollouts from verified level-5 frontiers."""
import random
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def eights(env):
    return tuple(
        blob.bbox[:2]
        for blob in p.connected_components(
            env.frame(), colors=(8,), min_area=4
        )
        if blob.bbox[0] < 53
    )


def has_898(env):
    pieces = []
    for color in (8, 9):
        for blob in p.connected_components(
            env.frame(), colors=(color,), min_area=4
        ):
            if blob.bbox[0] < 53:
                pieces.append((blob.bbox[0], blob.bbox[1], color))
    for row in {item[0] for item in pieces}:
        colors = tuple(
            color
            for _, _, color in sorted(
                (item for item in pieces if item[0] == row),
                key=lambda item: item[1],
            )
        )
        if any(colors[index:index + 3] == (8, 9, 8)
               for index in range(max(0, len(colors) - 2))):
            return True
    return False


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    transfer = (
        4, 4, 4, 4, 4, 1, 1, 1, 2, 3, 3, 2, 6,
        1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 6, 1, 1,
        4, 4, 4, 4, 4, 4, 2, 2,
    )
    candidate_suffix = (
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        4,
        3, 3, 3, 3,
        1,
        2, 2, 2, 2,
        6,
        3, 3, 3, 3,
        4, 4, 4,
        6,
        4, 4,
        6,
        1,
    )
    roots = (("candidate", transfer + candidate_suffix),)
    rng = random.Random(4805)
    for name, prefix in roots:
        root = p.replay(env, prefix)
        initial_eights = eights(root)
        candidate_reported = False
        for trial in range(700):
            branch = root.clone()
            path = []
            last_action = None
            for _ in range(28):
                choices = tuple(
                    action
                    for action in (1, 2, 3, 4, 6)
                    if action != last_action
                )
                action = rng.choice(choices)
                count = 1 if action == 6 else rng.randint(1, 6)
                for _ in range(count):
                    branch.step(action)
                    path.append(action)
                    if branch.levels_completed > env.levels_completed:
                        print(
                            "FOUND",
                            name,
                            list(prefix) + path,
                            branch.levels_completed,
                            eights(branch),
                        )
                        return
                    if has_898(branch) and not candidate_reported:
                        print("CANDIDATE", name, list(prefix) + path)
                        candidate_reported = True
                last_action = action
            if trial and trial % 100 == 0:
                print("PROGRESS", name, trial)
    print("NONE")


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
