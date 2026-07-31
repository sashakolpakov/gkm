import itertools
import json
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


OFFSETS = (
    (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0),
    (0, 2), (1, -1), (1, 0), (1, 1),
)
SEED = (
    (1, (1, 1)), (2, (1, 0)), (0, (0, 2)), (1, (1, 1)),
    (2, (0, -2)), (0, (0, 2)), (2, (1, 1)), (1, (1, 1)),
)
WAIT = (6, 56, 15)


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def labeled(groups):
    groups = list(groups)
    top = min(groups, key=lambda group: center(group)[0])
    groups.remove(top)
    left = min(groups, key=lambda group: center(group)[1])
    groups.remove(left)
    return top, groups[0], left


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    masks = tuple(
        {
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        }
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )[1:]
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    variants = {SEED}
    controls = tuple(itertools.product(range(3), OFFSETS)) + ((3, (0, 0)),)
    for index in range(len(SEED)):
        variants.add(SEED[:index] + SEED[index + 1:])
        for control in controls:
            variants.add(SEED[:index] + (control,) + SEED[index + 1:])
    for index in range(len(SEED) + 1):
        for control in controls:
            variants.add(SEED[:index] + (control,) + SEED[index:])
    for index in range(len(SEED) - 1):
        swapped = list(SEED)
        swapped[index], swapped[index + 1] = (
            swapped[index + 1], swapped[index]
        )
        variants.add(tuple(swapped))

    def score(groups):
        if len(groups) != 3:
            return -1
        pixels = {point for group in groups for point in group}
        return sum(point in mask for point in pixels for mask in masks)

    started = time.monotonic()
    clone_steps = 0
    ranked = []
    for genome in variants:
        node = root.clone()
        actions = []
        best = score(body_groups(node.frame()))
        valid = True
        for label, offset in genome:
            groups = body_groups(node.frame())
            if len(groups) != 3:
                valid = False
                break
            if label == 3:
                action = WAIT
            else:
                group = labeled(groups)[label]
                row, col = center(group)
                action = (6, col + offset[1], row + offset[0])
            node.step(*action)
            actions.append(action)
            clone_steps += 1
            delay = clone_steps / 300 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if int(node.levels_completed) > start_level:
                print("FOUND", PREFIX + actions, "GENOME", genome, flush=True)
                return
            best = max(best, score(body_groups(node.frame())))
        final = score(body_groups(node.frame())) if valid else -1
        ranked.append(((final, best), genome, actions))
    ranked.sort(reverse=True)
    print("TESTED", len(variants), "steps", clone_steps, flush=True)
    elite = {
        genome for (final, _), genome, _ in ranked if final >= 23
    }
    print("ELITE23", len(elite), flush=True)
    extensions = {
        genome + (control,)
        for genome in elite
        for control in controls
    }
    extension_ranked = []
    for genome in extensions:
        node = root.clone()
        actions = []
        best = score(body_groups(node.frame()))
        valid = True
        for label, offset in genome:
            groups = body_groups(node.frame())
            if len(groups) != 3:
                valid = False
                break
            if label == 3:
                action = WAIT
            else:
                group = labeled(groups)[label]
                row, col = center(group)
                action = (6, col + offset[1], row + offset[0])
            node.step(*action)
            actions.append(action)
            clone_steps += 1
            delay = clone_steps / 300 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if int(node.levels_completed) > start_level:
                print("FOUND", PREFIX + actions, "GENOME", genome, flush=True)
                return
            best = max(best, score(body_groups(node.frame())))
        final = score(body_groups(node.frame())) if valid else -1
        extension_ranked.append(((final, best), genome, actions))
    extension_ranked.sort(reverse=True)
    print(
        "EXTENSIONS", len(extensions), "steps", clone_steps,
        "best", extension_ranked[:5], flush=True,
    )
    for item in ranked[:20]:
        print("BEST", item, flush=True)


A.run_program("su15", inspect)
