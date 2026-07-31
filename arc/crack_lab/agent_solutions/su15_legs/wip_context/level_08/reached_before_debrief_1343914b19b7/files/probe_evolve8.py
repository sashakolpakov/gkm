import itertools
import json
import random
import time

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


OFFSETS = (
    (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0),
    (0, 2), (1, -1), (1, 0), (1, 1),
)
CONTROLS = tuple(itertools.product(range(3), OFFSETS)) + ((3, (0, 0)),)
SEED = (
    (1, (1, 1)), (2, (1, 0)), (0, (0, 2)), (1, (1, 1)),
    (2, (0, -2)), (0, (0, 2)), (2, (1, 1)), (1, (1, 1)),
)
WAIT = (6, 56, 15)
POPULATION = 900
GENERATIONS = 12


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
    template = body_groups(root.frame())[0]
    template_center = center(template)
    offsets = tuple(
        (row - template_center[0], col - template_center[1])
        for row, col in template
    )
    valid_centers = tuple(
        tuple(
            (row, col)
            for row in range(10, 64)
            for col in range(64)
            if all((row + dr, col + dc) in mask for dr, dc in offsets)
        )
        for mask in masks
    )

    def overlap(groups):
        if len(groups) != 3:
            return -1
        pixels = {point for group in groups for point in group}
        return sum(point in mask for point in pixels for mask in masks)

    def distance(groups):
        if len(groups) != 3:
            return 999
        centers = tuple(center(group) for group in groups)
        return min(
            sum(
                min(
                    max(abs(row - target_row), abs(col - target_col))
                    for target_row, target_col in valid_centers[target]
                )
                for (row, col), target in zip(centers, order)
            )
            for order in itertools.permutations(range(3))
        )

    rng = random.Random(8)
    started = time.monotonic()
    clone_steps = 0
    cache = {}

    def evaluate(genome):
        nonlocal clone_steps
        if genome in cache:
            return cache[genome]
        node = root.clone()
        actions = []
        trace = [overlap(body_groups(node.frame()))]
        prior_groups = body_groups(node.frame())
        groups = prior_groups
        for label, offset in genome:
            if len(groups) != 3:
                break
            if label == 3:
                action = WAIT
            else:
                group = labeled(groups)[label]
                row, col = center(group)
                action = (6, col + offset[1], row + offset[0])
            try:
                node.step(*action)
            except Exception:
                break
            actions.append(action)
            clone_steps += 1
            delay = clone_steps / 300 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if int(node.levels_completed) > start_level:
                print(
                    "FOUND", PREFIX + actions, "GENOME", genome,
                    "steps", clone_steps, flush=True,
                )
                raise SystemExit
            prior_groups, groups = groups, body_groups(node.frame())
            trace.append(overlap(groups))
            if node.terminal():
                break
        score = (
            max(trace),
            trace[-1],
            -distance(groups),
            len(actions) == len(genome),
            -len(genome),
        )
        signature = (
            groups, prior_groups, tuple(trace[-3:]), genome[-2:]
        )
        result = score, signature, tuple(actions), tuple(trace)
        cache[genome] = result
        return result

    def mutate(genome):
        child = list(genome)
        for _ in range(rng.randint(1, 3)):
            operation = rng.choice(("replace", "insert", "delete", "swap"))
            if operation == "replace" and child:
                child[rng.randrange(len(child))] = rng.choice(CONTROLS)
            elif operation == "insert" and len(child) < 16:
                child.insert(rng.randrange(len(child) + 1), rng.choice(CONTROLS))
            elif operation == "delete" and len(child) > 3:
                del child[rng.randrange(len(child))]
            elif operation == "swap" and len(child) > 1:
                index = rng.randrange(len(child) - 1)
                child[index], child[index + 1] = child[index + 1], child[index]
        return tuple(child)

    population = {SEED}
    while len(population) < POPULATION:
        population.add(mutate(SEED))
    for generation in range(GENERATIONS):
        ranked = sorted(
            ((evaluate(genome)[0], genome) for genome in population),
            reverse=True,
        )
        elites = []
        signatures = set()
        for score, genome in ranked:
            signature = evaluate(genome)[1]
            if signature in signatures:
                continue
            signatures.add(signature)
            elites.append(genome)
            if len(elites) >= 140:
                break
        best_genome = ranked[0][1]
        print(
            "GEN", generation, "best", ranked[0][0],
            "trace", evaluate(best_genome)[3],
            "genome", best_genome, "elites", len(elites),
            "cache", len(cache), "steps", clone_steps, flush=True,
        )
        population = set(elites)
        while len(population) < POPULATION - 80:
            population.add(mutate(rng.choice(elites)))
        while len(population) < POPULATION:
            length = rng.randint(4, 15)
            population.add(tuple(rng.choice(CONTROLS) for _ in range(length)))
    print("NO_PATH", len(cache), clone_steps, flush=True)


result = H.A.run_program("su15", inspect)
print("DONE", result[0], result[2])
