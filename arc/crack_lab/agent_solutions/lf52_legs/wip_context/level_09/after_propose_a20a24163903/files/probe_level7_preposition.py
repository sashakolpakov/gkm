"""Test carrier prepositioning across level 7's final viewport transition."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level7_crossed_keys import compact, legal_moves, variants


LEVEL_START = 331
LEVEL_END = 476
FIRST_LOAD = ("P", (42, 30), (30, 30))


def groups(segment):
    result = []; index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        result.append((tuple(keys), tuple(pair)))
    return result


def apply_pair(node, pair):
    for action in pair:
        safe_step(node, action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:LEVEL_START]:
        safe_step(env, action)
    level_groups = groups(path[LEVEL_START:LEVEL_END])
    for keys, pair in level_groups[:16]:
        for action in keys:
            safe_step(env, action)
        apply_pair(env, pair)
    pre_exit = env.clone()

    first = (2, 3, 3, 3, 2, 2)
    prepaths = {()}
    for variant in variants(first):
        prepaths.update(variant[:cut] for cut in range(1, min(6, len(variant) + 1)))

    alignments = {()}
    for variant in variants(first):
        alignments.update(variant[:cut] for cut in range(1, min(7, len(variant) + 1)))

    results = []
    for prepath in sorted(prepaths, key=lambda value: (len(value), value)):
        node = pre_exit.clone()
        for action in prepath:
            safe_step(node, action)
        apply_pair(node, level_groups[16][1])
        for group in level_groups[17:20]:
            apply_pair(node, group[1])
        for alignment in sorted(alignments, key=lambda value: (len(value), value)):
            if len(prepath) + len(alignment) > 6:
                continue
            candidate = node.clone()
            for action in alignment:
                safe_step(candidate, action)
            if FIRST_LOAD in legal_moves(candidate):
                results.append((len(prepath) + len(alignment), prepath, alignment, compact(candidate)))
                break

    results.sort(key=lambda item: (item[0], len(item[1]), item[1], item[2]))
    print("L7_PREPOSITION", len(prepaths), len(results))
    for result in results[:30]:
        print("L7_PREPOSITION_RESULT", result)


gkm_try.A.run_program("lf52", probe)
