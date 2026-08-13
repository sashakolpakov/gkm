"""Delete key actions when the complete level suffix still succeeds."""

import json

import gkm_try

from perception import safe_step


LEVEL_LENGTHS = (8, 34, 45, 50, 89, 90, 145, 68)


def reaches(entry, actions, target_level):
    node = entry.clone()
    used = []
    for action in actions:
        safe_step(node, action)
        used.append(action)
        if int(node.levels_completed) >= target_level:
            return True, tuple(used)
    return False, tuple(used)


def probe(env):
    with open("optimized_prefix_l4_l6_candidate.json") as candidate_file:
        full = json.load(candidate_file)["final_path"]
    offset = 0
    accepted_prefix = []
    for level, original_length in enumerate(LEVEL_LENGTHS, 1):
        segment = list(full[offset:offset + original_length])
        offset += original_length
        entry = env.clone()
        before = len(segment)
        changed = True
        trials = 0
        if level >= 4:
            while changed:
                changed = False
                for index, action in enumerate(tuple(segment)):
                    if not isinstance(action, int) or action not in (1, 2, 3, 4):
                        continue
                    trial = segment[:index] + segment[index + 1:]
                    success, used = reaches(entry, trial, level)
                    trials += 1
                    if success:
                        segment = list(used)
                        changed = True
                        print("GREEDY_DELETE", level, index, action, len(segment), flush=True)
                        break
        success, used = reaches(entry, segment, level)
        segment = list(used)
        print("GREEDY_LEVEL", level, before, len(segment), trials, success, flush=True)
        if not success:
            return
        for action in segment:
            safe_step(env, action)
            accepted_prefix.append(action)
            if int(env.levels_completed) >= level:
                break
    print("GREEDY_RESULT", int(env.levels_completed), len(accepted_prefix), tuple(accepted_prefix), flush=True)


gkm_try.A.run_program("lf52", probe)
