import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def find_exposure(node, max_padding=9, max_pump=14):
    reward_path, reach = fast_reach(node)
    if reward_path is not None:
        return reward_path, None, None
    base_fronts = _special_frontier(reach, node.frame())
    if not base_fronts:
        return None, None, None
    base_positions = {p for p, _ in base_fronts}
    bottom, bottom_walk = max(base_fronts, key=lambda item: len(item[1]))
    candidates = []
    for padding in range(max_padding + 1):
        macro = [2, 1] * padding + bottom_walk + [5]
        for pump in range(max_pump + 1):
            wait = [2, 1] * (pump // 2) + ([2] if pump % 2 else [])
            candidates.append((len(macro) + pump, padding, pump, macro + wait))
    for _, padding, pump, actions in sorted(candidates):
        child = apply(node, actions)
        reward_path, child_reach = fast_reach(child)
        if reward_path is not None:
            return actions + reward_path, None, child
        fresh = [
            (p, walk)
            for p, walk in _special_frontier(child_reach, child.frame())
            if p not in base_positions
        ]
        if fresh:
            return actions, max(fresh, key=lambda item: len(item[1])), child
    return None, None, None


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)
    node = env.clone()
    prefix = []
    for stage in range(1, 8):
        exposure, remote, child = find_exposure(node)
        print("stage", stage, "exposure", exposure,
              "remote", None if remote is None else
              (remote[0], len(remote[1])))
        if exposure is None:
            break
        prefix += exposure
        node = child
        if int(node.levels_completed) > 6 or remote is None:
            break
        pos, walk = remote
        macro = walk + [5]
        node = apply(node, macro)
        prefix += macro
        print("commit", pos, macro, "level", int(node.levels_completed))
        if int(node.levels_completed) > 6:
            break
    print("result", int(node.levels_completed), len(prefix), prefix)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
