"""Search the final L7 board with public undo and corrected bridge moves."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import arr, connected_components, frame_delta, safe_step


SUFFIX = (
    (6, 19, 37), (6, 31, 37),
    (6, 7, 43), (6, 19, 43),
    (6, 19, 43), (6, 31, 43),
    2, 3, 3, 3, 2, 2,
    (6, 31, 43), (6, 31, 31),
    1, 1, 4, 4, 4, 2, 4, 2,
    (6, 55, 25), (6, 55, 37),
    1, 3, 1, 3, 3, 2, 4, 2, 2,
    (6, 55, 43), (6, 55, 31),
)


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def units(path):
    result = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            result.append((path[index],))
            index += 1
        else:
            result.append((path[index], path[index + 1]))
            index += 2
    return tuple(result)


def click_macros(frame):
    blobs = connected_components(frame, colors=(1, 8, 12, 14, 15))
    cells = {
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 8, 12, 14)
    }
    cells |= {
        (blob.bbox[0] - 1, blob.bbox[1] - 1) for blob in blobs
        if blob.color in (1, 12) and blob.size == (2, 2)
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    movable = {
        blob.top_left for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    occupied = pegs | movable
    candidates = []
    for kind, sources in (("peg", pegs), ("bridge", movable)):
        for source in sorted(sources):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (midpoint not in occupied | fixed
                        or destination not in cells
                        or destination in occupied):
                    continue
                capture = kind == "peg" and midpoint in pegs
                macro = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                candidates.append((0 if capture else 1, macro))
    return tuple(macro for _, macro in sorted(set(candidates),
                                               key=lambda item: item[0]))


def macros(frame):
    return click_macros(frame) + tuple((action,) for action in (4, 3, 2, 1))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    for action in campaign[:440]:
        safe_step(env, action)
    node = env.clone()
    base_level = int(node.levels_completed)

    preferred = {}
    if os.environ.get("OPT_NO_REFERENCE") != "1":
        reference = node.clone()
        for macro in units(SUFFIX):
            preferred[state_key(reference)] = macro
            for action in macro:
                safe_step(reference, action)
        print("final_reference", base_level, len(SUFFIX),
              int(reference.levels_completed), flush=True)

    bound = int(os.environ.get("OPT_COST", "34"))
    max_expanded = int(os.environ.get("OPT_STATES", "60"))
    best = {state_key(node): 0}
    path = []
    expanded = 0
    failed = None
    solution = None
    root_option = int(os.environ.get("OPT_ROOT", "-1"))
    print("final_root_options", macros(node.frame()), flush=True)
    root_seed = node.clone()

    def visit(cost):
        nonlocal expanded, failed, solution, node
        if (solution is not None or failed is not None
                or expanded >= max_expanded):
            return
        expanded += 1
        before = state_key(node)
        options = list(macros(node.frame()))
        wanted = preferred.get(before)
        if wanted in options:
            options.remove(wanted)
            options.insert(0, wanted)
        if not path and root_option >= 0:
            options = options[root_option:root_option + 1]
        for macro in options:
            child_cost = cost + len(macro)
            if child_cost > bound:
                continue
            before_frame = arr(node.frame()).copy()
            changed_count = 0
            valid = True
            for action in macro:
                step_before = state_key(node)
                safe_step(node, action)
                if state_key(node) == step_before:
                    valid = False
                    break
                changed_count += 1
            after = state_key(node)
            if valid and int(node.levels_completed) > base_level:
                solution = tuple(path) + macro
                return
            if valid and after != before and child_cost < best.get(
                    after, 10 ** 9):
                best[after] = child_cost
                path.extend(macro)
                visit(child_cost)
                del path[-len(macro):]
            if solution is not None or failed is not None:
                return
            for _ in range(changed_count):
                safe_step(node, 7)
            if state_key(node) != before:
                # Wrapped-frontier transitions intentionally clear public
                # undo history.  Rebuild the current search node from the
                # immutable public root and its recorded action path.
                node = root_seed.clone()
                for action in path:
                    safe_step(node, action)
            if state_key(node) != before:
                after_frame = arr(node.frame()).copy()
                after_frame[0] = before_frame[0]
                failed = (
                    cost, tuple(path), macro, changed_count,
                    frame_delta(before_frame, after_frame),
                    _bridge_carrier_state(before_frame),
                    _bridge_carrier_state(after_frame),
                )
                return
            if expanded >= max_expanded:
                return

    visit(0)
    print("final_search", bound, expanded, len(best), failed,
          None if solution is None else len(solution), solution, flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
