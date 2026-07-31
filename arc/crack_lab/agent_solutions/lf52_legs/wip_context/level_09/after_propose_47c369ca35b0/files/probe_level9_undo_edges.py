"""Check whether action 7 exactly reverses level-9 search edges."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level9_shortest_suffix import (
    move_actions,
    physical_key,
    play,
    visible_moves,
)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        entry = json.load(candidate_file)[:28]
    for action in prefix + entry:
        play(env, action)
    base_key = physical_key(env)

    results = []
    for label, actions in [
        ("key3", [3]),
        ("key4", [4]),
        *[
            (
                f"{kind}:{source}->{destination}",
                list(move_actions(source, destination)),
            )
            for kind, source, destination in visible_moves(env.frame())
        ],
    ]:
        direct = env.clone()
        for action in actions:
            play(direct, action)
        node = env.clone()
        for action in actions:
            play(node, action)
        changed = physical_key(node) != base_key
        restore_steps = None
        for undo_count in range(1, 4):
            play(node, 7)
            if physical_key(node) == base_key:
                restore_steps = undo_count
                break
        for action in actions:
            play(node, action)
        same_child = physical_key(node) == physical_key(direct)
        play(node, 4)
        play(direct, 4)
        same_next_turn = physical_key(node) == physical_key(direct)
        results.append((
            label, changed, restore_steps, same_child, same_next_turn
        ))
    print("UNDO_EDGES", results, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
