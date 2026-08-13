"""Map bounded off-screen carrier turns by forcing a visible return."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)

EMPTY_OPENING = FIRST_RELAY[:-2]


def carrier(frame):
    borders = [
        blob for blob in connected_components(frame, colors=(11,))
        if blob.area >= 6
    ]
    if not borders:
        return None
    blob = borders[0]
    return blob.bbox, (blob.bbox[0] + 1, blob.bbox[1] + 1)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    empty_mode = os.environ.get("L9_EMPTY") == "1"
    for source, destination in (EMPTY_OPENING if empty_mode else FIRST_RELAY):
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))
    if not empty_mode:
        safe_step(env, (6, 23, 37))
        safe_step(env, (6, 11, 37))

    observations = {}
    selected_lead = os.environ.get("OPT_LEAD")
    leads = range(3, 9) if empty_mode else (7, 8, 9)
    if selected_lead is not None:
        leads = (int(selected_lead),)
    selected_turn = os.environ.get("OPT_TURN")
    turns = (int(selected_turn),) if selected_turn is not None else (1, 2)
    max_vertical = int(os.environ.get("OPT_VERTICAL", "12"))
    max_left = int(os.environ.get("OPT_LEFT", "16"))
    for lead in leads:
        exited = env.clone()
        for _ in range(lead):
            safe_step(exited, 4)
        for turn in turns:
            turned = exited.clone()
            for vertical in range(1, max_vertical + 1):
                safe_step(turned, turn)
                returning = turned.clone()
                for left in range(1, max_left + 1):
                    safe_step(returning, 3)
                    visible = carrier(returning.frame())
                    if visible is None:
                        continue
                    path = (4,) * lead + (turn,) * vertical + (3,) * left
                    key = visible
                    old = observations.get(key)
                    if old is None or len(path) < len(old):
                        observations[key] = path
                    break
    print("hidden_turn_returns", [
        (visible, path) for visible, path in sorted(observations.items())
    ], flush=True)


arena.run_program("lf52", probe)
