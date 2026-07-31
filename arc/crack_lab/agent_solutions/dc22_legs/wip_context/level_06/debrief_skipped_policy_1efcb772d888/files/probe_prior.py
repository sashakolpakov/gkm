"""Raw-frame comparison probes for mechanics already encoded in local legs."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import toggle_control, walk_segments
from perception import connected_components, frame_delta
from players import play_level_1, play_level_2, play_level_3, play_level_4


def avatar_position(frame):
    blobs = connected_components(frame, colors=(14,), min_area=4)
    return blobs[0].top_left if blobs else None


def compact_delta(before, after):
    first, second = np.asarray(before), np.asarray(after)
    changed = np.argwhere(first[:62] != second[:62])
    transitions = {}
    for row, col in changed:
        pair = (int(first[row, col]), int(second[row, col]))
        transitions[pair] = transitions.get(pair, 0) + 1
    bbox = None
    if len(changed):
        bbox = (
            int(changed[:, 0].min()),
            int(changed[:, 1].min()),
            int(changed[:, 0].max()),
            int(changed[:, 1].max()),
        )
    return len(changed), bbox, tuple(sorted(transitions.items()))


def probe(env):
    play_level_1(env)
    play_level_2(env)
    print("LEVEL", env.levels_completed, "AVATAR", avatar_position(env.frame()))

    remote = env.clone()
    before = np.asarray(remote.frame()).copy()
    remote.step(6, 51, 36)
    print("REMOTE_TELEPORT_CONTROL", frame_delta(before, remote.frame()))

    endpoint = env.clone()
    walk_segments(endpoint, ((3, 1),))
    toggle_control(endpoint, (51, 18))
    toggle_control(endpoint, (51, 27))
    walk_segments(endpoint, ((3, 3), (2, 3), (3, 1)))
    toggle_control(endpoint, (51, 18))
    walk_segments(endpoint, ((3, 4),))
    toggle_control(endpoint, (51, 27))
    walk_segments(endpoint, ((3, 3),))
    toggle_control(endpoint, (51, 27))
    walk_segments(endpoint, ((1, 4),))
    print("ENDPOINT_BEFORE", avatar_position(endpoint.frame()))
    before = np.asarray(endpoint.frame()).copy()
    toggle_control(endpoint, (51, 36))
    print(
        "ENDPOINT_AFTER",
        avatar_position(endpoint.frame()),
        "LEVEL", endpoint.levels_completed,
        "DELTA", frame_delta(before, endpoint.frame()),
    )
    walk_segments(endpoint, ((1, 2), (4, 2)))
    walk_segments(endpoint, ((1, 1), (4, 1)))
    toggle_control(endpoint, (51, 18))
    before = np.asarray(endpoint.frame()).copy()
    toggle_control(endpoint, (51, 45))
    print(
        "REVEALED_EXIT_CONTROL",
        "AVATAR", avatar_position(endpoint.frame()),
        "DELTA", frame_delta(before, endpoint.frame()),
    )
    before_level = endpoint.levels_completed
    walk_segments(endpoint, ((1, 3), (4, 6), (2, 2), (4, 1)))
    print(
        "EXIT_PATH",
        "AVATAR", avatar_position(endpoint.frame()),
        "LEVEL", before_level, "->", endpoint.levels_completed,
    )

    play_level_3(env)
    play_level_4(env)
    print("LEVEL_5_ENTRY", env.levels_completed, avatar_position(env.frame()))
    controls = {
        "builder": (46, 22),
        "crossing": (56, 22),
        "bridge": (52, 42),
        "teleport": (52, 46),
        "platform_up": (50, 28),
        "platform_right": (56, 28),
        "platform_rotate": (51, 35),
        "assembly_left": (44, 30),
        "assembly_down": (60, 30),
        "assembly_right": (54, 30),
        "assembly_up": (50, 30),
        "lower_rotate": (52, 14),
    }
    for name, point in controls.items():
        clone = env.clone()
        before = np.asarray(clone.frame()).copy()
        clone.step(6, *point)
        print(name, compact_delta(before, clone.frame()))


if __name__ == "__main__":
    arena.run_program("dc22", probe)
