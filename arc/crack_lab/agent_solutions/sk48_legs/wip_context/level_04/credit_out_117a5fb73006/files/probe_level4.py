"""Compact clean-room probes for sk48 level 4."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import action_deltas, bounded_bfs, color_counts, connected_components, frame_delta
from legs import reverse_row_train
from players import play_level_1, play_level_2, play_level_3


def blobs(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(env.frame(), min_area=4)
        if b.color != 1
    ]


def tokens(env):
    return [
        (b.color, b.bbox)
        for b in connected_components(env.frame(), colors=(8, 9, 12, 14), min_area=4)
        if b.bbox[0] < 53
    ]


def physical_key(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            env.frame(), colors=(0, 6, 8, 9, 12, 14), min_area=4
        )
        if b.bbox[0] < 53
    )


def ordered_target(env, _path):
    state = tokens(env)
    if len(state) != 4 or len({bbox[0] for _, bbox in state}) != 1:
        return False
    return [color for color, _ in sorted(state, key=lambda item: item[1][1])] == [8, 12, 9, 14]


def advance_to_level4(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)


def inspect(env):
    advance_to_level4(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    print("BLOBS", blobs(env))
    print("DELTAS", {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env, env.actions).items()
    })
    for action in env.actions:
        node = env.clone()
        before = node.frame()
        node.step(action)
        print("ACTION", action, "D", frame_delta(before, node.frame()),
              "B", blobs(node), "L", node.levels_completed)

    for action in (6, 7):
        for label, x, y in (
            ("avatar", 7, 44), ("token9", 13, 44), ("token14", 19, 44),
            ("token8", 25, 44), ("token12", 31, 44),
            ("goal8", 19, 58), ("goal12", 25, 58),
            ("top10", 25, 2), ("top11", 37, 2),
        ):
            node = env.clone()
            before = node.frame()
            try:
                node.step(action, x, y)
                delta = frame_delta(before, node.frame())
                print("COORD", action, label, delta["count"], delta["bbox"],
                      node.levels_completed, tokens(node))
            except Exception as exc:
                print("COORD_ERR", action, label, type(exc).__name__)
    def split_rows(node, _path):
        state = tokens(node)
        return len(state) == 4 and len({bbox[0] for _, bbox in state}) > 1

    path = bounded_bfs(
        env,
        split_rows,
        actions=(1, 2, 3, 4),
        key_fn=physical_key,
        max_states=1500,
        max_depth=16,
    )
    print("SPLIT_PATH", path)
    if path:
        node = env.clone()
        for action in path:
            node.step(action)
        print("SPLIT_RESULT", node.levels_completed, tokens(node), physical_key(node))
        finish = bounded_bfs(
            node,
            lambda child, child_path: (
                child.levels_completed > 3 or ordered_target(child, child_path)
            ),
            actions=(1, 2, 3, 4),
            key_fn=physical_key,
            max_states=6000,
            max_depth=30,
        )
        print("FINISH_PATH", finish)
        if finish:
            for action in finish:
                node.step(action)
            print("FINISH_RESULT", node.levels_completed, tokens(node))
    return

    tested = 0
    for approach in range(1, 6):
        for wall_reach in range(3, 8):
            for compact in range(2, 8):
                for final in range(1, 8):
                    node = env.clone()
                    reverse_row_train(
                        node, approach_lanes=approach,
                        stages=((2, wall_reach, compact),),
                        final_extension=final,
                    )
                    tested += 1
                    if node.levels_completed > 3:
                        print("STRUCTURED_WIN", approach, 2, wall_reach, compact, final,
                              "TESTED", tested)
                        return
    print("STRUCTURED_NONE", tested)


if __name__ == "__main__":
    levels, path, err = A.run_program("sk48", inspect)
    print("END", levels, len(path), err)
