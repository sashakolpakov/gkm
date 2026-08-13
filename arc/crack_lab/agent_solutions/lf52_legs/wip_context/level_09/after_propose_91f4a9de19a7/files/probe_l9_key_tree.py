"""Path-aware key tree for hidden synchronized carriers on level 9."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def frame_key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    horizontal = {frame_key(env)}
    nodes = [(env.clone(), ())]
    for _ in range(6):
        children = []
        for node, path in nodes:
            for action in (3, 4):
                child = node.clone()
                safe_step(child, action)
                horizontal.add(frame_key(child))
                children.append((child, path + (action,)))
        nodes = children

    visible = {frame_key(env): ((), compact(env.frame()))}
    nodes = [(env.clone(), ())]
    reward = None
    for depth in range(1, 7):
        children = []
        for node, path in nodes:
            for action in (1, 2, 3, 4):
                child = node.clone()
                safe_step(child, action)
                child_path = path + (action,)
                child_key = frame_key(child)
                visible.setdefault(child_key, (child_path, compact(child.frame())))
                if child.levels_completed > env.levels_completed:
                    reward = child_path
                    break
                children.append((child, child_path))
            if reward is not None:
                break
        print("key_tree_depth", depth, len(children), len(visible), flush=True)
        if reward is not None:
            break
        nodes = children
    novel = [
        value for key, value in visible.items() if key not in horizontal
    ]
    print("key_tree_result", reward, len(horizontal), len(visible), novel,
          flush=True)


arena.run_program("lf52", probe)
