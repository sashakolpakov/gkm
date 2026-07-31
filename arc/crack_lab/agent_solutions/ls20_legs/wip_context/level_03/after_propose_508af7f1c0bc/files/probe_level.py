import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import players
from perception import bounded_replay_bfs, connected_components, replay


def key(node):
    blobs = connected_components(node.frame(), colors={9, 11}, min_area=2)
    avatar = next((b.bbox for b in blobs if b.color == 9 and b.area == 15), None)
    pickups = tuple(b.bbox for b in blobs if b.color == 11 and b.bbox[0] < 55)
    return int(node.levels_completed), avatar, pickups


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    base = int(env.levels_completed)
    started = time.monotonic()
    path = bounded_replay_bfs(
        env,
        lambda node, _: int(node.levels_completed) > base,
        action_fn=lambda _: env.actions,
        key_fn=key,
        max_states=800,
        max_depth=80,
    )
    node = replay(env, path or [])
    print(path, len(path or []), int(node.levels_completed), round(time.monotonic() - started, 2))


arena.run_program("ls20", probe)
