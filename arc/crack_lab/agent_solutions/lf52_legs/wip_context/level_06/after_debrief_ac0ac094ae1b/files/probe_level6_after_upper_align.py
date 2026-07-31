import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import brief, pieces, stage_upper


def probe(env):
    stage_upper(env)
    print("upper exit", brief(env))
    path = P.bounded_replay_bfs(
        env,
        lambda node, _: any(
            source in pieces(node.frame())[2]
            for source, _, _ in brief(node)[3]
        ),
        lambda _: (1, 2, 3, 4),
        key_fn=lambda node: P.arr(node.frame())[1:].tobytes(),
        max_states=900,
        max_depth=32,
    )
    print("bridge alignment", path)
    if path is not None:
        print("bridge aligned", brief(P.replay(env, path)))


A.run_program("lf52", probe)
