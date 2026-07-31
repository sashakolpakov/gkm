"""Collect visible level-2 pegs as dense subgoals, then search for reward."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def peg_centers(frame):
    return tuple(
        (int(round(b.centroid[0])), int(round(b.centroid[1])))
        for b in perception.connected_components(frame, colors=(8,), min_area=4)
    )


def covered(node, peg):
    r, c = peg
    return int(perception.arr(node.frame())[r, c]) == 0


def avatar_key(node):
    return (perception.arr(node.frame())[1:] == 0).tobytes()


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    print("pegs", list(enumerate(pegs)))
    visited = set()
    total = []
    for phase in range(len(pegs)):
        node = perception.replay(root, total)
        before = len(visited)
        segment = perception.bounded_bfs(
            node,
            lambda child, _: child.levels_completed > 1
            or any(i not in visited and covered(child, peg) for i, peg in enumerate(pegs)),
            actions=(1, 2, 3, 4, 5),
            key_fn=avatar_key,
            max_states=6000,
            max_depth=35,
        )
        if segment is None:
            print("stuck", phase, "visited", sorted(visited), "path", total)
            return
        total.extend(segment)
        node = perception.replay(root, total)
        newly = {
            i
            for i, peg in enumerate(pegs)
            if i not in visited and covered(node, peg)
        }
        visited |= newly
        print(
            "phase",
            phase,
            "segment",
            segment,
            "new",
            [(i, pegs[i]) for i in sorted(newly)],
            "visited",
            len(visited),
            "moves",
            len(total),
            "level",
            node.levels_completed,
        )
        if node.levels_completed > 1:
            print("solved", len(total), total)
            return
    node = perception.replay(root, total)
    finish = perception.bounded_bfs(
        node,
        perception.level_goal(1),
        actions=(1, 2, 3, 4, 5),
        key_fn=avatar_key,
        max_states=12000,
        max_depth=max(0, 94 - len(total)),
    )
    print("finish", finish)
    if finish is not None:
        total.extend(finish)
        print("solved", len(total), total)


if __name__ == "__main__":
    arena.run_program("cn04", probe)
