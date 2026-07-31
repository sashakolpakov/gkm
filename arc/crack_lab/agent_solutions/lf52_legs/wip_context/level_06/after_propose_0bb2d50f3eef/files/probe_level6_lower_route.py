import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import LOCAL_MACROS, brief, click, pieces


def stage_remote(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))
    for source, destination in LOCAL_MACROS:
        click(env, source)
        click(env, destination)
    _, carriers, movable, _, _ = pieces(env.frame())
    click(env, next(iter(movable)))
    click(env, next(iter(carriers)))
    for _ in range(9):
        env.step(4)


def probe(env):
    stage_remote(env)
    for action in (1, 1, 3, 3, 3, 3, 2, 2):
        env.step(action)
    print("far lower", brief(env))
    path = P.bounded_replay_bfs(
        env,
        lambda node, _: any(
            max(source[0], destination[0]) >= 36
            for source, _, destination in brief(node)[3]
        ),
        lambda _: (1, 2, 3, 4),
        key_fn=lambda node: P.arr(node.frame())[1:].tobytes(),
        max_states=700,
        max_depth=28,
    )
    print("lower alignment path", path)
    if path is not None:
        aligned = P.replay(env, path)
        print("lower aligned", brief(aligned))
        source, _, destination = brief(aligned)[3][0]
        click(aligned, source)
        click(aligned, destination)
        print("lower crossed", brief(aligned))
        for key_action in (1, 2, 3, 4):
            key_test = aligned.clone()
            key_test.step(key_action)
            print("crossed key", key_action, brief(key_test))
        rightward = aligned.clone()
        rightward.step(4)
        print("right carrier", brief(rightward))
        bridge_return = next((
            move for move in brief(rightward)[3]
            if move[0] in pieces(rightward.frame())[2]
        ), None)
        if bridge_return is not None:
            click(rightward, bridge_return[0])
            click(rightward, bridge_return[2])
            print("bridge returned", brief(rightward))
        return
        peg_move = next(
            move for move in brief(aligned)[3]
            if move[0] in pieces(aligned.frame())[3]
        )
        click(aligned, peg_move[0])
        click(aligned, peg_move[2])
        print("peg entered", brief(aligned))
        reverse = (peg_move[2], peg_move[0])
        for index in range(10):
            forward = [
                move for move in brief(aligned)[3]
                if max(move[0][0], move[2][0]) >= 36
                and (move[0], move[2]) != reverse
            ]
            if not forward:
                break
            source, _, destination = forward[0]
            click(aligned, source)
            click(aligned, destination)
            reverse = (destination, source)
            print("inchworm", index, brief(aligned))
        next_path = P.bounded_replay_bfs(
            aligned,
            lambda node, search_path: bool(search_path) and any(
                max(next_source[0], next_destination[0]) >= 36
                for next_source, _, next_destination in brief(node)[3]
            ),
            lambda _: (1, 2, 3, 4),
            key_fn=lambda node: P.arr(node.frame())[1:].tobytes(),
            max_states=700,
            max_depth=28,
        )
        print("next lower alignment path", next_path)
        if next_path is not None:
            print("next lower aligned", brief(P.replay(aligned, next_path)))
    rail = env.clone()
    for horizontal in range(7):
        print("rail", horizontal, brief(rail))
        for vertical_action in (2, 1):
            node = rail.clone()
            trace = []
            previous = brief(node)
            for count in range(1, 6):
                node.step(vertical_action)
                current = brief(node)
                if current == previous:
                    break
                trace.append((count, current))
                previous = current
            if trace:
                print("vertical", horizontal, vertical_action, trace)
        rail.step(4)


if __name__ == "__main__":
    A.run_program("lf52", probe)
