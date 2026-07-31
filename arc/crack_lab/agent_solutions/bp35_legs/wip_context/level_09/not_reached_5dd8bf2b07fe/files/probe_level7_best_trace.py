import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, band_shift
from perception import arr
from probe_level7_no_control import PREFIX, advance, avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


ROUTE = [
    (3,), (6, 3, 27), (6, 3, 39), (4,),
    (6, 33, 45), (6, 3, 51), (3,),
]


def symbolic(frame):
    chars = {3: ".", 5: "#", 8: "g", 9: "A", 10: "h", 11: "a",
             12: "s", 14: "b", 15: "x"}
    rows = []
    for i, y in enumerate(ROW_ANCHORS):
        row = []
        for j in range(8):
            color, area = _cell_shape(frame, i, j)
            ch = chars.get(color, "?")
            if color in (12, 14, 15):
                ch = ch.upper() if area > 8 else ch
            row.append(ch)
        rows.append("".join(row))
    return "/".join(rows)


def nearby_shapes(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return ()
    ai, aj = avatar
    return tuple(
        (i, j, *_cell_shape(frame, i, j))
        for i in range(max(0, ai - 3), min(10, ai + 4))
        for j in range(max(0, aj - 2), min(8, aj + 3))
        if _cell_shape(frame, i, j)[0] not in (3, 5, 8, 9, 11)
    )


def report(label, node, gain):
    frame = node.frame()
    print(
        label, "gain", gain, "avatar", avatar_cell(frame),
        "controls", controls(frame), "near", nearby_shapes(frame),
    )
    print(symbolic(frame), flush=True)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    node = env.clone()
    total = advance(node, PREFIX)
    report("ROOT", node, total)
    for k, action in enumerate(ROUTE, 1):
        before = arr(node.frame()).copy()
        node.step(*action)
        if node.terminal():
            print("TERMINAL", k, action, node.levels_completed)
            return
        total += band_shift(before, node.frame())
        report(f"{k}:{action}", node, total)
    for action in [(4,), (6, 3, 45), (3,)]:
        child = node.clone()
        before = arr(child.frame()).copy()
        child.step(*action)
        gain = total if child.terminal() else total + band_shift(before, child.frame())
        report(f"TRIAL:{action}", child, gain)
    chain = node
    for action in [(4,), (6, 3, 39)]:
        before = arr(chain.frame()).copy()
        chain.step(*action)
        if chain.terminal():
            print("CHAIN_TERMINAL", action, chain.levels_completed)
            break
        total += band_shift(before, chain.frame())
        report(f"CHAIN:{action}", chain, total)
    if not chain.terminal():
        for action in [(3,), (4,), (6, 3, 27), (6, 3, 33),
                       (6, 3, 45), (6, 3, 51), (6, 3, 57)]:
            child = chain.clone()
            before = arr(child.frame()).copy()
            child.step(*action)
            gain = total if child.terminal() else total + band_shift(
                before, child.frame()
            )
            report(f"CHAIN_TRIAL:{action}", child, gain)
        patterns = [
            ((3,),) * n for n in range(2, 9)
        ] + [
            ((4,),) * n for n in range(2, 9)
        ] + [
            ((3,), (4,)) * n for n in range(1, 5)
        ] + [
            ((4,), (3,)) * n for n in range(1, 5)
        ]
        for pattern in patterns:
            child = chain.clone()
            gain = total + advance(child, pattern)
            print(
                "PATTERN", pattern, "level", child.levels_completed,
                "terminal", child.terminal(), "gain", gain,
                "avatar", None if child.terminal() else avatar_cell(child.frame()),
                "controls", () if child.terminal() else controls(child.frame()),
            )


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
