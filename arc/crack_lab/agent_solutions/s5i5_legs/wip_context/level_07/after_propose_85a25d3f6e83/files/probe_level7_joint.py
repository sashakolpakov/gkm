import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta


CONTROLS = (
    ("A<", (6, 4, 51)), ("A>", (6, 10, 51)), ("B", (6, 17, 51)),
    ("C<", (6, 26, 51)), ("C>", (6, 32, 51)), ("D", (6, 39, 51)),
    ("F<", (6, 4, 58)), ("F>", (6, 10, 58)), ("G", (6, 17, 58)),
    ("H<", (6, 26, 58)), ("H>", (6, 32, 58)),
)
LOOKUP = dict(CONTROLS)
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D",
    "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "C<", "C<", "C<",
    "F>", "F>", "F>", "F>",
    "D", "D",
    "C>", "C>", "C>", "C>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def cells(frame, color):
    grid = arr(frame)[:42]
    return tuple((int(r), int(c)) for r, c in zip(*((grid == color).nonzero())))


def joint(frame, first, second):
    one, two = cells(frame, first), cells(frame, second)
    if not one or not two:
        return (99, 99)
    a, b = min(
        ((p, q) for p in one for q in two),
        key=lambda pair: abs(pair[0][0] - pair[1][0])
        + abs(pair[0][1] - pair[1][1]),
    )
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def summary(env):
    moving = tuple(p for p in cells(env.frame(), 13) if p not in RINGS)
    return (
        joint(env.frame(), 11, 14),
        joint(env.frame(), 14, 9),
        joint(env.frame(), 9, 12),
        moving,
    )


def bodies(env):
    return tuple(
        (blob.color, blob.bbox) for blob in connected_components(
            env.frame(), colors={9, 11, 12, 14}, min_area=4
        ) if blob.bbox[0] < 42
    )


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PREFIX:
        env.step(*LOOKUP[name])
    print("root", summary(env), bodies(env))
    for first, action in CONTROLS:
        node = env.clone()
        states = []
        for second in range(4):
            before = node.frame()
            node.step(*action)
            states.append((frame_delta(before, node.frame())["count"], summary(node)))
            if states[-1][0] == 0 or node.levels_completed > 6:
                break
        print(first, states, "level", node.levels_completed)
    for first in ("C<", "C>", "D", "F<", "F>", "G", "H<", "H>"):
        for second in ("C<", "C>", "D", "F<", "F>", "G", "H<", "H>"):
            node = env.clone()
            before = node.frame()
            node.step(*LOOKUP[first])
            node.step(*LOOKUP[second])
            changed = frame_delta(before, node.frame())["count"]
            state = summary(node)
            if changed and (state[2][0] < 17 or node.levels_completed > 6):
                print("pair", first, second, changed, state, node.levels_completed)
    print("shift_then_turn")
    for shift_name in ("C<", "C>"):
        for count in range(1, 6):
            shifted = env.clone()
            for _ in range(count):
                shifted.step(*LOOKUP[shift_name])
            for turn in ("B", "D", "G"):
                node = shifted.clone()
                before = node.frame()
                node.step(*LOOKUP[turn])
                print(
                    shift_name, count, turn,
                    frame_delta(before, node.frame())["count"], summary(node),
                )
    print("lower_bridge")
    for count in range(1, 5):
        lowered = env.clone()
        for _ in range(count):
            lowered.step(*LOOKUP["A>"])
        for control in ("F<", "F>", "G", "B"):
            node = lowered.clone()
            states = []
            for _ in range(6):
                before = node.frame()
                node.step(*LOOKUP[control])
                states.append((frame_delta(before, node.frame())["count"], summary(node)))
                if states[-1][0] == 0:
                    break
            print("A>", count, control, states)
    path = (
        ("C<",) * 4 + ("D", "F<") + ("H>",) * 5
        + ("D", "F<", "H>", "F<", "F<", "A<")
        + ("C>", "A<") * 4
    )
    node = env.clone()
    print("known_trace", summary(node))
    for name in path:
        before = node.frame()
        node.step(*LOOKUP[name])
        print(name, frame_delta(before, node.frame())["count"], summary(node))
    print("high_bridge")
    for name, action in CONTROLS:
        child = node.clone()
        states = []
        for _ in range(4):
            before = child.frame()
            child.step(*action)
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print(name, states, child.levels_completed)
    west = env.clone()
    for name in ("C<",) * 4 + ("D", "C>"):
        west.step(*LOOKUP[name])
    print("west", summary(west), bodies(west))
    for name, action in CONTROLS:
        child = west.clone()
        states = []
        for _ in range(5):
            before = child.frame()
            child.step(*action)
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print("west", name, states, child.levels_completed)
    print("west_bridge_turns")
    for shift in ("F<", "F>"):
        for count in range(1, 7):
            shifted = west.clone()
            for _ in range(count):
                shifted.step(*LOOKUP[shift])
            for turn in ("A<", "B", "G"):
                child = shifted.clone()
                before = child.frame()
                child.step(*LOOKUP[turn])
                print(
                    shift, count, turn,
                    frame_delta(before, child.frame())["count"], summary(child),
                )
    stack = env.clone()
    for name in ("C<", "D", "C<", "C<") + ("F<",) * 5 + ("G",):
        stack.step(*LOOKUP[name])
    print("stack", summary(stack), bodies(stack))
    for name, action in CONTROLS:
        child = stack.clone()
        states = []
        for _ in range(5):
            before = child.frame()
            child.step(*action)
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0:
                break
        print("stack", name, states)
    print("stack_first_turn")
    for count in range(1, 10):
        shifted = stack.clone()
        for _ in range(count):
            shifted.step(*LOOKUP["A<"])
        child = shifted.clone()
        states = []
        for _ in range(4):
            before = child.frame()
            child.step(*LOOKUP["B"])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0:
                break
        print(count, states)
    print("folded_first_turn")
    for folds in (("G", "G"), ("D", "D"), ("G", "G", "D")):
        folded = stack.clone()
        for name in folds:
            folded.step(*LOOKUP[name])
        for count in range(0, 8):
            shifted = folded.clone()
            for _ in range(count):
                shifted.step(*LOOKUP["A<"])
            child = shifted.clone()
            before = child.frame()
            child.step(*LOOKUP["B"])
            changed = frame_delta(before, child.frame())["count"]
            if changed > 1:
                print(folds, count, changed, summary(child))
    flipped = env.clone()
    for name in ("C<",) * 4 + ("D", "C>", "D", "D"):
        flipped.step(*LOOKUP[name])
    print("flipped", summary(flipped), bodies(flipped))
    for name in ("C<", "C>", "H<", "H>", "D"):
        child = flipped.clone()
        states = []
        for _ in range(7):
            before = child.frame()
            child.step(*LOOKUP[name])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print("flipped", name, states, child.levels_completed)
    for _ in range(3):
        flipped.step(*LOOKUP["C>"])
    print("flipped_north", summary(flipped), bodies(flipped))
    for name in ("H<", "H>"):
        child = flipped.clone()
        states = []
        for _ in range(4):
            before = child.frame()
            child.step(*LOOKUP[name])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print("flipped_north", name, states, child.levels_completed)
    upstack = stack.clone()
    for name in ("D", "D") + ("A<",) * 5 + ("B", "B"):
        upstack.step(*LOOKUP[name])
    print("upstack", summary(upstack), bodies(upstack))
    for name, action in CONTROLS:
        child = upstack.clone()
        states = []
        for _ in range(5):
            before = child.frame()
            child.step(*action)
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0:
                break
        print("upstack", name, states)
    downstack = upstack.clone()
    downstack.step(*LOOKUP["B"])
    downstack.step(*LOOKUP["B"])
    print("downstack", summary(downstack), bodies(downstack))
    for name in ("A<", "A>", "B", "C<", "C>", "F<", "F>", "G"):
        child = downstack.clone()
        states = []
        for _ in range(7):
            before = child.frame()
            child.step(*LOOKUP[name])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0:
                break
        print("downstack", name, states)
    lowstack = downstack.clone()
    for _ in range(5):
        lowstack.step(*LOOKUP["A>"])
    print("lowstack", summary(lowstack), bodies(lowstack))
    for turns in range(1, 5):
        turned = lowstack.clone()
        for _ in range(turns):
            turned.step(*LOOKUP["B"])
        print("lowstack B", turns, summary(turned), bodies(turned))
        for name in ("A<", "A>"):
            child = turned.clone()
            states = []
            for _ in range(6):
                before = child.frame()
                child.step(*LOOKUP[name])
                states.append((frame_delta(before, child.frame())["count"], summary(child)))
                if states[-1][0] == 0:
                    break
            print("lowstack", turns, name, states)
    coupled = env.clone()
    coupled.step(6, 54, 51)
    coupled.step(6, 54, 51)
    print("coupled_before", summary(coupled), bodies(coupled))
    for count in range(1, 5):
        before = coupled.frame()
        coupled.step(6, 60, 58)
        print(
            "coupled_I", count,
            frame_delta(before, coupled.frame())["count"],
            summary(coupled), bodies(coupled),
        )
    south = env.clone()
    for name in ("C<",) * 4 + ("D", "C>", "D"):
        south.step(*LOOKUP[name])
    print("south", summary(south), bodies(south))
    for name in ("C<", "C>", "H<", "H>"):
        child = south.clone()
        states = []
        for _ in range(10):
            before = child.frame()
            child.step(*LOOKUP[name])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print("south", name, states, child.levels_completed)
    left = env.clone()
    for name in (
        ("C<",) * 4 + ("D", "C>")
        + ("C>",) * 2 + ("F>",) * 8
    ):
        left.step(*LOOKUP[name])
    print("left", summary(left), bodies(left))
    for name in ("A<", "A>", "C<", "C>", "D", "F<", "F>", "G", "H<", "H>"):
        child = left.clone()
        states = []
        for _ in range(8):
            before = child.frame()
            child.step(*LOOKUP[name])
            states.append((frame_delta(before, child.frame())["count"], summary(child)))
            if states[-1][0] == 0 or child.levels_completed > 6:
                break
        print("left", name, states, child.levels_completed)
    ratchet = env.clone()
    for name in ("C<",) * 4 + ("D", "C>"):
        ratchet.step(*LOOKUP[name])
    print("ratchet", summary(ratchet))
    for _ in range(8):
        for name in ("F>", "C>"):
            before = ratchet.frame()
            ratchet.step(*LOOKUP[name])
            print(
                "ratchet", name,
                frame_delta(before, ratchet.frame())["count"],
                summary(ratchet), ratchet.levels_completed,
            )
    ratchet.step(*LOOKUP["A>"])
    print("low_ratchet", summary(ratchet), bodies(ratchet))
    for _ in range(8):
        for name in ("F>", "C>"):
            before = ratchet.frame()
            ratchet.step(*LOOKUP[name])
            print(
                "low_ratchet", name,
                frame_delta(before, ratchet.frame())["count"],
                summary(ratchet), ratchet.levels_completed,
            )


arena.run_program("s5i5", run)
