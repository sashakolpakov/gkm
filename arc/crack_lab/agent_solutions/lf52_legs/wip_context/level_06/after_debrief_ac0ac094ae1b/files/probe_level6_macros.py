import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P


def stage(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))


def pieces(frame):
    blobs = P.connected_components(frame, colors=(1, 8, 12, 14))
    holes = {
        blob.top_left
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    carriers = {
        blob.top_left
        for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left
        for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    pegs = {
        blob.top_left
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    return holes, carriers, bridges, pegs


def click(env, cell):
    env.step(6, cell[1] + 1, cell[0] + 1)


def brief(frame):
    holes, carriers, bridges, pegs = pieces(frame)
    hole_box = (
        (min(holes), max(holes)) if holes else None
    )
    return {
        "H": (len(holes), hole_box),
        "C": sorted(carriers),
        "B": sorted(bridges),
        "P": sorted(pegs),
    }


def symbolic_solution(slots, carriers, bridges, pegs, max_states=20000):
    start = (frozenset(pegs), next(iter(bridges)))
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (state_pegs, bridge), path = queue.popleft()
        if len(state_pegs) == 1 and next(iter(state_pegs)) in carriers:
            return path, len(seen)
        occupied = state_pegs | {bridge}
        for kind, source in (
            *((("peg", peg) for peg in sorted(state_pegs))),
            ("bridge", bridge),
        ):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    destination not in slots
                    or destination in occupied
                    or midpoint not in occupied
                ):
                    continue
                if kind == "bridge" and midpoint not in state_pegs:
                    continue
                child_pegs = set(state_pegs)
                child_bridge = bridge
                if kind == "peg":
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    if midpoint in child_pegs:
                        child_pegs.remove(midpoint)
                else:
                    child_bridge = destination
                child = (frozenset(child_pegs), child_bridge)
                if child not in seen:
                    seen.add(child)
                    queue.append(
                        (child, path + ((kind, source, destination),))
                    )
    return None, len(seen)


def observed_macros(env):
    before = pieces(env.frame())
    holes, carriers, bridges, pegs = before
    found = []
    for source in sorted(bridges | pegs):
        for destination in sorted(holes | carriers):
            if (
                (source[0] == destination[0] and
                 abs(source[1] - destination[1]) == 12)
                or
                (source[1] == destination[1] and
                 abs(source[0] - destination[0]) == 12)
            ):
                clone = env.clone()
                click(clone, source)
                click(clone, destination)
                if pieces(clone.frame()) != before:
                    found.append((source, destination))
    return found


def attempted_leaps(env):
    before = pieces(env.frame())
    _, _, bridges, pegs = before
    occupied = bridges | pegs
    found = []
    for source in sorted(occupied):
        for midpoint in sorted(occupied - {source}):
            dr = midpoint[0] - source[0]
            dc = midpoint[1] - source[1]
            if (abs(dr), abs(dc)) not in ((6, 0), (0, 6)):
                continue
            destination = (midpoint[0] + dr, midpoint[1] + dc)
            if not (0 <= destination[0] < 64 and 0 <= destination[1] < 64):
                continue
            clone = env.clone()
            click(clone, source)
            click(clone, destination)
            if pieces(clone.frame()) != before:
                found.append((source, midpoint, destination))
    return found


def selection_context(env):
    _, _, bridges, pegs = pieces(env.frame())
    for label, source in (
        ("bridge", next(iter(bridges))),
        ("peg", min(pegs)),
    ):
        selected = env.clone()
        click(selected, source)
        print(label, "selected", P.frame_delta(env.frame(), selected.frame()))
        for action in (1, 2, 3, 4):
            branch = selected.clone()
            branch.step(action)
            control = env.clone()
            control.step(action)
            a = P.arr(branch.frame()).copy()
            b = P.arr(control.frame()).copy()
            a[0, 0] = b[0, 0]
            print(
                label, "then", action,
                "diff_control", int((a != b).sum()),
                "brief", brief(branch.frame()),
            )


def probe(env):
    stage(env)
    base = env.frame()
    holes, carriers, bridges, pegs = pieces(base)
    print("state", tuple(sorted(group) for group in pieces(base)))
    destinations = holes | carriers
    for source in sorted(bridges | pegs):
        for destination in sorted(destinations):
            if (
                (source[0] == destination[0] and
                 abs(source[1] - destination[1]) == 12)
                or
                (source[1] == destination[1] and
                 abs(source[0] - destination[0]) == 12)
            ):
                clone = env.clone()
                click(clone, source)
                click(clone, destination)
                after = pieces(clone.frame())
                if after != (holes, carriers, bridges, pegs):
                    print(
                        "macro", source, destination,
                        "reward", clone.levels_completed - env.levels_completed,
                        "state", tuple(sorted(group) for group in after),
                    )
    path, states = symbolic_solution(
        holes | carriers, carriers, bridges, pegs
    )
    print("search", states, "path", path)
    clone = env.clone()
    for _, source, destination in path or ():
        click(clone, source)
        click(clone, destination)
    print(
        "replay reward", clone.levels_completed - env.levels_completed,
        "state", tuple(sorted(group) for group in pieces(clone.frame())),
    )
    _, current_carriers, current_bridges, _ = pieces(clone.frame())
    click(clone, next(iter(current_bridges)))
    click(clone, next(iter(current_carriers)))
    print(
        "loaded reward", clone.levels_completed - env.levels_completed,
        "brief", brief(clone.frame()),
    )
    for action in env.actions:
        branch = clone.clone()
        branch.step(action)
        print(
            "loaded action", action,
            "reward", branch.levels_completed - env.levels_completed,
            "brief", brief(branch.frame()),
        )
    branch = clone.clone()
    for count in range(1, 13):
        branch.step(4)
        print(
            "right", count,
            "reward", branch.levels_completed - env.levels_completed,
            "brief", brief(branch.frame()),
        )
        if branch.levels_completed > env.levels_completed:
            break
    remote = clone.clone()
    for _ in range(9):
        remote.step(4)
    print("remote state", tuple(sorted(group) for group in pieces(remote.frame())))
    print("remote attempted leaps", attempted_leaps(remote))
    selection_context(remote)
    for action in env.actions:
        test = remote.clone()
        test.step(action)
        print(
            "remote action", action,
            "reward", test.levels_completed - env.levels_completed,
            "brief", brief(test.frame()),
        )
    for action in (1, 2, 3, 4):
        test = remote.clone()
        trace = []
        for count in range(1, 9):
            test.step(action)
            trace.append((
                count,
                tuple(sorted(pieces(test.frame())[2])),
                tuple(sorted(pieces(test.frame())[3])),
            ))
        print("remote repeat", action, trace)
    for prefix in ((1,), (1, 1)):
        junction = remote.clone()
        for action in prefix:
            junction.step(action)
        for action in (1, 2, 3, 4):
            test = junction.clone()
            trace = []
            for count in range(1, 9):
                before = brief(test.frame())
                test.step(action)
                after = brief(test.frame())
                trace.append((count, after["B"], after["P"]))
                if after == before:
                    break
            print("junction", prefix, "repeat", action, trace)
    for down_count in range(1, 9):
        test = remote.clone()
        route = (1, 1) + (3,) * 5 + (2,) * down_count
        for action in route:
            test.step(action)
        entries = []
        for left_count in range(1, 13):
            test.step(4)
            current = brief(test.frame())
            if current["B"]:
                entries.append((left_count, current["B"], current["P"]))
        if entries:
            print("circle down", down_count, "entries", entries)
    for outward_count in range(5, 11):
        test = remote.clone()
        route = (1, 1) + (3,) * outward_count + (2, 2)
        for action in route:
            test.step(action)
        reported_entry = False
        for return_count in range(1, 16):
            test.step(4)
            _, _, now_bridges, now_pegs = pieces(test.frame())
            if now_bridges and not reported_entry:
                reported_entry = True
                print(
                    "return entry", outward_count, return_count,
                    "B", sorted(now_bridges), "P", sorted(now_pegs),
                )
            if any(
                bridge[1] == peg[1] and peg[0] - bridge[0] == 6
                for bridge in now_bridges for peg in now_pegs
            ):
                print(
                    "lower entry", outward_count, return_count,
                    "B", sorted(now_bridges), "P", sorted(now_pegs),
                )
    for outward_count in (5, 7, 9, 11, 13):
        for down_count in (2, 4, 6, 8, 10):
            test = remote.clone()
            route = (
                (1, 1)
                + (3,) * outward_count
                + (2,) * down_count
            )
            for action in route:
                test.step(action)
            for return_count in range(1, 21):
                test.step(4)
                now = brief(test.frame())
                if now["B"]:
                    print(
                        "circle variant", outward_count, down_count,
                        return_count, now["B"], now["P"],
                    )
                    break
    for return_count in (1, 2, 3, 4):
        entry = remote.clone()
        route = (1, 1) + (3,) * 5 + (2, 2) + (4,) * return_count
        for action in route:
            entry.step(action)
        for action in (1, 2, 3, 4):
            test = entry.clone()
            trace = []
            for count in range(1, 13):
                test.step(action)
                now = brief(test.frame())
                trace.append((count, now["B"], now["P"]))
            print(
                "entry branch", return_count, action,
                "start", brief(entry.frame())["B"], "trace", trace,
            )
        print(
            "entry state", return_count,
            tuple(sorted(group) for group in pieces(entry.frame())),
            "macros", observed_macros(entry),
            "leaps", attempted_leaps(entry),
        )
    for outward_count in (15, 20, 25, 30):
        test = remote.clone()
        route = (1, 1) + (3,) * outward_count + (2,) * 10
        for action in route:
            test.step(action)
        for return_count in range(1, 41):
            test.step(4)
            now = brief(test.frame())
            if now["B"]:
                print(
                    "long circle", outward_count, return_count,
                    now["B"], now["P"],
                )
                break
    for up_count in range(1, 9):
        test = remote.clone()
        route = (1, 1) + (3,) * 5 + (1,) * up_count
        for action in route:
            test.step(action)
        for return_count in range(1, 13):
            test.step(4)
            now = brief(test.frame())
            if now["B"]:
                print(
                    "upper circle", up_count, return_count,
                    now["B"], now["P"],
                    "leaps", attempted_leaps(test),
                )
                break


A.run_program("lf52", probe)
