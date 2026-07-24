"""Compact level-4 observations through the documented local harness surface."""
import gkm_try
import players
from legs import cover_colored_ring_groups_with_selected_shapes
from perception import (
    ACTION_NAME, action_deltas, arr, color_counts, connected_components,
    object_candidates,
)


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", tuple(env.actions))
    print("COLORS", color_counts(env.frame()))
    print("OBJECTS")
    for obj in object_candidates(env.frame()):
        if obj["color"] != 15:
            print(obj)
    print("DELTAS")
    for action, delta in action_deltas(env, env.actions).items():
        print(ACTION_NAME.get(action, action), {
            "count": delta["count"], "bbox": delta["bbox"],
            "samples": delta["samples"][:8],
        })
    print("RINGS")
    frame = arr(env.frame())
    rings = [
        b for b in connected_components(frame, colors=(4,), min_area=8)
        if b.size == (3, 3) and b.area == 8
    ]
    print([(b.centroid, int(frame[int(b.centroid[0]), int(b.centroid[1])]))
           for b in rings])
    print("SELECTIONS")
    scout = env.clone()
    first = None
    groups = {}
    for b in rings:
        point = (int(b.centroid[0]), int(b.centroid[1]))
        groups.setdefault(int(frame[point]), set()).add(point)
    for _ in range(12):
        before = arr(scout.frame()).copy()
        black = [tuple(map(int, p)) for p in zip(*((before == 0).nonzero()))]
        center = black[0]
        if first is None:
            first = center
        elif center == first:
            break
        erased = []
        for action in (1, 2, 3, 4):
            moved = scout.clone()
            moved.step(action)
            after = arr(moved.frame())
            erased.extend(
                (int(r) - center[0], int(c) - center[1], int(before[r, c]))
                for r, c in zip(*((before != after).nonzero()))
                if int(after[r, c]) == 5
            )
        by_color = {}
        for r, c, color in erased:
            by_color.setdefault(color, []).append((r, c))
        print(center, {color: (len(pts), min(pts), max(pts))
                       for color, pts in sorted(by_color.items())})
        shape_color = max(by_color, key=lambda color: len(by_color[color]))
        offsets = set(by_color[shape_color]) | {(0, 0)}
        fits = {}
        for group_color, targets in groups.items():
            first_target = next(iter(targets))
            candidates = {
                (first_target[0] - dr, first_target[1] - dc)
                for dr, dc in offsets
            }
            fits[group_color] = sorted(
                p for p in candidates
                if (p[0] - center[0]) % 3 == 0
                and (p[1] - center[1]) % 3 == 0
                and all((t[0] - p[0], t[1] - p[1]) in offsets
                        for t in targets)
            )
        print("FITS", fits)
        for candidate in ((30, 15), (30, 39)):
            print("AT", candidate, {
                color: [(t[0] - candidate[0], t[1] - candidate[1],
                         (t[0] - candidate[0], t[1] - candidate[1]) in offsets)
                        for t in sorted(targets)]
                for color, targets in groups.items()
            })
        scout.step(5)
    placed = env.clone()
    cover_colored_ring_groups_with_selected_shapes(placed)
    after = arr(placed.frame())
    points = sorted(point for targets in groups.values() for point in targets)
    black = [tuple(map(int, p)) for p in zip(*((after == 0).nonzero()))]
    print("PLACED", placed.levels_completed, "BLACK", black,
          "CENTERS", [(p, int(frame[p]), int(after[p])) for p in points])
    print("POST_DELTAS", {
        ACTION_NAME.get(a, a): (d["count"], d["bbox"])
        for a, d in action_deltas(placed, placed.actions).items()
    })
    used = placed.clone()
    used.step(5)
    print("POST_USE", used.levels_completed,
          [tuple(map(int, p)) for p in zip(*((arr(used.frame()) == 0).nonzero()))])
    candidates = []
    for kind, radius, start in (("plus", 13, (36, 54)), ("x", 10, (21, 24))):
        found = []
        for row in range(radius, 64 - radius):
            for col in range(radius, 64 - radius):
                if (row - start[0]) % 3 or (col - start[1]) % 3:
                    continue
                covered = set()
                for point in points:
                    dr, dc = point[0] - row, point[1] - col
                    if kind == "plus":
                        hit = (dr == 0 or dc == 0) and max(abs(dr), abs(dc)) <= radius
                    else:
                        hit = abs(dr) == abs(dc) and abs(dr) <= radius
                    if hit:
                        covered.add(point)
                if len(covered) >= 2:
                    found.append(((row, col), covered))
        candidates.append(found)
    wins = []
    for plus, plus_cover in candidates[0]:
        plus_state = env.clone()
        for _ in range(abs(plus[0] - 36) // 3):
            plus_state.step(1 if plus[0] < 36 else 2)
        for _ in range(abs(plus[1] - 54) // 3):
            plus_state.step(3 if plus[1] < 54 else 4)
        plus_state.step(5)
        for cross, cross_cover in candidates[1]:
            trial = plus_state.clone()
            for _ in range(abs(cross[0] - 21) // 3):
                trial.step(1 if cross[0] < 21 else 2)
            for _ in range(abs(cross[1] - 24) // 3):
                trial.step(3 if cross[1] < 24 else 4)
            if trial.levels_completed > env.levels_completed:
                wins.append((plus, sorted(plus_cover), cross, sorted(cross_cover)))
    print("MARKER_SEARCH", len(candidates[0]), len(candidates[1]), "WINS", wins)


gkm_try.A.run_program("re86", probe)
