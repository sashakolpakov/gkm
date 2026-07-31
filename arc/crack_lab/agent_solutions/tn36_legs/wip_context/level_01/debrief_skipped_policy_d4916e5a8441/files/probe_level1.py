"""Compact clean-room probes for tn36 level 1."""

import argparse
from collections import Counter
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

from perception import color_counts, connected_components


def inspect_entry(env):
    frame = env.frame()
    print("actions", env.actions)
    print("shape", frame.shape, "colors", color_counts(frame))
    for blob in connected_components(frame, min_area=2):
        print(
            "blob",
            blob.color,
            "bbox",
            blob.bbox,
            "area",
            blob.area,
            "centroid",
            tuple(round(v, 2) for v in blob.centroid),
        )


def compact_delta(before, after):
    before = np.asarray(before)
    after = np.asarray(after)
    changed = before != after
    ys, xs = np.where(changed)
    if not len(ys):
        return (0, None, ())
    transitions = Counter(
        (int(a), int(b)) for a, b in zip(before[changed], after[changed])
    )
    return (
        int(changed.sum()),
        (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        tuple(sorted(transitions.items())),
    )


def render_region(frame, r0, c0, r1, c1):
    glyph = {0: ".", 1: "1", 3: "3", 4: "4", 5: "5", 9: "9", 11: "B"}
    frame = np.asarray(frame)
    for r in range(r0, r1 + 1):
        print(f"{r:02d}", "".join(glyph.get(int(v), "?") for v in frame[r, c0:c1 + 1]))


def inspect_scene(env):
    frame = np.asarray(env.frame())
    print("board_cells dominant/anomaly_count")
    for br in range(8):
        fields = []
        for bc in range(9):
            tile = frame[9 + 4 * br:13 + 4 * br, 14 + 4 * bc:18 + 4 * bc]
            counts = Counter(int(v) for v in tile.flat)
            dominant, count = max(counts.items(), key=lambda item: item[1])
            fields.append(f"{dominant:X}{16-count:X}")
        print(br, " ".join(fields))
    print("upper_piece")
    render_region(frame, 12, 28, 17, 35)
    print("lower_piece")
    render_region(frame, 32, 27, 41, 36)
    print("switches")
    render_region(frame, 41, 18, 47, 44)
    print("bottom_glyph")
    render_region(frame, 50, 31, 60, 41)


def inspect_single_clicks(env):
    base = np.asarray(env.frame()).copy()
    probes = [
        ("corner", 0, 0),
        ("outside", 5, 5),
        ("top_line", 31, 1),
        ("ordinary_a", 15, 10),
        ("ordinary_b", 19, 10),
        ("upper_11", 31, 14),
        ("lower_11", 31, 35),
        ("symbol_1", 26, 42),
        ("symbol_5", 21, 42),
        ("bottom_9", 36, 55),
    ]
    for name, x, y in probes:
        clone = env.clone()
        clone.step(6, x, y)
        print(
            "click",
            name,
            (x, y),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "delta",
            compact_delta(base, clone.frame()),
        )


def switch_bits(frame):
    frame = np.asarray(frame)
    return "".join(
        str(int(frame[y, 21 + 5 * i]))
        for y in (42, 45)
        for i in range(5)
    )


def inspect_switches(env):
    base = np.asarray(env.frame()).copy()
    print("initial_bits", switch_bits(base))
    for row_name, y in (("h", 42), ("v", 45)):
        for i in range(5):
            x = 21 + 5 * i
            clone = env.clone()
            clone.step(6, x, y)
            after = np.asarray(clone.frame())
            changed = np.argwhere((base != after) & ~(
                (np.indices(base.shape)[0] == 1) &
                (np.indices(base.shape)[1] == 61)
            ))
            print(
                "switch",
                row_name,
                i,
                "bits",
                switch_bits(after),
                "changed",
                [tuple(int(v) for v in pt) for pt in changed],
            )
    clone = env.clone()
    sequence = [(6, 21 + 5 * i, y) for y in (42, 45) for i in range(5)]
    for step_index, action in enumerate(sequence, 1):
        clone.step(*action)
        print(
            "all_once_step",
            step_index,
            "bits",
            switch_bits(clone.frame()),
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
        )


SWITCH_POINTS = tuple(
    (21 + 5 * i, y)
    for y in (42, 45)
    for i in range(5)
)


def inspect_bit_search(env):
    tested_steps = 0
    for submit_name, submit in (
        ("automatic", None),
        ("circle", (36, 55)),
    ):
        for mask in range(1 << len(SWITCH_POINTS)):
            clone = env.clone()
            chosen = []
            for bit, (x, y) in enumerate(SWITCH_POINTS):
                if mask & (1 << bit):
                    clone.step(6, x, y)
                    tested_steps += 1
                    chosen.append((6, x, y))
                    if clone.levels_completed:
                        print(
                            "FOUND",
                            submit_name,
                            "mask",
                            f"{mask:010b}",
                            "bits",
                            switch_bits(clone.frame()),
                            "path",
                            chosen,
                            "tested_steps",
                            tested_steps,
                        )
                        return
            if submit is not None and not clone.terminal():
                clone.step(6, *submit)
                tested_steps += 1
                if clone.levels_completed:
                    print(
                        "FOUND",
                        submit_name,
                        "mask",
                        f"{mask:010b}",
                        "pre_submit_bits",
                        switch_bits(env.clone().frame()) if not chosen else "see_mask",
                        "path",
                        chosen + [(6, *submit)],
                        "tested_steps",
                        tested_steps,
                    )
                    return
        print("searched", submit_name, "configs", 1 << len(SWITCH_POINTS))
    print("NOT_FOUND", "tested_steps", tested_steps)


def inspect_short_sequences(env):
    h0 = (6, 21, 42)
    h1 = (6, 26, 42)
    circle = (6, 36, 55)
    sequences = (
        ("h0", (h0,)),
        ("h1", (h1,)),
        ("h0_h1", (h0, h1)),
        ("circle", (circle,)),
        ("h0_circle", (h0, circle)),
        ("h1_circle", (h1, circle)),
        ("h0_h1_circle", (h0, h1, circle)),
        ("h1_h0_circle", (h1, h0, circle)),
    )
    for name, actions in sequences:
        clone = env.clone()
        states = []
        for action in actions:
            clone.step(*action)
            states.append(
                (
                    clone.levels_completed,
                    clone.terminal(),
                    switch_bits(clone.frame()) if not clone.levels_completed else "next_level",
                )
            )
        print("sequence", name, "states", states)


def inspect_gray_search(env):
    base_board = np.asarray(env.frame())[9:41, 14:50].copy()
    simulated_steps = 0
    board_change_masks = []
    for high_mask in range(32):
        clone = env.clone()
        path = []
        toggle_mask = 0

        def toggle(bit):
            nonlocal simulated_steps, toggle_mask
            x, y = SWITCH_POINTS[bit]
            action = (6, x, y)
            clone.step(*action)
            path.append(action)
            simulated_steps += 1
            toggle_mask ^= 1 << bit
            if clone.levels_completed:
                print(
                    "GRAY_FOUND",
                    "toggle_mask",
                    f"{toggle_mask:010b}",
                    "path",
                    path,
                    "simulated_steps",
                    simulated_steps,
                )
                return True
            board_delta = int(
                np.count_nonzero(
                    np.asarray(clone.frame())[9:41, 14:50] != base_board
                )
            )
            if board_delta:
                board_change_masks.append((toggle_mask, board_delta))
            return False

        for offset in range(5):
            if high_mask & (1 << offset):
                if toggle(5 + offset):
                    return
        previous_gray = 0
        for index in range(1, 32):
            gray = index ^ (index >> 1)
            changed = gray ^ previous_gray
            bit = changed.bit_length() - 1
            if toggle(bit):
                return
            previous_gray = gray
    print(
        "GRAY_NOT_FOUND",
        "simulated_steps",
        simulated_steps,
        "board_change_masks",
        board_change_masks[:20],
        "board_change_count",
        len(board_change_masks),
    )


SUBMIT_POINTS = {
    "circle": (36, 55),
    "upper": (31, 14),
    "lower": (31, 35),
}


def scan_submit_point(name):
    submit_x, submit_y = SUBMIT_POINTS[name]

    def callback(env):
        simulated_steps = 0
        responses = []
        previous_tested_mask = None
        for high_mask in range(32):
            clone = env.clone()
            toggle_mask = 0
            parent_path = []
            for offset in range(5):
                if high_mask & (1 << offset):
                    bit = 5 + offset
                    x, y = SWITCH_POINTS[bit]
                    action = (6, x, y)
                    clone.step(*action)
                    simulated_steps += 1
                    parent_path.append(action)
                    toggle_mask ^= 1 << bit
                    if clone.levels_completed:
                        print(
                            "SCAN_PARENT_LEVEL",
                            name,
                            "after_previous_mask",
                            previous_tested_mask,
                            "current_path",
                            parent_path,
                        )
                        return

            def test_current():
                nonlocal simulated_steps, previous_tested_mask
                before = np.asarray(clone.frame()).copy()
                child = clone.clone()
                child.step(6, submit_x, submit_y)
                simulated_steps += 1
                after = np.asarray(child.frame())
                changed = before != after
                changed[1, :] = False
                response_count = int(changed.sum())
                if response_count or child.levels_completed:
                    responses.append(
                        (
                            toggle_mask,
                            response_count,
                            child.levels_completed,
                            [
                                tuple(int(v) for v in point)
                                for point in np.argwhere(changed)[:20]
                            ],
                        )
                    )
                    print(
                        "SCAN_RESPONSE",
                        name,
                        "toggle_mask",
                        f"{toggle_mask:010b}",
                        "response_count",
                        response_count,
                        "level",
                        child.levels_completed,
                        "minimal_path",
                        [
                            (6, *SWITCH_POINTS[bit])
                            for bit in range(10)
                            if toggle_mask & (1 << bit)
                        ] + [(6, submit_x, submit_y)],
                    )
                    return True
                previous_tested_mask = toggle_mask
                return False

            if test_current():
                return
            previous_gray = 0
            for index in range(1, 32):
                gray = index ^ (index >> 1)
                changed_bit = gray ^ previous_gray
                bit = changed_bit.bit_length() - 1
                x, y = SWITCH_POINTS[bit]
                action = (6, x, y)
                clone.step(*action)
                simulated_steps += 1
                parent_path.append(action)
                toggle_mask ^= 1 << bit
                if clone.levels_completed:
                    print(
                        "SCAN_PARENT_LEVEL",
                        name,
                        "after_previous_mask",
                        previous_tested_mask,
                        "current_mask",
                        f"{toggle_mask:010b}",
                        "parent_path",
                        parent_path,
                    )
                    return
                if test_current():
                    return
                previous_gray = gray
        print(
            "SCAN_NO_RESPONSE",
            name,
            "simulated_steps",
            simulated_steps,
            "responses",
            responses,
        )

    return callback


DIRECT_SEQUENCES = {
    "h0": ((6, 21, 42),),
    "h1": ((6, 26, 42),),
    "h0_h1": ((6, 21, 42), (6, 26, 42)),
    "h0_circle": ((6, 21, 42), (6, 36, 55)),
    "h1_circle": ((6, 26, 42), (6, 36, 55)),
    "h0_circle_noop": ((6, 21, 42), (6, 36, 55), (6, 0, 0)),
    "h1_circle_noop": ((6, 26, 42), (6, 36, 55), (6, 0, 0)),
    "v0_circle": ((6, 21, 45), (6, 36, 55)),
    "all_one_noop": tuple(
        (6, *SWITCH_POINTS[bit]) for bit in (0, 2, 5, 7)
    ) + ((6, 0, 0),),
    "all_five_noop": tuple(
        (6, *SWITCH_POINTS[bit]) for bit in (1, 3, 4, 6, 8, 9)
    ) + ((6, 0, 0),),
    "all_five_circle": tuple(
        (6, *SWITCH_POINTS[bit]) for bit in (1, 3, 4, 6, 8, 9)
    ) + ((6, 36, 55),),
    "horizontal_inverse_noop": tuple(
        (6, *SWITCH_POINTS[bit]) for bit in range(5)
    ) + ((6, 0, 0),),
    "vertical_inverse_noop": tuple(
        (6, *SWITCH_POINTS[bit]) for bit in range(5, 10)
    ) + ((6, 0, 0),),
    "h0_h1_circle": ((6, 21, 42), (6, 26, 42), (6, 36, 55)),
    "h1_h0_circle": ((6, 26, 42), (6, 21, 42), (6, 36, 55)),
}


def direct_sequence(name):
    def callback(env):
        for step_index, action in enumerate(DIRECT_SEQUENCES[name], 1):
            before = np.asarray(env.frame()).copy()
            env.step(*action)
            pieces = [
                (blob.bbox, blob.area)
                for blob in connected_components(env.frame(), colors=(11,), min_area=1)
            ]
            print(
                "direct_step",
                step_index,
                action,
                "level",
                env.levels_completed,
                "terminal",
                env.terminal(),
                "delta",
                compact_delta(before, env.frame()),
                "color11",
                pieces,
            )
            if env.levels_completed:
                break

    return callback


def run_fresh_gray_search(submit_name):
    submit = None if submit_name == "automatic" else SUBMIT_POINTS.get(
        submit_name, (0, 0)
    )
    varying_bits = tuple(range(5 if submit is None else 4))
    fixed_bits = tuple(range(len(varying_bits), 10))
    total_actions = 0
    for fixed_mask in range(1 << len(fixed_bits)):
        hit = {"mask": None}

        def callback(env):
            toggle_mask = 0

            def act(action, bit=None):
                nonlocal toggle_mask
                env.step(*action)
                if bit is not None:
                    toggle_mask ^= 1 << bit
                if env.levels_completed:
                    hit["mask"] = toggle_mask
                    return True
                return False

            for offset, bit in enumerate(fixed_bits):
                if fixed_mask & (1 << offset):
                    if act((6, *SWITCH_POINTS[bit]), bit):
                        return
            if submit is not None and act((6, *submit)):
                return
            previous_gray = 0
            for index in range(1, 1 << len(varying_bits)):
                gray = index ^ (index >> 1)
                changed = gray ^ previous_gray
                bit = changed.bit_length() - 1
                if act((6, *SWITCH_POINTS[bit]), bit):
                    return
                if submit is not None and act((6, *submit)):
                    return
                previous_gray = gray

        levels, path, err = arena.run_program("tn36", callback)
        total_actions += len(path)
        if err:
            print(
                "FRESH_SCAN_ERROR",
                submit_name,
                "fixed_mask",
                fixed_mask,
                "err",
                err,
            )
            return
        if levels:
            print(
                "FRESH_SCAN_FOUND",
                submit_name,
                "toggle_mask",
                f"{hit['mask']:010b}" if hit["mask"] is not None else None,
                "levels",
                levels,
                "path",
                path,
                "replay_ok",
                arena.validate("tn36", path, levels),
                "total_actions",
                total_actions,
            )
            return
    print(
        "FRESH_SCAN_NOT_FOUND",
        submit_name,
        "entries",
        1 << len(fixed_bits),
        "total_actions",
        total_actions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "probe",
        choices=("entry", "scene", "clicks", "switches", "search", "short", "gray"),
        nargs="?",
        default="entry",
    )
    parser.add_argument("--sequence", choices=tuple(DIRECT_SEQUENCES))
    parser.add_argument("--submit-scan", choices=tuple(SUBMIT_POINTS))
    parser.add_argument(
        "--fresh-scan",
        choices=("automatic", "circle", "upper", "lower", "noop"),
    )
    args = parser.parse_args()
    if args.fresh_scan:
        run_fresh_gray_search(args.fresh_scan)
        raise SystemExit
    callback = {
        "entry": inspect_entry,
        "scene": inspect_scene,
        "clicks": inspect_single_clicks,
        "switches": inspect_switches,
        "search": inspect_bit_search,
        "short": inspect_short_sequences,
        "gray": inspect_gray_search,
    }[args.probe]
    if args.sequence:
        callback = direct_sequence(args.sequence)
    if args.submit_scan:
        callback = scan_submit_point(args.submit_scan)
    levels, path, err = arena.run_program("tn36", callback)
    replay_ok = arena.validate("tn36", path, levels) if path else False
    print("probe_result", levels, len(path), replay_ok, err, path)
