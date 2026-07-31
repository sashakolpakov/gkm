"""Compact pristine-entry observations for wa30 level 9."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, ACTIONS, block_signatures, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solver", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def macro_map(frame):
    sigs = block_signatures(frame)
    palette = {}
    labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for sig in sorted(set(sigs.values())):
        palette[sig] = labels[len(palette)]
    rows = []
    for r in range(16):
        rows.append("".join(palette[sigs[(r, c)]] for c in range(16)))
    return palette, rows


def tile_patterns(frame, palette):
    patterns = {}
    for r in range(16):
        for c in range(16):
            tile = frame[r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4]
            sig = tuple(sorted({int(value) for row in tile for value in row}))
            if sig != (1,):
                key = "".join("".join(f"{int(value):X}" for value in row) for row in tile)
                patterns.setdefault((palette[sig], key), []).append((r, c))
    return patterns


def changed_cells(before, after):
    a = block_signatures(before)
    b = block_signatures(after)
    return [(cell, a[cell], b[cell]) for cell in sorted(a) if a[cell] != b[cell]]


def signature_positions(frame, signature):
    return tuple(
        cell for cell, value in sorted(block_signatures(frame).items()) if value == signature
    )


def cargo_state(frame):
    normal = signature_positions(frame, (4, 9))
    return {
        "loose": normal,
        "carried": signature_positions(frame, (5, 9)),
        "couriers": signature_positions(frame, (12,)),
        "runner": signature_positions(frame, (15,)),
        "pad": tuple(cell for cell in normal if 3 <= cell[0] <= 5 and 5 <= cell[1] <= 7),
    }


def observe(env):
    if os.path.exists("checkpoint.json"):
        with open("checkpoint.json") as handle:
            checkpoint = json.load(handle)
        if (
            checkpoint.get("game") == "wa30"
            and checkpoint.get("validated")
            and checkpoint.get("final_path")
        ):
            for action in checkpoint["final_path"]:
                env.step(action)
            print(
                "RESUME",
                {
                    "checkpoint_moves": len(checkpoint["final_path"]),
                    "levels": env.levels_completed,
                },
            )
    solver.solve(env)
    base = env.frame()
    palette, rows = macro_map(base)
    print("ENTRY", {"level": env.levels_completed + 1, "actions": list(env.actions)})
    print("PALETTE", {v: k for k, v in palette.items()})
    print("MAP", *rows, sep="\n")
    for (label, pattern), positions in tile_patterns(base, palette).items():
        print("TILE", label, positions, [pattern[i : i + 4] for i in range(0, 16, 4)])
    components = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(base, colors=(2, 4, 7, 9, 14), min_area=4)
    ]
    print("COMPONENTS", components)
    for action in ACTIONS:
        clone = env.clone()
        clone.step(action)
        print(
            "DELTA",
            ACTION_NAME[action],
            {
                "pixels": frame_delta(base, clone.frame())["count"],
                "cells": changed_cells(base, clone.frame()),
                "level": clone.levels_completed,
            },
        )
    waiter = env.clone()
    previous_state = cargo_state(waiter.frame())
    print("WAIT_STATE", 0, previous_state)
    for turn in range(1, 181):
        waiter.step(5)
        current_state = cargo_state(waiter.frame())
        if (
            current_state["loose"] != previous_state["loose"]
            or current_state["pad"] != previous_state["pad"]
            or turn % 20 == 0
            or waiter.levels_completed != env.levels_completed
            or waiter.terminal()
        ):
            print(
                "WAIT_STATE",
                turn,
                current_state,
                {"level": waiter.levels_completed, "terminal": waiter.terminal()},
            )
        previous_state = current_state
        if waiter.levels_completed != env.levels_completed or waiter.terminal():
            break


levels, path, err = arena.run_program("wa30", observe)
print("RUN", {"levels": levels, "moves": len(path), "err": err})
