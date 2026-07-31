"""Reward-test the final descent through the pocket opening."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
from legs import click_action
from probe_l8_down1 import FLIPPED_ROUTE, HANDOFF, RELEASE
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar_cell(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    blob = blobs[0]
    row = min(range(10), key=lambda i: abs(3 + 6 * i - blob.centroid[0]))
    col = min(range(8), key=lambda j: abs(15 + 6 * j - blob.centroid[1]))
    return row, col


def run(node, route, trace=False):
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        node.step(*action)
        if trace:
            print(
                "STEP",
                index,
                action,
                "level",
                node.levels_completed,
                "terminal",
                node.terminal(),
                "avatar",
                None if node.terminal() else avatar_cell(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for releases in range(5, 11):
        prefix = [
            *FLIPPED_ROUTE,
            *HANDOFF,
            *([RELEASE] * releases),
        ]
        for rights in range(1, 5):
            node = env.clone()
            run(node, [*prefix, *([(4,)] * rights)])
            print(
                "CASE",
                releases,
                rights,
                node.levels_completed,
                node.terminal(),
                None if node.terminal() else avatar_cell(node.frame()),
                None if node.terminal() else target(node.frame()),
                "" if node.terminal() else lattice(node.frame()),
            )
            if node.levels_completed > 7:
                winning = [*prefix, *([(4,)] * rights)]
                print("WIN", len(winning), winning)
                print("DIRECT")
                run(env, winning, trace=True)
                return

    for releases in (6, 7):
        prefix = [
            *FLIPPED_ROUTE,
            *HANDOFF,
            *([RELEASE] * releases),
        ]
        suffix = [click_action(4, 5), (4,), (4,), (4,)]
        node = env.clone()
        run(node, [*prefix, *suffix])
        print(
            "HANDOFF_CASE",
            releases,
            node.levels_completed,
            node.terminal(),
            None if node.terminal() else avatar_cell(node.frame()),
            None if node.terminal() else target(node.frame()),
            "" if node.terminal() else lattice(node.frame()),
        )
        if node.levels_completed > 7:
            winning = [*prefix, *suffix]
            print("WIN", len(winning), winning)
            print("DIRECT")
            run(env, winning, trace=True)
            return

    releases = 7
    prefix = [
        *FLIPPED_ROUTE,
        *HANDOFF,
        *([RELEASE] * releases),
    ]
    suffix = [
        click_action(4, 5),
        (4,),
        click_action(4, 6),
        (4,),
        (4,),
    ]
    node = env.clone()
    run(node, [*prefix, *suffix])
    print(
        "DOUBLE_HANDOFF",
        node.levels_completed,
        node.terminal(),
        None if node.terminal() else avatar_cell(node.frame()),
        None if node.terminal() else target(node.frame()),
        "" if node.terminal() else lattice(node.frame()),
    )
    if node.levels_completed > 7:
        winning = [*prefix, *suffix]
        print("WIN", len(winning), winning)
        print("DIRECT")
        run(env, winning, trace=True)
        return

    base_suffix = [
        click_action(4, 5),
        (4,),
        click_action(4, 6),
        (4,),
    ]
    for support_clicks in (1, 2, 3):
        suffix = [
            *base_suffix,
            *([click_action(5, 6)] * support_clicks),
            (4,),
        ]
        node = env.clone()
        run(node, [*prefix, *suffix])
        print(
            "SUPPORT_CYCLE",
            support_clicks,
            node.levels_completed,
            node.terminal(),
            None if node.terminal() else avatar_cell(node.frame()),
            None if node.terminal() else target(node.frame()),
            "" if node.terminal() else lattice(node.frame()),
        )
        if node.levels_completed > 7:
            winning = [*prefix, *suffix]
            print("WIN", len(winning), winning)
            print("DIRECT")
            run(env, winning, trace=True)
            return


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
