#!/usr/bin/env python3
"""Drive a LIVE ARC-AGI-3 game and run the scene-discovery pipeline on real
frames. This replaces the synthetic GoalGame stub with real ARC data, so the
"is this really ARC?" confusion goes away — it genuinely is.

Requires ARC_API_KEY (get one at https://three.arcprize.org). With no key it
prints instructions and exits 0, so the offline test suite stays hermetic. It
commits no frames (the repo does not vendor datasets).

    ARC_API_KEY=... python3 arc/run_arc_live_probe.py --game ls20

What it does on a real game: RESET, run the connected-component scene functor
on the real frame (colours, objects), then attempt avatar discovery by action
response (issue each available action, look for an object that translates by
the action delta). It reports honestly what the perception primitives find on
real ARC frames — including where a primitive does NOT transfer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _domain in ("cone", "arc"):
    _p = REPO_ROOT / _domain
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_dotenv() -> None:
    """Load ARC_API_KEY (and any KEY=VALUE) from the gitignored .env if present
    and not already set in the environment."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

import arc_agi3_adapter as arc  # noqa: E402
import arc_scene_atoms as sa  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ls20", help="game short code (e.g. ls20, wa30)")
    parser.add_argument("--actions", type=int, default=4, help="how many directional actions to probe")
    parser.add_argument("--list", action="store_true", help="just list available games")
    parser.add_argument("--scan", default="", help="comma-separated game codes to survey for responsiveness")
    parser.add_argument("--clicks", type=int, default=6, help="how many object centroids to click (ACTION6 games)")
    return parser.parse_args()


def grid_objects(frame) -> dict:
    return {c: sa.count_color(frame, c) for c in sa.colors_present(frame)}


def main() -> None:
    args = parse_args()
    if not os.environ.get("ARC_API_KEY"):
        print("No ARC_API_KEY set. Get one at https://three.arcprize.org and run:")
        print("  ARC_API_KEY=... python3 arc/run_arc_live_probe.py --game ls20")
        return

    games = arc.ArcEnv.list_games()
    print(f"live games available: {len(games)}")
    if args.list:
        for g in games:
            print(f"  {g['game_id']}  {g.get('title','')}  tags={g.get('tags')}")
        return

    if args.scan:
        scan_games([g.strip() for g in args.scan.split(",") if g.strip()], args)
        return

    probe_game(args.game, args, verbose=True)


def diff_cells(a, b) -> int:
    return sum(1 for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x])


def probe_game(game: str, args: argparse.Namespace, verbose: bool = True) -> dict:
    """Run OUR approach on one real game: perceive the scene (scene functor),
    then act — perception-driven clicks (ACTION6) on discovered object centroids
    for click games, or arrow avatar-discovery otherwise — and report whether
    actions move the world. Returns a summary dict."""
    env = arc.ArcEnv(game)
    if verbose:
        print(f"RESET {game} (provisioning can take ~15s) ...")
    snap = env.reset()
    frame = snap.frame
    colours = sa.colors_present(frame)
    avail = env.available_actions
    if verbose:
        print(f"state={snap.state.name} score={snap.score} available_actions={avail}")
        print(f"REAL frame: {len(frame)}x{len(frame[0])}  colours={colours}")
        print(f"objects per colour (connected components): {grid_objects(frame)}")

    max_changed = 0
    best_score = snap.score
    mode = "click" if 6 in avail else "arrow"

    if mode == "click":
        # OUR APPROACH on click games: the scene functor finds objects; we act
        # on them by clicking their centroids (ACTION6). A frame/score change
        # means perception-driven action is doing something real.
        scene = arc.extract_scene(frame)
        targets = [obj.centroid for obj in scene.objects][: args.clicks]
        if verbose:
            print(f"\nperception-driven clicks (ACTION6) on {len(targets)} object centroids:")
        prev = frame
        for (x, y) in targets:
            try:
                snap = env.step(arc.GameAction.ACTION6, x=x, y=y)
            except RuntimeError as exc:
                if verbose:
                    print(f"  click ({x},{y}) failed: {exc}")
                break
            changed = diff_cells(snap.frame, prev)
            max_changed = max(max_changed, changed)
            best_score = max(best_score, snap.score)
            if verbose:
                print(f"  click ({x},{y}): changed_cells={changed} state={snap.state.name} score={snap.score}")
            prev = snap.frame
    else:
        actions = [a for a in avail if 1 <= a <= 5][: args.actions]
        if verbose:
            print(f"\narrow avatar-discovery over actions {actions}:")
        prev = frame
        votes: dict = {}
        for a in actions:
            before = {c: set(sa.color_centroids(prev, c)) for c in sa.colors_present(prev)}
            try:
                snap = env.step(arc.GameAction(a))
            except RuntimeError as exc:
                if verbose:
                    print(f"  ACTION{a} failed: {exc}")
                break
            changed = diff_cells(snap.frame, prev)
            max_changed = max(max_changed, changed)
            best_score = max(best_score, snap.score)
            dx, dy = arc.ACTION_TO_DELTA[arc.GameAction(a)]
            moved = []
            if (dx, dy) != (0, 0):
                for c, cents in before.items():
                    after = set(sa.color_centroids(snap.frame, c))
                    if after and after == {(px + dx, py + dy) for (px, py) in cents}:
                        moved.append(c); votes[c] = votes.get(c, 0) + 1
            if verbose:
                print(f"  ACTION{a} (delta {dx,dy}): changed_cells={changed} state={snap.state.name} "
                      f"score={snap.score} translated_colours={moved}")
            prev = snap.frame
        if verbose:
            if votes:
                print(f"avatar candidate (translator): colour {max(votes, key=lambda c: votes[c])} (votes {votes})")
            else:
                print("no translating avatar found under arrow actions (honest negative).")

    env.close_scorecard()
    summary = {"game": game, "mode": mode, "colours": colours, "objects": sum(grid_objects(frame).values()),
               "max_changed_cells": max_changed, "best_score": best_score, "available": avail}
    if verbose:
        verdict = ("actions MOVED the world" if max_changed > 0 else
                   "actions produced NO frame change (title screen / wrong modality)")
        print(f"\nresult [{game}, {mode} mode]: {verdict}; max_changed_cells={max_changed}, best_score={best_score}")
        print("closed scorecard. (No frames written to the repo.)")
    return summary


def scan_games(game_list, args: argparse.Namespace) -> None:
    """Survey several real games: does perception-driven action change the
    world? Reports a one-line result per game — the 'couple of test cases'."""
    import time
    print(f"\nscanning {len(game_list)} games for action responsiveness ...")
    print("game,mode,colours,objects,max_changed_cells,best_score")
    rows = []
    for i, g in enumerate(game_list):
        if i:
            time.sleep(20)  # space calls to respect rate limits
        try:
            s = probe_game(g, args, verbose=False)
            rows.append(s)
            print(f"{s['game']},{s['mode']},{len(s['colours'])},{s['objects']},{s['max_changed_cells']},{s['best_score']}")
        except Exception as exc:  # noqa: BLE001 - report and continue the survey
            print(f"{g},ERROR,-,-,-,- ({type(exc).__name__}: {str(exc)[:80]})")
    responsive = [r for r in rows if r["max_changed_cells"] > 0]
    print(f"\n{len(responsive)}/{len(rows)} games responded to perception-driven actions: "
          f"{[r['game'] for r in responsive]}")


if __name__ == "__main__":
    main()
