#!/usr/bin/env python3
"""Point the GKM colimit-cone machinery at a LOCALLY-running ARC-AGI-3 game.

This is the end-to-end local pipeline:

    arc_agi toolkit (arcengine, OFFLINE/NORMAL)   <- real ARC game, run locally
        -> LocalArcEnv (reset/step -> Snapshot)   <- same surface as the live API
        -> scene functor (frame -> objects)       <- perception
        -> witness_seek_leg bound to a colour      <- the GKM seek cone (a leg
                                                      bound to a perceptual slot)
        -> levels_completed                        <- the reward the binding is
                                                      priced by (cone selection)

No network and no key are needed once the game is downloaded (OFFLINE); the
first fetch uses NORMAL mode and a key (ARC_API_KEY / .env). ~2000 FPS locally,
no rate limits (docs.arcprize.org/local-vs-online).

What it does, honestly:
  1. Plays the game locally and auto-detects the avatar colour by action
     response (the colour whose cells all translate by k*delta under a move).
  2. Runs the GKM cone: binds the channel-blind witness_seek_leg to each
     candidate goal colour, drives the avatar, and PRICES each binding by the
     levels_completed it achieves (this is goal-induction / cone selection by
     reward). It reports a per-binding table.
  3. Runs a RANDOM-action control over the same horizon (the honesty baseline:
     a binding only "counts" if it beats random).
  4. Prints a short steering trace so you can SEE the loop move the avatar on
     real frames.

It claims a level only when levels_completed actually increases. On the 2026
keyboard games (ls20, wa30, ...) the games are structured puzzles (block
sliding / Sokoban-like) whose win condition is not "reach a coloured cell", so
the minimal greedy leg steers the avatar correctly but does not complete a
level — which this script reports rather than hides.

    python3 experiments/run_arc_local_gkm.py --game wa30
    python3 experiments/run_arc_local_gkm.py --game ls20 --steps 120
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENVIRONMENTS_DIR = str(REPO_ROOT / "environment_files")


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
logging.disable(logging.INFO)  # quiet the toolkit's INFO chatter

import arc_agi3_adapter as arc  # noqa: E402
import cone_foraging as cf  # noqa: E402

MOVE_DELTAS = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}  # ARC ACTION -> (dx,dy)


def make_env(game: str, mode: str):
    return arc.LocalArcEnv(game, operation_mode=mode, environments_dir=ENVIRONMENTS_DIR)


def cells_by_color(frame):
    out = {}
    h, w = len(frame), len(frame[0])
    for color in range(1, arc.NUM_COLORS):
        s = {(x, y) for y in range(h) for x in range(w) if frame[y][x] == color}
        if s:
            out[color] = s
    return out


def detect_avatar(game: str, mode: str) -> int | None:
    """Avatar = the colour whose entire cell-set translates by k*delta under
    some directional action (k>=1). Tried from a fresh reset per action so the
    probe never depends on accumulated state. Returns None if no colour cleanly
    translates (the game has no free-moving avatar)."""
    for action, (dx, dy) in MOVE_DELTAS.items():
        env = make_env(game, mode)
        snap = env.reset()
        if action not in env.available_actions:
            continue
        before = cells_by_color(snap.frame)
        after = cells_by_color(env.step(arc.GameAction(action)).frame)
        for color, b in before.items():
            a = after.get(color, set())
            if not b or len(a) != len(b):
                continue
            for k in range(1, 20):
                if a == {(px + k * dx, py + k * dy) for px, py in b}:
                    return color
    return None


def seek_binding(game: str, mode: str, goal_color: int, avatar_color: int, steps: int):
    """Run the GKM witness seek leg bound to `goal_color` and report the best
    levels_completed reached. Uses the connector's own run_seek_leg_on_game so
    the leg here is exactly the substrate leg, unchanged."""
    env = make_env(game, mode)
    snap = arc.run_seek_leg_on_game(
        env, cf.witness_seek_leg(), goal_color, avatar_color=avatar_color, max_steps=steps
    )
    return snap.levels_completed, snap.win_levels, snap.state.name


def random_control(game: str, mode: str, steps: int, seed: int = 0):
    env = make_env(game, mode)
    snap = env.reset()
    rng = random.Random(seed)
    moves = [a for a in env.available_actions if a in MOVE_DELTAS] or [1, 2, 3, 4]
    best = snap.levels_completed
    for _ in range(steps):
        if snap.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            break
        snap = env.step(arc.GameAction(rng.choice(moves)))
        best = max(best, snap.levels_completed)
    return best, snap.win_levels, snap.state.name


def steering_trace(game: str, mode: str, goal_color: int, avatar_color: int, n: int = 8):
    """Print n steps of the loop so the avatar's motion on real frames is
    visible: avatar centroid, observed azimuth to the goal colour, chosen move."""
    env = make_env(game, mode)
    leg_map = cf.witness_seek_leg().genome.rule_map()
    substate = 0
    snap = env.reset()
    print(f"  steering trace (avatar=colour {avatar_color}, goal=colour {goal_color}):")
    for t in range(n):
        scene = arc.extract_scene(snap.frame, avatar_color=avatar_color)
        obs = arc.slot_observation(scene, goal_color)
        rule = leg_map.get((substate, obs)) or leg_map.get((substate, cf.ANY_OBS))
        pos = scene.avatar.centroid if scene.avatar else None
        move = cf.action_name(rule.actions[0]) if rule else "HALT"
        print(f"    t={t:2d} avatar@{pos} obs={cf.observation_name(obs):>4s} -> {move}")
        if rule is None:
            break
        for action in rule.actions:
            if cf.is_move(action):
                snap = env.step(arc.MOVE_TO_ACTION[action])
            elif action == cf.RETURN_ACTION:
                print("    RETURN (avatar on goal)")
                return
        substate = rule.next_state
        if snap.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            print(f"    ended: {snap.state.name}")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", default="wa30", help="game short code (default wa30)")
    parser.add_argument("--mode", default="normal", choices=["normal", "offline"],
                        help="normal: download once then run local (needs key); "
                             "offline: local only, must already be downloaded")
    parser.add_argument("--avatar", type=int, default=None,
                        help="avatar colour (default: auto-detect by action response)")
    parser.add_argument("--goals", default="", help="comma-separated goal colours (default: all present)")
    parser.add_argument("--steps", type=int, default=80, help="max steps per binding / control")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "normal" and not os.environ.get("ARC_API_KEY"):
        print("normal mode needs ARC_API_KEY (get one at https://three.arcprize.org, put it in .env).")
        print("If the game is already under environment_files/, re-run with --mode offline.")
        return

    print(f"=== GKM on local ARC-AGI-3: {args.game} (mode={args.mode}) ===")
    try:
        env = make_env(args.game, args.mode)
        snap = env.reset()
    except Exception as exc:  # noqa: BLE001
        print(f"could not start {args.game} locally: {exc}")
        return
    colours = arc.extract_scene(snap.frame).colors_present()
    print(f"local frame {len(snap.frame)}x{len(snap.frame[0])}, colours={colours}, "
          f"win_levels={snap.win_levels}, available_actions={env.available_actions}")

    avatar = args.avatar if args.avatar is not None else detect_avatar(args.game, args.mode)
    if avatar is None:
        print("no free-moving avatar found (a colour translating by the action delta); "
              "this game is not navigation-shaped, so the seek cone does not apply. "
              "Pass --avatar to force one.")
        return
    print(f"avatar colour (auto-detected by action response): {avatar}")

    goals = [int(g) for g in args.goals.split(",") if g.strip()] or [c for c in colours if c != avatar]

    print(f"\nGKM seek cone — bind witness_seek_leg to each goal colour, price by levels_completed:")
    print(f"  {'goal':>4s} {'levels':>8s} {'state':>12s}")
    results = []
    for goal in goals:
        levels, win, state = seek_binding(args.game, args.mode, goal, avatar, args.steps)
        results.append((goal, levels))
        print(f"  {goal:>4d} {f'{levels}/{win}':>8s} {state:>12s}")

    rnd_best, rnd_win, rnd_state = random_control(args.game, args.mode, max(args.steps, 200))
    print(f"\nRANDOM control ({max(args.steps,200)} steps): best_levels={rnd_best}/{rnd_win} state={rnd_state}")

    best_goal, best_levels = max(results, key=lambda r: r[1]) if results else (None, 0)
    print()
    steering_trace(args.game, args.mode, best_goal if best_goal is not None else goals[0], avatar)

    print("\nverdict:")
    if best_levels > rnd_best:
        print(f"  GKM seek (goal colour {best_goal}) reached level {best_levels}, beating random ({rnd_best}). "
              f"The cone cracked levels via a single colour binding.")
    elif best_levels > 0:
        print(f"  GKM seek reached level {best_levels} but did not beat random ({rnd_best}).")
    else:
        print("  no level completed by the seek cone or by random. The loop runs on real frames and "
              "steers the avatar (see trace), but this game's win condition is not 'reach a coloured "
              "cell' — it is a structured puzzle needing obstacle routing / pattern matching beyond the "
              "minimal greedy leg. Honest negative; next step is a richer (multi-state / pushing) leg.")


if __name__ == "__main__":
    main()
