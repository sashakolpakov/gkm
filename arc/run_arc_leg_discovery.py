#!/usr/bin/env python3
"""Learn the legs from a LOCAL ARC-AGI-3 game, then form the cone over them.

This is the "learn fragments, then make the colimit cone over them" pipeline,
run on real local frames (no per-step network, no rate limits):

  1. cone_leg_discovery: mine channel-blind directional effect-legs from
     interaction using an INTRINSIC signal (replicable avatar displacement) —
     NOT extrinsic reward, which is never hit by a random policy. Two controls:
     a random-action floor, and held-out replication (pruned legs are shown).
  2. cone_leg_composition: glue the LEARNED legs into a seek cone for each
     candidate goal colour, and price each by levels_completed.
  3. a random-action control over the same horizon (the honesty floor).

It claims a level only when levels_completed increases. The point it proves is
the mechanism: legs are no longer hand-built — they are learned from the game,
with controls — and the cone is formed over the learned library. On the open
navigation stub this composed cone WINS (see tests); on the 2026 puzzle games it
learns the genuinely-reliable primitives and steers with them, reporting honestly
where the game needs primitives (push/slide) beyond a directional library.

    python3 arc/run_arc_leg_discovery.py --game wa30
    python3 arc/run_arc_leg_discovery.py --game ls20 --mode offline
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _domain in ("cone", "arc"):
    _p = REPO_ROOT / _domain
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
logging.disable(logging.INFO)

import arc_agi3_adapter as arc  # noqa: E402
import cone_leg_discovery as cld  # noqa: E402
import cone_leg_composition as clc  # noqa: E402


def make_env_factory(game: str, mode: str):
    return lambda: arc.LocalArcEnv(game, operation_mode=mode, environments_dir=ENVIRONMENTS_DIR)


def random_control(make_env, steps: int, seed: int = 0):
    env = make_env()
    snap = env.reset()
    moves = [a for a in (env.available_actions or [1, 2, 3, 4]) if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
    rng = random.Random(seed)
    best = snap.levels_completed
    for _ in range(steps):
        if snap.state in (arc.GameState.WIN, arc.GameState.GAME_OVER):
            break
        snap = env.step(arc.GameAction(rng.choice(moves)))
        best = max(best, snap.levels_completed)
    return best, snap.win_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", default="wa30")
    parser.add_argument("--mode", default="normal", choices=["normal", "offline"])
    parser.add_argument("--trials", type=int, default=20, help="probe states for discovery")
    parser.add_argument("--steps", type=int, default=80, help="max steps per composed-cone run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "normal" and not os.environ.get("ARC_API_KEY"):
        print("normal mode needs ARC_API_KEY (.env). If the game is already downloaded, use --mode offline.")
        return

    make_env = make_env_factory(args.game, args.mode)
    print(f"=== learn legs, then cone over them: {args.game} (local, mode={args.mode}) ===")
    try:
        snap0 = make_env().reset()
    except Exception as exc:  # noqa: BLE001
        print(f"could not start {args.game} locally: {exc}")
        return
    colours = arc.extract_scene(snap0.frame).colors_present()
    print(f"frame {len(snap0.frame)}x{len(snap0.frame[0])}, colours={colours}, win_levels={snap0.win_levels}")

    # ---- 1. discover legs (intrinsic signal + controls) ----
    res = cld.discover_effect_legs(make_env, trials=args.trials)
    if res.avatar_color is None:
        print("no free-moving avatar found; this game has no directional avatar to learn legs for.")
        return
    print(f"\navatar (learned by action response): colour {res.avatar_color}")
    print(f"random-action control floor (best-direction consistency): {res.random_consistency}")
    print("LEARNED legs (survived held-out replication AND beat the random floor):")
    if res.legs:
        for leg in res.legs:
            print(f"  {leg.name:26s} dir={leg.direction} consistency={leg.consistency} "
                  f"efficacy={leg.efficacy} mean_step={leg.mean_step} n={leg.support}")
    else:
        print("  (none cleared the bar)")
    print("rejected by a control (kept for honesty):")
    for leg in res.rejected:
        print(f"  {leg.name:26s} consistency={leg.consistency} efficacy={leg.efficacy}")
    if res.cooccurring:
        print(f"push candidates (a colour rigidly co-moved with the avatar): {res.cooccurring}")

    if not res.legs:
        print("\nverdict: no reliable directional leg learned — the game's avatar does not cleanly "
              "translate, so a directional library is the wrong primitive set (needs push/slide). "
              "Honest negative; the discovery + controls are the result.")
        return

    # ---- 2. cone over the LEARNED legs, priced by levels_completed ----
    print(f"\ncone over the learned legs — seek each goal colour, price by levels_completed:")
    print(f"  {'goal':>4s} {'reached':>8s} {'levels':>8s} {'steps':>6s}")
    best_levels = 0
    for goal in [c for c in colours if c != res.avatar_color]:
        snap, steps, reached = clc.run_composed_seek(
            make_env(), res.legs, goal_color=goal, avatar_color=res.avatar_color, max_steps=args.steps)
        best_levels = max(best_levels, snap.levels_completed)
        print(f"  {goal:>4d} {str(reached):>8s} {f'{snap.levels_completed}/{snap.win_levels}':>8s} {steps:>6d}")

    # ---- 3. random control ----
    rnd_best, rnd_win = random_control(make_env, max(args.steps, 200))
    print(f"\nRANDOM control ({max(args.steps,200)} steps): best_levels={rnd_best}/{rnd_win}")

    print("\nverdict:")
    if best_levels > rnd_best:
        print(f"  the cone over LEARNED legs reached level {best_levels}, beating random ({rnd_best}).")
    elif best_levels > 0:
        print(f"  cone over learned legs reached level {best_levels}, not beating random ({rnd_best}).")
    else:
        print("  the learned legs steer the avatar on real frames but complete no level (random also 0). "
              "The directional legs are real and validated; this game additionally needs higher-order "
              "legs (push/slide/align) and goal induction over them — the next fragment to learn.")


if __name__ == "__main__":
    main()
