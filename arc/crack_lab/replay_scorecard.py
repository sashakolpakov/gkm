#!/usr/bin/env python3
"""Replay the promoted, replay-validated artifact paths against the live
ARC-AGI-3 API to produce a scorecard — WITHOUT re-running any discovery.

The expensive part of GKM (proposer-driven search) already happened and its
result is a literal action path in each promoted checkpoint
(agent_solutions/<game>_legs/checkpoint.json, replay-validated locally). This
tool only replays those paths through the official `arc_agi` toolkit, so a
scorecard costs a few hundred API calls and zero LLM tokens.

Modes (docs.arcprize.org/toolkit/competition_mode):
  --mode online       dry run: same remote API, no competition constraints.
                      Use this FIRST to check the recorded paths reproduce
                      remotely (a desync here costs nothing).
  --mode competition  the real thing: single scorecard, each environment may
                      be made once, scoring is against ALL environments (the
                      untouched ones count as 0), game resets become level
                      resets. The closed scorecard is what the community
                      leaderboard links as scorecard_url.

    python3 arc/crack_lab/replay_scorecard.py --mode online
    python3 arc/crack_lab/replay_scorecard.py --mode competition
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
GKM = LAB.parents[1]

# load ARC_API_KEY from the repo .env (same convention as lab.py)
_env = GKM / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

DEFAULT_SOURCE_URL = "https://github.com/sashakolpakov/gkm"


def checkpoint(game: str) -> dict:
    path = LAB / "agent_solutions" / f"{game}_legs" / "checkpoint.json"
    with open(path) as f:
        return json.load(f)


def replay(env, actions, engine_action_cls, label: str, verbose: bool = True) -> int:
    fd = env.reset()
    if fd is None:
        raise RuntimeError(f"{label}: RESET failed")
    levels = int(fd.levels_completed or 0)
    for i, a in enumerate(actions):
        fd = env.step(engine_action_cls[f"ACTION{a}"])
        if fd is None:
            raise RuntimeError(f"{label}: step {i} (ACTION{a}) failed")
        new = int(fd.levels_completed or 0)
        if verbose and new != levels:
            print(f"  {label}: level {new} after {i + 1} moves")
        levels = new
        state = getattr(fd.state, "name", str(fd.state))
        if state == "WIN":
            break
    return levels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("online", "competition"), default="online",
                    help="online = remote dry run; competition = the single real scorecard run")
    ap.add_argument("--games", default="wa30,ls20",
                    help="comma-separated promoted games to replay (default: wa30,ls20)")
    ap.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    ap.add_argument("--tags", default="gkm,replay-validated")
    args = ap.parse_args()

    if not os.environ.get("ARC_API_KEY"):
        print("ARC_API_KEY required (repo .env or environment).")
        return 2

    from arc_agi import Arcade, OperationMode  # network toolkit; import late
    from arcengine import GameAction as EngineAction

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    plan = {g: checkpoint(g) for g in games}
    for g, ck in plan.items():
        print(f"{g}: replaying {len(ck['final_path'])} recorded actions "
              f"(locally validated reached={ck['reached']})")

    arcade = Arcade(arc_api_key=os.environ["ARC_API_KEY"],
                    operation_mode=OperationMode(args.mode))
    card_id = arcade.open_scorecard(source_url=args.source_url,
                                    tags=[t.strip() for t in args.tags.split(",") if t.strip()])
    print(f"scorecard opened: {card_id} (mode={args.mode})")

    results, ok = {}, True
    for g, ck in plan.items():
        env = arcade.make(g, scorecard_id=card_id)
        if env is None:
            print(f"{g}: make() failed"); ok = False
            continue
        try:
            reached = replay(env, ck["final_path"], EngineAction, g)
        except Exception as ex:
            print(f"{g}: replay aborted: {type(ex).__name__}: {ex}")
            reached, ok = -1, False
        results[g] = (reached, ck["reached"])
        status = "OK" if reached >= ck["reached"] else "DESYNC"
        if reached < ck["reached"]:
            ok = False
        print(f"{g}: remote levels_completed={reached} vs local {ck['reached']} -> {status}")

    card = arcade.close_scorecard(card_id)
    print(f"scorecard closed: {card_id}")
    print(f"scorecard_url: https://arcprize.org/scorecards/{card_id}")
    if card is not None:
        print("aggregate:", card)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
