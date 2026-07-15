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
import time
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


def decode_action(action) -> tuple[int, dict | None]:
    """Decode scalar keys and canonical ``[6, x, y]`` replay tokens."""
    if isinstance(action, (list, tuple)):
        if len(action) != 3 or int(action[0]) != 6:
            raise ValueError(f"invalid compound replay action: {action!r}")
        return 6, {"x": int(action[1]), "y": int(action[2])}
    action_id = int(action)
    if not 1 <= action_id <= 7:
        raise ValueError(f"invalid replay action: {action!r}")
    return action_id, None


def level_segments(game: str, actions) -> list:
    """Split the flat recorded path into per-level action segments by replaying
    it on the LOCAL engine (offline, ~2000 fps). Level boundaries let the remote
    replay recover from transient API failures: in competition mode RESET is a
    LEVEL reset, so a failed level can be restarted and its segment replayed
    without double-applying actions."""
    sys.path[:0] = [str(GKM / "arc"), str(GKM / "cone")]
    import arc_agi3_adapter as arc

    env = arc.LocalArcEnv(game, operation_mode="offline",
                          environments_dir=str(GKM / "environment_files"))
    snap = env.reset()
    levels = snap.levels_completed
    segments, start = [], 0
    for i, a in enumerate(actions):
        action_id, data = decode_action(a)
        snap = env.step(
            arc.GameAction(action_id),
            **({"x": data["x"], "y": data["y"]} if data else {}),
        )
        if snap.levels_completed > levels:
            segments.append(list(actions[start:i + 1]))
            start, levels = i + 1, snap.levels_completed
    if start < len(actions):  # trailing moves that close no level (not expected)
        segments.append(list(actions[start:]))
    return segments


def _reset_with_retry(env, label: str, tries: int = 5):
    for t in range(tries):
        fd = env.reset()
        if fd is not None:
            return fd
        print(f"  {label}: RESET failed (attempt {t + 1}/{tries}); retrying")
        time.sleep(3 * (t + 1))
    raise RuntimeError(f"{label}: RESET failed after {tries} attempts")


def replay(env, segments, engine_action_cls, label: str,
           level_retries: int = 4, verbose: bool = True) -> int:
    fd = _reset_with_retry(env, label)
    levels = int(fd.levels_completed or 0)
    moves = 0
    for k, seg in enumerate(segments, start=1):
        if levels >= k:
            continue
        for attempt in range(1, level_retries + 1):
            failed_at = None
            for i, a in enumerate(seg):
                action_id, data = decode_action(a)
                fd = env.step(engine_action_cls[f"ACTION{action_id}"], data=data)
                if fd is None:  # transient API failure; the level is now dirty
                    failed_at = i
                    break
                moves += 1
            now = int(fd.levels_completed or 0) if fd is not None else levels
            if failed_at is None and now >= k:
                levels = now
                if verbose:
                    print(f"  {label}: level {now} after {moves} moves")
                break
            why = (f"step {failed_at} failed" if failed_at is not None
                   else f"segment ended at levels={now}")
            print(f"  {label}: level {k} attempt {attempt}/{level_retries}: {why}; "
                  f"level-reset and retry")
            time.sleep(3 * attempt)
            fd = _reset_with_retry(env, label)  # competition: level reset
            levels = int(fd.levels_completed or 0)
        else:
            raise RuntimeError(f"{label}: level {k} failed {level_retries} attempts")
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
    segs = {}
    for g, ck in plan.items():
        segs[g] = level_segments(g, ck["final_path"])
        print(f"{g}: replaying {len(ck['final_path'])} recorded actions in "
              f"{len(segs[g])} level segments (locally validated reached={ck['reached']})")

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
            reached = replay(env, segs[g], EngineAction, g)
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
