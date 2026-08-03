#!/usr/bin/env python3
"""Replay the promoted, replay-validated artifact paths against the live
ARC-AGI-3 API to produce a scorecard — WITHOUT re-running any discovery.

The expensive part of GKM (proposer-driven search) already happened and its
result is a literal action path in each promoted checkpoint
(agent_solutions/<game>_legs/checkpoint.json, replay-validated locally). This
tool only replays those paths through the official `arc_agi` toolkit.  The
exact API-action count is determined by the frozen checkpoints; replay uses
zero LLM tokens.

Modes (docs.arcprize.org/toolkit/competition_mode):
  --mode online       dry run: same remote API, no competition constraints.
                      Use this FIRST to check the recorded paths reproduce
                      remotely (a desync here costs nothing).
  --mode competition  the real thing: single scorecard, each environment may
                      be made once, scoring is against ALL environments (the
                      untouched ones count as 0), game resets become level
                      resets. The closed scorecard is what the community
                      leaderboard links as scorecard_url.

Canonical all-game invocations must also provide ``--games all``, the frozen
``--artifact-root``, its content-addressed ``--release-receipt``, and the
expected claimed-level count.  See ``REPRODUCE_ARC.md`` for the exact frozen
release commands; bare mode-only examples intentionally select only the small
demonstration subset and are not publication runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

from verify_frozen_release import (
    FrozenReleaseError,
    load_receipt,
    verify_frozen_release,
)

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
DEFAULT_ARTIFACT_ROOT = LAB / "agent_solutions"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(artifact_root: Path, game: str) -> Path:
    return artifact_root / f"{game}_legs" / "checkpoint.json"


def load_checkpoint(
    game: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT
) -> tuple[dict, str]:
    """Read and hash one regular checkpoint from the same immutable byte string."""
    path = checkpoint_path(artifact_root, game)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size > MAX_CHECKPOINT_BYTES
    ):
        raise ValueError(f"checkpoint is not a bounded regular file: {game}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_CHECKPOINT_BYTES
        ):
            raise ValueError(f"checkpoint changed during bounded read: {game}")
        raw = handle.read(MAX_CHECKPOINT_BYTES + 1)
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint is too large: {game}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a JSON object: {game}")
    return value, hashlib.sha256(raw).hexdigest()


def checkpoint(game: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict:
    """Backward-compatible value-only checkpoint reader."""
    return load_checkpoint(game, artifact_root)[0]


def parse_games(value: str, artifact_root: Path) -> list[str]:
    if value.strip().lower() == "all":
        games = sorted(
            path.parent.name.removesuffix("_legs")
            for path in artifact_root.glob("*_legs/checkpoint.json")
        )
    else:
        games = [game.strip() for game in value.split(",") if game.strip()]
    if not games:
        raise ValueError("no games selected")
    if len(games) != len(set(games)):
        raise ValueError("duplicate game in --games")
    if any(
        len(game) != 4
        or not game.isascii()
        or not game.isalnum()
        or game.lower() != game
        for game in games
    ):
        raise ValueError(f"invalid game set: {games!r}")
    return games


def release_binding(
    receipt_path: Path,
    games: list[str],
    checkpoints: dict[str, dict],
    checkpoint_hashes: dict[str, str],
) -> dict:
    """Bind the exact already-loaded endpoint bytes after full verification."""
    receipt, _ = load_receipt(receipt_path)
    receipt_sha256 = receipt_path.stem
    inventory = receipt.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("release receipt has no authoritative inventory")
    claimed = receipt.get("claimed_inventory", inventory)
    evidence = receipt.get("evidence")
    if not isinstance(claimed, dict) or not isinstance(evidence, dict):
        raise ValueError("release receipt has no claimed inventory/evidence")
    if set(games) != set(inventory):
        raise ValueError(
            "scorecard game set differs from the receipt's authoritative set"
        )
    for game in games:
        expected = claimed.get(game)
        value = checkpoints[game]
        if value.get("game") != game or value.get("reached") != expected:
            raise ValueError(
                f"checkpoint frontier differs from release receipt: {game}"
            )
        rows = evidence.get(game)
        if (
            not isinstance(rows, list)
            or len(rows) != expected
            or not isinstance(rows[-1], dict)
        ):
            raise ValueError(f"release evidence is incomplete for {game}")
        actual = checkpoint_hashes.get(game)
        if rows[-1].get("checkpoint_sha256") != actual:
            raise ValueError(
                f"checkpoint bytes differ from release receipt: {game}"
            )
    claimed_total = sum(int(value) for value in claimed.values())
    recorded_total = receipt.get(
        "claimed_level_count", receipt.get("authoritative_level_count")
    )
    if claimed_total != recorded_total:
        raise ValueError("release receipt claimed-level total is inconsistent")
    return {
        "binding_scope": "endpoint_checkpoint_bytes_only_after_full_gate",
        "receipt_sha256": receipt_sha256,
        "canonical_tree_sha256": receipt.get("canonical_tree_sha256"),
        "claimed_inventory": claimed,
        "claimed_level_count": claimed_total,
        "authoritative_level_count": sum(
            int(value) for value in inventory.values()
        ),
    }


def write_new_json(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def decode_action(action) -> tuple[int, dict | None]:
    """Decode scalar keys and canonical ``[6, x, y]`` replay tokens."""
    if isinstance(action, (list, tuple)):
        if (
            len(action) != 3
            or action[0] != 6
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in action
            )
            or not all(0 <= value < 64 for value in action[1:])
        ):
            raise ValueError(f"invalid compound replay action: {action!r}")
        return 6, {"x": action[1], "y": action[2]}
    if not isinstance(action, int) or isinstance(action, bool):
        raise ValueError(f"invalid replay action: {action!r}")
    action_id = action
    if not 1 <= action_id <= 7 or action_id == 6:
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
           level_retries: int = 4, max_recovery_cycles: int = 20,
           verbose: bool = True) -> int:
    fd = _reset_with_retry(env, label)
    levels = int(fd.levels_completed or 0)
    moves = 0
    attempts_at_level: dict[int, int] = {}
    recovery_cycles = 0
    while levels < len(segments):
        k = levels + 1
        seg = segments[k - 1]
        failed_at = None
        for i, a in enumerate(seg):
            action_id, data = decode_action(a)
            fd = env.step(engine_action_cls[f"ACTION{action_id}"], data=data)
            if fd is None:  # transient API failure; remote state is uncertain
                failed_at = i
                break
            moves += 1
        now = int(fd.levels_completed or 0) if fd is not None else levels
        if failed_at is None and now >= k:
            levels = now
            attempts_at_level.pop(k, None)
            if verbose:
                print(f"  {label}: level {now} after {moves} moves")
            state = getattr(fd.state, "name", str(fd.state))
            if state == "WIN":
                break
            continue

        attempts_at_level[k] = attempts_at_level.get(k, 0) + 1
        attempt = attempts_at_level[k]
        why = (f"step {failed_at} failed" if failed_at is not None
               else f"segment ended at levels={now}")
        print(f"  {label}: level {k} attempt {attempt}/{level_retries}: {why}; "
              f"level-reset and recover")
        if attempt >= level_retries:
            raise RuntimeError(f"{label}: level {k} failed {level_retries} attempts")
        recovery_cycles += 1
        if recovery_cycles > max_recovery_cycles:
            raise RuntimeError(
                f"{label}: exceeded {max_recovery_cycles} total recovery cycles"
            )
        time.sleep(3 * attempt)
        fd = _reset_with_retry(env, label)
        recovered = int(fd.levels_completed or 0)
        if recovered < levels:
            print(f"  {label}: reset rolled back from level {levels} to "
                  f"{recovered}; rebuilding from level {recovered + 1}")
        levels = recovered
    return levels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("online", "competition"), default="online",
                    help="online = remote dry run; competition = the single real scorecard run")
    ap.add_argument("--games", default="wa30,ls20",
                    help="comma-separated games or 'all' (default: wa30,ls20)")
    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="frozen *_legs tree containing the scored checkpoints",
    )
    ap.add_argument(
        "--release-receipt",
        type=Path,
        help=(
            "content-addressed schema-v2 receipt; triggers full historical "
            "release-gate verification before endpoint binding"
        ),
    )
    ap.add_argument(
        "--release-verifier-root",
        type=Path,
        help=(
            "already extracted receipt-bound verifier tree; by default the "
            "exact source revision is read from local Git history"
        ),
    )
    ap.add_argument(
        "--expected-claimed-levels",
        type=int,
        help="fail unless the receipt and checkpoints claim exactly this depth",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        help="write a new machine-readable run receipt; never overwrite",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the frozen receipt, checkpoints, and local level segments without network access",
    )
    ap.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    ap.add_argument("--tags", default="gkm,replay-validated")
    args = ap.parse_args()

    artifact_root = args.artifact_root.resolve()
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        print("artifact root must be a non-symlink directory.")
        return 2
    try:
        games = parse_games(args.games, artifact_root)
    except ValueError as exc:
        print(f"invalid scorecard plan: {exc}")
        return 2
    if args.mode == "competition" and args.release_receipt is None:
        print("competition mode requires --release-receipt.")
        return 2

    binding = None
    release_verification = None
    if args.release_receipt is not None:
        try:
            release_verification = verify_frozen_release(
                receipt_path=args.release_receipt.resolve(),
                canonical_root=artifact_root,
                repo_root=GKM,
                verifier_root=(
                    args.release_verifier_root.resolve()
                    if args.release_verifier_root is not None
                    else None
                ),
            )
        except (
            FrozenReleaseError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            print(f"release receipt rejected: {exc}")
            return 2
    try:
        loaded = {game: load_checkpoint(game, artifact_root) for game in games}
        plan = {game: value for game, (value, _) in loaded.items()}
        checkpoint_hashes = {game: digest for game, (_, digest) in loaded.items()}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"cannot load frozen checkpoint plan: {exc}")
        return 2
    if args.release_receipt is not None:
        try:
            binding = release_binding(
                args.release_receipt.resolve(),
                games,
                plan,
                checkpoint_hashes,
            )
            if (
                release_verification is None
                or binding["receipt_sha256"]
                != release_verification.get("receipt_sha256")
            ):
                raise ValueError("release receipt changed after full verification")
        except (
            FrozenReleaseError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            print(f"release endpoint binding rejected: {exc}")
            return 2
    if args.mode == "competition" and (
        binding is None
        or len(games) != 25
        or binding["authoritative_level_count"] != 183
    ):
        print(
            "competition mode requires one receipt-bound 25-game/183-level "
            "authoritative plan."
        )
        return 2
    claimed_levels = sum(
        int(value.get("reached", -1)) for value in plan.values()
    )
    if (
        args.expected_claimed_levels is not None
        and claimed_levels != args.expected_claimed_levels
    ):
        print(
            "claimed-level mismatch: "
            f"expected {args.expected_claimed_levels}, found {claimed_levels}"
        )
        return 2
    if (
        binding is not None
        and claimed_levels != binding["claimed_level_count"]
    ):
        print("checkpoint total differs from release receipt.")
        return 2

    segs = {}
    for g, ck in plan.items():
        segs[g] = level_segments(g, ck["final_path"])
        if len(segs[g]) != ck["reached"]:
            print(
                f"{g}: local segmentation reached {len(segs[g])} "
                f"but checkpoint claims {ck['reached']}"
            )
            return 2
        print(f"{g}: replaying {len(ck['final_path'])} recorded actions in "
              f"{len(segs[g])} level segments (locally validated reached={ck['reached']})")

    if args.preflight_only:
        run_receipt = {
            "schema": 1,
            "mode": "preflight",
            "status": "PASS",
            "scorecard_id": None,
            "scorecard_url": None,
            "source_url": args.source_url,
            "artifact_root": str(artifact_root),
            "release_receipt": (
                str(args.release_receipt.resolve())
                if args.release_receipt is not None
                else None
            ),
            "release_verification": release_verification,
            "release_binding": binding,
            "checkpoint_sha256": checkpoint_hashes,
            "claimed_levels": claimed_levels,
            "authoritative_levels": (
                binding["authoritative_level_count"] if binding else None
            ),
            "stored_actions": sum(
                len(value["final_path"]) for value in plan.values()
            ),
            "results": {
                game: {
                    "locally_segmented": len(segs[game]),
                    "claimed": plan[game]["reached"],
                }
                for game in games
            },
            "aggregate": None,
        }
        if args.output_json is not None:
            try:
                write_new_json(args.output_json, run_receipt)
            except OSError as exc:
                print(f"cannot write run receipt: {exc}")
                return 1
        print(
            "preflight PASS: "
            f"{len(games)} games, {claimed_levels} claimed levels"
        )
        return 0

    if not os.environ.get("ARC_API_KEY"):
        print("ARC_API_KEY required (repo .env or environment).")
        return 2
    from arc_agi import Arcade, OperationMode  # network toolkit; import late
    from arcengine import GameAction as EngineAction

    arcade = Arcade(arc_api_key=os.environ["ARC_API_KEY"],
                    operation_mode=OperationMode(args.mode))
    card_id = arcade.open_scorecard(source_url=args.source_url,
                                    tags=[t.strip() for t in args.tags.split(",") if t.strip()])
    print(f"scorecard opened: {card_id} (mode={args.mode})")

    results, ok = {}, True
    card = None
    try:
        for g, ck in plan.items():
            try:
                env = arcade.make(g, scorecard_id=card_id)
            except Exception as ex:
                print(f"{g}: make() aborted: {type(ex).__name__}: {ex}")
                results[g] = {"remote": -1, "claimed": ck["reached"]}
                ok = False
                continue
            if env is None:
                print(f"{g}: make() failed")
                results[g] = {"remote": -1, "claimed": ck["reached"]}
                ok = False
                continue
            try:
                reached = replay(env, segs[g], EngineAction, g)
            except Exception as ex:
                print(f"{g}: replay aborted: {type(ex).__name__}: {ex}")
                reached, ok = -1, False
            results[g] = {"remote": reached, "claimed": ck["reached"]}
            status = "OK" if reached >= ck["reached"] else "DESYNC"
            if reached < ck["reached"]:
                ok = False
            print(
                f"{g}: remote levels_completed={reached} "
                f"vs local {ck['reached']} -> {status}"
            )
    finally:
        card = arcade.close_scorecard(card_id)

    print(f"scorecard closed: {card_id}")
    print(f"scorecard_url: https://arcprize.org/scorecards/{card_id}")
    if card is not None:
        print("aggregate:", card)
    after_hashes = {
        game: sha256_file(checkpoint_path(artifact_root, game))
        for game in games
    }
    if after_hashes != checkpoint_hashes:
        print("frozen checkpoint bytes changed during scorecard replay.")
        ok = False
    aggregate = None
    if card is not None:
        if hasattr(card, "model_dump"):
            aggregate = card.model_dump(mode="json", exclude={"api_key"})
        else:
            aggregate = str(card)
    run_receipt = {
        "schema": 1,
        "mode": args.mode,
        "status": "PASS" if ok else "FAIL",
        "scorecard_id": card_id,
        "scorecard_url": f"https://arcprize.org/scorecards/{card_id}",
        "source_url": args.source_url,
        "artifact_root": str(artifact_root),
        "release_receipt": (
            str(args.release_receipt.resolve())
            if args.release_receipt is not None
            else None
        ),
        "release_verification": release_verification,
        "release_binding": binding,
        "checkpoint_sha256": checkpoint_hashes,
        "claimed_levels": claimed_levels,
        "authoritative_levels": (
            binding["authoritative_level_count"] if binding else None
        ),
        "stored_actions": sum(
            len(value["final_path"]) for value in plan.values()
        ),
        "results": results,
        "aggregate": aggregate,
    }
    if args.output_json is not None:
        try:
            write_new_json(args.output_json, run_receipt)
        except OSError as exc:
            print(f"cannot write run receipt: {exc}")
            return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
