"""ARC-style artifact discipline for the semantic track.

Mirrors the unrestricted track's scheme:

- promoted artifacts live in ``agent_solutions/<tag>_semantic/`` and hold
  ``checkpoint.json``, ``promoted_cones.json``, harness-only ``results.json``
  (the ONLY place ground-truth concept names may exist) and ``README.md``;
- failed attempts are snapshotted append-only under
  ``wip_context/<opaque_id>/<timestamp>/`` and are never admitted;
- promotion is gated on a taint scan of the run workspace;
- replay is the verifier itself: ``verify_hypothesis`` is a pure function of
  (cone IR, panels), so re-running it on the promoted cones must reproduce
  every verdict.
"""
from __future__ import annotations

import json
import os
import shutil
import time

LAB_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_TAINT_MARKERS = (
    "downloads/bongard-logo",
    "get_action_string_list",
    "human_designed_shapes",
    "basic_sampler",
    "abstract_sampler",
    "action_program",
    "results.json",
)


class WorkspaceTainted(RuntimeError):
    pass


def artifact_dir(tag: str) -> str:
    return os.path.join(LAB_DIR, "agent_solutions", f"{tag}_semantic")


def taint_reason(root: str) -> str | None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")]
        for filename in filenames:
            if filename.endswith((".npy", ".png")):
                continue
            path = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                text = open(path, encoding="utf-8", errors="ignore").read().lower()
            except OSError:
                continue
            for marker in SOURCE_TAINT_MARKERS:
                if marker in text:
                    rel = os.path.relpath(path, root)
                    return f"{marker} in {rel}"
    return None


def assert_not_tainted(root: str) -> None:
    reason = taint_reason(root)
    if reason:
        raise WorkspaceTainted(reason)


def snapshot_wip(tag: str, out_dir: str, opaque_id: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    dest = os.path.join(artifact_dir(tag), "wip_context", opaque_id, stamp)
    os.makedirs(dest, exist_ok=True)
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        if name.startswith(opaque_id) or name == "checkpoint.json":
            shutil.copy2(path, os.path.join(dest, name))
    return dest


def promote(tag: str, out_dir: str, payload: dict, results: dict,
            promoted_cones: list[dict]) -> str:
    assert_not_tainted(out_dir)
    art = artifact_dir(tag)
    os.makedirs(art, exist_ok=True)
    promoted_payload = dict(payload)
    promoted_payload["artifact_state"] = "PROMOTED"
    with open(os.path.join(art, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(promoted_payload, f, indent=2)
    with open(os.path.join(art, "promoted_cones.json"), "w", encoding="utf-8") as f:
        json.dump(promoted_cones, f, indent=2)
    # Ground truth stays harness-side: concept names exist only here.
    with open(os.path.join(art, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    solved = sum(1 for r in results.values() if r.get("solved"))
    lines = [
        f"# Semantic artifact `{tag}`",
        "",
        f"solved {solved}/{len(results)} (semantic-pure typed cones)",
        "",
        "Replay: re-run `verify_hypothesis` on each promoted cone IR against",
        "the recorded problems; verdicts must reproduce exactly (the verifier",
        "is a pure function of cone IR and panels).",
        "",
        "| opaque_id | solved | status | rule |",
        "|---|---|---|---|",
    ]
    for oid in sorted(results):
        r = results[oid]
        lines.append(
            f"| {oid} | {r.get('solved')} | {r.get('status', '')} | {r.get('rule', '')} |")
    with open(os.path.join(art, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return art
