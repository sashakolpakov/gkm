"""Stage 1.5: describe-first A/B on the stage-1 cracked corpus
(bongard_crack_plan.md Section 9).

Rebuilds the exact stage-1 corpus (same seed, same interleave), selects the
problems stage 1 SOLVED (solvability-controlled), and runs each arm with
the Messages-API proposer from an EMPTY library under its own tag:

    arm A  tag ab_current        current prompt
    arm B  tag ab_describe       describe-first prompt

    python3 run_api_ab.py [--arm=current|describe_first|both]
                          [--minutes=8] [--turns=6] [--source-tag=logo_full]
                          [--model=claude-fable-5] [--tag-suffix=_fable]
                          [--max-tokens=8000] [--call-timeout=90]
                          [--ab-limit=25]

Scoring (solve rate, articulation name-match vs results.json ground truth,
marginal-C shape) is done by the report, not here.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_api_agent as API
import bongard_arena as A
import bongard_legs as L


def stage1_corpus(source_tag: str = "logo_full", max_problems: int = 80,
                  ab_limit: int = 0):
    dataset = os.path.join(L.LAB_DIR, "..", "..", "downloads", "Bongard-LOGO")
    problems = A.sample_problems(dataset, limit=627, seed=20260709, source="both")
    problems = L.interleave_corpus(
        [p for p in problems if p.category == "basic"],
        [p for p in problems if p.category == "abstract"])[:max_problems]
    ck = L._load_checkpoint(L.artifact_dir(source_tag))
    if ck is None:
        raise SystemExit(f"no checkpoint for source tag {source_tag}")
    solved = {r.opaque_id for r in ck.records if r.solved}
    corpus = [p for k, p in enumerate(problems) if f"problem_{k:02d}" in solved]
    if ab_limit:
        corpus = corpus[:ab_limit]
    print(f"A/B corpus: {len(corpus)} stage-1-cracked problems")
    return corpus


def run_arm(variant: str, corpus, minutes: int, turns: int,
            model: str, tag_suffix: str, max_tokens: int,
            call_timeout: float) -> None:
    tag = ("ab_current" if variant == "current" else "ab_describe") + tag_suffix
    print(f"=== arm {variant} (tag {tag}) ===")
    L.run(corpus, tag=tag, ws=os.path.join("/tmp", f"bongard_ws_{tag}"),
          propose_fn=API.api_propose(variant, max_turns=turns,
                                     max_tokens=max_tokens,
                                     per_call_timeout=call_timeout),
          ladder=(model,), minutes=minutes,
          git_checkpoints=True)


if __name__ == "__main__":
    arm, minutes, turns, source_tag = "both", 8, 6, "logo_full"
    model, tag_suffix = "sonnet", ""
    max_tokens, call_timeout, ab_limit = 8000, 90.0, 0
    for a in sys.argv[1:]:
        if a.startswith("--arm="):
            arm = a.split("=", 1)[1]
        elif a.startswith("--minutes="):
            minutes = int(a.split("=", 1)[1])
        elif a.startswith("--turns="):
            turns = int(a.split("=", 1)[1])
        elif a.startswith("--source-tag="):
            source_tag = a.split("=", 1)[1]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
        elif a.startswith("--tag-suffix="):
            tag_suffix = a.split("=", 1)[1]
        elif a.startswith("--max-tokens="):
            max_tokens = int(a.split("=", 1)[1])
        elif a.startswith("--call-timeout="):
            call_timeout = float(a.split("=", 1)[1])
        elif a.startswith("--ab-limit="):
            ab_limit = int(a.split("=", 1)[1])
    corpus = stage1_corpus(source_tag, ab_limit=ab_limit)
    for variant in (("current", "describe_first") if arm == "both" else (arm,)):
        run_arm(variant, corpus, minutes, turns, model, tag_suffix,
                max_tokens, call_timeout)
