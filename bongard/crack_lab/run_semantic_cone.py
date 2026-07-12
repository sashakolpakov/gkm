"""Run Bongard semantic-cone experiments with no predicate fallback."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cofibered_proposer import AnthropicCofiberedProposer
from dataset import sample_problems, write_panels
from semantic_legs import default_registry
from semantic_verifier import ConeVerification, verify_hypothesis


@dataclass
class ProblemResult:
    opaque_id: str
    category: str
    solved: bool
    selected_hypothesis: str
    selected_description: str
    selected_rule: str
    support_errors: int
    loo_errors: int
    n_examples: int
    complexity: int
    proposer_kind: str
    track: str
    status: str
    candidates: list[dict]


def _panel_pngs(workspace: str, opaque_id: str) -> list[str]:
    pdir = os.path.join(workspace, opaque_id)
    paths = []
    for side in ("pos", "neg"):
        for i in range(6):
            png = os.path.join(pdir, f"{side}_{i}.png")
            if os.path.exists(png):
                paths.append(png)
    return paths


def _select(candidates: list[ConeVerification]) -> ConeVerification | None:
    if not candidates:
        return None
    return min(candidates, key=lambda r: (
        not r.accepted,
        r.loo_errors,
        r.support_errors,
        r.complexity,
        r.hypothesis_id,
    ))


def run(args: argparse.Namespace) -> None:
    if args.proposer != "anthropic":
        raise SystemExit("semantic-cone experiments require --proposer anthropic")
    problems = sample_problems(args.dataset_dir, args.limit, args.seed, args.source)
    out_dir = os.path.abspath(args.out_dir)
    ws = os.path.join(out_dir, "workspace")
    os.makedirs(ws, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    proposer = AnthropicCofiberedProposer(args.model, args.max_tokens)
    registry = default_registry()
    records: list[ProblemResult] = []

    for idx, problem in enumerate(problems):
        oid = f"problem_{idx:02d}"
        write_panels(ws, problem, oid)
        bundle = proposer.propose(oid, _panel_pngs(ws, oid))
        proposal_path = os.path.join(out_dir, f"{oid}_proposal.txt")
        with open(proposal_path, "w", encoding="utf-8") as f:
            f.write(bundle.raw_text)

        verifications = [
            verify_hypothesis(h, registry, problem,
                              max_support_errors=args.max_support_errors,
                              max_loo_errors=args.max_loo_errors)
            for h in bundle.hypotheses
        ]
        selected = _select(verifications)
        if selected is None:
            record = ProblemResult(
                oid, problem.category, False, "", "", "", 12, 12, 12, 0,
                bundle.proposer_kind, "SEMANTIC-PURE", "NO_PROPOSALS", [])
        else:
            status = "SOLVED_SEMANTIC_PURE" if selected.accepted else (
                "MISSING_LEG" if selected.semantic_issue == "MISSING_LEG" else (
                "COMPILE_FAILED" if selected.compile_error else (
                    "MEASUREMENT_ONLY" if selected.semantic_issue else "COUNTEREXAMPLE_FAILURE"))
            )
            descriptions = {h.hypothesis_id: h.description for h in bundle.hypotheses}
            record = ProblemResult(
                oid, problem.category, selected.accepted, selected.hypothesis_id,
                descriptions.get(selected.hypothesis_id, ""), selected.rule,
                selected.support_errors, selected.loo_errors, selected.n_examples,
                selected.complexity, bundle.proposer_kind, "SEMANTIC-PURE", status,
                [v.to_dict() for v in verifications],
            )
        records.append(record)
        _write_checkpoint(out_dir, args, records)
        print(
            f"[{idx + 1:02d}/{len(problems):02d}] {oid} {record.status} "
            f"support_errors={record.support_errors}/{record.n_examples} "
            f"loo_errors={record.loo_errors}/{record.n_examples} "
            f"rule={record.selected_rule}",
            flush=True,
        )


def _write_checkpoint(out_dir: str, args: argparse.Namespace,
                      records: list[ProblemResult]) -> None:
    payload = {
        "runner": "semantic_cone",
        "tracks": ["UNRESTRICTED", "SEMANTIC-PURE", "HYBRID"],
        "active_track": "SEMANTIC-PURE",
        "proposer": args.proposer,
        "model": args.model,
        "solved": sum(r.solved for r in records),
        "attempted": len(records),
        "records": [asdict(r) for r in records],
    }
    with open(os.path.join(out_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="downloads/Bongard-LOGO")
    parser.add_argument("--source", choices=("basic", "abstract", "both"), default="both")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--out-dir", default="bongard/crack_lab/semantic_runs/latest")
    parser.add_argument("--proposer", choices=("anthropic",), default="anthropic")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--max-support-errors", type=int, default=1)
    parser.add_argument("--max-loo-errors", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
