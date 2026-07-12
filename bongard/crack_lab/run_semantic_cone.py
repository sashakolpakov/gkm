"""Run Bongard semantic-cone experiments with no predicate fallback.

Per problem the proposer gets up to ``--rounds`` verifier-in-the-loop turns:
each round's compile errors, MISSING_LEG structures, per-panel score tables
and invariance violations are fed back mechanically.  Solved problems are
promoted into the semantic artifact; failed attempts are snapshotted as WIP.
Ground-truth concept names never enter the run workspace.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_artifacts
from cofibered_proposer import AnthropicCofiberedProposer, ProposalBundle
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
    rounds_used: int
    proposer_kind: str
    track: str
    status: str
    proposer_error: str
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
        r.naturality_errors + r.cofibration_errors,
        r.complexity,
        r.hypothesis_id,
    ))


def _panel_name(index: int) -> str:
    return f"pos_{index}" if index < 6 else f"neg_{index - 6}"


def _score_table(v: ConeVerification) -> str:
    if not v.scores:
        return ""
    pos = ", ".join(f"{s:.4g}" for s in v.scores[:6])
    neg = ", ".join(f"{s:.4g}" for s in v.scores[6:])
    return f"  pos scores: [{pos}]\n  neg scores: [{neg}]"


def _misses(v: ConeVerification) -> str:
    if not v.scores:
        return ""
    order_low = v.rule.find("<=") >= 0
    names = []
    for i, s in enumerate(v.scores):
        predicted = (s <= v.threshold) if order_low else (s >= v.threshold)
        expected = i < 6
        if predicted != expected:
            names.append(f"{_panel_name(i)}(score={s:.4g})")
    if not names:
        return ""
    return "  misclassified: " + ", ".join(names[:6])


def _feedback_text(bundle: ProposalBundle,
                   verifications: list[ConeVerification]) -> str:
    lines = ["Verifier diagnostics for the last round (mechanical output):"]
    for v in verifications:
        if v.semantic_issue == "MISSING_LEG":
            lines.append(f"- {v.hypothesis_id}:\n{v.compile_error}")
        elif v.compile_error:
            lines.append(f"- {v.hypothesis_id}: COMPILE_ERROR: {v.compile_error}")
        else:
            lines.append(
                f"- {v.hypothesis_id}: accepted={v.accepted} "
                f"support_errors={v.support_errors}/{v.n_examples} "
                f"loo_errors={v.loo_errors}/{v.n_examples} "
                f"predicate_errors={v.predicate_errors} "
                f"naturality_errors={v.naturality_errors} "
                f"cofibration_errors={v.cofibration_errors} "
                f"rule={v.rule} fold_t=[{v.fold_threshold_min:.4g}, "
                f"{v.fold_threshold_max:.4g}]")
            table = _score_table(v)
            if table:
                lines.append(table)
            misses = _misses(v)
            if misses:
                lines.append(misses)
            if v.unchecked_morphisms:
                lines.append(
                    "  unchecked morphisms (no exact pixel action): "
                    + ", ".join(v.unchecked_morphisms))
    if bundle.parse_error:
        lines.append(f"Schema issues in your last submission: {bundle.parse_error}")
    lines.append(
        "Submit a full replacement set of 3-8 hypotheses. Keep the semantic "
        "object if the score table looks promising and improve the typed "
        "evidence path; otherwise propose different semantics. Do not weaken "
        "rich terms into scalar proxies; if a leg is missing, keep naming it "
        "so the MISSING_LEG demand stays visible.")
    return "\n".join(lines)


def _status_of(selected: ConeVerification | None,
               bundle: ProposalBundle) -> str:
    if selected is None:
        return "PROPOSER_PARSE_FAILED" if bundle.parse_error else "NO_PROPOSALS"
    if selected.accepted:
        return "SOLVED_SEMANTIC_PURE"
    if selected.semantic_issue == "MISSING_LEG":
        return "MISSING_LEG"
    if selected.compile_error:
        return "COMPILE_FAILED"
    if selected.semantic_issue:
        return "MEASUREMENT_ONLY"
    if selected.naturality_errors:
        return "NATURALITY_FAILURE"
    if selected.cofibration_errors:
        return "COFIBRATION_FAILURE"
    return "COUNTEREXAMPLE_FAILURE"


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
    results: dict[str, dict] = {}
    promoted_cones: list[dict] = []

    for idx, problem in enumerate(problems):
        oid = f"problem_{idx:02d}"
        write_panels(ws, problem, oid)
        pngs = _panel_pngs(ws, oid)

        all_verifications: list[ConeVerification] = []
        descriptions: dict[str, str] = {}
        hypotheses_by_id: dict[str, dict] = {}
        bundle: ProposalBundle | None = None
        rounds_used = 0
        infra_error = ""
        for rnd in range(args.rounds):
            try:
                if rnd == 0:
                    bundle = proposer.propose(oid, pngs)
                else:
                    bundle = proposer.refine(
                        oid, _feedback_text(bundle, round_verifications))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # record per problem; never kill the batch
                infra_error = f"{type(exc).__name__}: {exc}"
                break
            rounds_used = rnd + 1
            with open(os.path.join(out_dir, f"{oid}_round{rnd:02d}_proposal.txt"),
                      "w", encoding="utf-8") as f:
                f.write(bundle.raw_text +
                        (f"\n\nSCHEMA_ISSUES: {bundle.parse_error}\n"
                         if bundle.parse_error else ""))
            round_verifications = [
                verify_hypothesis(h, registry, problem,
                                  max_support_errors=args.max_support_errors,
                                  max_loo_errors=args.max_loo_errors)
                for h in bundle.hypotheses
            ]
            for h in bundle.hypotheses:
                descriptions[h.hypothesis_id] = h.description
                hypotheses_by_id[h.hypothesis_id] = h.to_dict()
            all_verifications.extend(round_verifications)
            selected = _select(all_verifications)
            if selected is not None and selected.accepted:
                break

        selected = _select(all_verifications)
        if infra_error and selected is None:
            status = "PROPOSER_INFRA_FAILURE"
            record = ProblemResult(
                oid, problem.category, False, "", "", "", 12, 12, 12, 0,
                rounds_used, "anthropic", "SEMANTIC-PURE", status, infra_error, [])
        elif selected is None:
            status = _status_of(None, bundle)
            record = ProblemResult(
                oid, problem.category, False, "", "", "", 12, 12, 12, 0,
                rounds_used, bundle.proposer_kind, "SEMANTIC-PURE", status,
                bundle.parse_error, [])
        else:
            status = _status_of(selected, bundle)
            record = ProblemResult(
                oid, problem.category, selected.accepted, selected.hypothesis_id,
                descriptions.get(selected.hypothesis_id, ""), selected.rule,
                selected.support_errors, selected.loo_errors, selected.n_examples,
                selected.complexity, rounds_used, bundle.proposer_kind,
                "SEMANTIC-PURE", status, bundle.parse_error,
                [v.to_dict() for v in all_verifications],
            )
        records.append(record)
        # Ground truth stays harness-side; it is written only into the
        # promoted artifact directory, never into the run workspace.
        results[oid] = {
            "problem_id": problem.problem_id,
            "category": problem.category,
            "concept": problem.concept,
            "solved": record.solved,
            "status": record.status,
            "rule": record.selected_rule,
        }
        if record.solved:
            promoted_cones.append({
                "opaque_id": oid,
                "hypothesis": hypotheses_by_id.get(record.selected_hypothesis, {}),
                "verification": selected.to_dict(),
                "rounds_used": rounds_used,
            })
        payload = _checkpoint_payload(args, records)
        _write_checkpoint(out_dir, payload)
        if record.solved:
            semantic_artifacts.promote(args.tag, out_dir, payload, results,
                                       promoted_cones)
        else:
            semantic_artifacts.snapshot_wip(args.tag, out_dir, oid)
        print(
            f"[{idx + 1:02d}/{len(problems):02d}] {oid} {record.status} "
            f"rounds={record.rounds_used} "
            f"support_errors={record.support_errors}/{record.n_examples} "
            f"loo_errors={record.loo_errors}/{record.n_examples} "
            f"rule={record.selected_rule}",
            flush=True,
        )


def _checkpoint_payload(args: argparse.Namespace,
                        records: list[ProblemResult]) -> dict:
    return {
        "runner": "semantic_cone",
        "artifact_state": "WIP",
        "promotion_policy": (
            "semantic_runs are local WIP; promote only replayable clean artifacts "
            "with typed traces, risk vectors, complexity breakdowns, and no taint"
        ),
        "tracks": ["UNRESTRICTED", "SEMANTIC-PURE", "HYBRID"],
        "active_track": "SEMANTIC-PURE",
        "proposer": args.proposer,
        "model": args.model,
        "rounds": args.rounds,
        "tag": args.tag,
        "solved": sum(r.solved for r in records),
        "attempted": len(records),
        "records": [asdict(r) for r in records],
    }


def _write_checkpoint(out_dir: str, payload: dict) -> None:
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
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--tag", default="typed")
    parser.add_argument("--max-support-errors", type=int, default=1)
    parser.add_argument("--max-loo-errors", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
