"""Run the describe-first arm on the exact 25 solved stage-1 problems.

This is a local comparison run, not the production stage-1.5 orchestrator.
Arm A is the already-promoted ``logo_full`` run. This driver executes only
arm B, in the same opaque-ID order, with one bounded API proposer session per
problem and no infrastructure wait or retry loop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bongard_api_agent as API
import bongard_arena as A
import bongard_legs as L


EMPTY_LIBRARY = """\
# Describe-first B-arm shared predicate library.
# p_<name>(panel) -> float | bool; panel is a 128x128 uint8 array.
"""

# Frozen at the stage-1 auto-stop checkpoint (commit a9f3427). The live
# logo_full artifact later continued to 51 solves, so its summary is not the
# experimental sampling boundary for this comparison.
STAGE1_25_IDS = (
    "problem_00", "problem_01", "problem_02", "problem_03", "problem_04",
    "problem_05", "problem_06", "problem_07", "problem_08", "problem_09",
    "problem_10", "problem_13", "problem_23", "problem_33", "problem_35",
    "problem_40", "problem_41", "problem_45", "problem_50", "problem_55",
    "problem_61", "problem_62", "problem_65", "problem_71", "problem_73",
)


@dataclass
class BRecord:
    opaque_id: str
    category: str
    a_rule: str
    a_rule_cost: float
    solved: bool
    cv_accuracy: float
    train_accuracy: float
    rule: str
    marginal_C: int
    predicate_errors: int
    cv_errors: int
    train_errors: int
    n_examples: int
    max_cv_errors: int
    max_train_errors: int
    threshold: float
    fold_threshold_min: float
    fold_threshold_max: float
    attempts: int
    infra_failure: bool
    description_present: bool
    failure_diagnoses: List[str]
    transcript_path: str
    description_path: str
    valid_evidence: bool = True
    invalid_reason: str = ""


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def write_json(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def semantic_candidate_names(ws: str, oid: str) -> List[str] | None:
    path = os.path.join(ws, f"semantic_candidate_names_{oid}.json")
    try:
        data = json.loads(read_text(path) or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    names = [str(item) for item in data if str(item).startswith("p_")]
    return names or None


def stage1_records(source_tag: str) -> List[dict]:
    path = os.path.join(L.artifact_dir(source_tag), "checkpoint.json")
    data = json.loads(read_text(path) or "{}")
    by_id = {r.get("opaque_id"): r for r in data.get("records", [])}
    records = [by_id[oid] for oid in STAGE1_25_IDS
               if oid in by_id and by_id[oid].get("solved")]
    if len(records) != 25:
        raise RuntimeError(
            f"{source_tag} must contain exactly 25 solved records; got {len(records)}")
    return records


def stage1_problem_map(max_problems: int = 80) -> Dict[str, A.Problem]:
    dataset = os.path.join(L.LAB_DIR, "..", "..", "downloads", "Bongard-LOGO")
    problems = A.sample_problems(dataset, limit=627, seed=20260709, source="both")
    ordered = L.interleave_corpus(
        [p for p in problems if p.category == "basic"],
        [p for p in problems if p.category == "abstract"],
    )[:max_problems]
    return {f"problem_{i:02d}": problem for i, problem in enumerate(ordered)}


def clean_problem_state(ws: str, keep_library: bool = True) -> None:
    for name in os.listdir(ws):
        path = os.path.join(ws, name)
        if keep_library and name == "predicates.py":
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def snapshot_attempt(out_dir: str, ws: str, oid: str,
                     attempt: int) -> tuple[str, str]:
    problem_dir = os.path.join(out_dir, "attempts", oid, f"attempt_{attempt:02d}")
    if os.path.exists(problem_dir):
        shutil.rmtree(problem_dir)
    os.makedirs(problem_dir, exist_ok=True)
    for name in os.listdir(ws):
        src = os.path.join(ws, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(problem_dir, name), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(problem_dir, name))
    transcript = os.path.join(problem_dir, f"api_transcript_{oid}.md")
    description = os.path.join(problem_dir, f"descriptions_{oid}.md")
    return os.path.relpath(transcript, out_dir), os.path.relpath(description, out_dir)


def failure_diagnosis(result: API.SemanticCVResult,
                      description_present: bool) -> str:
    facts = [result.result_line()]
    if result.rule.startswith("SEMANTIC_CV_TIMEOUT"):
        facts.append(
            "Candidate predicate execution exceeded the semantic-CV wall; replace "
            "expensive iterative measurements with simpler bounded image statistics.")
    elif result.predicate_errors:
        facts.append(
            f"Fix predicate execution first: {result.predicate_errors} evaluations failed.")
    elif result.train_errors > result.max_train_errors:
        facts.append(
            "The current measurement does not cleanly separate the visible panels; "
            "revise the semantic hypothesis or add a measurement for the observed "
            "counterexamples.")
    elif result.cv_errors > result.max_cv_errors:
        facts.append(
            "The measurement separates the visible panels but exceeds the allowed "
            "one-image-out error budget; keep the semantic concept only if a more "
            "stable raw measurement gives a common cutoff.")
    else:
        facts.append("Semantic CV did not admit the candidate; inspect code loading and rule output.")
    if not description_present:
        facts.append(
            "The required semantic section was missing; propose semantic separators "
            "before code and restate the surviving rule after CV.")
    return " ".join(facts)


def save_artifact(out_dir: str, out_tag: str, source_tag: str, model: str,
                  records: List[BRecord], library: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predicates.py"), "w", encoding="utf-8") as f:
        f.write(library)
    solved = sum(r.solved for r in records)
    total_c = sum(r.marginal_C for r in records)
    payload = {
        "tag": out_tag,
        "arm": "B_describe_first",
        "source_a_tag": source_tag,
        "model": model,
        "attempted": len(records),
        "solved": solved,
        "valid_attempted": sum(r.valid_evidence for r in records),
        "valid_solved": sum(r.solved for r in records if r.valid_evidence),
        "invalid_or_infra": sum(not r.valid_evidence for r in records),
        "total_marginal_C": total_c,
        "free_energy": A.free_energy(solved, total_c),
        "infra_failures": sum(r.infra_failure for r in records),
        "descriptions_present": sum(r.description_present for r in records),
        "records": [asdict(r) for r in records],
    }
    write_json(os.path.join(out_dir, "checkpoint.json"), payload)


def load_records(previous: dict) -> List[BRecord]:
    allowed = {f.name for f in fields(BRecord)}
    records = []
    for raw in previous.get("records", []):
        filtered = {k: v for k, v in raw.items() if k in allowed}
        if "cv_accuracy" not in filtered and "semantic_cv_accuracy" in raw:
            filtered["cv_accuracy"] = raw["semantic_cv_accuracy"]
        if "train_accuracy" not in filtered and "semantic_cv_train_accuracy" in raw:
            filtered["train_accuracy"] = raw["semantic_cv_train_accuracy"]
        filtered.setdefault("cv_accuracy", 0.0)
        filtered.setdefault("train_accuracy", 0.0)
        if "rule" not in filtered and "semantic_cv_rule" in raw:
            filtered["rule"] = raw["semantic_cv_rule"]
        filtered.setdefault("rule", "")
        if "threshold" not in filtered and "semantic_cv_threshold" in raw:
            filtered["threshold"] = raw["semantic_cv_threshold"]
        if "fold_threshold_min" not in filtered and "semantic_cv_fold_threshold_min" in raw:
            filtered["fold_threshold_min"] = raw["semantic_cv_fold_threshold_min"]
        if "fold_threshold_max" not in filtered and "semantic_cv_fold_threshold_max" in raw:
            filtered["fold_threshold_max"] = raw["semantic_cv_fold_threshold_max"]
        if "cv_errors" not in filtered:
            n = int(filtered.get("n_examples", 12))
            filtered["cv_errors"] = int(round(n * (1.0 - filtered.get("cv_accuracy", 0.0))))
        if "train_errors" not in filtered:
            n = int(filtered.get("n_examples", 12))
            filtered["train_errors"] = int(round(n * (1.0 - filtered.get("train_accuracy", 0.0))))
        filtered.setdefault("n_examples", 12)
        filtered.setdefault("max_cv_errors", 1)
        filtered.setdefault("max_train_errors", 0)
        filtered.setdefault("threshold", 0.0)
        filtered.setdefault("fold_threshold_min", 0.0)
        filtered.setdefault("fold_threshold_max", 0.0)
        filtered.setdefault("predicate_errors", 0)
        records.append(BRecord(**filtered))
    return records


def run(args: argparse.Namespace) -> None:
    a_records = stage1_records(args.source_tag)
    problems = stage1_problem_map()
    missing = [r["opaque_id"] for r in a_records if r["opaque_id"] not in problems]
    if missing:
        raise RuntimeError(f"stage-1 IDs missing from reconstructed corpus: {missing}")

    out_dir = L.artifact_dir(args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")
    previous = {} if args.clean else json.loads(read_text(checkpoint_path) or "{}")
    records = load_records(previous)
    completed = {r.opaque_id for r in records}

    ws = os.path.join("/tmp", f"bongard_ws_{args.out_tag}")
    os.makedirs(ws, exist_ok=True)
    clean_problem_state(ws, keep_library=False)
    admitted_library = (
        EMPTY_LIBRARY if args.clean
        else read_text(os.path.join(out_dir, "predicates.py")) or EMPTY_LIBRARY
    )
    with open(os.path.join(ws, "predicates.py"), "w", encoding="utf-8") as f:
        f.write(admitted_library)

    proposer = API.api_propose(
        "describe_first",
        max_turns=args.turns,
        max_tokens=args.max_tokens,
        per_call_timeout=args.call_timeout,
        max_cv_errors=args.semantic_max_cv_errors,
        max_train_errors=args.semantic_max_train_errors,
    )
    print(f"B25 describe-first: source A={args.source_tag} model={args.model}", flush=True)
    print("IDs: " + ", ".join(r["opaque_id"] for r in a_records), flush=True)

    for index, a_record in enumerate(a_records, start=1):
        oid = a_record["opaque_id"]
        if oid in completed:
            print(f"[{index:02d}/25] {oid}: checkpointed; skipping", flush=True)
            continue

        problem = problems[oid]
        clean_problem_state(ws)
        with open(os.path.join(ws, "current_problem.txt"), "w", encoding="utf-8") as f:
            f.write(oid)
        A.write_panels(ws, problem, oid)
        before = admitted_library
        base_task = L.build_task(oid, f"{sys.executable} bongard_try.py")
        infra_failure = False
        diagnoses: List[str] = []
        transcript_path = ""
        description_path = ""
        attempt_count = 0
        any_infra_failure = False
        valid_attempt_seen = False
        last_valid_semantic = API.SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12, "NO_VALID_B_ATTEMPT",
            0.0, 0.0, 0.0, args.semantic_max_cv_errors,
            args.semantic_max_train_errors, 0)
        last_valid_transcript_path = ""
        last_valid_description_path = ""
        last_valid_description_present = False
        invalid_reason = ""
        for attempt in range(1, args.attempts + 1):
            attempt_count = attempt
            if diagnoses:
                task = (
                    base_task
                    + "\n\nPREVIOUS ATTEMPT DIAGNOSIS:\n"
                    + diagnoses[-1]
                    + "\nContinue from the current candidate library and correct this failure."
                )
            else:
                task = base_task
            transcript = proposer(task, ws, args.model, args.minutes) or ""
            attempt_tainted = API.TAINT_MARKER.lower() in transcript.lower()
            attempt_no_valid_code = API.NO_VALID_CODE_MARKER.lower() in transcript.lower()
            attempt_infra = "api failure:" in transcript.lower() or attempt_tainted
            any_infra_failure = any_infra_failure or attempt_infra
            semantic = API.semantic_cv_ws_with_timeout(
                ws,
                args.verify_timeout,
                max_cv_errors=args.semantic_max_cv_errors,
                max_train_errors=args.semantic_max_train_errors,
                allowed_names=semantic_candidate_names(ws, oid),
            )
            description_file = os.path.join(ws, f"descriptions_{oid}.md")
            description_present = bool(read_text(description_file).strip())
            transcript_path, description_path = snapshot_attempt(
                out_dir, ws, oid, attempt)
            if attempt_infra or attempt_no_valid_code or not description_present:
                if attempt_tainted:
                    invalid_reason = "tainted_assistant_fabrication"
                elif attempt_infra:
                    invalid_reason = "infra_failure"
                elif attempt_no_valid_code:
                    invalid_reason = "missing_valid_code"
                else:
                    invalid_reason = "missing_description"
                diagnosis = (
                    f"INVALID_ATTEMPT {invalid_reason}; retrying without scoring "
                    "this as B evidence.")
                diagnoses.append(diagnosis)
                print(
                    f"[{index:02d}/25] {oid} attempt {attempt}/{args.attempts}: "
                    f"{diagnosis}",
                    flush=True,
                )
                with open(os.path.join(ws, "predicates.py"), "w", encoding="utf-8") as f:
                    f.write(admitted_library)
                try:
                    os.unlink(os.path.join(ws, f"semantic_candidate_names_{oid}.json"))
                except OSError:
                    pass
                continue
            valid_attempt_seen = True
            last_valid_semantic = semantic
            last_valid_transcript_path = transcript_path
            last_valid_description_path = description_path
            last_valid_description_present = description_present
            if semantic.accepted:
                break
            diagnosis = failure_diagnosis(semantic, description_present)
            diagnoses.append(diagnosis)
            print(
                f"[{index:02d}/25] {oid} attempt {attempt}/{args.attempts}: "
                f"{diagnosis}",
                flush=True,
            )
        semantic = last_valid_semantic if valid_attempt_seen else API.SemanticCVResult(
            False, 0.0, 0.0, 12, 12, 12, "NO_VALID_B_ATTEMPT",
            0.0, 0.0, 0.0, args.semantic_max_cv_errors,
            args.semantic_max_train_errors, 0)
        infra_failure = any_infra_failure
        if infra_failure and not valid_attempt_seen and not args.keep_going_on_infra:
            print(
                f"[{index:02d}/25] {oid}: stopping on infrastructure failure; "
                "no valid B evidence recorded. Re-run resumes at this ID.",
                flush=True,
            )
            break

        candidate = read_text(os.path.join(ws, "predicates.py"))
        if valid_attempt_seen:
            transcript_path = last_valid_transcript_path
            description_path = last_valid_description_path
            description_present = last_valid_description_present
        else:
            description_file = os.path.join(ws, f"descriptions_{oid}.md")
            description_present = bool(read_text(description_file).strip())

        valid_evidence = valid_attempt_seen
        admitted = valid_evidence and semantic.accepted
        if admitted:
            admitted_library = candidate
            marginal_c = L.marginal_complexity(before, candidate)
        else:
            marginal_c = 0
            with open(os.path.join(ws, "predicates.py"), "w", encoding="utf-8") as f:
                f.write(admitted_library)
            try:
                os.unlink(os.path.join(ws, f"semantic_candidate_names_{oid}.json"))
            except OSError:
                pass

        records.append(BRecord(
            opaque_id=oid,
            category=problem.category,
            a_rule=a_record.get("rule", ""),
            a_rule_cost=float(a_record.get("rule_cost", 0.0)),
            solved=admitted,
            cv_accuracy=semantic.cv_accuracy,
            train_accuracy=semantic.train_accuracy,
            rule=semantic.rule,
            marginal_C=marginal_c,
            predicate_errors=semantic.predicate_errors,
            cv_errors=semantic.cv_errors,
            train_errors=semantic.train_errors,
            n_examples=semantic.n_examples,
            max_cv_errors=semantic.max_cv_errors,
            max_train_errors=semantic.max_train_errors,
            threshold=semantic.threshold,
            fold_threshold_min=semantic.fold_threshold_min,
            fold_threshold_max=semantic.fold_threshold_max,
            attempts=attempt_count,
            infra_failure=infra_failure,
            description_present=description_present,
            failure_diagnoses=diagnoses,
            transcript_path=transcript_path,
            description_path=description_path if description_present else "",
            valid_evidence=valid_evidence,
            invalid_reason="" if valid_evidence else (invalid_reason or "no_valid_attempt"),
        ))
        save_artifact(out_dir, args.out_tag, args.source_tag, args.model,
                      records, admitted_library)
        status = "SOLVED" if admitted else "FAILED"
        print(
            f"[{index:02d}/25] {oid}: {status} "
            f"cv_errors={semantic.cv_errors}/{semantic.n_examples} "
            f"train_errors={semantic.train_errors}/{semantic.n_examples} "
            f"C+={marginal_c} description={description_present} "
            f"infra={infra_failure} rule={semantic.rule}",
            flush=True,
        )

    save_artifact(out_dir, args.out_tag, args.source_tag, args.model,
                  records, admitted_library)
    print(
        f"COMPLETE attempted={len(records)}/25 solved={sum(r.solved for r in records)}/25 "
        f"described={sum(r.description_present for r in records)}/25 "
        f"infra={sum(r.infra_failure for r in records)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tag", default="logo_full")
    parser.add_argument("--out-tag", default="ab_describe_sonnet_b25_local")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--minutes", type=int, default=8)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--call-timeout", type=float, default=120.0)
    parser.add_argument("--verify-timeout", type=float, default=45.0)
    parser.add_argument("--semantic-max-cv-errors", type=int, default=1)
    parser.add_argument("--semantic-max-train-errors", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ignore any checkpoint/predicate library for the output tag and start at problem_00",
    )
    parser.add_argument(
        "--keep-going-on-infra",
        action="store_true",
        help="record no-valid-attempt infrastructure failures and continue to later IDs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
