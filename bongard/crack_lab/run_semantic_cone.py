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
import math
import os
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_artifacts
import semantic_replay
import phase_d_protocol
import collect_phase_d as artifact_io
from cofibered_proposer import (
    AnthropicCofiberedProposer,
    CODEX_DEFAULT_MODEL,
    CodexCofiberedProposer,
    MODEL_MAP,
    ProposalBundle,
)
from dataset import write_panels
from replay_semantic_runspec import verifier_related_sources
from semantic_legs import default_registry
from semantic_selection import (
    CandidateEvaluation,
    RISK_FIELDS,
    Track,
    rank_candidates,
)
from semantic_ir import SemanticHypothesis
from semantic_verifier import ConeVerification, verify_hypothesis


SELECTION_RISK_FIELDS = (
    "R_support",
    "R_rotated_LOO",
    "R_naturality",
    "R_parser_stability",
)

TERMINAL_EVIDENCE_SCHEMA = "bongard.semantic-terminal-evidence/v1"
SEMANTIC_RUNS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "semantic_runs")


def _phase_python_hash_runtime(execution_policy: dict) -> dict:
    runtime = execution_policy.get("runtime", {})
    return {
        "python_hash_seed_env": runtime.get("python_hash_seed_env"),
        "python_hash_probes": runtime.get("python_hash_probes"),
    }


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
    rotated_loo_errors: int
    rotated_loo_checks: int
    n_examples: int
    complexity: int
    rounds_used: int
    proposer_kind: str
    track: str
    condition: str
    sharing_policy: str
    corpus_digest: str
    panel_set_digest: str
    control_digest: str
    status: str
    proposer_error: str
    candidates: list[dict]
    candidate_manifest: list[dict]
    selection: dict = field(default_factory=dict)
    terminal_evidence: dict = field(default_factory=dict)
    terminal_evidence_digest: str = ""
    replay_spec_digest: str = ""
    phase_execution_binding_digest: str = ""


def _panel_pngs(workspace: str, opaque_id: str) -> list[str]:
    pdir = os.path.join(workspace, opaque_id)
    paths = []
    for side in ("pos", "neg"):
        for i in range(6):
            png = os.path.join(pdir, f"{side}_{i}.png")
            if os.path.exists(png):
                paths.append(png)
    if len(paths) != 12:
        raise RuntimeError(
            f"{opaque_id}: proposer presentation requires exactly 12 PNGs, "
            f"found {len(paths)}")
    return paths


def _write_replay_spec(args: argparse.Namespace, out_dir: str, opaque_id: str,
                       problem, hypothesis: dict,
                       verification: ConeVerification,
                       registry,
                       candidate_verifications: list[ConeVerification],
                       candidate_hypotheses: list[dict],
                       candidate_origins: list[dict],
                       corpus_manifest: dict,
                       corpus_entry: dict,
                       control_manifest: dict | None,
                       control_entry: dict | None,
                       corpus_bundle_digest: str,
                       terminal_evidence: dict,
                       ) -> semantic_replay.SemanticRunSpec:
    policy = semantic_replay.VerifierPolicy(
        max_support_errors=args.max_support_errors,
        max_loo_errors=args.max_loo_errors,
        max_rotated_loo_errors=args.max_rotated_loo_errors,
        require_threshold_overlap=False,
        transform_policy=(
            "execute declared exact panel morphisms for admission; record "
            "rotation/reflection/translation battery as selection risk"),
        unexecuted_checks=(
            "contrast", "counterfactual", "archive_regression"),
    )
    build_kwargs = {
        "opaque_id": opaque_id,
        "problem": problem,
        "cones": [hypothesis],
        "registry": registry,
        "verifier": verify_hypothesis,
        "policy": policy,
        "expected_verifications": {
            verification.hypothesis_id: verification.to_dict(),
        },
        "provenance": {
            "runner": "semantic_cone",
            "dataset": {
                "source": args.source,
                "seed": args.seed,
                "limit_per_source": args.limit,
                "count_policy": phase_d_protocol.COUNT_POLICY,
                "order_policy": corpus_manifest["sampling"]["order_policy"],
                "repository_commit": corpus_manifest["sampling"][
                    "dataset_revision"],
                "corpus_digest": corpus_manifest["corpus_digest"],
                "corpus_bundle_digest": corpus_bundle_digest,
                "panel_set_digest": (
                    control_entry["controlled_panel_set_digest"]
                    if control_entry is not None else
                    corpus_entry["panel_set_digest"]),
                "panels": "self-contained; source identifier redacted",
            },
            "experiment": {
                "track": "SEMANTIC-PURE",
                "condition": args.condition,
                "sharing_policy": phase_d_protocol.SHARED,
                "phase_execution_binding": getattr(
                    args, "phase_execution_binding", {}),
                "control": (
                    {
                        "schema": control_manifest["schema"],
                        "control_digest": control_manifest["control_digest"],
                        "base_corpus_digest": control_manifest[
                            "base_corpus_digest"],
                        "seed": control_manifest["seed"],
                        "replicate": control_manifest["replicate"],
                        "assignment_policy": control_manifest[
                            "assignment_policy"],
                        "problem_assignment": control_entry,
                    }
                    if control_manifest is not None else None),
            },
            "proposer": {
                "kind": args.proposer,
                "model": args.model,
                "round_limit": args.rounds,
            },
            "python_hash_runtime": getattr(
                args, "phase_python_hash_runtime", {}),
            "selection": _selection_evidence(
                candidate_verifications, candidate_hypotheses,
                verification, args.lambda_value, candidate_origins),
            "terminal": {
                "schema": terminal_evidence["schema"],
                "proposal_outcome": terminal_evidence["proposal_outcome"],
                "rounds": terminal_evidence["rounds"],
                "evidence_digest": semantic_replay.canonical_json_digest(
                    terminal_evidence),
            },
            "scope_note": (
                "The proposer saw all 12 labeled panels. LOO diagnostics "
                "refit only the scalar threshold; they are not an untouched "
                "representation-level generalization estimate."
            ),
        },
        "include_ground_truth": False,
    }
    # Fingerprint these imported modules in addition to
    # semantic_verifier.py, closing provenance over the compiler/gate.
    build_kwargs["verifier_sources"] = verifier_related_sources()
    spec = semantic_replay.build_runspec(**build_kwargs)
    path = os.path.join(out_dir, "replay_specs", f"{opaque_id}.json")
    semantic_replay.save_runspec(path, spec, create_parents=True)
    return spec


def _dataset_revision(dataset_dir: str) -> str:
    return phase_d_protocol.dataset_revision(dataset_dir)


def _candidate_evaluations(
        candidates: list[ConeVerification],
        lambda_value: float) -> list[tuple[ConeVerification, CandidateEvaluation]]:
    return [
        (verification, CandidateEvaluation(
            candidate_id=f"{verification.hypothesis_id}@{index:03d}",
            track=Track.SEMANTIC_PURE,
            # MDL breaks ties among candidates that passed the active exact
            # verifier policy; it must not trade a support/CV failure for a
            # shorter cone and hide an available exact solve.
            semantic_admissible=verification.accepted,
            risk=verification.risk,
            complexity=verification.complexity_breakdown,
            lambda_value=lambda_value,
            diagnostics=tuple(
                item for item in (
                    verification.compile_error,
                    verification.semantic_issue,
                ) if item),
            metadata={
                "hypothesis_id": verification.hypothesis_id,
                "candidate_index": index,
            },
        ))
        for index, verification in enumerate(candidates)
    ]


def _selection_evidence(
        candidates: list[ConeVerification], candidate_hypotheses: list[dict],
        selected: ConeVerification | None, lambda_value: float,
        candidate_origins: list[dict] | None = None) -> dict:
    """Cold-replayable evidence for the complete MDL selection decision."""
    if len(candidates) != len(candidate_hypotheses):
        raise ValueError("selection candidates and hypotheses must align")
    if candidate_origins is None:
        candidate_origins = [
            {
                "round": 0,
                "round_candidate_index": index,
                "round_candidate_count": len(candidates),
            }
            for index in range(len(candidates))
        ]
    if len(candidate_origins) != len(candidates):
        raise ValueError("selection candidate origins must align")
    if not candidates:
        if selected is not None:
            raise ValueError("an empty candidate set cannot have a selection")
        return {
            "lambda": lambda_value,
            "risk_fields": list(SELECTION_RISK_FIELDS),
            "unmeasured_risks": [
                name for name in RISK_FIELDS
                if name not in SELECTION_RISK_FIELDS
            ],
            "selected_candidate_id": "",
            "selected_record": {},
            "candidate_manifest": [],
            "selector_fingerprint": semantic_replay.callable_fingerprint(
                _select, require_source=True),
            "candidates": [],
        }
    if selected is None:
        raise ValueError("a nonempty candidate set must have a selection")
    pairs = _candidate_evaluations(candidates, lambda_value)
    records = []
    selected_id = ""
    for (verification, evaluation), hypothesis, origin in zip(
            pairs, candidate_hypotheses, candidate_origins):
        if verification is selected:
            selected_id = evaluation.candidate_id
        records.append({
            "candidate_id": evaluation.candidate_id,
            "hypothesis": hypothesis,
            "expected_verification": verification.to_dict(),
            "evaluation": evaluation.to_dict(),
            "origin": dict(origin),
        })
    if not selected_id:
        raise ValueError("selected verification is absent from candidate set")
    return {
        "lambda": lambda_value,
        "risk_fields": list(SELECTION_RISK_FIELDS),
        "unmeasured_risks": [
            name for name in RISK_FIELDS
            if name not in SELECTION_RISK_FIELDS
        ],
        "selected_candidate_id": selected_id,
        "selected_record": _selection_record(
            selected, candidates, lambda_value),
        "candidate_manifest": _candidate_manifest(
            candidates, candidate_hypotheses, candidate_origins),
        "selector_fingerprint": semantic_replay.callable_fingerprint(
            _select, require_source=True),
        "candidates": records,
    }


def _proposal_round_evidence(bundle: ProposalBundle, round_index: int) -> dict:
    """Bind one completed proposer turn to its exact semantic cones."""
    return {
        "round": round_index,
        "proposer_kind": bundle.proposer_kind,
        "parse_error": bundle.parse_error,
        "candidate_count": len(bundle.hypotheses),
        "candidate_ids": [
            hypothesis.hypothesis_id for hypothesis in bundle.hypotheses],
        "hypothesis_digests": [
            semantic_replay.semantic_cone_digest(hypothesis.to_dict())
            for hypothesis in bundle.hypotheses],
        "model_receipts": [dict(item) for item in bundle.model_receipts],
    }


def _terminal_evidence(
        rounds: list[dict], candidates: list[ConeVerification],
        candidate_hypotheses: list[dict],
        selected: ConeVerification | None, lambda_value: float,
        candidate_origins: list[dict]) -> dict:
    """Create replay-complete evidence for solved and failed outcomes alike."""
    if not rounds:
        raise ValueError("a terminal outcome must contain a completed proposer round")
    selection = _selection_evidence(
        candidates, candidate_hypotheses, selected, lambda_value,
        candidate_origins)
    if candidates:
        proposal_outcome = "CANDIDATES"
    elif rounds[-1]["parse_error"]:
        proposal_outcome = "PROPOSER_PARSE_FAILED"
    else:
        proposal_outcome = "NO_PROPOSALS"
    return {
        "schema": TERMINAL_EVIDENCE_SCHEMA,
        "proposal_outcome": proposal_outcome,
        "rounds": [dict(item) for item in rounds],
        "selection": selection,
    }


def _same_json(left, right) -> bool:
    return semantic_replay.canonical_json_digest([left]) == \
        semantic_replay.canonical_json_digest([right])


def _replay_terminal_record(
        record: ProblemResult | dict, problem, *,
        max_support_errors: int, max_loo_errors: int,
        max_rotated_loo_errors: int, lambda_value: float,
        round_limit: int | None = None,
        registry=None) -> dict:
    """Recompute a terminal record solely from panels and recorded cones.

    The returned payload is useful to promotion code as well as resume.  Every
    user-visible scientific summary is compared here; callers cannot validate
    the cone evidence and then accidentally trust an unrelated checkpoint
    status, rule, error count, description, or selection.
    """
    raw = asdict(record) if isinstance(record, ProblemResult) else dict(record)
    evidence = raw.get("terminal_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
            "schema", "proposal_outcome", "rounds", "selection"}:
        raise ValueError("terminal_evidence has an invalid shape")
    if evidence.get("schema") != TERMINAL_EVIDENCE_SCHEMA:
        raise ValueError("terminal_evidence uses an unsupported schema")
    evidence_digest = semantic_replay.canonical_json_digest(evidence)
    if raw.get("terminal_evidence_digest") != evidence_digest:
        raise ValueError("terminal_evidence_digest does not reproduce")

    rounds = evidence.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("terminal evidence must contain completed rounds")
    if round_limit is not None and (
            isinstance(round_limit, bool) or not isinstance(round_limit, int)
            or round_limit < 1 or len(rounds) > round_limit):
        raise ValueError("terminal evidence exceeds the configured round limit")
    round_keys = {
        "round", "proposer_kind", "parse_error", "candidate_count",
        "candidate_ids", "hypothesis_digests",
    }
    for index, round_record in enumerate(rounds):
        if not isinstance(round_record, dict) \
                or frozenset(round_record) not in {
                    frozenset(round_keys),
                    frozenset(round_keys | {"model_receipts"}),
                }:
            raise ValueError(f"terminal round {index} has an invalid shape")
        candidate_count = round_record.get("candidate_count")
        if isinstance(candidate_count, int) \
                and not isinstance(candidate_count, bool) \
                and candidate_count > 8:
            raise ValueError(
                f"terminal round {index} exceeds the proposer candidate bound")
        model_receipts = round_record.get("model_receipts")
        if isinstance(model_receipts, list) and len(model_receipts) > 3:
            raise ValueError(
                f"terminal round {index} exceeds the proposer receipt bound")
        if round_record["round"] != index \
                or isinstance(round_record["candidate_count"], bool) \
                or not isinstance(round_record["candidate_count"], int) \
                or not 0 <= round_record["candidate_count"] <= 8 \
                or not isinstance(round_record["proposer_kind"], str) \
                or not round_record["proposer_kind"] \
                or not isinstance(round_record["parse_error"], str) \
                or not isinstance(round_record["candidate_ids"], list) \
                or not isinstance(round_record["hypothesis_digests"], list) \
                or ("model_receipts" in round_record and not isinstance(
                    round_record["model_receipts"], list)) \
                or len(round_record["candidate_ids"]) != \
                round_record["candidate_count"] \
                or len(round_record["hypothesis_digests"]) != \
                round_record["candidate_count"]:
            raise ValueError(f"terminal round {index} is internally inconsistent")
    if round_limit is not None and sum(
            item["candidate_count"] for item in rounds) > 8 * round_limit:
        raise ValueError("terminal candidates exceed the configured proposer bound")

    selection = evidence.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("terminal selection evidence must be an object")
    candidate_records = selection.get("candidates")
    if not isinstance(candidate_records, list):
        raise ValueError("terminal selection candidates must be a list")
    active_registry = registry or default_registry()
    hypotheses: list[SemanticHypothesis] = []
    hypothesis_payloads: list[dict] = []
    verifications: list[ConeVerification] = []
    origins: list[dict] = []
    expected_candidate_keys = {
        "candidate_id", "hypothesis", "expected_verification",
        "evaluation", "origin",
    }
    for index, candidate in enumerate(candidate_records):
        if not isinstance(candidate, dict) \
                or set(candidate) != expected_candidate_keys:
            raise ValueError(f"terminal candidate {index} has an invalid shape")
        try:
            hypothesis = SemanticHypothesis.from_dict(
                dict(candidate["hypothesis"]))
        except Exception as exc:
            raise ValueError(
                f"terminal candidate {index} has an invalid cone: {exc}") from exc
        payload = hypothesis.to_dict()
        if semantic_replay.semantic_cone_digest(payload) != \
                semantic_replay.semantic_cone_digest(candidate["hypothesis"]):
            raise ValueError(
                f"terminal candidate {index} cone does not normalize exactly")
        observed = verify_hypothesis(
            hypothesis, active_registry, problem,
            max_support_errors=max_support_errors,
            max_loo_errors=max_loo_errors,
            max_rotated_loo_errors=max_rotated_loo_errors,
        )
        if not _same_json(
                observed.to_dict(), candidate.get("expected_verification")):
            raise ValueError(
                f"terminal candidate {index} verification does not replay")
        origin = candidate.get("origin")
        if not isinstance(origin, dict):
            raise ValueError(f"terminal candidate {index} lacks an origin")
        hypotheses.append(hypothesis)
        hypothesis_payloads.append(payload)
        verifications.append(observed)
        origins.append(dict(origin))

    selected = _select(verifications, lambda_value)
    reproduced_selection = _selection_evidence(
        verifications, hypothesis_payloads, selected, lambda_value, origins)
    if not _same_json(selection, reproduced_selection):
        raise ValueError("terminal selection evidence does not reproduce")

    for round_index, round_record in enumerate(rounds):
        round_candidates = [
            (candidate, hypothesis_payloads[index])
            for index, candidate in enumerate(candidate_records)
            if candidate["origin"].get("round") == round_index
        ]
        round_candidates.sort(
            key=lambda item: item[0]["origin"].get(
                "round_candidate_index", -1))
        expected_ids = [
            payload.get("hypothesis_id") for _candidate, payload
            in round_candidates]
        expected_digests = [
            semantic_replay.semantic_cone_digest(payload)
            for _candidate, payload in round_candidates]
        if round_record["candidate_count"] != len(round_candidates) \
                or not _same_json(round_record["candidate_ids"], expected_ids) \
                or not _same_json(
                    round_record["hypothesis_digests"], expected_digests):
            raise ValueError(
                f"terminal round {round_index} does not bind its candidates")
    if sum(item["candidate_count"] for item in rounds) != len(candidate_records):
        raise ValueError("terminal rounds do not cover every candidate")

    last_round = rounds[-1]
    if candidate_records:
        proposal_outcome = "CANDIDATES"
    elif last_round["parse_error"]:
        proposal_outcome = "PROPOSER_PARSE_FAILED"
    else:
        proposal_outcome = "NO_PROPOSALS"
    if evidence.get("proposal_outcome") != proposal_outcome:
        raise ValueError("terminal proposal outcome does not reproduce")

    n_examples = len(problem.pos) + len(problem.neg)
    if selected is None:
        status = proposal_outcome
        canonical = {
            "solved": False,
            "selected_hypothesis": "",
            "selected_description": "",
            "selected_rule": "",
            "support_errors": n_examples,
            "loo_errors": n_examples,
            "rotated_loo_errors": 2 * len(problem.pos) * len(problem.neg),
            "rotated_loo_checks": 2 * len(problem.pos) * len(problem.neg),
            "n_examples": n_examples,
            "complexity": 0,
            "status": status,
            "candidates": [],
            "candidate_manifest": [],
            "selection": {},
        }
        selected_payload = None
        selected_verification = None
    else:
        selected_index = next(
            index for index, item in enumerate(verifications)
            if item is selected)
        status = _status_of(selected, ProposalBundle(
            problem_id=str(raw.get("opaque_id", "")),
            hypotheses=(), raw_text="",
            proposer_kind=last_round["proposer_kind"],
            parse_error=last_round["parse_error"],
        ))
        exact_run_policy = (
            max_support_errors == 0 and max_loo_errors == 0
            and max_rotated_loo_errors == 0)
        if status.startswith("SOLVED_SEMANTIC_PURE") \
                and not exact_run_policy:
            status = "EXACT_SEMANTIC_FIT_TOLERANT_RUN_POLICY"
        solved = status.startswith("SOLVED_SEMANTIC_PURE")
        selected_payload = hypothesis_payloads[selected_index]
        selected_verification = selected.to_dict()
        canonical = {
            "solved": solved,
            "selected_hypothesis": selected.hypothesis_id,
            "selected_description": hypotheses[selected_index].description,
            "selected_rule": selected.rule,
            "support_errors": selected.support_errors,
            "loo_errors": selected.loo_errors,
            "rotated_loo_errors": selected.rotated_loo_errors,
            "rotated_loo_checks": selected.rotated_loo_checks,
            "n_examples": selected.n_examples,
            "complexity": selected.complexity,
            "status": status,
            "candidates": [item.to_dict() for item in verifications],
            "candidate_manifest": reproduced_selection["candidate_manifest"],
            "selection": reproduced_selection["selected_record"],
        }
    canonical.update({
        "rounds_used": len(rounds),
        "proposer_kind": last_round["proposer_kind"],
        "proposer_error": last_round["parse_error"],
    })
    for name, expected in canonical.items():
        if name not in raw or not _same_json(raw[name], expected):
            raise ValueError(
                f"terminal summary field {name!r} does not replay")
    if canonical["solved"]:
        digest = raw.get("replay_spec_digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("solved terminal record lacks a replay spec digest")
    elif raw.get("replay_spec_digest") != "":
        raise ValueError("failed terminal record must not claim a replay spec")
    return {
        "summary": canonical,
        "selected_hypothesis": selected_payload,
        "selected_verification": selected_verification,
        "selection_evidence": reproduced_selection,
        "terminal_evidence_digest": evidence_digest,
    }


def _result_payload(problem, record: ProblemResult) -> dict:
    """Harness-side result row with a fully reconciled scientific summary."""
    return {
        "problem_id": problem.problem_id,
        "category": problem.category,
        "concept": problem.concept,
        "track": record.track,
        "condition": record.condition,
        "sharing_policy": record.sharing_policy,
        "corpus_digest": record.corpus_digest,
        "panel_set_digest": record.panel_set_digest,
        "control_digest": record.control_digest,
        "solved": record.solved,
        "status": record.status,
        "rule": record.selected_rule,
        "selected_hypothesis": record.selected_hypothesis,
        "selected_description": record.selected_description,
        "selected_rule": record.selected_rule,
        "support_errors": record.support_errors,
        "loo_errors": record.loo_errors,
        "rotated_loo_errors": record.rotated_loo_errors,
        "rotated_loo_checks": record.rotated_loo_checks,
        "n_examples": record.n_examples,
        "complexity": record.complexity,
        "rounds_used": record.rounds_used,
        "proposer_kind": record.proposer_kind,
        "proposer_error": record.proposer_error,
    }


def _candidate_manifest(
        candidates: list[ConeVerification], candidate_hypotheses: list[dict],
        candidate_origins: list[dict]) -> list[dict]:
    """Bind candidate order, proposal origin, cone, and verifier payload."""
    if not (len(candidates) == len(candidate_hypotheses)
            == len(candidate_origins)):
        raise ValueError("candidate manifest inputs must align")
    manifest = []
    round_indices: dict[int, set[int]] = {}
    round_counts: dict[int, int] = {}
    for index, (verification, hypothesis, origin) in enumerate(zip(
            candidates, candidate_hypotheses, candidate_origins)):
        if not isinstance(hypothesis, dict) \
                or hypothesis.get("hypothesis_id") != verification.hypothesis_id:
            raise ValueError(
                f"candidate {index} hypothesis and verification IDs differ")
        if not isinstance(origin, dict) or set(origin) != {
                "round", "round_candidate_index", "round_candidate_count"}:
            raise ValueError(f"candidate {index} has an invalid origin")
        round_index = origin["round"]
        round_candidate_index = origin["round_candidate_index"]
        round_candidate_count = origin["round_candidate_count"]
        if isinstance(round_index, bool) or not isinstance(round_index, int) \
                or round_index < 0 \
                or isinstance(round_candidate_index, bool) \
                or not isinstance(round_candidate_index, int) \
                or round_candidate_index < 0 \
                or isinstance(round_candidate_count, bool) \
                or not isinstance(round_candidate_count, int) \
                or round_candidate_count <= 0 \
                or round_candidate_index >= round_candidate_count:
            raise ValueError(f"candidate {index} has an invalid origin")
        previous_count = round_counts.setdefault(
            round_index, round_candidate_count)
        if previous_count != round_candidate_count:
            raise ValueError(
                f"proposal round {round_index} has inconsistent counts")
        indices = round_indices.setdefault(round_index, set())
        if round_candidate_index in indices:
            raise ValueError(
                f"proposal round {round_index} repeats candidate index "
                f"{round_candidate_index}")
        indices.add(round_candidate_index)
        manifest.append({
            "candidate_id": f"{verification.hypothesis_id}@{index:03d}",
            "candidate_index": index,
            "round": round_index,
            "round_candidate_index": round_candidate_index,
            "round_candidate_count": round_candidate_count,
            "hypothesis_id": verification.hypothesis_id,
            "hypothesis_digest": semantic_replay.semantic_cone_digest(
                hypothesis),
            "verification_digest": semantic_replay.canonical_json_digest(
                verification.to_dict()),
        })
    for round_index, count in round_counts.items():
        if round_indices[round_index] != set(range(count)):
            raise ValueError(
                f"proposal round {round_index} candidate indices are incomplete")
    return manifest


def _selection_record(
        candidate: ConeVerification | None,
        candidates: list[ConeVerification],
        lambda_value: float) -> dict:
    if candidate is None:
        return {}
    for verification, evaluation in _candidate_evaluations(
            candidates, lambda_value):
        if verification is candidate:
            payload = evaluation.to_dict()
            payload["conditional_risk_fields"] = list(SELECTION_RISK_FIELDS)
            conditional_missing = [
                name for name in SELECTION_RISK_FIELDS
                if getattr(evaluation.risk, name) is None
            ]
            payload["conditional_unmeasured_risks"] = conditional_missing
            payload["conditional_free_energy"] = (
                None if conditional_missing else evaluation.score(
                    risk_fields=SELECTION_RISK_FIELDS)
            )
            return payload
    return {}


def _select(candidates: list[ConeVerification],
            lambda_value: float = 0.02) -> ConeVerification | None:
    if not candidates:
        return None
    pairs = _candidate_evaluations(candidates, lambda_value)
    ranked = rank_candidates(
        [evaluation for _, evaluation in pairs],
        risk_fields=SELECTION_RISK_FIELDS,
    )
    if ranked:
        selected = ranked[0]
        return next(
            verification for verification, evaluation in pairs
            if evaluation is selected)
    # Compilation/semantic failures cannot enter the MDL selector, but the
    # runner still needs one deterministic diagnostic candidate for feedback.
    return min(candidates, key=lambda r: (
        not r.semantic_admissible,
        r.rotated_loo_errors,
        r.loo_errors,
        r.support_errors,
        r.naturality_errors + r.cofibration_errors,
        r.complexity,
        r.hypothesis_id,
    ))


def _panel_name(index: int, n_pos: int = 6) -> str:
    return f"pos_{index}" if index < n_pos else f"neg_{index - n_pos}"


def _score_table(v: ConeVerification) -> str:
    if not v.scores:
        return ""
    def shown(index: int, score: float | None) -> str:
        if score is not None:
            return f"{score:.4g}"
        if index < len(v.score_dispositions):
            return v.score_dispositions[index].upper()
        return "UNKNOWN"

    pos = ", ".join(shown(i, s) for i, s in enumerate(v.scores[:6]))
    neg = ", ".join(
        shown(i, s) for i, s in enumerate(v.scores[6:], start=6))
    return f"  pos scores: [{pos}]\n  neg scores: [{neg}]"


def _misses(v: ConeVerification) -> str:
    if not v.scores \
            or len(v.support_predictions) != len(v.scores) \
            or len(v.support_labels) != len(v.scores):
        return ""
    names = []
    n_pos = sum(v.support_labels)
    for i, (s, predicted, expected) in enumerate(zip(
            v.scores, v.support_predictions, v.support_labels)):
        if predicted != expected:
            shown = (v.score_dispositions[i].upper()
                     if s is None and i < len(v.score_dispositions)
                     else "UNKNOWN" if s is None else f"{s:.4g}")
            names.append(f"{_panel_name(i, n_pos)}(score={shown})")
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
            nat = f"naturality_errors={v.naturality_errors}"
            if v.naturality_errors and v.worst_transform:
                nat += (f" (ROTATION/REFLECTION-UNSTABLE: this measurement "
                        f"changes the decision under '{v.worst_transform}'; "
                        f"drop it for a rotation-invariant one)")
            lines.append(
                f"- {v.hypothesis_id}: accepted={v.accepted} "
                f"support_errors={v.support_errors}/{v.n_examples} "
                f"threshold_loo_errors={v.loo_errors}/{v.n_examples} "
                f"pair_threshold_loo_errors={v.rotated_loo_errors}/"
                f"{v.rotated_loo_checks} "
                f"predicate_errors={v.predicate_errors} "
                f"{nat} "
                f"stress_errors={v.stress_errors} "
                f"structural_absences={v.structural_absences} "
                f"indeterminate_evaluations={v.indeterminate_evaluations} "
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
        "so the MISSING_LEG demand stays visible. Declare every true nuisance "
        "morphism: declared transformations are admission checks, while the "
        "broader rotation/reflection battery is reported separately as stress "
        "risk. For randomly oriented shape concepts prefer measurements that "
        "do not depend on orientation "
        "(residuals of fitted circles/arcs, endpoint/part counts computed "
        "after thinning, principal-axis elongation) over axis-aligned "
        "bounding-box measures or raw skeleton branch/cycle counts, which the "
        "battery rejects as unstable.")
    return "\n".join(lines)


def _status_of(selected: ConeVerification | None,
               bundle: ProposalBundle) -> str:
    if selected is None:
        return "PROPOSER_PARSE_FAILED" if bundle.parse_error else "NO_PROPOSALS"
    if selected.accepted:
        if (selected.support_errors == 0 and selected.loo_errors == 0
                and selected.rotated_loo_errors == 0):
            return (
                "SOLVED_SEMANTIC_PURE"
                if selected.stress_errors == 0 else
                "SOLVED_SEMANTIC_PURE_STRESS_FLAGGED")
        return "APPROXIMATE_SEMANTIC_FIT"
    if selected.semantic_issue == "MISSING_LEG":
        return "MISSING_LEG"
    if selected.compile_error:
        return "COMPILE_FAILED"
    if selected.indeterminate_evaluations:
        return "INDETERMINATE"
    if selected.semantic_issue:
        return "MEASUREMENT_ONLY"
    if selected.unchecked_morphisms:
        return "MORPHISM_UNCHECKED"
    if selected.naturality_errors:
        return "NATURALITY_FAILURE"
    if selected.cofibration_errors:
        return "COFIBRATION_FAILURE"
    return "COUNTEREXAMPLE_FAILURE"


def _make_live_proposer(args: argparse.Namespace):
    """Build the selected transport while keeping one semantic IR contract."""
    if args.proposer == "anthropic":
        return AnthropicCofiberedProposer(args.model, args.max_tokens)
    if args.proposer == "codex":
        return CodexCofiberedProposer(args.model)
    raise SystemExit(f"unsupported semantic proposer {args.proposer!r}")


def run(args: argparse.Namespace) -> None:
    if args.proposer not in {"anthropic", "codex"}:
        raise SystemExit("semantic-cone experiments require a live proposer")
    if args.condition not in {
            phase_d_protocol.OBSERVED, phase_d_protocol.SHUFFLED_SIDES}:
        raise SystemExit("unsupported experiment condition")
    out_dir = os.path.abspath(args.out_dir)
    bongard_root = os.path.realpath(str(semantic_replay.BONGARD_ROOT))
    if os.path.commonpath((bongard_root, os.path.realpath(out_dir))) != bongard_root:
        raise SystemExit("--out-dir must stay inside the bongard working tree")
    for name in ("max_support_errors", "max_loo_errors",
                 "max_rotated_loo_errors"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("limit", "rounds", "max_tokens"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.corpus_size < 0:
        raise SystemExit("--corpus-size must be nonnegative")
    if args.control_seed < 0 or args.control_replicate < 0:
        raise SystemExit("control seed/replicate must be nonnegative")
    if not math.isfinite(args.lambda_value) or args.lambda_value < 0.0:
        raise SystemExit("--lambda-value must be finite and nonnegative")
    all_problems = phase_d_protocol.sample_corpus(
        args.dataset_dir,
        limit_per_source=args.limit,
        seed=args.seed,
        source=args.source,
    )
    corpus_manifest = phase_d_protocol.build_corpus_manifest(
        all_problems,
        source=args.source,
        seed=args.seed,
        limit_per_source=args.limit,
        dataset_revision=_dataset_revision(args.dataset_dir),
        dataset_inputs_digest=phase_d_protocol.dataset_content_digest(
            args.dataset_dir),
    )
    corpus_bundle = phase_d_protocol.build_corpus_bundle(
        all_problems, corpus_manifest)
    control_manifest: dict | None = None
    experiment_problems = all_problems
    if args.condition == phase_d_protocol.SHUFFLED_SIDES:
        control = phase_d_protocol.build_shuffled_sides_control(
            all_problems,
            corpus_manifest,
            seed=args.control_seed,
            replicate=args.control_replicate,
        )
        control_manifest = control.manifest
        experiment_problems = list(control.problems)
    active_size = args.corpus_size or len(all_problems)
    if active_size > len(all_problems):
        raise SystemExit(
            f"--corpus-size {active_size} exceeds frozen maximum corpus "
            f"size {len(all_problems)}")
    problems = experiment_problems[:active_size]
    preregistration_path = getattr(args, "preregistration", "")
    arm_id = getattr(args, "arm_id", "")
    if bool(preregistration_path) != bool(arm_id):
        raise SystemExit(
            "--preregistration and --arm-id must be supplied together")
    preregistration = None
    preregistered_arm = None
    previous_active_size = None
    phase_execution_binding: dict = {}
    phase_predecessor_execution_binding: dict = {}
    phase_execution_binding_history: list[dict] = []
    if preregistration_path:
        preregistration, preregistered_arm = _load_preregistered_semantic_arm(
            preregistration_path,
            arm_id,
            corpus_manifest=corpus_manifest,
            args=args,
            condition=args.condition,
            scale=len(problems),
            control_manifest=control_manifest,
        )
        previous_active_size = _previous_preregistered_family_scale(
            preregistration, preregistered_arm)
        phase_execution_binding_history = \
            phase_d_protocol.execution_binding_family(
                preregistration, preregistered_arm)
        phase_execution_binding = phase_execution_binding_history[-1]
        if len(phase_execution_binding_history) > 1:
            phase_predecessor_execution_binding = \
                phase_execution_binding_history[-2]
    args.phase_execution_binding = phase_execution_binding
    args.phase_predecessor_execution_binding = \
        phase_predecessor_execution_binding
    args.phase_execution_binding_history = phase_execution_binding_history
    args.phase_python_hash_runtime = (
        _phase_python_hash_runtime(preregistration["execution_policy"])
        if preregistration is not None else {})
    # All preregistration and existing-run checks precede every write.  The
    # preflight also cold-replays an existing terminal prefix, so a late
    # checkpoint/policy conflict cannot leave an earlier manifest, control,
    # workspace, or checkpoint mutation behind.
    records, results, promoted_cones = _preflight_existing_run(
        out_dir, args, corpus_manifest, corpus_bundle, control_manifest,
        active_size, all_problems,
        previous_active_size=previous_active_size)
    os.makedirs(out_dir, exist_ok=True)
    _bind_corpus_manifest(out_dir, corpus_manifest)
    _bind_corpus_bundle(out_dir, corpus_bundle, corpus_manifest)
    if control_manifest is not None:
        _bind_control_manifest(out_dir, control_manifest, corpus_manifest)
    print(
        f"frozen corpus {corpus_manifest['corpus_digest']} | "
        f"condition {args.condition} | active prefix "
        f"{active_size}/{len(all_problems)}",
        flush=True,
    )
    if args.prepare_only:
        return

    ws = os.path.join(out_dir, "workspace")
    os.makedirs(ws, exist_ok=True)

    proposer = (
        _make_live_proposer(args) if len(records) < active_size else None)
    registry = default_registry()
    interrupted = False

    for idx, problem in enumerate(problems):
        oid = f"problem_{idx:02d}"
        base_problem = all_problems[idx]
        corpus_entry = corpus_manifest["problems"][idx]
        control_entry = (
            control_manifest["problems"][idx]
            if control_manifest is not None else None)
        if idx < len(records):
            continue
        write_panels(ws, problem, oid)
        pngs = _panel_pngs(ws, oid)

        all_verifications: list[ConeVerification] = []
        descriptions_by_verification: dict[int, str] = {}
        hypotheses_by_verification: dict[int, dict] = {}
        origins_by_verification: dict[int, dict] = {}
        round_trace: list[dict] = []
        bundle: ProposalBundle | None = None
        rounds_used = 0
        infra_error = ""
        for rnd in range(args.rounds):
            try:
                if rnd == 0:
                    assert proposer is not None
                    bundle = proposer.propose(oid, pngs)
                else:
                    assert proposer is not None
                    bundle = proposer.refine(
                        oid, _feedback_text(bundle, round_verifications))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # record per problem; never kill the batch
                infra_error = f"{type(exc).__name__}: {exc}"
                break
            rounds_used = rnd + 1
            round_trace.append(_proposal_round_evidence(bundle, rnd))
            with open(os.path.join(out_dir, f"{oid}_round{rnd:02d}_proposal.txt"),
                      "w", encoding="utf-8") as f:
                f.write(bundle.raw_text +
                        (f"\n\nSCHEMA_ISSUES: {bundle.parse_error}\n"
                         if bundle.parse_error else ""))
            round_verifications = [
                verify_hypothesis(h, registry, problem,
                                  max_support_errors=args.max_support_errors,
                                  max_loo_errors=args.max_loo_errors,
                                  max_rotated_loo_errors=(
                                      args.max_rotated_loo_errors))
                for h in bundle.hypotheses
            ]
            for round_candidate_index, (hypothesis, verification) in enumerate(
                    zip(bundle.hypotheses, round_verifications)):
                descriptions_by_verification[id(verification)] = \
                    hypothesis.description
                hypotheses_by_verification[id(verification)] = \
                    hypothesis.to_dict()
                origins_by_verification[id(verification)] = {
                    "round": rnd,
                    "round_candidate_index": round_candidate_index,
                    "round_candidate_count": len(round_verifications),
                }
            all_verifications.extend(round_verifications)
            selected = _select(all_verifications, args.lambda_value)
            if (selected is not None and selected.accepted
                    and selected.support_errors == 0
                    and selected.loo_errors == 0
                    and selected.rotated_loo_errors == 0):
                break

        selected = _select(all_verifications, args.lambda_value)
        if infra_error:
            semantic_artifacts.snapshot_wip(args.tag, out_dir, oid)
            print(
                f"[{idx + 1:02d}/{len(problems):02d}] {oid} "
                f"PROPOSER_INFRA_PENDING (not counted; resume from round zero): "
                f"{infra_error}",
                flush=True,
            )
            interrupted = True
            break
        protocol_fields = {
            "condition": args.condition,
            "sharing_policy": phase_d_protocol.SHARED,
            "corpus_digest": corpus_manifest["corpus_digest"],
            "panel_set_digest": (
                control_entry["controlled_panel_set_digest"]
                if control_entry is not None else
                corpus_entry["panel_set_digest"]),
            "control_digest": (
                control_manifest["control_digest"]
                if control_manifest is not None else ""),
            "phase_execution_binding_digest": (
                phase_execution_binding.get("binding_digest", "")),
        }
        assert bundle is not None
        if selected is None:
            status = _status_of(None, bundle)
            n_examples = len(problem.pos) + len(problem.neg)
            terminal_evidence = _terminal_evidence(
                round_trace, [], [], None, args.lambda_value, [])
            record = ProblemResult(
                opaque_id=oid, category=problem.category, solved=False,
                selected_hypothesis="", selected_description="",
                selected_rule="", support_errors=n_examples,
                loo_errors=n_examples,
                rotated_loo_errors=2 * len(problem.pos) * len(problem.neg),
                rotated_loo_checks=2 * len(problem.pos) * len(problem.neg),
                n_examples=n_examples, complexity=0,
                rounds_used=rounds_used,
                proposer_kind=bundle.proposer_kind,
                track="SEMANTIC-PURE", status=status,
                proposer_error=bundle.parse_error, candidates=[],
                candidate_manifest=[],
                terminal_evidence=terminal_evidence,
                terminal_evidence_digest=(
                    semantic_replay.canonical_json_digest(terminal_evidence)),
                **protocol_fields)
        else:
            status = _status_of(selected, bundle)
            exact_run_policy = (
                args.max_support_errors == 0
                and args.max_loo_errors == 0
                and args.max_rotated_loo_errors == 0
            )
            if status.startswith("SOLVED_SEMANTIC_PURE") \
                    and not exact_run_policy:
                status = "EXACT_SEMANTIC_FIT_TOLERANT_RUN_POLICY"
            exact_solve = status.startswith("SOLVED_SEMANTIC_PURE")
            candidate_hypothesis_payloads = [
                hypotheses_by_verification[id(candidate)]
                for candidate in all_verifications]
            candidate_origins = [
                origins_by_verification[id(candidate)]
                for candidate in all_verifications]
            selection_evidence = _selection_evidence(
                all_verifications, candidate_hypothesis_payloads,
                selected, args.lambda_value, candidate_origins)
            terminal_evidence = _terminal_evidence(
                round_trace, all_verifications,
                candidate_hypothesis_payloads, selected,
                args.lambda_value, candidate_origins)
            record = ProblemResult(
                opaque_id=oid, category=problem.category,
                solved=exact_solve,
                selected_hypothesis=selected.hypothesis_id,
                selected_description=descriptions_by_verification.get(
                    id(selected), ""),
                selected_rule=selected.rule,
                support_errors=selected.support_errors,
                loo_errors=selected.loo_errors,
                rotated_loo_errors=selected.rotated_loo_errors,
                rotated_loo_checks=selected.rotated_loo_checks,
                n_examples=selected.n_examples,
                complexity=selected.complexity,
                rounds_used=rounds_used,
                proposer_kind=bundle.proposer_kind,
                track="SEMANTIC-PURE", status=status,
                proposer_error=infra_error or bundle.parse_error,
                candidates=[v.to_dict() for v in all_verifications],
                candidate_manifest=selection_evidence["candidate_manifest"],
                selection=selection_evidence["selected_record"],
                terminal_evidence=terminal_evidence,
                terminal_evidence_digest=(
                    semantic_replay.canonical_json_digest(terminal_evidence)),
                **protocol_fields,
            )
        records.append(record)
        # Ground truth stays harness-side; it is written only into the
        # promoted artifact directory, never into the run workspace.
        results[oid] = _result_payload(base_problem, record)
        if record.solved:
            hypothesis_payload = hypotheses_by_verification.get(id(selected), {})
            spec = _write_replay_spec(
                args, out_dir, oid, problem, hypothesis_payload,
                selected, registry, all_verifications,
                candidate_hypothesis_payloads, candidate_origins,
                corpus_manifest, corpus_entry,
                control_manifest, control_entry,
                corpus_bundle["bundle_digest"], terminal_evidence)
            record.replay_spec_digest = spec.spec_digest
            promoted_cones.append({
                "opaque_id": oid,
                "hypothesis": hypothesis_payload,
                "verification": selected.to_dict(),
                "selection": record.selection,
                "runspec_digest": spec.spec_digest,
                "rounds_used": rounds_used,
            })
        payload = _checkpoint_payload(
            args, records, corpus_manifest, active_size, control_manifest,
            corpus_bundle)
        _write_checkpoint(out_dir, payload)
        if record.solved:
            semantic_artifacts.promote(args.tag, out_dir, payload, results,
                                       promoted_cones,
                                       control_manifest=control_manifest)
        else:
            semantic_artifacts.snapshot_wip(args.tag, out_dir, oid)
        print(
            f"[{idx + 1:02d}/{len(problems):02d}] {oid} {record.status} "
            f"rounds={record.rounds_used} "
            f"support_errors={record.support_errors}/{record.n_examples} "
            f"threshold_loo_errors={record.loo_errors}/{record.n_examples} "
            f"pair_threshold_loo_errors={record.rotated_loo_errors}/"
            f"{record.rotated_loo_checks} "
            f"rule={record.selected_rule}",
            flush=True,
        )

    if interrupted:
        # A provider outage is not a scientific failure and must not enter the
        # attempted denominator or overwrite the last complete run artifact.
        _write_checkpoint(
            out_dir,
            _checkpoint_payload(
                args, records, corpus_manifest, active_size,
                control_manifest, corpus_bundle),
        )
        return

    # Finalize the scientific denominator even when every problem failed.
    # If exact cones exist, repeat promotion once with the final checkpoint so
    # failures after the last solve cannot disappear from the artifact.
    final_payload = _checkpoint_payload(
        args, records, corpus_manifest, active_size, control_manifest,
        corpus_bundle)
    _write_checkpoint(out_dir, final_payload)
    _finalize_semantic_artifact(
        args, out_dir, final_payload, results, promoted_cones,
        corpus_manifest, corpus_bundle, control_manifest)
    if preregistration is not None and preregistered_arm is not None:
        path = _publish_phase_d_track_report(
            args.tag, preregistration, preregistered_arm, records)
        print(f"published preregistered track report: {path}", flush=True)


def _checkpoint_payload(args: argparse.Namespace,
                        records: list[ProblemResult],
                        corpus_manifest: dict,
                        active_size: int,
                        control_manifest: dict | None,
                        corpus_bundle: dict) -> dict:
    return {
        "runner": "semantic_cone",
        "artifact_state": "WIP",
        "promotion_policy": (
            "promote only exact typed fits after taint scan and fresh-process "
            "RunSpec replay; persist explicit risk nulls and harness-derived "
            "conditional complexity"
        ),
        "tracks": ["UNRESTRICTED", "SEMANTIC-PURE", "HYBRID"],
        "active_track": "SEMANTIC-PURE",
        "condition": args.condition,
        "sharing_policy": phase_d_protocol.SHARED,
        "phase_execution_binding": getattr(
            args, "phase_execution_binding", {}),
        "phase_execution_binding_history": getattr(
            args, "phase_execution_binding_history", []),
        "phase_python_hash_runtime": getattr(
            args, "phase_python_hash_runtime", {}),
        "control": (
            {
                "schema": control_manifest["schema"],
                "control_digest": control_manifest["control_digest"],
                "seed": control_manifest["seed"],
                "replicate": control_manifest["replicate"],
                "assignment_policy": control_manifest["assignment_policy"],
                "manifest": "control_manifest.json",
            }
            if control_manifest is not None else None),
        "proposer": args.proposer,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "rounds": args.rounds,
        "tag": args.tag,
        "dataset": {
            "source": args.source,
            "seed": args.seed,
            "count_policy": phase_d_protocol.COUNT_POLICY,
            "limit_per_source": args.limit,
            "active_prefix_size": active_size,
            "frozen_problem_count": corpus_manifest["problem_count"],
            "order_policy": corpus_manifest["sampling"]["order_policy"],
            "repository_commit": corpus_manifest["sampling"][
                "dataset_revision"],
            "corpus_digest": corpus_manifest["corpus_digest"],
            "corpus_manifest": "corpus_manifest.json",
            "corpus_bundle_digest": corpus_bundle["bundle_digest"],
            "corpus_bundle": "corpus_panels.json",
            "panel_bytes": (
                "all records bind corpus panel-set digests; solved replay_specs "
                "also embed canonical panel bytes"),
        },
        "verifier_policy": {
            "max_support_errors": args.max_support_errors,
            "max_threshold_loo_errors": args.max_loo_errors,
            "max_pair_threshold_loo_errors": args.max_rotated_loo_errors,
            "exact_status_requires_all_three_zero": True,
            "representation_selection_scope": "all_12_labeled_panels",
        },
        "selection": {
            "method": "conditional_free_energy",
            "lambda": args.lambda_value,
            "risk_fields": list(SELECTION_RISK_FIELDS),
            "unmeasured_risks": [
                name for name in RISK_FIELDS
                if name not in SELECTION_RISK_FIELDS
            ],
        },
        "replay_schema": semantic_replay.RUNSPEC_SCHEMA,
        "solved": sum(r.solved for r in records),
        "attempted": len(records),
        "records": [asdict(r) for r in records],
    }


def _finalize_semantic_artifact(
        args: argparse.Namespace, out_dir: str, final_payload: dict,
        results: dict, promoted_cones: list[dict], corpus_manifest: dict,
        corpus_bundle: dict, control_manifest: dict | None) -> str:
    """Commit one final state without a destructive intermediate artifact."""
    if promoted_cones:
        return semantic_artifacts.promote(
            args.tag, out_dir, final_payload, results, promoted_cones,
            control_manifest=control_manifest)
    return semantic_artifacts.publish_run_report(
        args.tag, final_payload, results, corpus_manifest,
        control_manifest=control_manifest,
        corpus_bundle=corpus_bundle)


def _assert_checkpoint_protocol(
        checkpoint: dict, args: argparse.Namespace,
        corpus_manifest: dict, active_size: int,
        control_manifest: dict | None, corpus_bundle: dict, *,
        artifact_states: tuple[str, ...] = ("WIP",),
        previous_active_size: int | None = None) -> None:
    """Require the exact immutable checkpoint policy before any mutation."""
    if not isinstance(checkpoint, dict):
        raise SystemExit("existing checkpoint must be a JSON object")
    expected = _checkpoint_payload(
        args, [], corpus_manifest, active_size, control_manifest,
        corpus_bundle)
    if set(checkpoint) != set(expected):
        raise SystemExit(
            "existing checkpoint fields differ from the semantic run schema; "
            "choose a fresh --out-dir")
    observed_dataset = checkpoint.get("dataset")
    expected_dataset = expected["dataset"]
    if not isinstance(observed_dataset, dict) \
            or set(observed_dataset) != set(expected_dataset):
        raise SystemExit(
            "existing checkpoint dataset fields differ from the run schema")
    observed_active_size = observed_dataset.get("active_prefix_size")
    if isinstance(observed_active_size, bool) \
            or not isinstance(observed_active_size, int) \
            or observed_active_size <= 0:
        raise SystemExit(
            "existing checkpoint active prefix is invalid")
    if observed_active_size > active_size:
        raise SystemExit(
            "existing checkpoint active prefix exceeds the requested scale")
    current_binding = getattr(args, "phase_execution_binding", {})
    predecessor_binding = getattr(
        args, "phase_predecessor_execution_binding", {})
    full_binding_history = getattr(
        args, "phase_execution_binding_history", [])
    observed_binding = checkpoint.get("phase_execution_binding")
    observed_binding_history = checkpoint.get(
        "phase_execution_binding_history")
    observed_hash_runtime = checkpoint.get("phase_python_hash_runtime")
    if current_binding:
        desired_binding = (
            current_binding if observed_active_size == active_size
            else predecessor_binding)
        desired_history = (
            full_binding_history if desired_binding == current_binding
            else full_binding_history[:-1])
        if observed_binding != desired_binding \
                or observed_binding_history != desired_history:
            raise SystemExit(
                "existing checkpoint Phase execution binding differs from "
                "the current/immediate-predecessor arm")
        try:
            phase_d_protocol.validate_execution_binding(observed_binding)
            for binding in observed_binding_history:
                phase_d_protocol.validate_execution_binding(binding)
        except phase_d_protocol.PhaseDProtocolError as exc:
            raise SystemExit(
                f"existing checkpoint Phase execution binding is invalid: {exc}") \
                from exc
        records_for_binding = checkpoint.get("records")
        if not isinstance(records_for_binding, list) \
                or len(records_for_binding) > observed_binding["scale"]:
            raise SystemExit(
                "existing checkpoint has more records than its Phase scale")
        for index, record in enumerate(records_for_binding):
            expected_binding = next(
                (binding for binding in observed_binding_history
                 if index < binding["scale"]), None)
            if not isinstance(record, dict) or expected_binding is None \
                    or record.get("phase_execution_binding_digest") != \
                    expected_binding["binding_digest"]:
                raise SystemExit(
                    "existing semantic record Phase execution tranche differs")
        expected_hash_runtime = getattr(
            args, "phase_python_hash_runtime", {})
        if observed_hash_runtime != expected_hash_runtime \
                or set(expected_hash_runtime) != {
                    "python_hash_seed_env", "python_hash_probes"} \
                or not isinstance(
                    expected_hash_runtime["python_hash_seed_env"], str) \
                or not isinstance(
                    expected_hash_runtime["python_hash_probes"], list) \
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in expected_hash_runtime[
                           "python_hash_probes"]):
            raise SystemExit(
                "existing checkpoint Python hash runtime differs from Phase")
    elif observed_binding != {} or observed_binding_history != []:
        raise SystemExit(
            "unpreregistered semantic run carries Phase execution provenance")
    elif observed_hash_runtime != {}:
        raise SystemExit(
            "unpreregistered semantic run carries Phase hash provenance")
    elif any(
            isinstance(record, dict)
            and record.get("phase_execution_binding_digest")
            for record in checkpoint.get("records", [])):
        raise SystemExit(
            "unbound semantic checkpoint record claims Phase provenance")
    if observed_active_size != active_size:
        if previous_active_size is None \
                or observed_active_size != previous_active_size \
                or observed_active_size >= active_size:
            raise SystemExit(
                "existing checkpoint active prefix is not the same or the "
                "immediately preceding preregistered scale")
        records = checkpoint.get("records")
        attempted = checkpoint.get("attempted")
        if isinstance(attempted, bool) or not isinstance(attempted, int) \
                or attempted != observed_active_size \
                or not isinstance(records, list) \
                or len(records) != observed_active_size:
            raise SystemExit(
                "smaller checkpoint must be a completed terminal prefix "
                "before monotone growth")
    for name in sorted(set(expected_dataset) - {"active_prefix_size"}):
        if not _same_json(observed_dataset[name], expected_dataset[name]):
            raise SystemExit(
                "existing checkpoint dataset/corpus policy differs from this "
                "run")

    dynamic = {
        "solved", "attempted", "records", "dataset",
        "phase_execution_binding", "phase_execution_binding_history",
    }
    for name in sorted(set(expected) - dynamic):
        if name == "artifact_state":
            if checkpoint[name] not in artifact_states:
                raise SystemExit(
                    "existing checkpoint has an incompatible artifact state")
            continue
        if not _same_json(checkpoint[name], expected[name]):
            raise SystemExit(
                "existing checkpoint uses a different corpus/control/active-"
                "prefix/run policy; choose a fresh --out-dir")


def _load_resume_state(
        out_dir: str,
        args: argparse.Namespace,
        corpus_manifest: dict,
        control_manifest: dict | None,
        active_size: int,
        base_problems: list,
        corpus_bundle: dict, *,
        artifact_states: tuple[str, ...] = ("WIP",),
        previous_active_size: int | None = None) -> tuple[
            list[ProblemResult], dict[str, dict], list[dict]]:
    """Load a contiguous terminal prefix after proving protocol identity.

    Conversations are deliberately not resumed.  An interrupted problem has
    no record and restarts from round zero; terminal records are immutable.
    """
    path = os.path.join(out_dir, "checkpoint.json")
    try:
        checkpoint = artifact_io._load_json(path, "semantic checkpoint")
    except artifact_io.CampaignCollectionError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return [], {}, []
        raise SystemExit(f"cannot resume invalid checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise SystemExit("cannot resume invalid checkpoint: expected JSON object")
    _assert_checkpoint_protocol(
        checkpoint, args, corpus_manifest, active_size,
        control_manifest, corpus_bundle, artifact_states=artifact_states,
        previous_active_size=previous_active_size)
    expected_control_digest = (
        control_manifest["control_digest"]
        if control_manifest is not None else None)
    raw_records = checkpoint.get("records")
    if not isinstance(raw_records, list) or len(raw_records) > active_size:
        raise SystemExit("checkpoint record prefix is invalid for active scale")
    if checkpoint.get("attempted") != len(raw_records) \
            or checkpoint.get("solved") != sum(
                bool(record.get("solved")) for record in raw_records
                if isinstance(record, dict)):
        raise SystemExit("checkpoint aggregate counts do not reproduce")

    try:
        replay_base = phase_d_protocol.problems_from_corpus_bundle(
            corpus_bundle, corpus_manifest)
        if control_manifest is None:
            replay_problems = replay_base
        else:
            replay_control = phase_d_protocol.build_shuffled_sides_control(
                replay_base, corpus_manifest,
                seed=control_manifest["seed"],
                replicate=control_manifest["replicate"],
            )
            if replay_control.manifest["control_digest"] != \
                    control_manifest["control_digest"]:
                raise ValueError("control digest does not reproduce")
            replay_problems = replay_control.problems
    except Exception as exc:
        raise SystemExit(
            f"cannot reconstruct checkpoint replay panels: {exc}") from exc

    records: list[ProblemResult] = []
    results: dict[str, dict] = {}
    promoted_cones: list[dict] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise SystemExit(f"checkpoint record {index} is not an object")
        try:
            record = ProblemResult(**raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"checkpoint record {index} is malformed: {exc}") \
                from exc
        oid = f"problem_{index:02d}"
        control_entry = (
            control_manifest["problems"][index]
            if control_manifest is not None else None)
        expected_panel_digest = (
            control_entry["controlled_panel_set_digest"]
            if control_entry is not None else
            corpus_manifest["problems"][index]["panel_set_digest"])
        if record.opaque_id != oid \
                or record.track != "SEMANTIC-PURE" \
                or record.condition != args.condition \
                or record.sharing_policy != phase_d_protocol.SHARED \
                or record.corpus_digest != corpus_manifest["corpus_digest"] \
                or record.panel_set_digest != expected_panel_digest \
                or record.control_digest != (expected_control_digest or "") \
                or record.status == "PROPOSER_INFRA_FAILURE":
            raise SystemExit(f"checkpoint record {oid} violates corpus/arm identity")
        if getattr(args, "phase_execution_binding", {}):
            try:
                phase_d_protocol.validate_semantic_proposer_receipts(
                    raw, oid, {
                        "rounds": args.rounds,
                        "concrete_model": MODEL_MAP.get(args.model, args.model),
                        "max_model_attempts_per_round": (
                            phase_d_protocol.SEMANTIC_MAX_MODEL_ATTEMPTS_PER_ROUND
                        ),
                    })
            except (phase_d_protocol.PhaseDProtocolError,
                    RuntimeError, ValueError) as exc:
                raise SystemExit(
                    f"checkpoint semantic proposer receipt is invalid for "
                    f"{oid}: {exc}") from exc
        try:
            replayed = _replay_terminal_record(
                record, replay_problems[index],
                max_support_errors=args.max_support_errors,
                max_loo_errors=args.max_loo_errors,
                max_rotated_loo_errors=args.max_rotated_loo_errors,
                lambda_value=args.lambda_value,
                round_limit=args.rounds,
            )
        except Exception as exc:
            raise SystemExit(
                f"checkpoint terminal record {oid} does not replay: {exc}") \
                from exc
        records.append(record)
        problem = base_problems[index]
        results[oid] = _result_payload(problem, record)
        if not record.solved:
            continue
        spec_path = os.path.join(out_dir, "replay_specs", f"{oid}.json")
        try:
            spec = semantic_replay.load_runspec(spec_path)
        except Exception as exc:
            raise SystemExit(f"cannot resume solved record {oid}: {exc}") from exc
        if record.replay_spec_digest != spec.spec_digest \
                or len(spec.cones) != 1 \
                or spec.panel_set_digest != record.panel_set_digest:
            raise SystemExit(f"solved record {oid} differs from its RunSpec")
        cone = spec.cones[0]
        if cone.expected_verification is None:
            raise SystemExit(f"solved record {oid} lacks replay verification")
        provenance = dict(spec.provenance)
        terminal_provenance = provenance.get("terminal")
        experiment_provenance = provenance.get("experiment")
        proposer_provenance = provenance.get("proposer")
        expected_record_binding = next(
            (binding for binding in getattr(
                args, "phase_execution_binding_history", [])
             if index < binding["scale"]),
            {},
        )
        if not isinstance(terminal_provenance, dict) \
                or not isinstance(experiment_provenance, dict) \
                or proposer_provenance != {
                    "kind": args.proposer,
                    "model": args.model,
                    "round_limit": args.rounds,
                } \
                or provenance.get("python_hash_runtime", {}) != getattr(
                    args, "phase_python_hash_runtime", {}) \
                or experiment_provenance.get(
                    "phase_execution_binding", {}) != \
                expected_record_binding \
                or record.phase_execution_binding_digest != \
                expected_record_binding.get("binding_digest", "") \
                or terminal_provenance.get("schema") != \
                TERMINAL_EVIDENCE_SCHEMA \
                or terminal_provenance.get("evidence_digest") != \
                record.terminal_evidence_digest \
                or not _same_json(
                    terminal_provenance.get("rounds"),
                    record.terminal_evidence.get("rounds")) \
                or terminal_provenance.get("proposal_outcome") != \
                record.terminal_evidence.get("proposal_outcome") \
                or not _same_json(
                    provenance.get("selection"),
                    replayed["selection_evidence"]) \
                or not _same_json(cone.cone, replayed["selected_hypothesis"]) \
                or not _same_json(
                    cone.expected_verification,
                    replayed["selected_verification"]):
            raise SystemExit(
                f"solved record {oid} terminal evidence differs from its RunSpec")
        promoted_cones.append({
            "opaque_id": oid,
            "hypothesis": dict(cone.cone),
            "verification": dict(cone.expected_verification),
            "selection": dict(record.selection),
            "runspec_digest": spec.spec_digest,
            "rounds_used": record.rounds_used,
        })
    return records, results, promoted_cones


def _write_checkpoint(out_dir: str, payload: dict) -> None:
    semantic_artifacts.atomic_json(
        os.path.join(out_dir, "checkpoint.json"), payload)


def _bind_corpus_manifest(out_dir: str, manifest: dict) -> None:
    """Write once, or prove an existing output directory has the same corpus."""
    path = os.path.join(out_dir, "corpus_manifest.json")
    phase_d_protocol.validate_corpus_manifest(manifest)
    if semantic_artifacts.create_json_once(path, manifest):
        return
    existing = _read_preflight_json(path, "corpus manifest")
    try:
        phase_d_protocol.validate_corpus_manifest(existing)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise SystemExit(
            f"existing corpus manifest is invalid; choose a fresh --out-dir: "
            f"{exc}") from exc
    if existing["corpus_digest"] != manifest["corpus_digest"]:
        raise SystemExit(
            "existing --out-dir is bound to a different corpus; choose a "
            "fresh directory")


def _bind_control_manifest(out_dir: str, control_manifest: dict,
                           corpus_manifest: dict) -> None:
    phase_d_protocol.validate_shuffled_control_manifest(
        control_manifest, corpus_manifest)
    path = os.path.join(out_dir, "control_manifest.json")
    if semantic_artifacts.create_json_once(path, control_manifest):
        return
    existing = _read_preflight_json(path, "control manifest")
    try:
        phase_d_protocol.validate_shuffled_control_manifest(
            existing, corpus_manifest)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise SystemExit(
            f"existing control manifest is invalid; choose a fresh "
            f"--out-dir: {exc}") from exc
    if existing["control_digest"] != control_manifest["control_digest"]:
        raise SystemExit(
            "existing --out-dir is bound to a different control; choose "
            "a fresh directory")


def _bind_corpus_bundle(out_dir: str, bundle: dict,
                        corpus_manifest: dict) -> None:
    phase_d_protocol.validate_corpus_bundle(bundle, corpus_manifest)
    path = os.path.join(out_dir, "corpus_panels.json")
    if semantic_artifacts.create_json_once(path, bundle):
        return
    existing = _read_preflight_json(path, "corpus panel bundle")
    try:
        phase_d_protocol.validate_corpus_bundle(existing, corpus_manifest)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise SystemExit(
            f"existing corpus panel bundle is invalid; choose a fresh "
            f"--out-dir: {exc}") from exc
    if existing["bundle_digest"] != bundle["bundle_digest"]:
        raise SystemExit(
            "existing --out-dir is bound to different corpus panel bytes")


def _read_preflight_json(path: str, label: str) -> dict:
    try:
        value = artifact_io._load_json(path, label)
    except artifact_io.CampaignCollectionError as exc:
        raise SystemExit(
            f"existing {label} is invalid; choose a fresh --out-dir: {exc}") \
            from exc
    if not isinstance(value, dict):
        raise SystemExit(
            f"existing {label} must be a JSON object; choose a fresh --out-dir")
    return value


def _read_optional_preflight_json(path: str, label: str) -> dict | None:
    """Return a stable bounded JSON object, or ``None`` only for ENOENT."""
    try:
        value = artifact_io._load_json(path, label)
    except artifact_io.CampaignCollectionError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise SystemExit(
            f"existing {label} is invalid; choose a fresh --out-dir: {exc}") \
            from exc
    if not isinstance(value, dict):
        raise SystemExit(
            f"existing {label} must be a JSON object; choose a fresh --out-dir")
    return value


def _preflight_existing_run(
        out_dir: str, args: argparse.Namespace,
        corpus_manifest: dict, corpus_bundle: dict,
        control_manifest: dict | None, active_size: int,
        base_problems: list, *,
        previous_active_size: int | None = None) -> tuple[
            list[ProblemResult], dict[str, dict], list[dict]]:
    """Read-only validation of every binding and terminal checkpoint.

    No output path is created or updated here.  Consequently any discoverable
    conflict is reported before the bind/write phase begins, and callers can
    prove the pre-existing tree stayed byte-identical.
    """
    phase_d_protocol.validate_corpus_manifest(corpus_manifest)
    phase_d_protocol.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    if control_manifest is not None:
        phase_d_protocol.validate_shuffled_control_manifest(
            control_manifest, corpus_manifest)
    proposed_checkpoint = _checkpoint_payload(
        args, [], corpus_manifest, active_size, control_manifest,
        corpus_bundle)
    try:
        destination_artifact = \
            semantic_artifacts.assert_artifact_binding_compatible(
                args.tag, proposed_checkpoint, corpus_manifest,
                corpus_bundle, control_manifest)
    except (ValueError, semantic_artifacts.ReplayCertificationError) as exc:
        raise SystemExit(
            f"semantic destination artifact conflicts with this run: {exc}") \
            from exc
    destination_checkpoint = os.path.join(
        destination_artifact, "checkpoint.json")
    destination_state = None
    if _read_optional_preflight_json(
            destination_checkpoint, "destination checkpoint") is not None:
        destination_state = _load_resume_state(
            destination_artifact, args, corpus_manifest, control_manifest,
            active_size, base_problems, corpus_bundle,
            artifact_states=("PROMOTED", "RUN_COMPLETE"),
            previous_active_size=previous_active_size,
        )
    if not os.path.exists(out_dir):
        if previous_active_size is not None:
            raise SystemExit(
                "later preregistered scale requires the complete immediate "
                "predecessor checkpoint")
        return [], {}, []
    if not os.path.isdir(out_dir):
        raise SystemExit("existing --out-dir is not a directory")
    try:
        semantic_artifacts.assert_not_tainted(out_dir)
    except semantic_artifacts.WorkspaceTainted as exc:
        raise SystemExit(
            f"existing --out-dir is tainted before mutation: {exc}") from exc

    manifest_path = os.path.join(out_dir, "corpus_manifest.json")
    bundle_path = os.path.join(out_dir, "corpus_panels.json")
    control_path = os.path.join(out_dir, "control_manifest.json")
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    existing_manifest = _read_optional_preflight_json(
        manifest_path, "corpus manifest")
    if existing_manifest is not None:
        try:
            phase_d_protocol.validate_corpus_manifest(existing_manifest)
        except phase_d_protocol.PhaseDProtocolError as exc:
            raise SystemExit(
                f"existing corpus manifest is invalid; choose a fresh "
                f"--out-dir: {exc}") from exc
        if not _same_json(existing_manifest, corpus_manifest):
            raise SystemExit(
                "existing --out-dir is bound to a different corpus")

    existing_bundle = _read_optional_preflight_json(
        bundle_path, "corpus panel bundle")
    if existing_bundle is not None:
        try:
            phase_d_protocol.validate_corpus_bundle(
                existing_bundle, corpus_manifest)
        except phase_d_protocol.PhaseDProtocolError as exc:
            raise SystemExit(
                f"existing corpus panel bundle is invalid; choose a fresh "
                f"--out-dir: {exc}") from exc
        if not _same_json(existing_bundle, corpus_bundle):
            raise SystemExit(
                "existing --out-dir is bound to different corpus panel bytes")

    if args.condition == phase_d_protocol.OBSERVED:
        if control_manifest is not None:
            raise SystemExit("observed run cannot carry a control manifest")
        existing_control = _read_optional_preflight_json(
            control_path, "control manifest")
        if existing_control is not None:
            raise SystemExit(
                "observed run cannot reuse an output directory containing a "
                "control manifest")
    else:
        if control_manifest is None:
            raise SystemExit("shuffled run lacks a proposed control manifest")
        existing_control = _read_optional_preflight_json(
            control_path, "control manifest")
        if existing_control is not None:
            try:
                phase_d_protocol.validate_shuffled_control_manifest(
                    existing_control, corpus_manifest)
            except phase_d_protocol.PhaseDProtocolError as exc:
                raise SystemExit(
                    f"existing control manifest is invalid; choose a fresh "
                    f"--out-dir: {exc}") from exc
            if not _same_json(existing_control, control_manifest):
                raise SystemExit(
                    "existing --out-dir is bound to a different control")

    checkpoint = _read_optional_preflight_json(
        checkpoint_path, "checkpoint")
    if checkpoint is None:
        if previous_active_size is not None:
            raise SystemExit(
                "later preregistered scale requires the complete immediate "
                "predecessor checkpoint")
        return [], {}, []
    _assert_checkpoint_protocol(
        checkpoint, args, corpus_manifest, active_size,
        control_manifest, corpus_bundle,
        previous_active_size=previous_active_size)
    missing = []
    if existing_manifest is None:
        missing.append(os.path.basename(manifest_path))
    if existing_bundle is None:
        missing.append(os.path.basename(bundle_path))
    if control_manifest is not None and existing_control is None:
        missing.append(os.path.basename(control_path))
    if missing:
        raise SystemExit(
            "existing checkpoint is missing bound evidence: "
            + ", ".join(missing))
    # `_load_resume_state` performs the exact static-policy comparison and
    # canonical terminal replay.  It is read-only and must remain inside this
    # preflight boundary.
    run_state = _load_resume_state(
        out_dir, args, corpus_manifest, control_manifest, active_size,
        base_problems, corpus_bundle,
        previous_active_size=previous_active_size)
    if destination_state is not None:
        destination_records = [
            asdict(record) for record in destination_state[0]]
        run_records = [asdict(record) for record in run_state[0]]
        destination_results = destination_state[1]
        run_results = run_state[1]
        destination_cones = {
            item.get("opaque_id"): item for item in destination_state[2]
            if isinstance(item, dict)
        }
        run_cones = {
            item.get("opaque_id"): item for item in run_state[2]
            if isinstance(item, dict)
        }
        nested = len(destination_records) <= len(run_records) \
            and _same_json(
                destination_records,
                run_records[:len(destination_records)]) \
            and all(
                key in run_results
                and _same_json(value, run_results[key])
                for key, value in destination_results.items()) \
            and all(
                key in run_cones and _same_json(value, run_cones[key])
                for key, value in destination_cones.items())
        if not nested:
            raise SystemExit(
                "semantic destination artifact and run directory carry "
                "divergent predecessor histories")
    return run_state


def _previous_preregistered_family_scale(
        preregistration: dict, arm: dict) -> int | None:
    """Return only the immediately preceding scale in one execution family."""
    family_scales = sorted({
        candidate["scale"] for candidate in preregistration["arms"]
        if candidate["execution_tag"] == arm["execution_tag"]
        and candidate["track"] == arm["track"]
        and candidate["condition"] == arm["condition"]
        and candidate["replicate"] == arm["replicate"]
        and candidate["control_digest"] == arm["control_digest"]
    })
    if arm["scale"] not in family_scales:
        raise SystemExit("preregistered arm is absent from its execution family")
    index = family_scales.index(arm["scale"])
    return None if index == 0 else family_scales[index - 1]


def _load_preregistered_semantic_arm(
        path: str, arm_id: str, *, corpus_manifest: dict,
        args: argparse.Namespace,
        condition: str, scale: int,
        control_manifest: dict | None) -> tuple[dict, dict]:
    try:
        preregistration = artifact_io._load_json(
            path, "Phase D preregistration")
        if not isinstance(preregistration, dict):
            raise phase_d_protocol.PhaseDProtocolError(
                "preregistration must be a JSON object")
        phase_d_protocol.validate_preregistration(
            preregistration, corpus_manifest=corpus_manifest)
    except (artifact_io.CampaignCollectionError,
            phase_d_protocol.PhaseDProtocolError) as exc:
        raise SystemExit(f"invalid Phase D preregistration: {exc}") from exc
    matches = [
        arm for arm in preregistration["arms"] if arm["arm_id"] == arm_id]
    if len(matches) != 1:
        raise SystemExit("requested semantic Phase D arm is not preregistered")
    arm = matches[0]
    execution_tag = arm["execution_tag"]
    if getattr(args, "tag", None) != execution_tag:
        raise SystemExit(
            "semantic preregistered run tag must equal arm.execution_tag")
    canonical_out_dir = os.path.abspath(os.path.join(
        SEMANTIC_RUNS_DIR, execution_tag))
    invoked_out_dir = os.path.abspath(str(getattr(args, "out_dir", "")))
    if invoked_out_dir != canonical_out_dir:
        raise SystemExit(
            "semantic preregistered --out-dir must equal the canonical arm "
            f"path {canonical_out_dir}")
    report_condition = (
        "primary" if condition == phase_d_protocol.OBSERVED else condition)
    if preregistration["corpus_digest"] != corpus_manifest["corpus_digest"] \
            or arm["track"] != "SEMANTIC-PURE" \
            or arm["condition"] != report_condition \
            or arm["label_policy"] != condition \
            or arm["sharing_policy"] != phase_d_protocol.SHARED \
            or arm["scale"] != scale:
        raise SystemExit("semantic runner arguments differ from preregistered arm")
    policy = preregistration["execution_policy"]["semantic_pure"]
    invocation_policy = {
        "runner": "run_semantic_cone.py",
        "proposer": args.proposer,
        "model": args.model,
        "concrete_model": MODEL_MAP.get(args.model, args.model),
        "proposer_receipt_schema": (
            phase_d_protocol.SEMANTIC_PROPOSER_RECEIPT_SCHEMA),
        "max_tokens": args.max_tokens,
        "rounds": args.rounds,
        "max_model_attempts_per_round": (
            phase_d_protocol.SEMANTIC_MAX_MODEL_ATTEMPTS_PER_ROUND),
        "max_support_errors": args.max_support_errors,
        "max_loo_errors": args.max_loo_errors,
        "max_rotated_loo_errors": args.max_rotated_loo_errors,
        "selection_method": "conditional_free_energy",
        "lambda": args.lambda_value,
        "selection_risk_fields": list(SELECTION_RISK_FIELDS),
        "selection_unmeasured_risks": [
            name for name in RISK_FIELDS
            if name not in SELECTION_RISK_FIELDS],
    }
    if invocation_policy != policy:
        raise SystemExit(
            "semantic runner execution policy differs from preregistration")
    if condition == phase_d_protocol.SHUFFLED_SIDES:
        if control_manifest is None \
                or arm["replicate"] != control_manifest["replicate"] \
                or arm["control_digest"] != \
                control_manifest["control_digest"] \
                or preregistration["shuffled_sides"]["seed"] != \
                control_manifest["seed"]:
            raise SystemExit("semantic shuffled control differs from preregistration")
    elif arm["replicate"] is not None or control_manifest is not None:
        raise SystemExit("semantic primary arm cannot carry a control replicate")
    return preregistration, arm


def _publish_phase_d_track_report(
        tag: str, preregistration: dict, arm: dict,
        records: list[ProblemResult]) -> str:
    if tag != arm.get("execution_tag"):
        raise SystemExit(
            "semantic track-report tag must equal arm.execution_tag")
    report_source_trace_digest = semantic_replay.canonical_json_digest(
        [asdict(record) for record in records])
    report_records: list[dict] = []
    for record in records:
        value = asdict(record)
        value["runner_condition"] = value["condition"]
        value["condition"] = arm["condition"]
        value["label_policy"] = arm["label_policy"]
        value["sharing_policy"] = arm["sharing_policy"]
        report_records.append(value)
    report = phase_d_protocol.build_track_report(
        preregistration,
        arm_id=arm["arm_id"],
        records=report_records,
        report_source_trace_digest=report_source_trace_digest,
    )
    filename = arm["arm_id"].replace(":", "__") + ".json"
    path = os.path.join(
        semantic_artifacts.artifact_dir(tag), "track_reports", filename)
    if semantic_artifacts.create_json_once(path, report):
        return path
    try:
        existing = artifact_io._load_json(path, "semantic track report")
        if not isinstance(existing, dict):
            raise phase_d_protocol.PhaseDProtocolError(
                "track report must be a JSON object")
        phase_d_protocol.validate_track_report(existing, preregistration)
    except (artifact_io.CampaignCollectionError,
            phase_d_protocol.PhaseDProtocolError) as exc:
        raise SystemExit("existing semantic track report is invalid") from exc
    if existing != report:
        raise SystemExit(
            "semantic artifact tag has a different Phase D arm report")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--dataset-dir", default=os.path.join(repo_root, "downloads", "Bongard-LOGO"))
    parser.add_argument("--source", choices=("basic", "abstract", "both"), default="both")
    parser.add_argument(
        "--limit", type=int, default=5,
        help=("maximum problems sampled per selected source; source=both may "
              "freeze up to twice this many"))
    parser.add_argument(
        "--corpus-size", type=int, default=0,
        help=("active ordered prefix of the frozen maximum corpus; 0 uses all. "
              "Keep --limit fixed when scaling 1->5->25."))
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--condition",
        choices=(phase_d_protocol.OBSERVED, phase_d_protocol.SHUFFLED_SIDES),
        default=phase_d_protocol.OBSERVED,
        help="run original labels or the full adaptive balanced shuffled-side control")
    parser.add_argument("--control-seed", type=int, default=20260805)
    parser.add_argument("--control-replicate", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "semantic_runs", "latest"))
    parser.add_argument(
        "--proposer", choices=("anthropic", "codex"), default="anthropic")
    parser.add_argument(
        "--model", default=None,
        help=("provider model; defaults to sonnet for Anthropic and "
              f"{CODEX_DEFAULT_MODEL} for Codex"))
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--tag", default="typed")
    parser.add_argument("--max-support-errors", type=int, default=0)
    parser.add_argument("--max-loo-errors", type=int, default=0)
    parser.add_argument("--max-rotated-loo-errors", type=int, default=0)
    parser.add_argument("--lambda-value", type=float, default=0.02)
    parser.add_argument(
        "--preregistration", default="",
        help="prepared phase_d_preregistration.json to bind before API use")
    parser.add_argument(
        "--arm-id", default="",
        help="exact SEMANTIC-PURE arm ID from --preregistration")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="freeze/validate corpus_manifest.json and exit before creating a proposer")
    args = parser.parse_args()
    if args.model is None:
        args.model = (
            "sonnet" if args.proposer == "anthropic" else CODEX_DEFAULT_MODEL)
    return args


if __name__ == "__main__":
    run(parse_args())
