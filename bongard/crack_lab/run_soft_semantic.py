#!/usr/bin/env python3
"""Run the prose-grounded ``SEMANTIC-SOFT`` Bongard pipeline.

One joint labelled Codex vision turn freezes 3-8 side-free rubrics.  Exactly
twelve subsequent Codex turns score one neutrally named panel apiece against
all frozen rubrics.  No target score or verifier feedback can rewrite a rubric.
The artifact stores exact proposer/scorer receipts and all atomic evidence so
thresholding, pair-LOO, polarity checks, and selection replay without a model
call.

This exploratory track deliberately does not claim ``SEMANTIC-PURE`` proof or
representation-level holdout: the proposer saw all twelve support panels.
Confirmatory reporting additionally requires an independently frozen hidden
query corpus or external calibration suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Protocol, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase_d_protocol
import semantic_replay
from dataset import write_panels
from semantic_soft_pipeline import (
    BlindSoftBatchScorer,
    CodexBlindSoftBatchScorer,
    CodexSoftHypothesisProposer,
    SoftProposalBundle,
    panel_soft_score_from_dict,
    replay_soft_verification,
    select_soft_verification,
    verify_soft_predicates_batched,
)


CAMPAIGN_SCHEMA = "bongard.semantic-soft-campaign/v1"
CONDITION_OBSERVED = phase_d_protocol.OBSERVED
CONDITION_SHUFFLED = phase_d_protocol.SHUFFLED_SIDES


class SoftProposer(Protocol):
    def propose(self, problem_id: str,
                panel_png_paths: Sequence[str]) -> SoftProposalBundle:
        ...


def _panel_paths(workspace: str, opaque_id: str) -> list[str]:
    directory = os.path.join(workspace, opaque_id)
    paths = [
        os.path.abspath(os.path.join(directory, f"{side}_{index}.png"))
        for side in ("pos", "neg") for index in range(6)
    ]
    if any(not os.path.isfile(path) for path in paths):
        raise RuntimeError(f"{opaque_id} does not have twelve rendered PNGs")
    return paths


def _receipt_usage(receipts: Sequence[dict[str, Any]]) -> dict[str, int]:
    unique: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        digest = receipt.get("receipt_digest")
        if isinstance(digest, str) and digest:
            unique.setdefault(digest, receipt)
    names = (
        "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_output_tokens",
    )
    return {
        name: sum(
            value for receipt in unique.values()
            if isinstance((value := receipt.get(name)), int)
            and not isinstance(value, bool) and value >= 0)
        for name in names
    } | {"turns": len(unique)}


def evaluate_problem(
        opaque_id: str, panel_png_paths: Sequence[str],
        proposer: SoftProposer, scorer: BlindSoftBatchScorer,
        *, scorer_workers: int = 1,
        ) -> dict[str, Any]:
    """Run one immutable propose→blind-score→verify problem transaction."""
    bundle = proposer.propose(opaque_id, panel_png_paths)
    labels = (True,) * 6 + (False,) * 6
    verifications = verify_soft_predicates_batched(
        bundle.hypotheses, panel_png_paths, labels, scorer,
        max_workers=scorer_workers)
    selected = select_soft_verification(verifications)
    scorer_errors = sum(
        state == "error"
        for verification in verifications for state in verification.states)
    scorer_absences = sum(
        state == "absent"
        for verification in verifications for state in verification.states)
    infrastructure_valid = scorer_errors == 0
    receipts: list[dict[str, Any]] = [dict(bundle.receipt)]
    for verification in verifications:
        receipts.extend(
            dict(panel.receipt) for panel in verification.evidence
            if panel.receipt)
    selected_dict = selected.to_dict() if selected is not None else {}
    return {
        "opaque_id": opaque_id,
        "track": "SEMANTIC-SOFT",
        "status": (
            "SOLVED_SEMANTIC_SOFT"
            if infrastructure_valid and selected is not None
            and selected.accepted else
            "INVALID_SEMANTIC_SOFT"
            if not infrastructure_valid else
            "UNSOLVED_SEMANTIC_SOFT"),
        "solved": bool(
            infrastructure_valid and selected is not None
            and selected.accepted),
        "infrastructure_valid": infrastructure_valid,
        "scorer_error_measurements": scorer_errors,
        "scorer_absent_measurements": scorer_absences,
        "proposal": bundle.to_dict(),
        "candidate_count": len(verifications),
        "candidates": [item.to_dict() for item in verifications],
        "selected_hypothesis": (
            selected.hypothesis_id if selected is not None else ""),
        "selected": selected_dict,
        "usage": _receipt_usage(receipts),
        "information_boundary": {
            "proposer": "all-twelve-labelled-direct-images/one-turn",
            "rubric_commitment": "before-target-scoring/content-addressed",
            "scorer": "one-neutral-image/no-labels/no-neighbours/no-feedback",
            "scoring_turns": 12,
            "scorer_workers": scorer_workers,
            "aggregation": "harness-owned-unweighted",
            "polarity": "prose-declared/no-selector-reversal",
            "representation_level_holdout": False,
        },
    }


def _write_json(path: str, value: Any) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    temporary = path + ".tmp"
    with open(temporary, "xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bind_json(path: str, value: Any) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != value:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    _write_json(path, value)


def _same_canonical(left: Any, right: Any) -> bool:
    return semantic_replay.canonical_json_digest(left) == \
        semantic_replay.canonical_json_digest(right)


def replay_campaign_artifact(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Cold-validate evidence and recompute every downstream soft decision."""
    required_campaign = {
        "schema", "track", "condition", "model", "reasoning_effort",
        "corpus_digest", "control_digest", "record_count", "records",
        "claim_boundary", "campaign_digest",
    }
    if not isinstance(campaign, Mapping) or set(campaign) != required_campaign:
        raise ValueError("semantic-soft campaign fields differ")
    if campaign["schema"] != CAMPAIGN_SCHEMA \
            or campaign["track"] != "SEMANTIC-SOFT" \
            or campaign["condition"] not in {
                CONDITION_OBSERVED, CONDITION_SHUFFLED} \
            or not isinstance(campaign["records"], list) \
            or campaign["record_count"] != len(campaign["records"]):
        raise ValueError("semantic-soft campaign violates its schema")
    unsigned = {
        key: value for key, value in campaign.items()
        if key != "campaign_digest"}
    if campaign["campaign_digest"] != \
            semantic_replay.canonical_json_digest(unsigned):
        raise ValueError("semantic-soft campaign digest does not reproduce")

    expected_record_keys = {
        "opaque_id", "track", "status", "solved", "infrastructure_valid",
        "scorer_error_measurements", "scorer_absent_measurements", "proposal",
        "candidate_count", "candidates", "selected_hypothesis", "selected",
        "usage", "information_boundary", "category", "panel_set_digest",
    }
    replayed_records = 0
    accepted_records = 0
    for record_index, record in enumerate(campaign["records"]):
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise ValueError(
                f"semantic-soft record {record_index} fields differ")
        if record["track"] != "SEMANTIC-SOFT" \
                or record["opaque_id"] != f"problem_{record_index:02d}" \
                or not isinstance(record["candidates"], list):
            raise ValueError("semantic-soft record identity is malformed")
        proposal = SoftProposalBundle.from_dict(record["proposal"])
        if proposal.problem_id != record["opaque_id"] \
                or record["candidate_count"] != len(proposal.hypotheses) \
                or len(record["candidates"]) != len(proposal.hypotheses):
            raise ValueError("semantic-soft proposal/candidate binding differs")

        labels = (True,) * 6 + (False,) * 6
        replayed = []
        receipts: list[dict[str, Any]] = [dict(proposal.receipt)]
        for spec, stored in zip(proposal.hypotheses, record["candidates"]):
            if not isinstance(stored, Mapping) \
                    or stored.get("hypothesis_id") != spec.hypothesis_id \
                    or not isinstance(stored.get("evidence"), list) \
                    or len(stored["evidence"]) != 12:
                raise ValueError("semantic-soft candidate evidence is malformed")
            evidence = tuple(
                panel_soft_score_from_dict(spec, item)
                for item in stored["evidence"])
            verification = replay_soft_verification(spec, evidence, labels)
            if not _same_canonical(verification.to_dict(), stored):
                raise ValueError(
                    "semantic-soft stored decision does not replay exactly")
            replayed.append(verification)
            receipts.extend(
                dict(panel.receipt) for panel in evidence if panel.receipt)

        selected = select_soft_verification(replayed)
        expected_selected = selected.to_dict() if selected is not None else {}
        expected_selected_id = (
            selected.hypothesis_id if selected is not None else "")
        if record["selected_hypothesis"] != expected_selected_id \
                or not _same_canonical(record["selected"], expected_selected):
            raise ValueError("semantic-soft selected candidate does not replay")
        scorer_errors = sum(
            state == "error"
            for verification in replayed for state in verification.states)
        scorer_absences = sum(
            state == "absent"
            for verification in replayed for state in verification.states)
        infrastructure_valid = scorer_errors == 0
        solved = bool(
            infrastructure_valid and selected is not None
            and selected.accepted)
        status = (
            "SOLVED_SEMANTIC_SOFT" if solved else
            "INVALID_SEMANTIC_SOFT" if not infrastructure_valid else
            "UNSOLVED_SEMANTIC_SOFT")
        if record["scorer_error_measurements"] != scorer_errors \
                or record["scorer_absent_measurements"] != scorer_absences \
                or record["infrastructure_valid"] is not infrastructure_valid \
                or record["solved"] is not solved \
                or record["status"] != status \
                or record["usage"] != _receipt_usage(receipts):
            raise ValueError("semantic-soft record summary does not replay")
        replayed_records += 1
        accepted_records += int(solved)
    return {
        "schema": "bongard.semantic-soft-replay-report/v1",
        "campaign_digest": campaign["campaign_digest"],
        "record_count": replayed_records,
        "solved_count": accepted_records,
        "valid": True,
    }


def replay_campaign_directory(directory: str) -> dict[str, Any]:
    """Load a persisted campaign and bind it back to its frozen corpus."""
    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "campaign.json"), encoding="utf-8") \
            as handle:
        campaign = json.load(handle)
    with open(os.path.join(directory, "corpus_manifest.json"),
              encoding="utf-8") as handle:
        corpus_manifest = json.load(handle)
    phase_d_protocol.validate_corpus_manifest(corpus_manifest)
    if campaign.get("corpus_digest") != corpus_manifest["corpus_digest"]:
        raise ValueError("semantic-soft campaign uses a different corpus")
    control_manifest = None
    if campaign.get("condition") == CONDITION_SHUFFLED:
        with open(os.path.join(directory, "shuffled_sides.json"),
                  encoding="utf-8") as handle:
            control_manifest = json.load(handle)
        phase_d_protocol.validate_shuffled_control_manifest(
            control_manifest, corpus_manifest)
        if campaign.get("control_digest") != control_manifest["control_digest"]:
            raise ValueError("semantic-soft campaign uses a different control")
    elif campaign.get("control_digest") != "":
        raise ValueError("observed semantic-soft campaign claims a control")

    if len(campaign.get("records", ())) > len(corpus_manifest["problems"]):
        raise ValueError("semantic-soft campaign exceeds its frozen corpus")
    for index, record in enumerate(campaign.get("records", ())):
        corpus_entry = corpus_manifest["problems"][index]
        expected_digest = (
            control_manifest["problems"][index]["controlled_panel_set_digest"]
            if control_manifest is not None else corpus_entry["panel_set_digest"])
        if record.get("category") != corpus_entry["category"] \
                or record.get("panel_set_digest") != expected_digest:
            raise ValueError("semantic-soft record is not bound to its panel set")
    return replay_campaign_artifact(campaign)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = os.path.abspath(args.out_dir)
    bongard_root = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if os.path.commonpath((bongard_root, os.path.realpath(out_dir))) != \
            bongard_root:
        raise SystemExit("--out-dir must stay inside the bongard working tree")
    if args.limit <= 0 or args.corpus_size < 0 \
            or not 1 <= args.minutes <= 120 \
            or not 1 <= args.scorer_workers <= 12:
        raise SystemExit(
            "limit must be positive, corpus-size nonnegative, minutes in "
            "1..120, and scorer-workers in 1..12")
    base_problems = phase_d_protocol.sample_corpus(
        args.dataset_dir,
        limit_per_source=args.limit,
        seed=args.seed,
        source=args.source,
    )
    corpus_manifest = phase_d_protocol.build_corpus_manifest(
        base_problems,
        source=args.source,
        seed=args.seed,
        limit_per_source=args.limit,
        dataset_revision=phase_d_protocol.dataset_revision(args.dataset_dir),
        dataset_inputs_digest=phase_d_protocol.dataset_content_digest(
            args.dataset_dir),
    )
    corpus_bundle = phase_d_protocol.build_corpus_bundle(
        base_problems, corpus_manifest)
    problems = list(base_problems)
    control_manifest = None
    if args.condition == CONDITION_SHUFFLED:
        control = phase_d_protocol.build_shuffled_sides_control(
            base_problems,
            corpus_manifest,
            seed=args.control_seed,
            replicate=args.control_replicate,
        )
        problems = list(control.problems)
        control_manifest = control.manifest
    active_size = args.corpus_size or len(problems)
    if active_size > len(problems):
        raise SystemExit("--corpus-size exceeds the frozen corpus")

    os.makedirs(out_dir, exist_ok=True)
    _bind_json(os.path.join(out_dir, "corpus_manifest.json"), corpus_manifest)
    _bind_json(os.path.join(out_dir, "corpus_panels.json"), corpus_bundle)
    if control_manifest is not None:
        _bind_json(
            os.path.join(out_dir, "shuffled_sides.json"), control_manifest)
    campaign_path = os.path.join(out_dir, "campaign.json")
    if os.path.exists(campaign_path):
        raise RuntimeError(
            "campaign.json already exists; semantic-soft runs are immutable")
    if args.prepare_only:
        return {
            "corpus_digest": corpus_manifest["corpus_digest"],
            "prepared": True,
        }

    workspace = os.path.join(out_dir, "workspace")
    os.makedirs(workspace, exist_ok=True)
    proposer = CodexSoftHypothesisProposer(
        args.model, minutes=args.minutes,
        reasoning_effort=args.reasoning_effort)
    scorer = CodexBlindSoftBatchScorer(
        args.model, minutes=args.minutes,
        reasoning_effort=args.reasoning_effort)
    records = []
    for index, problem in enumerate(problems[:active_size]):
        opaque_id = f"problem_{index:02d}"
        write_panels(workspace, problem, opaque_id)
        record = evaluate_problem(
            opaque_id, _panel_paths(workspace, opaque_id), proposer, scorer,
            scorer_workers=args.scorer_workers)
        record["category"] = corpus_manifest["problems"][index]["category"]
        record["panel_set_digest"] = (
            control_manifest["problems"][index]["controlled_panel_set_digest"]
            if control_manifest is not None else
            corpus_manifest["problems"][index]["panel_set_digest"])
        records.append(record)
        print(
            f"[{index + 1:02d}/{active_size:02d}] {opaque_id} "
            f"{record['status']} selected={record['selected_hypothesis'] or '-'}",
            flush=True,
        )

    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "track": "SEMANTIC-SOFT",
        "condition": args.condition,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "corpus_digest": corpus_manifest["corpus_digest"],
        "control_digest": (
            control_manifest["control_digest"]
            if control_manifest is not None else ""),
        "record_count": len(records),
        "records": records,
        "claim_boundary": (
            "frozen prose-conditioned blind VLM evidence; deterministic "
            "downstream replay; not SEMANTIC-PURE and not a representation-"
            "level holdout"),
    }
    campaign["campaign_digest"] = semantic_replay.canonical_json_digest(campaign)
    _write_json(campaign_path, campaign)
    replay_campaign_directory(out_dir)
    return campaign


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(repo_root, "downloads", "Bongard-LOGO"))
    parser.add_argument(
        "--source", choices=("basic", "abstract", "both"), default="basic")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--corpus-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--condition", choices=(CONDITION_OBSERVED, CONDITION_SHUFFLED),
        default=CONDITION_OBSERVED)
    parser.add_argument("--control-seed", type=int, default=20260805)
    parser.add_argument("--control-replicate", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(__file__), "semantic_soft_runs", "latest"))
    parser.add_argument(
        "--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=tuple(sorted(codex_reasoning_efforts())))
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--scorer-workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--replay-artifact", metavar="DIRECTORY",
        help="cold-validate an existing campaign without model calls")
    return parser.parse_args(argv)


def codex_reasoning_efforts() -> frozenset[str]:
    # Publicly expose the transport's closed values without importing private
    # constants into the campaign artifact.
    return frozenset({
        "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.replay_artifact:
        print(json.dumps(
            replay_campaign_directory(parsed.replay_artifact),
            sort_keys=True))
    else:
        run(parsed)
