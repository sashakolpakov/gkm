#!/usr/bin/env python3
"""Run a content-distinct latent-program/style-pose HYBRID benchmark.

One isolated support turn commits one affirmative prose claim.  Six positive
support panels and six negative support panels become a content-addressed
contrastive oracle bundle.  A query-latent commitment is written before the
proposal, and a second freeze record is written before query pixels exist.
Only then are six held-out positive and six held-out negative action-string
programs rendered.  In the basic bird6 task these are content-distinct
style/pose programs for the same fixed template; this does not test semantic-
instance or cross-template generalization.

Each neutrally named query target is separately evaluated by the frozen
oracle.  Its typed Boolean observation is elaborated through the grounded IR
as ``oracle_value == true``; the executable predicate is operationally
``OperationalResemblance(bundle, target)``, not a proof that the prose claim is
true of the pixels.  The resulting predicate must be HYBRID.  There is no
target-conditioned threshold, candidate fallback, or polarity reversal.

Cold replay validates stored oracle receipts and categorical evidence and
recomputes decoding, grounded-IR traces, labels, and status.  It deliberately
does not claim to reproduce perception without new model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codex_proposer
import grounded_contrastive_oracle as C
import grounded_predicate_ir as G
from dataset import PANEL_SIZE, Problem
from hybrid_claim_proposer import (
    ClaimProposalBundle,
    CodexHybridClaimProposer,
    validate_claim_proposal_receipt,
)
from hybrid_program_split import (
    DEFAULT_POOL_SIZE,
    HybridProgramSplit,
    canonical_digest,
    canonical_json,
    file_digest,
    program_digest,
    sample_basic_program_splits,
)


CAMPAIGN_SCHEMA = "bongard.hybrid-contrastive-campaign/v1"
PANEL_SET_SCHEMA = "bongard.hybrid-panel-set/v1"
LATENT_COMMITMENT_SCHEMA = "bongard.hybrid-query-latent-commitment/v1"
FREEZE_SCHEMA = "bongard.hybrid-oracle-freeze/v1"
QUERY_RELEASE_SCHEMA = "bongard.hybrid-query-release/v1"
EVALUATION_SCHEMA = "bongard.hybrid-query-evaluation/v1"
TRACK = "SEMANTIC-HYBRID-EXPLORATORY"
QUERY_ORDER_POLICY = "content-ranked-side-blind-query-order/v1"

_SOURCE_FILES = (
    "run_hybrid_contrastive.py",
    "hybrid_program_split.py",
    "hybrid_claim_proposer.py",
    "grounded_contrastive_oracle.py",
    "grounded_predicate_ir.py",
    "codex_proposer.py",
    "semantic_replay.py",
    "dataset.py",
)


class ClaimProposer(Protocol):
    def propose(
        self, problem_id: str, support_png_paths: Sequence[str],
    ) -> ClaimProposalBundle:
        ...


class ContrastiveBackend(Protocol):
    """Small seam used only for offline tests; production uses CodexBackend."""

    def create_contract(
        self, claim: str, positive_paths: Sequence[str], foil_paths: Sequence[str],
        *, model: str, reasoning_effort: str,
    ) -> Any:
        ...

    def restore_contract(self, value: Mapping[str, Any]) -> Any:
        ...

    def create_oracle(
        self, contract: Any, positive_paths: Sequence[str],
        foil_paths: Sequence[str], *, minutes: int, executable: str,
        verbose: bool,
    ) -> Any:
        ...

    def replay_evaluation(
        self, contract: Any, value: Mapping[str, Any], target_png_path: str,
    ) -> Any:
        ...


class CodexBackend:
    def create_contract(
        self, claim: str, positive_paths: Sequence[str], foil_paths: Sequence[str],
        *, model: str, reasoning_effort: str,
    ) -> C.ContrastiveOracleContract:
        return C.ContrastiveOracleContract.create(
            claim, positive_paths, foil_paths,
            model=model, reasoning_effort=reasoning_effort,
        )

    def restore_contract(
        self, value: Mapping[str, Any],
    ) -> C.ContrastiveOracleContract:
        return C.ContrastiveOracleContract.from_dict(value)

    def create_oracle(
        self, contract: C.ContrastiveOracleContract,
        positive_paths: Sequence[str], foil_paths: Sequence[str],
        *, minutes: int, executable: str, verbose: bool,
    ) -> C.CodexContrastiveOracle:
        return C.CodexContrastiveOracle(
            contract, positive_paths, foil_paths,
            minutes=minutes, executable=executable, verbose=verbose,
        )

    def replay_evaluation(
        self, contract: C.ContrastiveOracleContract,
        value: Mapping[str, Any], target_png_path: str,
    ) -> C.ContrastiveOracleEvaluation:
        return C.replay_evaluation(
            contract, value, target_png_path=target_png_path)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: str) -> str:
    return file_digest(path)


def _array_digest(panel: np.ndarray) -> str:
    array = np.ascontiguousarray(panel)
    header = canonical_json({
        "dtype": array.dtype.str, "shape": list(array.shape),
    }).encode("utf-8")
    return _sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def _source_bindings() -> list[dict[str, str]]:
    directory = os.path.dirname(os.path.abspath(__file__))
    bindings = []
    for name in _SOURCE_FILES:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"required HYBRID source is absent: {name}")
        bindings.append({"path": name, "sha256": _sha256_file(path)})
    return bindings


def _environment_binding() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pillow": PIL_VERSION,
    }


def _write_exclusive_json(path: str, value: Any) -> dict[str, str]:
    encoded = canonical_json(value).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": path, "sha256": _sha256_bytes(encoded)}


def _write_campaign(path: str, value: Any) -> None:
    encoded = canonical_json(value).encode("utf-8")
    temporary = path + ".tmp"
    with open(temporary, "xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_path(root: str, relative: Any) -> str:
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise ValueError("artifact path must be a nonempty relative path")
    real_root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(real_root, relative))
    try:
        inside = os.path.commonpath((real_root, path)) == real_root
    except ValueError:
        inside = False
    if not inside:
        raise ValueError("artifact path escapes the campaign directory")
    return path


def _raw_programs(
    latent: HybridProgramSplit, split: str,
) -> tuple[tuple[Any, bool, str, int], ...]:
    if split == "support":
        groups = ((latent.support_pos, True, "pos"),
                  (latent.support_neg, False, "neg"))
    elif split == "query":
        groups = ((latent.query_pos, True, "pos"),
                  (latent.query_neg, False, "neg"))
    else:
        raise ValueError("unknown split")
    return tuple(
        (program, label, side, index)
        for programs, label, side in groups
        for index, program in enumerate(programs)
    )


def _query_rank(latent: HybridProgramSplit, program: Any) -> str:
    return canonical_digest({
        "policy": QUERY_ORDER_POLICY,
        "program_split_digest": latent.to_manifest()["program_split_digest"],
        "program_digest": program_digest(program),
    })


def _ordered_programs(
    latent: HybridProgramSplit, split: str,
) -> tuple[tuple[Any, bool, str, int], ...]:
    result = _raw_programs(latent, split)
    if split == "query":
        result = tuple(sorted(
            result,
            key=lambda item: (_query_rank(latent, item[0]),
                              program_digest(item[0])),
        ))
    return result


def _materialize_panel_set(
    run_directory: str,
    workspace: str,
    opaque_id: str,
    split: str,
    problem: Problem,
    latent: HybridProgramSplit,
    render_seed: int,
) -> tuple[dict[str, Any], list[str]]:
    directory = os.path.join(workspace, opaque_id, split)
    if os.path.exists(directory) and os.listdir(directory):
        raise RuntimeError(f"{split} panel directory is not empty")
    os.makedirs(directory, exist_ok=True)
    panels = problem.panels()
    raw_programs = _raw_programs(latent, split)
    combined = list(zip(panels, raw_programs))
    if split == "query":
        combined.sort(key=lambda item: (
            _query_rank(latent, item[1][0]), program_digest(item[1][0])))
    if len(combined) != 12:
        raise RuntimeError("HYBRID split must contain twelve panels")
    entries: list[dict[str, Any]] = []
    png_paths: list[str] = []
    for ordinal, ((panel, label), (program, expected, side, index)) in enumerate(
            combined):
        if label is not expected:
            raise RuntimeError("HYBRID panel/program ordering differs")
        name = f"{side}_{index}" if split == "support" else f"target_{ordinal:02d}"
        array = np.ascontiguousarray(panel)
        if array.dtype != np.uint8 or array.shape != (PANEL_SIZE, PANEL_SIZE) \
                or not np.isin(array, (0, 1)).all():
            raise RuntimeError(f"invalid {split} panel {ordinal}")
        npy_path = os.path.join(directory, name + ".npy")
        png_path = os.path.join(directory, name + ".png")
        np.save(npy_path, array, allow_pickle=False)
        Image.fromarray(np.where(array == 1, 0, 255).astype(np.uint8), mode="L") \
            .save(png_path)
        png_paths.append(os.path.abspath(png_path))
        entries.append({
            "ordinal": ordinal,
            "slot": f"{side}_{index}",
            "label": label,
            "oracle_name": f"target_{ordinal:02d}.png" if split == "query" else "",
            "program_digest": program_digest(program),
            "npy_path": os.path.relpath(npy_path, run_directory),
            "png_path": os.path.relpath(png_path, run_directory),
            "array_digest": _array_digest(array),
            "npy_sha256": _sha256_file(npy_path),
            "png_sha256": _sha256_file(png_path),
        })
    if split == "support":
        presentation_digest_schema = "labelled-semantic-panel-set/v1"
        presentation_digest = codex_proposer.semantic_panel_set_digest(png_paths)
    else:
        presentation_digest_schema = codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
        presentation_digest = codex_proposer.named_image_set_digest(
            png_paths, [f"target_{ordinal:02d}.png" for ordinal in range(12)])
    body: dict[str, Any] = {
        "schema": PANEL_SET_SCHEMA,
        "split": split,
        "presentation": (
            "labelled-support" if split == "support"
            else "neutral-content-distinct-latent-program-style-pose-targets"
        ),
        "render_seed": render_seed,
        "program_split_digest": latent.to_manifest()["program_split_digest"],
        "presentation_digest_schema": presentation_digest_schema,
        "presentation_digest": presentation_digest,
        "panels": entries,
    }
    body["panel_set_digest"] = canonical_digest(body)
    return body, png_paths


def _load_panel_set(
    run_directory: str,
    stored: Mapping[str, Any],
    expected_split: str,
    latent: HybridProgramSplit,
) -> tuple[tuple[np.ndarray, bool, str, str], ...]:
    keys = {
        "schema", "split", "presentation", "render_seed",
        "program_split_digest", "presentation_digest_schema",
        "presentation_digest", "panels", "panel_set_digest",
    }
    if not isinstance(stored, Mapping) or set(stored) != keys:
        raise ValueError(f"{expected_split} panel-set fields differ")
    unsigned = {key: item for key, item in stored.items()
                if key != "panel_set_digest"}
    expected_presentation = (
        "labelled-support" if expected_split == "support"
        else "neutral-content-distinct-latent-program-style-pose-targets"
    )
    if stored["schema"] != PANEL_SET_SCHEMA \
            or stored["split"] != expected_split \
            or stored["presentation"] != expected_presentation \
            or stored["program_split_digest"] != \
            latent.to_manifest()["program_split_digest"] \
            or stored["panel_set_digest"] != canonical_digest(unsigned):
        raise ValueError(f"{expected_split} panel-set binding differs")
    raw_entries = stored["panels"]
    programs = _ordered_programs(latent, expected_split)
    if not isinstance(raw_entries, list) or len(raw_entries) != 12:
        raise ValueError(f"{expected_split} must contain twelve panel entries")
    expected_keys = {
        "ordinal", "slot", "label", "oracle_name", "program_digest",
        "npy_path", "png_path", "array_digest", "npy_sha256", "png_sha256",
    }
    result: list[tuple[np.ndarray, bool, str, str]] = []
    png_paths: list[str] = []
    for ordinal, (raw, (program, label, side, index)) in enumerate(
            zip(raw_entries, programs)):
        slot = f"{side}_{index}"
        oracle_name = (
            f"target_{ordinal:02d}.png" if expected_split == "query" else "")
        if not isinstance(raw, Mapping) or set(raw) != expected_keys \
                or (raw["ordinal"], raw["slot"], raw["label"],
                    raw["oracle_name"], raw["program_digest"]) != (
                    ordinal, slot, label, oracle_name, program_digest(program)):
            raise ValueError(f"{expected_split} panel ordering differs")
        npy_path = _safe_path(run_directory, raw["npy_path"])
        png_path = _safe_path(run_directory, raw["png_path"])
        if _sha256_file(npy_path) != raw["npy_sha256"] \
                or _sha256_file(png_path) != raw["png_sha256"]:
            raise ValueError(f"{expected_split}/{slot} file digest mismatch")
        panel = np.load(npy_path, allow_pickle=False)
        if panel.dtype != np.uint8 or panel.shape != (PANEL_SIZE, PANEL_SIZE) \
                or not np.isin(panel, (0, 1)).all() \
                or _array_digest(panel) != raw["array_digest"]:
            raise ValueError(f"{expected_split}/{slot} array mismatch")
        with Image.open(png_path) as encoded:
            presentation = np.asarray(encoded.convert("L"))
        if presentation.shape != panel.shape \
                or not np.isin(presentation, (0, 255)).all() \
                or not np.array_equal(
                    (presentation == 0).astype(np.uint8), panel):
            raise ValueError(f"{expected_split}/{slot} PNG differs from NPY")
        png_paths.append(png_path)
        result.append((np.ascontiguousarray(panel), label, slot, png_path))
    if expected_split == "support":
        expected_digest_schema = "labelled-semantic-panel-set/v1"
        observed_digest = codex_proposer.semantic_panel_set_digest(png_paths)
    else:
        expected_digest_schema = codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
        observed_digest = codex_proposer.named_image_set_digest(
            png_paths, [f"target_{ordinal:02d}.png" for ordinal in range(12)])
    if stored["presentation_digest_schema"] != expected_digest_schema \
            or stored["presentation_digest"] != observed_digest:
        raise ValueError(f"{expected_split} presentation digest differs")
    rendered = latent.render(expected_split, stored["render_seed"])
    rendered_combined = list(zip(
        rendered.panels(), _raw_programs(latent, expected_split)))
    if expected_split == "query":
        rendered_combined.sort(key=lambda item: (
            _query_rank(latent, item[1][0]), program_digest(item[1][0])))
    for (observed, _label, slot, _path), \
            ((expected, _expected_label), _program_case) in zip(
                result, rendered_combined):
        if not np.array_equal(observed, expected):
            raise ValueError(f"{expected_split}/{slot} does not rerender")
    return tuple(result)


def _artifact_reference(
    run_directory: str, absolute_path: str, body: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "path": os.path.relpath(absolute_path, run_directory),
        "sha256": _sha256_file(absolute_path),
        "body_digest": canonical_digest(body),
    }


def _validate_bound_json(
    run_directory: str, reference: Mapping[str, Any], expected: Mapping[str, Any],
) -> None:
    if not isinstance(reference, Mapping) or set(reference) != {
            "path", "sha256", "body_digest"}:
        raise ValueError("commitment reference fields differ")
    path = _safe_path(run_directory, reference["path"])
    if _sha256_file(path) != reference["sha256"] \
            or reference["body_digest"] != canonical_digest(expected):
        raise ValueError("commitment file binding differs")
    with open(path, encoding="utf-8") as handle:
        observed = json.load(handle)
    if observed != dict(expected):
        raise ValueError("commitment body differs")


def _contract_artifact(contract: Any) -> dict[str, Any]:
    body = contract.to_dict()
    digest = contract.digest()
    if canonical_digest(body) != digest:
        raise RuntimeError("oracle contract digest is not canonical")
    return {"contract": body, "contract_digest": digest}


def _restore_contract(
    artifact: Mapping[str, Any], backend: ContrastiveBackend,
) -> Any:
    if not isinstance(artifact, Mapping) or set(artifact) != {
            "contract", "contract_digest"} \
            or not isinstance(artifact["contract"], Mapping):
        raise ValueError("oracle contract artifact fields differ")
    contract = backend.restore_contract(artifact["contract"])
    if contract.to_dict() != dict(artifact["contract"]) \
            or contract.digest() != artifact["contract_digest"] \
            or canonical_digest(contract.to_dict()) != contract.digest():
        raise ValueError("oracle contract digest does not reproduce")
    return contract


def _observation_contract(oracle: Any, observable_id: str) \
        -> G.ObservableContract:
    live = oracle.observable_contract(observable_id)
    # Apply the typed IR to the already content-bound oracle evaluation.  This
    # evaluator cannot trigger perception and has the same declarative ORACLE
    # contract/version as the live leaf.
    replay = replace(live, evaluator=lambda evaluation: evaluation.observation)
    if replay.source is not G.ObservableSource.ORACLE \
            or replay.value_type is not G.ValueType.BOOLEAN \
            or replay.unit is not G.Unit.BOOLEAN \
            or replay.taint is not G.Taint.HYBRID:
        raise RuntimeError("contrastive observable is not a HYBRID Boolean")
    return replay


def _compile_affirmative_formula(
    oracle: Any, contract_digest: str,
) -> G.CompiledPredicate:
    observable_id = "oracle.contrastive/bundle-" + \
        contract_digest.removeprefix("sha256:")
    observation_contract = _observation_contract(oracle, observable_id)
    registry = G.ObservableRegistry()
    registry.register(observation_contract)
    compiled = G.compile_predicate(
        G.Compare(
            observable_id,
            G.ComparisonOperator.EQ,
            G.Literal(True, G.Unit.BOOLEAN),
        ),
        registry,
    )
    if compiled.taint is not G.Taint.HYBRID:
        raise RuntimeError("open-vocabulary oracle formula lost HYBRID taint")
    return compiled


def _formula_artifact(compiled: G.CompiledPredicate) -> dict[str, Any]:
    body = compiled.canonical_dict()
    if compiled.digest != G.canonical_digest(body):
        raise RuntimeError("HYBRID formula digest is inconsistent")
    return {"compiled": body, "formula_digest": compiled.digest}


def _truth(trace: G.EvaluationTrace) -> bool | None:
    result = trace.result
    if isinstance(result, G.Present) and result.unit is G.Unit.BOOLEAN \
            and isinstance(result.value, bool):
        return result.value
    return None


def _evaluate_joined(
    contract: Any,
    compiled: G.CompiledPredicate,
    serialized_evaluations: Sequence[Mapping[str, Any]],
    query_cases: Sequence[tuple[np.ndarray, bool, str, str]],
    backend: ContrastiveBackend,
    expected_oracle_mode: str,
) -> dict[str, Any]:
    if len(serialized_evaluations) != 12 or len(query_cases) != 12:
        raise RuntimeError("HYBRID evaluation requires twelve query targets")
    decisions = []
    correct = errors = abstentions = 0
    for ordinal, (stored, (_panel, label, slot, path)) in enumerate(
            zip(serialized_evaluations, query_cases)):
        # Labels are first attached here, after every target future has joined.
        _validate_oracle_evaluation_mode(stored, expected_oracle_mode)
        replayed = backend.replay_evaluation(contract, stored, path)
        trace = compiled.evaluate_with_trace(replayed)
        is_error = isinstance(trace.result, G.Error)
        is_abstain = isinstance(trace.result, G.Indeterminate)
        predicted = None if is_error or is_abstain else _truth(trace)
        is_correct = predicted is not None and predicted is label
        correct += int(is_correct)
        errors += int(is_error)
        abstentions += int(is_abstain)
        decisions.append({
            "ordinal": ordinal,
            "slot": slot,
            "label": label,
            "predicted": predicted,
            "correct": is_correct,
            "oracle_evaluation": dict(stored),
            "formula_trace": trace.to_dict(),
        })
    exact = (
        correct == 12 and errors == 0 and abstentions == 0
        and len(decisions) == 12
    )
    body: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "case_count": len(decisions),
        "correct_count": correct,
        "error_count": errors,
        "abstention_count": abstentions,
        "exact": exact,
        "decisions": decisions,
    }
    body["evaluation_digest"] = canonical_digest(body)
    return body


def _validate_oracle_evaluation_mode(
    stored: Mapping[str, Any], expected_mode: str,
) -> None:
    """Keep offline transport fixtures out of live campaign evidence."""
    if expected_mode not in {"codex-cli", "offline-fixture"}:
        raise ValueError("oracle evidence mode is unsupported")
    evidence = stored.get("evidence")
    if "evidence" in stored and evidence is None:
        # Typed infrastructure errors may occur before a receipt exists.
        return
    sources: list[Any]
    if isinstance(evidence, Mapping) and isinstance(evidence.get("trials"), list):
        trials = evidence["trials"]
        if len(trials) != 2 or any(not isinstance(trial, Mapping)
                                   for trial in trials):
            raise ValueError("oracle trial evidence is malformed")
        sources = [
            trial.get("receipt", {}).get("source")
            if isinstance(trial.get("receipt"), Mapping) else None
            for trial in trials
        ]
    elif isinstance(stored.get("receipt"), Mapping):
        # Narrow injected backend seam used only by runner unit tests.
        sources = [stored["receipt"].get("source")]
    else:
        raise ValueError("oracle evidence has no bound receipts")
    if any(source != expected_mode for source in sources):
        raise ValueError("oracle receipt source differs from campaign mode")


def _outcome(evaluation: Mapping[str, Any]) -> tuple[bool, str]:
    if evaluation.get("error_count", 0) != 0:
        return False, "INVALID_HYBRID_EXPLORATORY"
    if evaluation.get("exact") is True \
            and evaluation.get("abstention_count") == 0:
        return True, "SOLVED_HYBRID_EXPLORATORY"
    return False, "UNSOLVED_HYBRID_EXPLORATORY"


def _same(left: Any, right: Any) -> bool:
    return canonical_digest(left) == canonical_digest(right)


def replay_campaign_artifact(
    campaign: Mapping[str, Any],
    run_directory: str,
    *,
    backend: ContrastiveBackend | None = None,
) -> dict[str, Any]:
    """Replay hashes, oracle evidence, decoder, IR, and hidden-label join.

    No proposer or visual-oracle process is started.  ``valid`` means the
    original live evidence replays; it is not perceptual reproducibility.
    """
    backend = backend or CodexBackend()
    campaign_keys = {
        "schema", "track", "sampling_seed", "support_render_seed",
        "query_render_seed", "pool_size", "record_count", "model",
        "reasoning_effort", "proposal_mode", "oracle_mode", "source_bindings",
        "source_bindings_digest", "environment", "records",
        "information_boundary", "campaign_digest",
    }
    if not isinstance(campaign, Mapping) or set(campaign) != campaign_keys:
        raise ValueError("HYBRID campaign fields differ")
    unsigned = {key: item for key, item in campaign.items()
                if key != "campaign_digest"}
    if campaign["schema"] != CAMPAIGN_SCHEMA or campaign["track"] != TRACK \
            or campaign["campaign_digest"] != canonical_digest(unsigned) \
            or campaign["record_count"] != len(campaign["records"]):
        raise ValueError("HYBRID campaign digest/schema differs")
    if campaign["source_bindings"] != _source_bindings() \
            or campaign["source_bindings_digest"] != canonical_digest(
                campaign["source_bindings"]):
        raise ValueError("HYBRID source bindings differ")
    if campaign["environment"] != _environment_binding():
        raise ValueError("HYBRID runtime environment binding differs")
    if campaign["support_render_seed"] == campaign["query_render_seed"]:
        raise ValueError("HYBRID support/query render seeds must differ")
    if campaign["proposal_mode"] not in {"codex-cli", "offline-fixture"}:
        raise ValueError("HYBRID proposal mode is unsupported")
    if campaign["oracle_mode"] not in {"codex-cli", "offline-fixture"}:
        raise ValueError("HYBRID oracle mode is unsupported")
    information_boundary = campaign["information_boundary"]
    if not isinstance(information_boundary, Mapping) \
            or not isinstance(information_boundary.get("dataset_root"), str):
        raise ValueError("HYBRID dataset root binding is malformed")
    # Re-run the deterministic public sampler exactly once. Stored programs
    # are executable evidence, but are not allowed to select a self-consistent
    # cherry-picked subset after the fact.
    resampled = sample_basic_program_splits(
        information_boundary["dataset_root"],
        limit=campaign["record_count"],
        seed=campaign["sampling_seed"],
        pool_size=campaign["pool_size"],
    )
    resampled_manifests = [item.to_manifest() for item in resampled]
    if len(resampled_manifests) != campaign["record_count"]:
        raise ValueError("HYBRID deterministic sampler record count differs")

    solved_count = 0
    record_keys = {
        "opaque_id", "program_split", "query_latent_commitment",
        "query_latent_commitment_file", "support_panel_set", "proposal",
        "oracle_contract", "formula", "oracle_freeze",
        "oracle_freeze_file", "query_panel_set", "query_release",
        "query_release_file", "query_evaluation", "solved", "status",
    }
    for index, record in enumerate(campaign["records"]):
        opaque_id = f"problem_{index:02d}"
        if not isinstance(record, Mapping) or set(record) != record_keys \
                or record["opaque_id"] != opaque_id:
            raise ValueError(f"HYBRID record {index} fields differ")
        if record["program_split"] != resampled_manifests[index]:
            raise ValueError(
                "HYBRID stored program split differs from deterministic sampler")
        latent = HybridProgramSplit.from_manifest(record["program_split"])
        if latent.sampling_seed != campaign["sampling_seed"] \
                or latent.pool_size != campaign["pool_size"]:
            raise ValueError("HYBRID latent sampling binding differs")
        for relative, digest in latent.dataset_inputs:
            dataset_path = os.path.join(
                information_boundary["dataset_root"], relative)
            if _sha256_file(dataset_path) != digest:
                raise ValueError("HYBRID dataset input digest differs")

        latent_commitment = record["query_latent_commitment"]
        expected_latent_commitment = {
            "schema": LATENT_COMMITMENT_SCHEMA,
            "opaque_id": opaque_id,
            "program_split_digest": record["program_split"][
                "program_split_digest"],
            "query_programs_digest": canonical_digest(
                record["program_split"]["query"]),
            "query_order_policy": QUERY_ORDER_POLICY,
            "query_presentation_order": [
                program_digest(program)
                for program, _label, _side, _index in
                _ordered_programs(latent, "query")
            ],
            "sampling_seed": campaign["sampling_seed"],
            "query_render_seed": campaign["query_render_seed"],
            "source_bindings_digest": campaign["source_bindings_digest"],
        }
        if latent_commitment != expected_latent_commitment:
            raise ValueError("HYBRID query latent commitment differs")
        _validate_bound_json(
            run_directory, record["query_latent_commitment_file"],
            latent_commitment)

        support = _load_panel_set(
            run_directory, record["support_panel_set"], "support", latent)
        query = _load_panel_set(
            run_directory, record["query_panel_set"], "query", latent)
        if record["support_panel_set"]["render_seed"] != \
                campaign["support_render_seed"] \
                or record["query_panel_set"]["render_seed"] != \
                campaign["query_render_seed"]:
            raise ValueError("HYBRID render-seed binding differs")
        proposal = ClaimProposalBundle.from_dict(record["proposal"])
        if proposal.problem_id != opaque_id:
            raise ValueError("HYBRID proposal problem binding differs")
        positive_paths = [case[3] for case in support if case[1] is True]
        foil_paths = [case[3] for case in support if case[1] is False]
        validate_claim_proposal_receipt(
            proposal,
            positive_paths + foil_paths,
            model=campaign["model"],
            reasoning_effort=campaign["reasoning_effort"],
            allow_offline_fixture=(
                campaign["proposal_mode"] == "offline-fixture"),
        )
        if proposal.receipt.get("source") != campaign["proposal_mode"]:
            raise ValueError("HYBRID proposal receipt/mode differs")
        contract = _restore_contract(record["oracle_contract"], backend)
        recreated = backend.create_contract(
            proposal.claim, positive_paths, foil_paths,
            model=campaign["model"],
            reasoning_effort=campaign["reasoning_effort"],
        )
        if recreated.to_dict() != contract.to_dict() \
                or recreated.digest() != contract.digest():
            raise ValueError("HYBRID contract does not bind support/claim")
        oracle = backend.create_oracle(
            contract, positive_paths, foil_paths,
            minutes=1, executable="codex", verbose=False)
        compiled = _compile_affirmative_formula(oracle, contract.digest())
        if record["formula"] != _formula_artifact(compiled) \
                or record["formula"]["compiled"].get("taint") != "HYBRID":
            raise ValueError("HYBRID grounded formula differs")

        freeze = record["oracle_freeze"]
        expected_freeze = {
            "schema": FREEZE_SCHEMA,
            "opaque_id": opaque_id,
            "query_latent_commitment_digest": canonical_digest(
                latent_commitment),
            "support_panel_set_digest": record["support_panel_set"][
                "panel_set_digest"],
            "proposal_digest": record["proposal"]["proposal_digest"],
            "oracle_contract_digest": contract.digest(),
            "formula_digest": compiled.digest,
            "polarity": "literal-affirmative-eq-true/no-reversal",
            "executable_predicate": (
                "OperationalResemblance(frozen_bundle,target)"
            ),
        }
        if freeze != expected_freeze:
            raise ValueError("HYBRID oracle freeze differs")
        _validate_bound_json(
            run_directory, record["oracle_freeze_file"], freeze)
        release = record["query_release"]
        expected_release = {
            "schema": QUERY_RELEASE_SCHEMA,
            "opaque_id": opaque_id,
            "query_latent_commitment_digest": canonical_digest(
                latent_commitment),
            "oracle_freeze_digest": canonical_digest(freeze),
            "query_panel_set_digest": record["query_panel_set"][
                "panel_set_digest"],
        }
        if release != expected_release:
            raise ValueError("HYBRID query release differs")
        _validate_bound_json(
            run_directory, record["query_release_file"], release)

        stored_evaluation = record["query_evaluation"]
        if not isinstance(stored_evaluation, Mapping) \
                or not isinstance(stored_evaluation.get("decisions"), list) \
                or len(stored_evaluation["decisions"]) != 12:
            raise ValueError("HYBRID stored query evaluation is malformed")
        serialized = [decision["oracle_evaluation"]
                      for decision in stored_evaluation["decisions"]]
        replayed = _evaluate_joined(
            contract, compiled, serialized, query, backend,
            campaign["oracle_mode"])
        if not _same(replayed, stored_evaluation):
            raise ValueError("HYBRID downstream query decision does not replay")
        solved, status = _outcome(replayed)
        if record["solved"] is not solved or record["status"] != status:
            raise ValueError("HYBRID query outcome does not replay")
        solved_count += int(solved)
    return {
        "schema": "bongard.hybrid-evidence-replay-report/v1",
        "campaign_digest": campaign["campaign_digest"],
        "record_count": len(campaign["records"]),
        "solved_count": solved_count,
        "valid": True,
        "status": "LIVE_EVIDENCE_REPLAY_VALID",
        "live_oracle_calls": 0,
        "perception_reexecuted": False,
        "claim": (
            "stored receipts/evidence, decoder, typed IR, and hidden-label "
            "join replay; live visual judgments are not reproduced"
        ),
    }


def replay_campaign_directory(
    directory: str, *, backend: ContrastiveBackend | None = None,
) -> dict[str, Any]:
    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "campaign.json"), encoding="utf-8") \
            as handle:
        campaign = json.load(handle)
    return replay_campaign_artifact(
        campaign, directory, backend=backend)


def run(
    args: argparse.Namespace,
    *,
    proposer: ClaimProposer | None = None,
    backend: ContrastiveBackend | None = None,
) -> dict[str, Any]:
    oracle_mode = "offline-fixture" if backend is not None else "codex-cli"
    backend = backend or CodexBackend()
    out_dir = os.path.abspath(args.out_dir)
    campaign_path = os.path.join(out_dir, "campaign.json")
    if os.path.exists(campaign_path):
        raise RuntimeError("campaign.json already exists; HYBRID runs are immutable")
    if args.support_seed == args.query_seed:
        raise SystemExit("support-seed and query-seed must differ")
    if args.limit <= 0 or args.pool_size < 12 \
            or not 1 <= args.minutes <= 120 \
            or not 1 <= args.scorer_workers <= 12:
        raise SystemExit(
            "limit must be positive, pool-size >=12, minutes in 1..120, "
            "and scorer-workers in 1..12")
    os.makedirs(out_dir, exist_ok=True)
    workspace = os.path.join(out_dir, "workspace")
    if os.path.exists(workspace) and os.listdir(workspace):
        raise RuntimeError("HYBRID workspace is nonempty")
    os.makedirs(workspace, exist_ok=True)
    latents = sample_basic_program_splits(
        args.dataset_dir, limit=args.limit, seed=args.sampling_seed,
        pool_size=args.pool_size)
    proposal_mode = "offline-fixture" if proposer is not None else "codex-cli"
    if proposer is None:
        proposer = CodexHybridClaimProposer(
            args.model, minutes=args.minutes,
            reasoning_effort=args.reasoning_effort,
            executable=args.executable,
        )
    source_bindings = _source_bindings()
    source_bindings_digest = canonical_digest(source_bindings)
    records = []
    for index, latent in enumerate(latents):
        opaque_id = f"problem_{index:02d}"
        record_directory = os.path.join(workspace, opaque_id)
        os.makedirs(record_directory, exist_ok=True)
        program_manifest = latent.to_manifest()

        support_problem = latent.render("support", args.support_seed)
        support_manifest, support_png_paths = _materialize_panel_set(
            out_dir, workspace, opaque_id, "support", support_problem,
            latent, args.support_seed)

        # Commit the still-unrendered query latents before the proposer turn.
        latent_commitment = {
            "schema": LATENT_COMMITMENT_SCHEMA,
            "opaque_id": opaque_id,
            "program_split_digest": program_manifest["program_split_digest"],
            "query_programs_digest": canonical_digest(program_manifest["query"]),
            "query_order_policy": QUERY_ORDER_POLICY,
            "query_presentation_order": [
                program_digest(program)
                for program, _label, _side, _index in
                _ordered_programs(latent, "query")
            ],
            "sampling_seed": args.sampling_seed,
            "query_render_seed": args.query_seed,
            "source_bindings_digest": source_bindings_digest,
        }
        latent_path = os.path.join(record_directory, "query_latent_commitment.json")
        _write_exclusive_json(latent_path, latent_commitment)
        latent_reference = _artifact_reference(
            out_dir, latent_path, latent_commitment)
        if os.path.exists(os.path.join(record_directory, "query")):
            raise RuntimeError("query pixels exist before the proposal")

        proposal_bundle = proposer.propose(opaque_id, support_png_paths)
        proposal = proposal_bundle.to_dict()
        validated_proposal = ClaimProposalBundle.from_dict(proposal)
        if validated_proposal.problem_id != opaque_id:
            raise RuntimeError("claim proposal uses a different problem ID")
        validate_claim_proposal_receipt(
            validated_proposal,
            support_png_paths,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            allow_offline_fixture=(proposal_mode == "offline-fixture"),
        )
        if validated_proposal.receipt.get("source") != proposal_mode:
            raise RuntimeError("claim proposer receipt/mode differs")
        positive_paths = support_png_paths[:6]
        foil_paths = support_png_paths[6:]
        contract = backend.create_contract(
            validated_proposal.claim, positive_paths, foil_paths,
            model=args.model, reasoning_effort=args.reasoning_effort)
        contract_artifact = _contract_artifact(contract)
        oracle = backend.create_oracle(
            contract, positive_paths, foil_paths,
            minutes=args.minutes, executable=args.executable,
            verbose=args.verbose_oracle)
        compiled = _compile_affirmative_formula(oracle, contract.digest())
        formula = _formula_artifact(compiled)

        # This O_EXCL record seals every executable decision before query
        # pixels are permitted to exist.
        freeze = {
            "schema": FREEZE_SCHEMA,
            "opaque_id": opaque_id,
            "query_latent_commitment_digest": canonical_digest(
                latent_commitment),
            "support_panel_set_digest": support_manifest["panel_set_digest"],
            "proposal_digest": proposal["proposal_digest"],
            "oracle_contract_digest": contract.digest(),
            "formula_digest": compiled.digest,
            "polarity": "literal-affirmative-eq-true/no-reversal",
            "executable_predicate": (
                "OperationalResemblance(frozen_bundle,target)"
            ),
        }
        freeze_path = os.path.join(record_directory, "oracle_freeze.json")
        _write_exclusive_json(freeze_path, freeze)
        freeze_reference = _artifact_reference(out_dir, freeze_path, freeze)
        if os.path.exists(os.path.join(record_directory, "query")):
            raise RuntimeError("query pixels exist before the oracle freeze")

        query_problem = latent.render("query", args.query_seed)
        query_manifest, query_paths = _materialize_panel_set(
            out_dir, workspace, opaque_id, "query", query_problem,
            latent, args.query_seed)
        query_release = {
            "schema": QUERY_RELEASE_SCHEMA,
            "opaque_id": opaque_id,
            "query_latent_commitment_digest": canonical_digest(
                latent_commitment),
            "oracle_freeze_digest": canonical_digest(freeze),
            "query_panel_set_digest": query_manifest["panel_set_digest"],
        }
        release_path = os.path.join(record_directory, "query_release.json")
        _write_exclusive_json(release_path, query_release)
        release_reference = _artifact_reference(
            out_dir, release_path, query_release)

        # Worker inputs are target paths only.  They are neutrally named and
        # carry neither labels nor Bongard side.  Stable map ordering is kept;
        # labels are joined only after every result has returned.
        with ThreadPoolExecutor(max_workers=args.scorer_workers) as executor:
            live_results = list(executor.map(oracle.evaluate, query_paths))
        serialized = [result.to_dict() for result in live_results]
        query_cases = _load_panel_set(
            out_dir, query_manifest, "query", latent)
        evaluation = _evaluate_joined(
            contract, compiled, serialized, query_cases, backend, oracle_mode)
        solved, status = _outcome(evaluation)
        records.append({
            "opaque_id": opaque_id,
            "program_split": program_manifest,
            "query_latent_commitment": latent_commitment,
            "query_latent_commitment_file": latent_reference,
            "support_panel_set": support_manifest,
            "proposal": proposal,
            "oracle_contract": contract_artifact,
            "formula": formula,
            "oracle_freeze": freeze,
            "oracle_freeze_file": freeze_reference,
            "query_panel_set": query_manifest,
            "query_release": query_release,
            "query_release_file": release_reference,
            "query_evaluation": evaluation,
            "solved": solved,
            "status": status,
        })
        print(
            f"[{index + 1:02d}/{len(latents):02d}] {opaque_id} {status} "
            f"claim={validated_proposal.claim!r}", flush=True)

    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "track": TRACK,
        "sampling_seed": args.sampling_seed,
        "support_render_seed": args.support_seed,
        "query_render_seed": args.query_seed,
        "pool_size": args.pool_size,
        "record_count": len(records),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "proposal_mode": proposal_mode,
        "oracle_mode": oracle_mode,
        "source_bindings": source_bindings,
        "source_bindings_digest": source_bindings_digest,
        "environment": _environment_binding(),
        "records": records,
        "information_boundary": {
            "dataset_root": os.path.abspath(args.dataset_dir),
            "proposer": "one-turn/twelve-labelled-support-images/one-claim",
            "query_latents": (
                "content-distinct-latent-program/style-pose-holdout;"
                "same-basic-category-template;committed-before-proposal"
            ),
            "query_pixels": "materialized-after-contract-and-formula-freeze",
            "query_presentation": (
                "side-blind-content-ranked-order/neutral-target-ordinals"
            ),
            "oracle": "one-neutral-target/frozen-contrastive-bundle/stateless",
            "reference_pool": "six-affirmative/six-hard-negative/content-bound",
            "reference_selection": (
                "claim-seeded-content-rank/freezes-three-anchor-foil-pairs"
            ),
            "oracle_trials_per_target": (
                "two-fresh-stateless-turns/full-pair-order-and-side-swap"
            ),
            "pair_decoder": (
                "role-agreement-across-both-turns;true-iff-two-of-three-"
                "anchor-and-zero-foil;false-symmetric;otherwise-abstain"
            ),
            "worker_arguments": "neutral-target-path-only/no-labels",
            "target_evaluations": 12,
            "threshold_fits": 0,
            "polarity_flips": 0,
            "support_self_scoring": 0,
            "adaptive_semantic_retries": 0,
            "taint": "HYBRID",
            "executable_predicate": (
                "OperationalResemblance(frozen_bundle,target)"
            ),
            "prose_claim_boundary": (
                "frozen oracle context only; benchmark does not establish "
                "the prose claim as pixel-level truth"
            ),
            "calibration": (
                "none; fixed decoder is exploratory until bound to an "
                "independent calibration manifest"
            ),
            "replay": (
                "stored evidence and downstream decisions only; no claim of "
                "cold perceptual reproduction"
            ),
        },
    }
    campaign["campaign_digest"] = canonical_digest(campaign)
    _write_campaign(campaign_path, campaign)
    replay_report = replay_campaign_directory(out_dir, backend=backend)
    if replay_report["status"] != "LIVE_EVIDENCE_REPLAY_VALID":
        raise RuntimeError("HYBRID campaign did not cold-replay")
    return campaign


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--dataset-dir", default=os.path.join(repo_root, "downloads", "Bongard-LOGO"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--sampling-seed", type=int, default=20260805)
    parser.add_argument("--support-seed", type=int, default=20260805)
    parser.add_argument("--query-seed", type=int, default=20260806)
    parser.add_argument(
        "--out-dir", default=os.path.join(
            os.path.dirname(__file__), "semantic_hybrid_runs", "latest"))
    parser.add_argument("--model", default=codex_proposer.DEFAULT_CODEX_MODEL)
    parser.add_argument(
        "--reasoning-effort", default=codex_proposer.DEFAULT_REASONING_EFFORT,
        choices=tuple(sorted(codex_proposer.REASONING_EFFORTS)))
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--scorer-workers", type=int, default=4)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--verbose-oracle", action="store_true")
    parser.add_argument(
        "--replay-artifact", metavar="DIRECTORY",
        help="validate live evidence and downstream decisions without model calls")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.replay_artifact:
        print(canonical_json(replay_campaign_directory(parsed.replay_artifact)))
    else:
        run(parsed)
