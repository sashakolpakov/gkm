#!/usr/bin/env python3
"""Run and cold-replay the grounded support/query Bongard benchmark.

Codex sees one labelled support rendering and may select only entries from a
closed observable catalog.  Numeric fitting and Boolean synthesis consume the
support rendering only.  The selected formula is frozen before a second
nuisance rendering is created, then evaluated on that hidden query rendering.

The resulting artifact is deliberately executable evidence, rather than a
claim that the image extractor itself has been proved correct: it binds the
exact raster files, observable contracts, implementation sources, proposal,
formula, and evaluation traces needed for model-free cold replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codex_proposer
import grounded_predicate_ir as G
import grounded_synthesis as S
from dataset import Problem, ProgrammedProblem, sample_problem_programs, write_panels
from grounded_observables import (
    GroundedPanelContext,
    ObservableDescriptor,
    default_grounded_observables,
)
from grounded_proposer import (
    CodexGroundedIntentProposer,
    GROUNDED_PROPOSAL_SCHEMA,
    GroundedProposalBundle,
    grounded_catalog_digest,
)


CAMPAIGN_SCHEMA = "bongard.grounded-semantic-campaign/v1"
PANEL_SET_SCHEMA = "bongard.grounded-panel-files/v1"
EVALUATION_SCHEMA = "bongard.grounded-evaluation/v1"
TRACK = "SEMANTIC-GROUNDED"

_SOURCE_FILES = (
    "run_grounded_semantic.py",
    "grounded_synthesis.py",
    "grounded_predicate_ir.py",
    "grounded_observables.py",
    "grounded_proposer.py",
    "codex_proposer.py",
    "semantic_legs.py",
    "visual_witnesses.py",
    "dataset.py",
)
_SIDES = (("pos", True), ("neg", False))


class GroundedProposer(Protocol):
    def propose(
        self, problem_id: str, panel_png_paths: Sequence[str]
    ) -> GroundedProposalBundle:
        ...


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _array_digest(panel: np.ndarray) -> str:
    array = np.ascontiguousarray(panel)
    header = G.canonical_json({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }).encode("utf-8")
    return _sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def _write_json(path: str, value: Any) -> None:
    encoded = G.canonical_json(value).encode("utf-8")
    temporary = path + ".tmp"
    with open(temporary, "xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_bindings() -> list[dict[str, str]]:
    directory = os.path.dirname(os.path.abspath(__file__))
    result = []
    for name in _SOURCE_FILES:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"required grounded source is absent: {name}")
        result.append({"path": name, "sha256": _sha256_file(path)})
    return result


def _panel_paths(directory: str, suffix: str) -> list[str]:
    paths = [
        os.path.abspath(os.path.join(directory, f"{side}_{index}.{suffix}"))
        for side, _label in _SIDES for index in range(6)
    ]
    if any(not os.path.isfile(path) for path in paths):
        raise RuntimeError(f"panel directory lacks twelve {suffix} files")
    return paths


def _panel_file_set(
    run_directory: str,
    panel_directory: str,
    split: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for side, label in _SIDES:
        for index in range(6):
            stem = f"{side}_{index}"
            npy_path = os.path.join(panel_directory, stem + ".npy")
            png_path = os.path.join(panel_directory, stem + ".png")
            panel = np.load(npy_path, allow_pickle=False)
            if panel.dtype != np.uint8 or panel.shape != (128, 128) \
                    or not np.isin(panel, (0, 1)).all():
                raise RuntimeError(f"invalid materialized panel {split}/{stem}")
            entries.append({
                "slot": stem,
                "side": side,
                "index": index,
                "label": label,
                "npy_path": os.path.relpath(npy_path, run_directory),
                "png_path": os.path.relpath(png_path, run_directory),
                "array_digest": _array_digest(panel),
                "npy_sha256": _sha256_file(npy_path),
                "png_sha256": _sha256_file(png_path),
            })
    png_paths = _panel_paths(panel_directory, "png")
    body = {
        "schema": PANEL_SET_SCHEMA,
        "split": split,
        "panels": entries,
        "semantic_panel_set_digest": codex_proposer.semantic_panel_set_digest(
            png_paths),
    }
    body["panel_set_digest"] = G.canonical_digest(body)
    return body


def _materialize_split(
    run_directory: str,
    workspace: str,
    opaque_id: str,
    split: str,
    problem: Problem,
) -> tuple[dict[str, Any], list[str]]:
    panel_directory = write_panels(
        workspace, problem, os.path.join(opaque_id, split))
    manifest = _panel_file_set(run_directory, panel_directory, split)
    return manifest, _panel_paths(panel_directory, "png")


def _safe_artifact_path(run_directory: str, relative: Any) -> str:
    if not isinstance(relative, str) or not relative \
            or os.path.isabs(relative):
        raise ValueError("panel artifact path must be a nonempty relative path")
    root = os.path.realpath(run_directory)
    path = os.path.realpath(os.path.join(root, relative))
    try:
        within = os.path.commonpath((root, path)) == root
    except ValueError:
        within = False
    if not within:
        raise ValueError("panel artifact path escapes the campaign directory")
    return path


def _load_panel_set(
    run_directory: str, stored: Mapping[str, Any], expected_split: str
) -> tuple[tuple[np.ndarray, bool, str], ...]:
    if not isinstance(stored, Mapping) or set(stored) != {
        "schema", "split", "panels", "semantic_panel_set_digest",
        "panel_set_digest"
    }:
        raise ValueError("panel-set fields differ")
    unsigned = {key: value for key, value in stored.items()
                if key != "panel_set_digest"}
    if stored["schema"] != PANEL_SET_SCHEMA \
            or stored["split"] != expected_split \
            or stored["panel_set_digest"] != G.canonical_digest(unsigned):
        raise ValueError(f"{expected_split} panel-set digest does not reproduce")
    raw_panels = stored["panels"]
    if not isinstance(raw_panels, list) or len(raw_panels) != 12:
        raise ValueError(f"{expected_split} must bind twelve panels")
    result: list[tuple[np.ndarray, bool, str]] = []
    png_paths: list[str] = []
    expected_slots = [
        (f"{side}_{index}", side, index, label)
        for side, label in _SIDES for index in range(6)
    ]
    entry_keys = {
        "slot", "side", "index", "label", "npy_path", "png_path",
        "array_digest", "npy_sha256", "png_sha256",
    }
    for raw, (slot, side, index, label) in zip(raw_panels, expected_slots):
        if not isinstance(raw, Mapping) or set(raw) != entry_keys \
                or (raw["slot"], raw["side"], raw["index"], raw["label"]) \
                != (slot, side, index, label):
            raise ValueError(f"{expected_split} panel ordering/labels differ")
        npy_path = _safe_artifact_path(run_directory, raw["npy_path"])
        png_path = _safe_artifact_path(run_directory, raw["png_path"])
        png_paths.append(png_path)
        if _sha256_file(npy_path) != raw["npy_sha256"] \
                or _sha256_file(png_path) != raw["png_sha256"]:
            raise ValueError(f"{expected_split}/{slot} file digest mismatch")
        panel = np.load(npy_path, allow_pickle=False)
        if panel.dtype != np.uint8 or panel.shape != (128, 128) \
                or not np.isin(panel, (0, 1)).all() \
                or _array_digest(panel) != raw["array_digest"]:
            raise ValueError(f"{expected_split}/{slot} array digest mismatch")
        with Image.open(png_path) as encoded:
            presentation = np.asarray(encoded.convert("L"))
        if presentation.shape != panel.shape \
                or not np.isin(presentation, (0, 255)).all() \
                or not np.array_equal((presentation == 0).astype(np.uint8), panel):
            raise ValueError(f"{expected_split}/{slot} PNG differs from NPY")
        result.append((np.ascontiguousarray(panel), label, slot))
    if codex_proposer.semantic_panel_set_digest(png_paths) != \
            stored["semantic_panel_set_digest"]:
        raise ValueError(f"{expected_split} semantic panel digest mismatch")
    return tuple(result)


def _truth(trace: G.EvaluationTrace) -> bool | None:
    result = trace.result
    if isinstance(result, G.Present) and result.unit is G.Unit.BOOLEAN \
            and isinstance(result.value, bool):
        return result.value
    return None


def _evaluate_compiled(
    compiled: G.CompiledPredicate,
    cases: Sequence[tuple[np.ndarray, bool, str]],
    split: str,
) -> dict[str, Any]:
    decisions = []
    errors = indeterminate = 0
    correct = 0
    for panel, label, slot in cases:
        trace = compiled.evaluate_with_trace(GroundedPanelContext(panel))
        observations = tuple(value for _key, value in trace.observations)
        has_error = isinstance(trace.result, G.Error) or any(
            isinstance(value, G.Error) for value in observations)
        has_indeterminate = isinstance(trace.result, G.Indeterminate) or any(
            isinstance(value, G.Indeterminate) for value in observations)
        predicted = None if has_error or has_indeterminate else _truth(trace)
        errors += int(has_error)
        indeterminate += int(has_indeterminate)
        is_correct = predicted is not None and predicted is label
        correct += int(is_correct)
        decisions.append({
            "slot": slot,
            "label": label,
            "predicted": predicted,
            "correct": is_correct,
            "trace": trace.to_dict(),
        })
    summary = {
        "schema": EVALUATION_SCHEMA,
        "split": split,
        "case_count": len(decisions),
        "correct_count": correct,
        "error_count": errors,
        "indeterminate_count": indeterminate,
        "exact": (
            len(decisions) == 12 and correct == 12
            and errors == 0 and indeterminate == 0
        ),
        "decisions": decisions,
    }
    summary["evaluation_digest"] = G.canonical_digest(summary)
    return summary


def _formula_artifact(compiled: G.CompiledPredicate) -> dict[str, Any]:
    body = compiled.canonical_dict()
    if G.canonical_digest(body) != compiled.digest:
        raise RuntimeError("compiled predicate digest is internally inconsistent")
    return {"digest": compiled.digest, "compiled": body}


def _compile_formula(
    stored: Mapping[str, Any], registry: G.ObservableRegistry
) -> G.CompiledPredicate:
    if not isinstance(stored, Mapping) or set(stored) != {"digest", "compiled"} \
            or not isinstance(stored["compiled"], Mapping):
        raise ValueError("formula artifact fields differ")
    compiled_body = stored["compiled"]
    if set(compiled_body) != {"schema", "predicate", "contracts", "taint"} \
            or compiled_body["schema"] != G.PREDICATE_IR_SCHEMA:
        raise ValueError("formula compiled body differs")
    compiled = G.compile_predicate(compiled_body["predicate"], registry)
    if compiled.canonical_dict() != dict(compiled_body) \
            or compiled.digest != stored["digest"]:
        raise ValueError("formula does not compile to its stored binding")
    return compiled


def _same(left: Any, right: Any) -> bool:
    return G.canonical_digest(left) == G.canonical_digest(right)


def _campaign_status(
    support: Mapping[str, Any], query: Mapping[str, Any]
) -> tuple[bool, str]:
    valid = all(
        isinstance(item.get(key), int) and item[key] == 0
        for item in (support, query)
        for key in ("error_count", "indeterminate_count")
    )
    solved = bool(valid and support.get("exact") is True
                  and query.get("exact") is True)
    return solved, (
        "SOLVED_SEMANTIC_GROUNDED" if solved else
        "INVALID_SEMANTIC_GROUNDED" if not valid else
        "UNSOLVED_SEMANTIC_GROUNDED"
    )


def _not_evaluated(split: str, reason: str) -> dict[str, Any]:
    body = {
        "schema": EVALUATION_SCHEMA,
        "split": split,
        "case_count": 0,
        "correct_count": 0,
        "error_count": 0,
        "indeterminate_count": 0,
        "exact": False,
        "decisions": [],
        "not_evaluated_reason": reason,
    }
    body["evaluation_digest"] = G.canonical_digest(body)
    return body


def _validate_proposal(
    value: Any,
    opaque_id: str,
    descriptors: Sequence[ObservableDescriptor],
    expected_panel_set_digest: str,
) -> None:
    keys = {
        "schema", "problem_id", "analysis", "intents", "catalog_digest",
        "receipt",
    }
    if not isinstance(value, Mapping) or set(value) != keys \
            or value["schema"] != GROUNDED_PROPOSAL_SCHEMA \
            or value["problem_id"] != opaque_id \
            or value["catalog_digest"] != grounded_catalog_digest(descriptors) \
            or not isinstance(value["analysis"], str) \
            or not isinstance(value["intents"], list) \
            or not isinstance(value["receipt"], Mapping):
        raise ValueError("grounded proposal binding differs")
    if value["receipt"].get("panel_set_digest") != \
            expected_panel_set_digest:
        raise ValueError("grounded proposal receipt is bound to different panels")
    allowed = {
        descriptor.contract.observable_id: set(descriptor.admissible_shapes)
        for descriptor in descriptors
    }
    seen: set[tuple[str, str]] = set()
    for position, raw in enumerate(value["intents"]):
        if not isinstance(raw, Mapping) or set(raw) != {
            "intent_id", "observable_id", "shape", "rationale"
        }:
            raise ValueError("grounded proposal intent fields differ")
        key = (raw["observable_id"], raw["shape"])
        if raw["intent_id"] != f"intent-{position:02d}" \
                or key in seen or raw["observable_id"] not in allowed \
                or raw["shape"] not in allowed[raw["observable_id"]] \
                or not isinstance(raw["rationale"], str) \
                or not raw["rationale"]:
            raise ValueError("grounded proposal intent is invalid")
        seen.add(key)
    if not seen:
        raise ValueError("grounded proposal contains no intents")


def _synthesis_artifact(
    result_payload: Mapping[str, Any],
    support_panel_set_digest: str,
    proposal: Mapping[str, Any],
    formula_digest: str,
) -> dict[str, Any]:
    body = {
        "support_panel_set_digest": support_panel_set_digest,
        "proposal_intents_digest": G.canonical_digest(proposal["intents"]),
        "formula_digest": formula_digest,
        "result": dict(result_payload),
    }
    body["synthesis_digest"] = G.canonical_digest(body)
    return body


def _validate_synthesis_binding(
    value: Any,
    support_panel_set_digest: str,
    proposal: Mapping[str, Any],
    formula_digest: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "support_panel_set_digest", "proposal_intents_digest",
        "formula_digest", "result", "synthesis_digest",
    } or not isinstance(value["result"], Mapping):
        raise ValueError("synthesis artifact fields differ")
    unsigned = {key: item for key, item in value.items()
                if key != "synthesis_digest"}
    if value["synthesis_digest"] != G.canonical_digest(unsigned) \
            or value["support_panel_set_digest"] != support_panel_set_digest \
            or value["proposal_intents_digest"] != G.canonical_digest(
                proposal["intents"]) \
            or value["formula_digest"] != formula_digest:
        raise ValueError("synthesis artifact binding does not reproduce")


def replay_campaign_artifact(
    campaign: Mapping[str, Any], run_directory: str
) -> dict[str, Any]:
    """Cold-recompute all formula decisions from persisted panel files."""
    campaign_keys = {
        "schema", "track", "source", "program_seed", "support_seed",
        "query_seed", "limit_per_source", "record_count", "model",
        "reasoning_effort", "catalog", "catalog_digest", "registry_digest",
        "source_bindings", "records", "information_boundary",
        "campaign_digest",
    }
    if not isinstance(campaign, Mapping) or set(campaign) != campaign_keys:
        raise ValueError("grounded campaign fields differ")
    unsigned = {key: value for key, value in campaign.items()
                if key != "campaign_digest"}
    if campaign["schema"] != CAMPAIGN_SCHEMA or campaign["track"] != TRACK \
            or campaign["campaign_digest"] != G.canonical_digest(unsigned) \
            or not isinstance(campaign["records"], list) \
            or campaign["record_count"] != len(campaign["records"]):
        raise ValueError("grounded campaign digest/schema does not reproduce")
    if campaign["support_seed"] == campaign["query_seed"]:
        raise ValueError("support and query render seeds must differ")
    observed_sources = _source_bindings()
    if campaign["source_bindings"] != observed_sources:
        raise ValueError("grounded implementation source digest mismatch")

    registry, descriptors = default_grounded_observables()
    catalog = [descriptor.prompt_dict() for descriptor in descriptors]
    if campaign["catalog"] != catalog \
            or campaign["catalog_digest"] != grounded_catalog_digest(descriptors) \
            or campaign["registry_digest"] != registry.version_digest():
        raise ValueError("grounded observable catalog/registry mismatch")

    record_keys = {
        "opaque_id", "generator_metadata", "support_panel_set", "proposal",
        "synthesis", "formula", "support_evaluation", "query_panel_set",
        "query_evaluation", "solved", "status",
    }
    solved_count = 0
    for index, record in enumerate(campaign["records"]):
        opaque_id = f"problem_{index:02d}"
        if not isinstance(record, Mapping) or set(record) != record_keys \
                or record["opaque_id"] != opaque_id \
                or not isinstance(record["generator_metadata"], Mapping):
            raise ValueError(f"grounded record {index} fields differ")
        _validate_proposal(
            record["proposal"], opaque_id, descriptors,
            record["support_panel_set"].get("semantic_panel_set_digest", ""))
        support_cases = _load_panel_set(
            run_directory, record["support_panel_set"], "support")
        query_cases = _load_panel_set(
            run_directory, record["query_panel_set"], "query")
        if record["support_panel_set"]["semantic_panel_set_digest"] == \
                record["query_panel_set"]["semantic_panel_set_digest"]:
            raise ValueError("hidden query rendering is identical to support")
        if record["formula"] is None:
            _validate_synthesis_binding(
                record["synthesis"],
                record["support_panel_set"]["panel_set_digest"],
                record["proposal"], "")
            intents = tuple(S.MeasurementIntent(
                raw["intent_id"], raw["observable_id"], raw["shape"])
                for raw in record["proposal"]["intents"])
            try:
                S.synthesize_grounded_predicate(
                    intents, _support_cases(support_cases), registry)
            except S.NoGroundedSeparator as exc:
                diagnostic = str(exc)
            else:
                raise ValueError("stored no-separator result now synthesizes")
            expected_support = _not_evaluated(
                "support", "no-grounded-support-separator")
            expected_query = _not_evaluated(
                "query", "no-grounded-support-separator")
            payload = record["synthesis"]["result"]
            if payload != {
                "status": "no-grounded-separator",
                "diagnostic": diagnostic,
            } or not _same(expected_support, record["support_evaluation"]) \
                    or not _same(expected_query, record["query_evaluation"]) \
                    or record["solved"] is not False \
                    or record["status"] != "UNSOLVED_SEMANTIC_GROUNDED":
                raise ValueError("stored no-separator result does not replay")
            continue

        compiled = _compile_formula(record["formula"], registry)
        _validate_synthesis_binding(
            record["synthesis"],
            record["support_panel_set"]["panel_set_digest"],
            record["proposal"], compiled.digest)
        intents = tuple(S.MeasurementIntent(
            raw["intent_id"], raw["observable_id"], raw["shape"])
            for raw in record["proposal"]["intents"])
        try:
            frozen = S.synthesize_grounded_predicate(
                intents, _support_cases(support_cases), registry)
        except S.NoGroundedSeparator as exc:
            raise ValueError(
                "stored formula no longer synthesizes from support") from exc
        synthesis_payload = record["synthesis"]["result"]
        if not isinstance(synthesis_payload, Mapping) \
                or set(synthesis_payload) != {"frozen", "hidden_query"} \
                or frozen.compiled.digest != compiled.digest \
                or not _same(frozen.to_dict(), synthesis_payload["frozen"]):
            raise ValueError("stored frozen synthesis does not replay")
        hidden = S.evaluate_hidden_queries(
            frozen, _support_cases(query_cases))
        if not _same(hidden.to_dict(), synthesis_payload["hidden_query"]):
            raise ValueError("stored hidden-query synthesis trace does not replay")
        support = _evaluate_compiled(compiled, support_cases, "support")
        query = _evaluate_compiled(compiled, query_cases, "query")
        if not _same(support, record["support_evaluation"]) \
                or not _same(query, record["query_evaluation"]):
            raise ValueError("stored grounded evaluation does not replay")
        solved, status = _campaign_status(support, query)
        if record["solved"] is not solved or record["status"] != status:
            raise ValueError("stored grounded result status does not replay")
        solved_count += int(solved)
    return {
        "schema": "bongard.grounded-semantic-replay-report/v1",
        "campaign_digest": campaign["campaign_digest"],
        "record_count": len(campaign["records"]),
        "solved_count": solved_count,
        "valid": True,
    }


def replay_campaign_directory(directory: str) -> dict[str, Any]:
    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "campaign.json"), encoding="utf-8") \
            as handle:
        campaign = json.load(handle)
    return replay_campaign_artifact(campaign, directory)


def _measurement_intents(
    bundle: GroundedProposalBundle,
) -> tuple[S.MeasurementIntent, ...]:
    return tuple(S.MeasurementIntent(
        intent.intent_id, intent.observable_id, intent.shape)
                 for intent in bundle.intents)


def _support_cases(
    cases: Sequence[tuple[np.ndarray, bool, str]],
) -> tuple[S.SupportCase, ...]:
    return tuple(S.SupportCase(
        case_id=slot,
        context=GroundedPanelContext(panel),
        label=label,
    ) for panel, label, slot in cases)


def run(
    args: argparse.Namespace,
    *,
    proposer: GroundedProposer | None = None,
) -> dict[str, Any]:
    out_dir = os.path.abspath(args.out_dir)
    campaign_path = os.path.join(out_dir, "campaign.json")
    if os.path.exists(campaign_path):
        raise RuntimeError(
            "campaign.json already exists; grounded runs are immutable")
    if args.support_seed == args.query_seed:
        raise SystemExit("support-seed and query-seed must differ")
    if args.limit <= 0 or args.corpus_size < 0 \
            or not 1 <= args.minutes <= 120:
        raise SystemExit(
            "limit must be positive, corpus-size nonnegative, and minutes in 1..120")

    latent_problems = sample_problem_programs(
        args.dataset_dir,
        limit=args.limit,
        seed=args.program_seed,
        source=args.source,
    )
    active_size = args.corpus_size or len(latent_problems)
    if active_size > len(latent_problems):
        raise SystemExit("--corpus-size exceeds the sampled latent corpus")
    os.makedirs(out_dir, exist_ok=True)
    workspace = os.path.join(out_dir, "workspace")
    if os.path.exists(workspace) and os.listdir(workspace):
        raise RuntimeError("grounded workspace is nonempty; choose a fresh out-dir")
    os.makedirs(workspace, exist_ok=True)

    registry, descriptors = default_grounded_observables()
    if proposer is None:
        proposer = CodexGroundedIntentProposer(
            descriptors,
            model=args.model,
            minutes=args.minutes,
            reasoning_effort=args.reasoning_effort,
        )

    records: list[dict[str, Any]] = []
    for index, latent in enumerate(latent_problems[:active_size]):
        if not isinstance(latent, ProgrammedProblem):
            raise RuntimeError("latent sampler returned a non-programmed problem")
        opaque_id = f"problem_{index:02d}"

        # The hidden rendering is intentionally not even generated here.  The
        # isolated proposer receives only these twelve canonical support PNGs.
        support_problem = latent.render(args.support_seed)
        support_manifest, support_png_paths = _materialize_split(
            out_dir, workspace, opaque_id, "support", support_problem)
        bundle = proposer.propose(opaque_id, support_png_paths)
        proposal = bundle.to_dict()
        _validate_proposal(
            proposal, opaque_id, descriptors,
            support_manifest["semantic_panel_set_digest"])

        support_arrays = _load_panel_set(out_dir, support_manifest, "support")
        try:
            frozen = S.synthesize_grounded_predicate(
                _measurement_intents(bundle),
                _support_cases(support_arrays),
                registry,
            )
        except S.NoGroundedSeparator as exc:
            frozen = None
            compiled = None
            formula = None
            support_evaluation = _not_evaluated(
                "support", "no-grounded-support-separator")
            no_separator_diagnostic = str(exc)
        else:
            compiled = frozen.compiled
            if not isinstance(compiled, G.CompiledPredicate):
                raise RuntimeError(
                    "grounded synthesis did not freeze a compiled predicate")
            formula = _formula_artifact(compiled)
            support_evaluation = _evaluate_compiled(
                compiled, support_arrays, "support")

        # This is the first point at which query pixels exist.  The proposal
        # and every fitted threshold above are already frozen.
        query_problem = latent.render(args.query_seed)
        query_manifest, _query_png_paths = _materialize_split(
            out_dir, workspace, opaque_id, "query", query_problem)
        if query_manifest["semantic_panel_set_digest"] == \
                support_manifest["semantic_panel_set_digest"]:
            raise RuntimeError("query rendering did not differ from support")
        query_arrays = _load_panel_set(out_dir, query_manifest, "query")
        if frozen is None:
            query_evaluation = _not_evaluated(
                "query", "no-grounded-support-separator")
            solved, status = False, "UNSOLVED_SEMANTIC_GROUNDED"
            synthesis_payload = {
                "status": "no-grounded-separator",
                "diagnostic": no_separator_diagnostic,
            }
            formula_digest = ""
        else:
            assert compiled is not None
            hidden_evaluation = S.evaluate_hidden_queries(
                frozen, _support_cases(query_arrays))
            query_evaluation = _evaluate_compiled(
                compiled, query_arrays, "query")
            solved, status = _campaign_status(
                support_evaluation, query_evaluation)
            synthesis_payload = {
                "frozen": frozen.to_dict(),
                "hidden_query": hidden_evaluation.to_dict(),
            }
            formula_digest = compiled.digest
        synthesis = _synthesis_artifact(
            synthesis_payload,
            support_manifest["panel_set_digest"],
            proposal,
            formula_digest,
        )
        record = {
            "opaque_id": opaque_id,
            "generator_metadata": {
                "problem_id": latent.problem_id,
                "category": latent.category,
                "concept": latent.concept,
            },
            "support_panel_set": support_manifest,
            "proposal": proposal,
            "synthesis": synthesis,
            "formula": formula,
            "support_evaluation": support_evaluation,
            "query_panel_set": query_manifest,
            "query_evaluation": query_evaluation,
            "solved": solved,
            "status": status,
        }
        records.append(record)
        print(
            f"[{index + 1:02d}/{active_size:02d}] {opaque_id} {status} "
            f"formula={formula_digest or '-'}",
            flush=True,
        )

    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "track": TRACK,
        "source": args.source,
        "program_seed": args.program_seed,
        "support_seed": args.support_seed,
        "query_seed": args.query_seed,
        "limit_per_source": args.limit,
        "record_count": len(records),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "catalog": [descriptor.prompt_dict() for descriptor in descriptors],
        "catalog_digest": grounded_catalog_digest(descriptors),
        "registry_digest": registry.version_digest(),
        "source_bindings": _source_bindings(),
        "records": records,
        "information_boundary": {
            "proposer": "exactly-twelve-labelled-support-pngs/one-turn",
            "proposer_catalog": "closed-registered-observable-ids",
            "synthesis": "support-labels-and-support-observations-only",
            "query_creation": "after-formula-freeze",
            "query_model_calls": 0,
            "query_threshold_fits": 0,
            "polarity": "positive-atoms-and-conjunctions/no-negated-rescue",
            "claim": (
                "deterministic typed evaluation conditional on registered "
                "pixel extractors; not a proof of the visual ontology"
            ),
        },
    }
    campaign["campaign_digest"] = G.canonical_digest(campaign)
    _write_json(campaign_path, campaign)
    replay_campaign_directory(out_dir)
    return campaign


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(repo_root, "downloads", "Bongard-LOGO"),
    )
    parser.add_argument(
        "--source", choices=("basic", "abstract", "both"), default="basic")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--corpus-size", type=int, default=1)
    parser.add_argument("--program-seed", type=int, default=20260805)
    parser.add_argument("--support-seed", type=int, default=20260805)
    parser.add_argument("--query-seed", type=int, default=20260806)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(__file__), "semantic_grounded_runs", "latest"),
    )
    parser.add_argument("--model", default=codex_proposer.DEFAULT_CODEX_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default=codex_proposer.DEFAULT_REASONING_EFFORT,
        choices=tuple(sorted(codex_proposer.REASONING_EFFORTS)),
    )
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument(
        "--replay-artifact", metavar="DIRECTORY",
        help="cold-validate an existing campaign without model calls",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.replay_artifact:
        print(G.canonical_json(replay_campaign_directory(
            parsed.replay_artifact)))
    else:
        run(parsed)
