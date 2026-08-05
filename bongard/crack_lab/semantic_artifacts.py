"""ARC-style artifact discipline for the semantic track.

Mirrors the unrestricted track's scheme:

- promoted artifacts live in ``agent_solutions/<tag>_semantic/`` and hold
  ``checkpoint.json``, ``promoted_cones.json``, harness-only ``results.json``
  (the ONLY place ground-truth concept names may exist) and ``README.md``;
- failed attempts are snapshotted append-only under
  ``wip_context/<opaque_id>/<timestamp>/`` and are never admitted;
- promotion is gated on a taint scan plus one data-self-contained RunSpec per
  cone;
- every RunSpec is executed through ``replay_semantic_runspec.py`` in a fresh
  interpreter, and the artifact receives the spec and a PASS receipt only when
  panel, cone, complete checkpoint candidate manifest, full selected record,
  policy, registry/source, environment, verifier verdicts, and selection all
  reproduce exactly.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import fields

CODE_LAB_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = CODE_LAB_DIR
REPLAY_SCRIPT = os.path.join(CODE_LAB_DIR, "replay_semantic_runspec.py")

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


class ReplayCertificationError(RuntimeError):
    pass


def artifact_dir(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
        raise ValueError("artifact tag must be a simple 1-64 character name")
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
    stamp = (time.strftime("%Y%m%dT%H%M%S", time.gmtime())
             + f"_{time.time_ns()}")
    dest = os.path.join(artifact_dir(tag), "wip_context", opaque_id, stamp)
    os.makedirs(dest, exist_ok=False)
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        if name.startswith(opaque_id) or name == "checkpoint.json":
            shutil.copy2(path, os.path.join(dest, name))
    return dest


def _atomic_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(path),
                prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                delete=False) as handle:
            temp_name = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def atomic_json(path: str, payload: object) -> None:
    """Public atomic JSON writer for runner manifests and checkpoints."""
    _atomic_json(path, payload)


def create_json_once(path: str, payload: object) -> bool:
    """Atomically create canonical JSON without replacing a concurrent winner."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory_fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_text(path: str, payload: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(path),
                prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                delete=False) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _atomic_copy(source: str, destination: str) -> None:
    """Copy one certified JSON document without exposing a partial file."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp_name = None
    try:
        with open(source, "rb") as source_handle, tempfile.NamedTemporaryFile(
                mode="wb", dir=os.path.dirname(destination),
                prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
                delete=False) as destination_handle:
            temp_name = destination_handle.name
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _strict_count(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayCertificationError(
            f"{label} must be an integer >= {minimum}")
    return value


def _validate_run_inputs(payload: dict, results: dict,
                         corpus_manifest: dict,
                         corpus_bundle: dict | None,
                         control_manifest: dict | None, *,
                         require_complete: bool = False) -> dict:
    """Validate the full report/promotion denominator against Phase D bytes."""
    from phase_d_protocol import (
        COUNT_POLICY,
        OBSERVED,
        SHARED,
        SHUFFLED_SIDES,
        build_shuffled_sides_control,
        problems_from_corpus_bundle,
        validate_corpus_bundle,
        validate_corpus_manifest,
        validate_shuffled_control_manifest,
    )

    if not isinstance(payload, dict):
        raise ReplayCertificationError("checkpoint payload must be an object")
    if not isinstance(results, dict):
        raise ReplayCertificationError("results payload must be an object")
    try:
        validate_corpus_manifest(corpus_manifest)
    except Exception as exc:
        raise ReplayCertificationError(
            f"run report requires a valid frozen corpus manifest: {exc}") from exc
    if corpus_bundle is None:
        raise ReplayCertificationError(
            "run report requires a valid embedded corpus bundle")
    try:
        validate_corpus_bundle(corpus_bundle, corpus_manifest)
    except Exception as exc:
        raise ReplayCertificationError(
            f"run report requires a valid embedded corpus bundle: {exc}") from exc

    condition = payload.get("condition")
    sharing_policy = payload.get("sharing_policy")
    if condition not in {OBSERVED, SHUFFLED_SIDES}:
        raise ReplayCertificationError(
            "checkpoint condition must be observed or shuffled-sides")
    if sharing_policy != SHARED:
        raise ReplayCertificationError(
            "semantic artifacts require the shared Phase D policy")
    if "control" not in payload:
        raise ReplayCertificationError(
            "checkpoint must explicitly record control evidence or null")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) \
            or dataset.get("corpus_digest") != corpus_manifest["corpus_digest"]:
        raise ReplayCertificationError(
            "checkpoint corpus identity differs from corpus manifest")
    if dataset.get("corpus_bundle_digest") != corpus_bundle["bundle_digest"]:
        raise ReplayCertificationError(
            "checkpoint corpus bundle identity differs from embedded bytes")
    frozen_count = _strict_count(
        dataset.get("frozen_problem_count"),
        "dataset.frozen_problem_count", minimum=1)
    if frozen_count != corpus_manifest["problem_count"]:
        raise ReplayCertificationError(
            "checkpoint frozen problem count differs from corpus manifest")
    active_size = _strict_count(
        dataset.get("active_prefix_size"),
        "dataset.active_prefix_size", minimum=0)
    if active_size > frozen_count:
        raise ReplayCertificationError(
            "checkpoint active prefix exceeds the frozen corpus")
    sampling = corpus_manifest["sampling"]
    expected_dataset = {
        "source": sampling["source"],
        "seed": sampling["seed"],
        "count_policy": COUNT_POLICY,
        "limit_per_source": sampling["limit_per_source"],
        "active_prefix_size": active_size,
        "frozen_problem_count": corpus_manifest["problem_count"],
        "order_policy": sampling["order_policy"],
        "repository_commit": sampling["dataset_revision"],
        "corpus_digest": corpus_manifest["corpus_digest"],
        "corpus_manifest": "corpus_manifest.json",
        "corpus_bundle_digest": corpus_bundle["bundle_digest"],
        "corpus_bundle": "corpus_panels.json",
        "panel_bytes": (
            "all records bind corpus panel-set digests; solved replay_specs "
            "also embed canonical panel bytes"),
    }
    if dataset != expected_dataset:
        raise ReplayCertificationError(
            "checkpoint sampling provenance differs from frozen corpus")

    expected_control_digest = ""
    if condition == SHUFFLED_SIDES:
        if control_manifest is None:
            raise ReplayCertificationError(
                "shuffled-side artifact requires its control manifest")
        try:
            validate_shuffled_control_manifest(
                control_manifest, corpus_manifest)
        except Exception as exc:
            raise ReplayCertificationError(
                f"invalid shuffled-side control manifest: {exc}") from exc
        expected_control_digest = control_manifest["control_digest"]
        control = payload.get("control")
        expected_control_fields = {
            "schema": control_manifest["schema"],
            "control_digest": expected_control_digest,
            "seed": control_manifest["seed"],
            "replicate": control_manifest["replicate"],
            "assignment_policy": control_manifest["assignment_policy"],
        }
        if not isinstance(control, dict) or any(
                control.get(name) != value
                for name, value in expected_control_fields.items()):
            raise ReplayCertificationError(
                "checkpoint control identity differs from control manifest")
    elif control_manifest is not None or payload.get("control") is not None:
        raise ReplayCertificationError(
            "observed artifact must not carry control evidence")

    try:
        replay_base = problems_from_corpus_bundle(
            corpus_bundle, corpus_manifest)
        if control_manifest is None:
            replay_problems = replay_base
        else:
            replay_control = build_shuffled_sides_control(
                replay_base, corpus_manifest,
                seed=control_manifest["seed"],
                replicate=control_manifest["replicate"],
            )
            if replay_control.manifest["control_digest"] != \
                    control_manifest["control_digest"]:
                raise ValueError("control digest does not reproduce")
            replay_problems = replay_control.problems
    except Exception as exc:
        raise ReplayCertificationError(
            f"cannot reconstruct terminal replay panels: {exc}") from exc

    verifier_policy = payload.get("verifier_policy")
    selection_policy = payload.get("selection")
    if not isinstance(verifier_policy, dict) \
            or not isinstance(selection_policy, dict):
        raise ReplayCertificationError(
            "checkpoint lacks verifier/selection replay policy")
    max_support_errors = _strict_count(
        verifier_policy.get("max_support_errors"),
        "verifier_policy.max_support_errors")
    max_loo_errors = _strict_count(
        verifier_policy.get("max_threshold_loo_errors"),
        "verifier_policy.max_threshold_loo_errors")
    max_rotated_loo_errors = _strict_count(
        verifier_policy.get("max_pair_threshold_loo_errors"),
        "verifier_policy.max_pair_threshold_loo_errors")
    round_limit = _strict_count(
        payload.get("rounds"), "rounds", minimum=1)
    lambda_value = selection_policy.get("lambda")
    if isinstance(lambda_value, bool) or not isinstance(
            lambda_value, (int, float)) \
            or not math.isfinite(float(lambda_value)) \
            or float(lambda_value) < 0.0:
        raise ReplayCertificationError(
            "selection.lambda must be finite and nonnegative for terminal replay")
    from run_semantic_cone import ProblemResult, _replay_terminal_record
    from semantic_replay import canonical_json_digest
    expected_record_fields = {item.name for item in fields(ProblemResult)}
    expected_result_fields = {
        "problem_id", "category", "concept", "track", "condition",
        "sharing_policy", "corpus_digest", "panel_set_digest",
        "control_digest", "solved", "status", "rule",
        "selected_hypothesis", "selected_description", "selected_rule",
        "support_errors", "loo_errors", "rotated_loo_errors",
        "rotated_loo_checks", "n_examples", "complexity", "rounds_used",
        "proposer_kind", "proposer_error",
    }

    records = payload.get("records")
    if not isinstance(records, list):
        raise ReplayCertificationError(
            "checkpoint payload must contain a record list")
    attempted = _strict_count(payload.get("attempted"), "attempted")
    solved = _strict_count(payload.get("solved"), "solved")
    if attempted != len(records) or attempted > active_size:
        raise ReplayCertificationError(
            "checkpoint attempted count does not match its active record prefix")
    if require_complete and attempted != active_size:
        raise ReplayCertificationError(
            "run report must contain the complete active record prefix")
    if set(results) != {
            f"problem_{index:02d}" for index in range(attempted)}:
        raise ReplayCertificationError(
            "results keys must exactly match the contiguous checkpoint prefix")

    records_by_oid: dict[str, dict] = {}
    terminal_replays: dict[str, dict] = {}
    solved_oids: set[str] = set()
    for index, record in enumerate(records):
        oid = f"problem_{index:02d}"
        if not isinstance(record, dict) or record.get("opaque_id") != oid:
            raise ReplayCertificationError(
                f"checkpoint records must be contiguous at {oid}")
        if set(record) != expected_record_fields:
            raise ReplayCertificationError(
                f"checkpoint record {oid} fields differ from the terminal schema")
        record_solved = record.get("solved")
        if not isinstance(record_solved, bool):
            raise ReplayCertificationError(f"{oid}.solved must be boolean")
        manifest_entry = corpus_manifest["problems"][index]
        control_entry = (
            control_manifest["problems"][index]
            if control_manifest is not None else None)
        expected_panel_digest = (
            control_entry["controlled_panel_set_digest"]
            if control_entry is not None else manifest_entry["panel_set_digest"])
        expected_identity = {
            "track": "SEMANTIC-PURE",
            "condition": condition,
            "sharing_policy": SHARED,
            "corpus_digest": corpus_manifest["corpus_digest"],
            "panel_set_digest": expected_panel_digest,
            "control_digest": expected_control_digest,
        }
        if any(record.get(name) != value
               for name, value in expected_identity.items()):
            raise ReplayCertificationError(
                f"checkpoint record {oid} violates corpus/arm identity")
        if record.get("category") != manifest_entry["category"]:
            raise ReplayCertificationError(
                f"checkpoint record {oid} category differs from the manifest")
        if not isinstance(record.get("status"), str) \
                or not record["status"] \
                or not isinstance(record.get("selected_rule"), str):
            raise ReplayCertificationError(
                f"checkpoint record {oid} has invalid result fields")
        try:
            terminal_replay = _replay_terminal_record(
                record, replay_problems[index],
                max_support_errors=max_support_errors,
                max_loo_errors=max_loo_errors,
                max_rotated_loo_errors=max_rotated_loo_errors,
                lambda_value=float(lambda_value),
                round_limit=round_limit,
            )
        except Exception as exc:
            raise ReplayCertificationError(
                f"checkpoint terminal record {oid} does not replay: {exc}") \
                from exc

        result = results.get(oid)
        if not isinstance(result, dict):
            raise ReplayCertificationError(f"result {oid} must be an object")
        if set(result) != expected_result_fields:
            raise ReplayCertificationError(
                f"result {oid} fields differ from the canonical summary schema")
        result_identity = {
            **expected_identity,
            "solved": record_solved,
            "status": record["status"],
            "rule": record["selected_rule"],
        }
        if any(result.get(name) != value
               for name, value in result_identity.items()):
            raise ReplayCertificationError(
                f"result {oid} differs from its checkpoint record")
        summary_fields = (
            "selected_hypothesis", "selected_description", "selected_rule",
            "support_errors", "loo_errors", "rotated_loo_errors",
            "rotated_loo_checks", "n_examples", "complexity", "rounds_used",
            "proposer_kind", "proposer_error",
        )
        for name in summary_fields:
            if canonical_json_digest([result.get(name)]) != \
                    canonical_json_digest([record.get(name)]):
                raise ReplayCertificationError(
                    f"result {oid}.{name} differs from canonical replay")
        if "category" in result and result["category"] != manifest_entry["category"]:
            raise ReplayCertificationError(
                f"result {oid} category differs from the manifest")
        records_by_oid[oid] = record
        terminal_replays[oid] = terminal_replay
        if record_solved:
            solved_oids.add(oid)
    if solved != len(solved_oids):
        raise ReplayCertificationError(
            "checkpoint solved count does not reproduce from its records")
    return {
        "condition": condition,
        "control_digest": expected_control_digest,
        "records_by_oid": records_by_oid,
        "terminal_replays": terminal_replays,
        "solved_oids": solved_oids,
    }


def _artifact_binding(payload: dict, corpus_manifest: dict,
                      corpus_bundle: dict,
                      control_manifest: dict | None) -> dict:
    return {
        "schema": "bongard.semantic-artifact-binding/v1",
        "track": "SEMANTIC-PURE",
        "condition": payload["condition"],
        "sharing_policy": payload["sharing_policy"],
        "corpus_digest": corpus_manifest["corpus_digest"],
        "corpus_bundle_digest": corpus_bundle["bundle_digest"],
        "control_digest": (
            control_manifest["control_digest"]
            if control_manifest is not None else ""),
    }


def _check_artifact_binding(art: str, expected: dict) -> bool:
    """Return whether an exact write-once binding already exists."""
    from phase_d_protocol import (
        validate_corpus_bundle,
        validate_corpus_manifest,
        validate_shuffled_control_manifest,
    )

    binding_path = os.path.join(art, "artifact_binding.json")
    if not os.path.exists(art):
        return False
    if not os.path.isdir(art):
        raise ReplayCertificationError("semantic artifact path is not a directory")
    if not os.path.exists(binding_path):
        unexpected = set(os.listdir(art)) - {"wip_context"}
        if unexpected:
            raise ReplayCertificationError(
                "existing semantic artifact lacks a write-once arm binding")
        return False
    try:
        with open(binding_path, encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayCertificationError(
            f"semantic artifact arm binding is invalid: {exc}") from exc
    if observed != expected:
        raise ReplayCertificationError(
            "semantic artifact tag is already bound to a different arm")

    manifest_path = os.path.join(art, "corpus_manifest.json")
    bundle_path = os.path.join(art, "corpus_panels.json")
    control_path = os.path.join(art, "control_manifest.json")
    try:
        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            validate_corpus_manifest(manifest)
            if manifest.get("corpus_digest") != expected["corpus_digest"]:
                raise ReplayCertificationError(
                    "artifact corpus manifest contradicts its arm binding")
        if os.path.exists(bundle_path):
            if manifest is None:
                raise ReplayCertificationError(
                    "artifact corpus bundle lacks its bound manifest")
            with open(bundle_path, encoding="utf-8") as handle:
                bundle = json.load(handle)
            validate_corpus_bundle(bundle, manifest)
            if bundle.get("bundle_digest") != expected["corpus_bundle_digest"]:
                raise ReplayCertificationError(
                    "artifact corpus bundle contradicts its arm binding")
        if os.path.exists(control_path):
            if manifest is None:
                raise ReplayCertificationError(
                    "artifact control manifest lacks its bound corpus")
            with open(control_path, encoding="utf-8") as handle:
                control = json.load(handle)
            validate_shuffled_control_manifest(control, manifest)
            if not expected["control_digest"] \
                    or control.get("control_digest") != expected["control_digest"]:
                raise ReplayCertificationError(
                    "artifact control manifest contradicts its arm binding")
    except ReplayCertificationError:
        raise
    except Exception as exc:
        raise ReplayCertificationError(
            f"artifact protocol evidence is invalid: {exc}") from exc
    return True


def assert_artifact_binding_compatible(
        tag: str, payload: dict, corpus_manifest: dict,
        corpus_bundle: dict,
        control_manifest: dict | None = None) -> str:
    """Read-only preflight for the destination artifact arm binding."""
    art = artifact_dir(tag)
    expected = _artifact_binding(
        payload, corpus_manifest, corpus_bundle, control_manifest)
    _check_artifact_binding(art, expected)
    return art


def _commit_artifact_binding(art: str, expected: dict) -> None:
    os.makedirs(art, exist_ok=True)
    if not _check_artifact_binding(art, expected):
        _atomic_json(os.path.join(art, "artifact_binding.json"), expected)


def _write_protocol_evidence(art: str, corpus_manifest: dict,
                             corpus_bundle: dict,
                             control_manifest: dict | None) -> None:
    _atomic_json(os.path.join(art, "corpus_manifest.json"), corpus_manifest)
    _atomic_json(os.path.join(art, "corpus_panels.json"), corpus_bundle)
    if control_manifest is not None:
        _atomic_json(os.path.join(art, "control_manifest.json"), control_manifest)


def _prune_json_directory(path: str, keep_oids: set[str]) -> None:
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.endswith(".json") and name[:-5] not in keep_oids:
            os.unlink(os.path.join(path, name))


def _cold_replay_specs(out_dir: str, promoted_cones: list[dict],
                       checkpoint_payload: dict, *,
                       corpus_manifest: dict | None = None,
                       corpus_bundle: dict | None = None,
                       control_manifest: dict | None = None) -> list[dict]:
    """Replay every promoted cone in a fresh interpreter and issue receipts."""
    from phase_d_protocol import (
        COUNT_POLICY,
        OBSERVED,
        PhaseDProtocolError,
        SHARED,
        SHUFFLED_SIDES,
        validate_execution_binding,
        validate_corpus_bundle,
        validate_corpus_manifest,
        validate_shuffled_control_manifest,
    )
    from semantic_replay import (
        canonical_json_digest,
        load_runspec,
        semantic_cone_digest,
    )

    if not isinstance(checkpoint_payload, dict):
        raise ReplayCertificationError("checkpoint payload must be an object")
    condition = checkpoint_payload.get("condition")
    try:
        if corpus_manifest is None:
            with open(os.path.join(out_dir, "corpus_manifest.json"),
                      encoding="utf-8") as handle:
                corpus_manifest = json.load(handle)
        validate_corpus_manifest(corpus_manifest)
        if corpus_bundle is None:
            with open(os.path.join(out_dir, "corpus_panels.json"),
                      encoding="utf-8") as handle:
                corpus_bundle = json.load(handle)
        validate_corpus_bundle(corpus_bundle, corpus_manifest)
        if condition == SHUFFLED_SIDES:
            if control_manifest is None:
                with open(os.path.join(out_dir, "control_manifest.json"),
                          encoding="utf-8") as handle:
                    control_manifest = json.load(handle)
            validate_shuffled_control_manifest(
                control_manifest, corpus_manifest)
        elif condition != OBSERVED or control_manifest is not None:
            raise ReplayCertificationError(
                "cold replay condition/control evidence is inconsistent")
    except ReplayCertificationError:
        raise
    except Exception as exc:
        raise ReplayCertificationError(
            f"cold replay requires valid corpus/control evidence: {exc}") from exc
    assert corpus_manifest is not None and corpus_bundle is not None

    checkpoint_dataset = checkpoint_payload.get("dataset")
    if not isinstance(checkpoint_dataset, dict) \
            or checkpoint_dataset.get("corpus_digest") != corpus_manifest[
                "corpus_digest"] \
            or checkpoint_dataset.get("corpus_bundle_digest") != corpus_bundle[
                "bundle_digest"]:
        raise ReplayCertificationError(
            "checkpoint corpus/bundle identity differs from frozen evidence")
    expected_control_digest = (
        control_manifest["control_digest"]
        if control_manifest is not None else "")
    checkpoint_control = checkpoint_payload.get("control")
    checkpoint_control_digest = (
        checkpoint_control.get("control_digest")
        if isinstance(checkpoint_control, dict) else "")
    if checkpoint_control_digest != expected_control_digest \
            or (condition == OBSERVED and checkpoint_control is not None):
        raise ReplayCertificationError(
            "checkpoint control identity differs from frozen evidence")
    expected_entries = {
        entry["opaque_id"]: entry for entry in corpus_manifest["problems"]}
    expected_control_entries = (
        {entry["opaque_id"]: entry
         for entry in control_manifest["problems"]}
        if control_manifest is not None else {})

    by_oid: dict[str, dict] = {}
    if not isinstance(promoted_cones, list):
        raise ReplayCertificationError("promoted cones must be a list")
    for promoted in promoted_cones:
        if not isinstance(promoted, dict):
            raise ReplayCertificationError("promoted cones must be objects")
        oid = str(promoted.get("opaque_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", oid) \
                or oid in by_oid:
            raise ReplayCertificationError(
                "promoted cones must have unique non-empty opaque_id values")
        by_oid[oid] = promoted
    if not by_oid:
        raise ReplayCertificationError(
            "promotion requires at least one replay-certified cone")

    checkpoint_records = checkpoint_payload.get("records") \
        if isinstance(checkpoint_payload, dict) else None
    if not isinstance(checkpoint_records, list):
        raise ReplayCertificationError(
            "checkpoint payload must contain a record list")
    checkpoint_by_oid: dict[str, dict] = {}
    checkpoint_index_by_oid: dict[str, int] = {}
    for record_index, record in enumerate(checkpoint_records):
        if not isinstance(record, dict):
            raise ReplayCertificationError(
                "checkpoint records must be objects")
        record_oid = str(record.get("opaque_id", ""))
        if not record_oid or record_oid in checkpoint_by_oid:
            raise ReplayCertificationError(
                "checkpoint records must have unique non-empty opaque IDs")
        checkpoint_by_oid[record_oid] = record
        checkpoint_index_by_oid[record_oid] = record_index

    phase_binding = checkpoint_payload.get("phase_execution_binding", {})
    phase_binding_history = checkpoint_payload.get(
        "phase_execution_binding_history", [])
    phase_hash_runtime = checkpoint_payload.get(
        "phase_python_hash_runtime", {})
    if phase_binding:
        if not isinstance(phase_binding_history, list) \
                or not phase_binding_history \
                or phase_binding_history[-1] != phase_binding:
            raise ReplayCertificationError(
                "checkpoint Phase execution binding history is inconsistent")
        try:
            for binding in phase_binding_history:
                validate_execution_binding(binding)
        except PhaseDProtocolError as exc:
            raise ReplayCertificationError(
                f"checkpoint Phase execution binding is invalid: {exc}") \
                from exc
        family_fields = (
            "preregistration_digest", "execution_policy_digest", "track",
            "condition", "execution_tag",
        )
        if any(
                binding.get("track") != "SEMANTIC-PURE"
                or binding.get("execution_tag") != checkpoint_payload.get("tag")
                or any(binding.get(name) != phase_binding.get(name)
                       for name in family_fields)
                for binding in phase_binding_history) \
                or [binding["scale"] for binding in phase_binding_history] != \
                sorted({binding["scale"] for binding in phase_binding_history}) \
                or len(checkpoint_records) > phase_binding["scale"]:
            raise ReplayCertificationError(
                "checkpoint Phase execution binding family is inconsistent")
        if not isinstance(phase_hash_runtime, dict) \
                or set(phase_hash_runtime) != {
                    "python_hash_seed_env", "python_hash_probes"} \
                or not isinstance(
                    phase_hash_runtime["python_hash_seed_env"], str) \
                or not isinstance(
                    phase_hash_runtime["python_hash_probes"], list) \
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in phase_hash_runtime["python_hash_probes"]):
            raise ReplayCertificationError(
                "checkpoint Phase Python hash runtime is malformed")
    elif phase_binding != {} or phase_binding_history != []:
        raise ReplayCertificationError(
            "unpreregistered checkpoint carries Phase execution provenance")
    elif phase_hash_runtime != {}:
        raise ReplayCertificationError(
            "unpreregistered checkpoint carries Phase hash provenance")

    specs_dir = os.path.join(out_dir, "replay_specs")
    receipts_dir = os.path.join(out_dir, "replay_receipts")
    receipts: list[dict] = []
    for oid, promoted in sorted(by_oid.items()):
        checkpoint = checkpoint_by_oid.get(oid)
        if checkpoint is None or checkpoint.get("solved") is not True:
            raise ReplayCertificationError(
                f"promoted cone {oid} lacks a solved checkpoint record")
        manifest_entry = expected_entries.get(oid)
        control_entry = expected_control_entries.get(oid)
        if manifest_entry is None \
                or (condition == SHUFFLED_SIDES and control_entry is None):
            raise ReplayCertificationError(
                f"promoted cone {oid} is outside the frozen corpus")
        expected_panel_digest = (
            control_entry["controlled_panel_set_digest"]
            if control_entry is not None else manifest_entry["panel_set_digest"])
        path = os.path.join(specs_dir, f"{oid}.json")
        try:
            spec = load_runspec(path)
        except Exception as exc:
            raise ReplayCertificationError(
                f"cannot load replay spec for {oid}: {exc}") from exc
        if spec.problem.get("opaque_id") != oid:
            raise ReplayCertificationError(
                f"replay spec opaque_id mismatch for {oid}")
        if promoted.get("runspec_digest") != spec.spec_digest:
            raise ReplayCertificationError(
                f"promoted runspec_digest differs from replay spec for {oid}")
        if checkpoint.get("replay_spec_digest") != spec.spec_digest:
            raise ReplayCertificationError(
                f"checkpoint runspec_digest differs from replay spec for {oid}")
        provenance = dict(spec.provenance)
        dataset_provenance = provenance.get("dataset")
        experiment_provenance = provenance.get("experiment")
        sampling = corpus_manifest["sampling"]
        expected_dataset_provenance = {
            "source": sampling["source"],
            "seed": sampling["seed"],
            "limit_per_source": sampling["limit_per_source"],
            "count_policy": COUNT_POLICY,
            "order_policy": sampling["order_policy"],
            "repository_commit": sampling["dataset_revision"],
            "corpus_digest": corpus_manifest["corpus_digest"],
            "corpus_bundle_digest": corpus_bundle["bundle_digest"],
            "panel_set_digest": expected_panel_digest,
            "panels": "self-contained; source identifier redacted",
        }
        if dataset_provenance != expected_dataset_provenance \
                or spec.panel_set_digest != expected_panel_digest \
                or checkpoint.get("corpus_digest") != corpus_manifest[
                    "corpus_digest"] \
                or checkpoint.get("panel_set_digest") != expected_panel_digest:
            raise ReplayCertificationError(
                f"checkpoint corpus/panel identity differs from replay spec for {oid}")
        if not isinstance(experiment_provenance, dict) \
                or experiment_provenance.get("track") != "SEMANTIC-PURE" \
                or experiment_provenance.get("condition") != condition \
                or experiment_provenance.get("sharing_policy") != SHARED \
                or checkpoint.get("track") != "SEMANTIC-PURE" \
                or checkpoint.get("condition") != condition \
                or checkpoint.get("sharing_policy") != SHARED:
            raise ReplayCertificationError(
                f"checkpoint experiment arm differs from replay spec for {oid}")
        if provenance.get("proposer") != {
                "kind": checkpoint_payload.get("proposer"),
                "model": checkpoint_payload.get("model"),
                "round_limit": checkpoint_payload.get("rounds"),
                }:
            raise ReplayCertificationError(
                f"checkpoint proposer policy differs from replay spec for {oid}")
        if provenance.get("python_hash_runtime", {}) != phase_hash_runtime:
            raise ReplayCertificationError(
                f"checkpoint Python hash runtime differs from replay spec for {oid}")
        record_index = checkpoint_index_by_oid[oid]
        expected_phase_binding = (
            next((binding for binding in phase_binding_history
                  if record_index < binding["scale"]), None)
            if phase_binding else {})
        if expected_phase_binding is None \
                or checkpoint.get(
                    "phase_execution_binding_digest", "") != \
                expected_phase_binding.get("binding_digest", "") \
                or experiment_provenance.get(
                    "phase_execution_binding", {}) != expected_phase_binding:
            raise ReplayCertificationError(
                f"checkpoint Phase execution tranche differs from replay "
                f"spec for {oid}")
        replay_control = experiment_provenance.get("control")
        expected_replay_control = (
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
            if control_manifest is not None else None)
        if canonical_json_digest(replay_control) != canonical_json_digest(
                expected_replay_control) \
                or checkpoint.get("control_digest", "") != \
                expected_control_digest:
            raise ReplayCertificationError(
                f"checkpoint control identity differs from replay spec for {oid}")
        policy = dict(spec.verifier).get("policy", {})
        if policy.get("acceptance_mode") != "exact":
            raise ReplayCertificationError(
                f"promotion requires an exact verifier policy for {oid}")
        selection_evidence = dict(spec.provenance).get("selection")
        if not isinstance(selection_evidence, dict) \
                or not selection_evidence.get("candidates") \
                or not selection_evidence.get("selected_candidate_id") \
                or not isinstance(selection_evidence.get("selected_record"),
                                  dict):
            raise ReplayCertificationError(
                f"promotion requires replayable selection evidence for {oid}")
        checkpoint_terminal = checkpoint.get("terminal_evidence")
        checkpoint_terminal_digest = checkpoint.get(
            "terminal_evidence_digest")
        if not isinstance(checkpoint_terminal, dict) \
                or checkpoint_terminal.get("schema") != \
                "bongard.semantic-terminal-evidence/v1" \
                or canonical_json_digest(checkpoint_terminal) != \
                checkpoint_terminal_digest \
                or canonical_json_digest(
                    checkpoint_terminal.get("selection")) != \
                canonical_json_digest(selection_evidence):
            raise ReplayCertificationError(
                f"checkpoint terminal evidence differs from selection for {oid}")
        terminal_provenance = dict(spec.provenance).get("terminal")
        expected_terminal_provenance = {
            "schema": checkpoint_terminal["schema"],
            "proposal_outcome": checkpoint_terminal.get("proposal_outcome"),
            "rounds": checkpoint_terminal.get("rounds"),
            "evidence_digest": checkpoint_terminal_digest,
        }
        if canonical_json_digest(terminal_provenance) != \
                canonical_json_digest(expected_terminal_provenance) \
                or checkpoint.get("rounds_used") != len(
                    checkpoint_terminal.get("rounds", ())) \
                or promoted.get("rounds_used") != checkpoint.get("rounds_used"):
            raise ReplayCertificationError(
                f"terminal round/proposer evidence differs from replay spec "
                f"for {oid}")
        promoted_selection = promoted.get("selection")
        if not isinstance(promoted_selection, dict) \
                or promoted_selection.get("candidate_id") != \
                selection_evidence.get("selected_candidate_id") \
                or canonical_json_digest(promoted_selection) != \
                canonical_json_digest(selection_evidence["selected_record"]):
            raise ReplayCertificationError(
                f"promoted selection differs from replay spec for {oid}")
        if canonical_json_digest(checkpoint.get("selection")) != \
                canonical_json_digest(promoted_selection):
            raise ReplayCertificationError(
                f"checkpoint selection differs from promoted selection for {oid}")
        candidate_verdicts = [
            candidate.get("expected_verification")
            for candidate in selection_evidence["candidates"]
            if isinstance(candidate, dict)
        ]
        if len(candidate_verdicts) != len(selection_evidence["candidates"]) \
                or canonical_json_digest(checkpoint.get("candidates")) != \
                canonical_json_digest(candidate_verdicts):
            raise ReplayCertificationError(
                f"checkpoint candidates differ from replay spec for {oid}")
        candidate_manifest = selection_evidence.get("candidate_manifest")
        if not isinstance(candidate_manifest, list) \
                or len(candidate_manifest) != len(candidate_verdicts) \
                or canonical_json_digest(
                    checkpoint.get("candidate_manifest")) != \
                canonical_json_digest(candidate_manifest):
            raise ReplayCertificationError(
                f"checkpoint candidate manifest differs from replay spec "
                f"for {oid}")
        if len(spec.cones) != 1:
            raise ReplayCertificationError(
                f"replay spec for {oid} must contain exactly one cone")
        record = spec.cones[0]
        if semantic_cone_digest(promoted.get("hypothesis", {})) \
                != record.cone_digest:
            raise ReplayCertificationError(
                f"promoted cone payload differs from replay spec for {oid}")
        if checkpoint.get("selected_hypothesis") != record.cone_id:
            raise ReplayCertificationError(
                f"checkpoint winner differs from replay spec for {oid}")
        expected = record.expected_verification
        if expected is None or canonical_json_digest(expected) != \
                canonical_json_digest(promoted.get("verification", {})):
            raise ReplayCertificationError(
                f"promoted verification differs from replay spec for {oid}")
        exact_zero_fields = (
            "support_errors", "loo_errors", "rotated_loo_errors",
            "predicate_errors", "indeterminate_evaluations",
            "naturality_errors", "cofibration_errors",
        )
        if not expected.get("accepted") \
                or not expected.get("semantic_admissible") \
                or any(expected.get(name) != 0 for name in exact_zero_fields) \
                or expected.get("unchecked_morphisms") \
                or expected.get("compile_error") \
                or expected.get("semantic_issue"):
            raise ReplayCertificationError(
                f"promoted verification is not exact and fully admissible "
                f"for {oid}")

        replay_env = os.environ.copy()
        replay_seed = phase_hash_runtime.get("python_hash_seed_env") \
            if phase_hash_runtime else None
        if replay_seed and replay_seed != "random":
            replay_env["PYTHONHASHSEED"] = replay_seed
        replay_env["PYTHONPATH"] = os.pathsep.join(
            path for path in sys.path if path and os.path.isabs(path))
        proc = subprocess.run(
            [sys.executable, REPLAY_SCRIPT, path],
            cwd=CODE_LAB_DIR,
            env=replay_env,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise ReplayCertificationError(
                f"cold replay failed for {oid} (exit {proc.returncode}): {detail}")
        try:
            receipt = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ReplayCertificationError(
                f"cold replay emitted invalid receipt for {oid}") from exc
        if receipt.get("status") != "PASS" \
                or receipt.get("process_mode") != "fresh_python_subprocess" \
                or receipt.get("spec_digest") != spec.spec_digest:
            raise ReplayCertificationError(
                f"cold replay receipt did not certify {oid}")
        verdicts = receipt.get("verdicts")
        expected_verification_digest = canonical_json_digest(expected)
        if not isinstance(verdicts, list) or len(verdicts) != 1 \
                or verdicts[0].get("cone_id") != record.cone_id \
                or verdicts[0].get("accepted") is not True \
                or verdicts[0].get("verification_digest") \
                != expected_verification_digest:
            raise ReplayCertificationError(
                f"cold replay receipt has the wrong cone verdict for {oid}")
        replayed_selection = receipt.get("selection")
        if not isinstance(replayed_selection, dict) \
                or replayed_selection.get("selected_candidate_id") != \
                selection_evidence.get("selected_candidate_id") \
                or replayed_selection.get("candidate_count") != len(
                    selection_evidence["candidates"]) \
                or replayed_selection.get("candidate_manifest_digest") != \
                canonical_json_digest(candidate_manifest) \
                or replayed_selection.get("evidence_digest") != \
                canonical_json_digest(selection_evidence) \
                or replayed_selection.get("selected_record_digest") != \
                canonical_json_digest(selection_evidence["selected_record"]):
            raise ReplayCertificationError(
                f"cold replay receipt did not reproduce selection for {oid}")
        receipt["opaque_id"] = oid
        _atomic_json(os.path.join(receipts_dir, f"{oid}.json"), receipt)
        receipts.append(receipt)
    return receipts


def promote(tag: str, out_dir: str, payload: dict, results: dict,
            promoted_cones: list[dict],
            control_manifest: dict | None = None) -> str:
    assert_not_tainted(out_dir)
    corpus_path = os.path.join(out_dir, "corpus_manifest.json")
    try:
        with open(corpus_path, encoding="utf-8") as handle:
            corpus_manifest = json.load(handle)
    except Exception as exc:
        raise ReplayCertificationError(
            f"promotion cannot read its frozen corpus manifest: {exc}") from exc
    bundle_path = os.path.join(out_dir, "corpus_panels.json")
    try:
        with open(bundle_path, encoding="utf-8") as handle:
            corpus_bundle = json.load(handle)
    except Exception as exc:
        raise ReplayCertificationError(
            f"promotion cannot read its embedded corpus bundle: {exc}") from exc
    validated = _validate_run_inputs(
        payload, results, corpus_manifest, corpus_bundle, control_manifest)
    if not isinstance(promoted_cones, list) \
            or any(not isinstance(cone, dict) for cone in promoted_cones):
        raise ReplayCertificationError("promoted cones must be a list of objects")
    cone_oids = [str(cone.get("opaque_id", "")) for cone in promoted_cones]
    if validated["solved_oids"] != set(cone_oids) \
            or len(cone_oids) != len(set(cone_oids)):
        raise ReplayCertificationError(
            "solved result IDs must match promoted replay cone IDs")
    art = artifact_dir(tag)
    binding = _artifact_binding(
        payload, corpus_manifest, corpus_bundle, control_manifest)
    _check_artifact_binding(art, binding)
    receipts = _cold_replay_specs(
        out_dir, promoted_cones, payload,
        corpus_manifest=corpus_manifest,
        corpus_bundle=corpus_bundle,
        control_manifest=control_manifest)

    _commit_artifact_binding(art, binding)
    _write_protocol_evidence(
        art, corpus_manifest, corpus_bundle, control_manifest)
    cone_oid_set = set(cone_oids)
    for dirname in ("replay_specs", "replay_receipts"):
        source = os.path.join(out_dir, dirname)
        destination = os.path.join(art, dirname)
        os.makedirs(destination, exist_ok=True)
        for oid in sorted(cone_oid_set):
            _atomic_copy(
                os.path.join(source, f"{oid}.json"),
                os.path.join(destination, f"{oid}.json"),
            )
        _prune_json_directory(destination, cone_oid_set)

    # Ground truth stays harness-side: concept names exist only here.
    _atomic_json(os.path.join(art, "results.json"), results)
    _atomic_json(os.path.join(art, "promoted_cones.json"), promoted_cones)
    promoted_payload = dict(payload)
    promoted_payload["artifact_state"] = "PROMOTED"
    solved = len(validated["solved_oids"])
    recorded_hash_seed = str(
        payload.get("phase_python_hash_runtime", {}).get(
            "python_hash_seed_env", "random"))
    lines = [
        f"# Semantic artifact `{tag}`",
        "",
        f"solved {solved}/{len(results)} (semantic-pure typed cones)",
        "",
        f"Cold replay certified {len(receipts)} cone(s) in fresh Python processes.",
        "Each self-contained RunSpec pins panel/cone digests, verifier policy,",
        "registry/source fingerprints, Python, and scientific dependencies.",
        "Re-run from this directory with:",
        f"`PYTHONHASHSEED={recorded_hash_seed} python "
        "../../replay_semantic_runspec.py replay_specs/<opaque_id>.json`",
        "",
        "| opaque_id | solved | status | rule |",
        "|---|---|---|---|",
    ]
    for oid in sorted(results):
        r = results[oid]
        lines.append(
            f"| {oid} | {r.get('solved')} | {r.get('status', '')} | {r.get('rule', '')} |")
    _atomic_text(os.path.join(art, "README.md"), "\n".join(lines) + "\n")
    # The checkpoint is the commit marker and is therefore written last.
    _atomic_json(os.path.join(art, "checkpoint.json"), promoted_payload)
    return art


def publish_run_report(tag: str, payload: dict, results: dict,
                       corpus_manifest: dict,
                       control_manifest: dict | None = None,
                       corpus_bundle: dict | None = None) -> str:
    """Persist the full attempted denominator, including zero-solve controls.

    This is a run report, not cone certification.  Exact solved cones still
    require :func:`promote`; failures and infrastructure statuses are retained
    so a trailing failure cannot disappear from the promoted denominator.
    """
    validated = _validate_run_inputs(
        payload, results, corpus_manifest, corpus_bundle, control_manifest,
        require_complete=True)
    assert corpus_bundle is not None
    art = artifact_dir(tag)
    binding = _artifact_binding(
        payload, corpus_manifest, corpus_bundle, control_manifest)
    _check_artifact_binding(art, binding)
    _commit_artifact_binding(art, binding)
    _write_protocol_evidence(
        art, corpus_manifest, corpus_bundle, control_manifest)

    # A run report does not certify cones.  Clear any certification from an
    # earlier report with the same arm before committing this denominator.
    _atomic_json(os.path.join(art, "promoted_cones.json"), [])
    for dirname in ("replay_specs", "replay_receipts"):
        _prune_json_directory(os.path.join(art, dirname), set())
    _atomic_json(os.path.join(art, "results.json"), results)
    report_payload = dict(payload)
    report_payload["artifact_state"] = "RUN_COMPLETE"
    solved = len(validated["solved_oids"])
    lines = [
        f"# Semantic run report `{tag}`",
        "",
        f"attempted {len(results)}; exact semantic solves {solved}",
        "",
        "This report preserves the complete attempted denominator. Exact solved",
        "cones, when present, are separately cold-replay certified during promotion.",
        "",
        "| opaque_id | solved | status | rule |",
        "|---|---|---|---|",
    ]
    for oid in sorted(results):
        result = results[oid]
        lines.append(
            f"| {oid} | {result.get('solved')} | "
            f"{result.get('status', '')} | {result.get('rule', '')} |")
    _atomic_text(os.path.join(art, "README.md"), "\n".join(lines) + "\n")
    # The checkpoint is the commit marker and is therefore written last.
    _atomic_json(os.path.join(art, "checkpoint.json"), report_payload)
    return art
