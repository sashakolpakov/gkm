"""Sealed one-turn shared-witness nomination on the exposed v10 cohort.

The command opens only the exact twelve released historical panels pinned by
the whole-panel calibration source.  It persists a fresh runtime precommit,
makes one journaled headless Codex call through the structured shared-witness
semantic protocol, and cold-replays the journal, pixels, parser, renderer, and
artifact without transport.  It cannot open query, broad-cohort, or official
test pixels and does not authorize calibration by itself.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard import object_bongard_rubric_nomination_command as _durable
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    ObjectBongardPanelRubricCalibrationSource,
    load_object_bongard_panel_rubric_calibration_source,
    object_bongard_panel_rubric_calibration_source_digest,
)
from bongard.object_bongard_shared_witness import (
    build_shared_witness_rubric_specs,
    object_bongard_shared_witness_source_digest,
)
from bongard.object_bongard_shared_witness_semantics import (
    GROUP_SIZE,
    ObjectBongardSharedWitnessSemanticArtifact,
    describe_object_bongard_shared_witness_support,
    object_bongard_shared_witness_semantics_output_schema,
    object_bongard_shared_witness_semantics_prompt,
    object_bongard_shared_witness_semantics_protocol_digest,
    object_bongard_shared_witness_semantics_source_digest,
    verify_object_bongard_shared_witness_semantic_artifact,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.prototype_object_scene_observer import (
    PrototypeSceneObserverStatus,
    prototype_scene_transport_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


AUTHORIZATION_SCHEMA = "gkm.bongard-shared-witness-nomination-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-shared-witness-nomination-precommit.v1"
REPLAY_SCHEMA = "gkm.bongard-shared-witness-nomination-cold-replay.v1"
RESULT_SCHEMA = "gkm.bongard-shared-witness-nomination-result.v1"
COMMAND_ID = "bongard.shared-witness-nomination/exposed-twelve-one-turn-v1"

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
ARTIFACT_FILENAME = "semantic_artifact.json"
REPLAY_FILENAME = "cold_replay.json"
RESULT_FILENAME = "result.json"
JOURNAL_DIRECTORY = "journals"

NOMINATION_MODEL = "gpt-5.6-sol"
NOMINATION_REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardSharedWitnessNominationCommandError(RuntimeError):
    """The nomination boundary, runtime seal, or cold replay differs."""


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardSharedWitnessNomination:
    artifact: ObjectBongardSharedWitnessSemanticArtifact
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    result_digest: str
    source_digest: str
    accepted: bool
    output_root: Path

    def __post_init__(self) -> None:
        for name in (
            "authorization_digest",
            "execution_precommit_digest",
            "cold_replay_digest",
            "result_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
                raise ObjectBongardSharedWitnessNominationCommandError(
                    f"verified {name} differs"
                )
        if (
            not isinstance(self.source_digest, str)
            or _RAW_DIGEST.fullmatch(self.source_digest) is None
            or self.accepted
            is not (self.artifact.status is PrototypeSceneObserverStatus.SUCCESS)
        ):
            raise ObjectBongardSharedWitnessNominationCommandError(
                "verified nomination summary differs"
            )

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-shared-witness-nomination-summary.v1",
            "output_root": str(self.output_root),
            "authorization_digest": self.authorization_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "semantic_artifact_digest": self.artifact.artifact_digest,
            "cold_replay_digest": self.cold_replay_digest,
            "result_digest": self.result_digest,
            "source_digest": self.source_digest,
            "accepted": self.accepted,
            "candidate_count": len(self.artifact.contrast_candidates),
            **_authority_data(),
        }


def object_bongard_shared_witness_nomination_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "structured_shared_witness_proposer": True,
        "model_emits_free_form_group_cue_pairs": False,
        "python_renders_descriptions": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "support_roles_visible_to_model": False,
        "historical_exposed_panel_count": 12,
        "physical_model_call_count": 1,
        "query_pixels_used": False,
        "fresh_broad_cohort_pixels_used": False,
        "official_test_pixels_used": False,
        "calibration_authorized_by_this_command": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _fresh_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or os.path.lexists(root):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination output root must be fresh"
        )
    root.mkdir(mode=0o700)
    _durable._fsync_directory(parent)
    return root


def _existing_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination root cannot be a symlink"
        )
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination root is not a directory"
        )
    return root


def _source_identities() -> list[dict[str, str]]:
    rows = {
        "command_source_sha256": (
            object_bongard_shared_witness_nomination_command_source_digest()
        ),
        "historical_panel_source_sha256": (
            object_bongard_panel_rubric_calibration_source_digest()
        ),
        "shared_witness_ir_source_sha256": (
            object_bongard_shared_witness_source_digest()
        ),
        "shared_witness_semantics_source_sha256": (
            object_bongard_shared_witness_semantics_source_digest()
        ),
        "turn_journal_source_sha256": object_bongard_turn_journal_source_digest(),
        "transport_source_sha256": prototype_scene_transport_source_digest(),
        "durable_record_helper_source_sha256": (
            _durable.object_bongard_rubric_nomination_command_source_digest()
        ),
    }
    return [{"role": key, "sha256": rows[key]} for key in sorted(rows)]


def _groups(
    source: ObjectBongardPanelRubricCalibrationSource,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(source, ObjectBongardPanelRubricCalibrationSource):
        raise TypeError("source must be the exact typed historical panel source")
    groups = (
        tuple(sorted(item.panel_id for item in source.group_0_panels)),
        tuple(sorted(item.panel_id for item in source.group_1_panels)),
    )
    if (
        tuple(len(group) for group in groups) != (GROUP_SIZE, GROUP_SIZE)
        or set(groups[0]) & set(groups[1])
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "historical neutral group inventory differs"
        )
    return groups


def _support(
    source: ObjectBongardPanelRubricCalibrationSource,
) -> dict[str, bytes]:
    expected = set().union(*_groups(source))
    support = {
        item.panel_id: item.exact_png_bytes
        for item in source.panels
        if item.panel_id in expected
    }
    if set(support) != expected:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "historical support PNG inventory differs"
        )
    return support


def _images(
    source: ObjectBongardPanelRubricCalibrationSource,
) -> tuple[tuple[str, bytes], ...]:
    support = _support(source)
    return tuple(
        (f"group_{group_index}_ref_{index:02d}.png", support[panel_id])
        for group_index, group in enumerate(_groups(source))
        for index, panel_id in enumerate(group)
    )


def _panel_commitments(
    source: ObjectBongardPanelRubricCalibrationSource,
) -> list[dict[str, object]]:
    return [
        {
            "neutral_group_name": f"group_{group_index}",
            "panel_ids": list(group),
            "png_sha256": [
                next(
                    item.png_sha256
                    for item in source.panels
                    if item.panel_id == panel_id
                )
                for panel_id in group
            ],
            "released_record_digests": [
                next(
                    item.released_record_digest
                    for item in source.panels
                    if item.panel_id == panel_id
                )
                for panel_id in group
            ],
        }
        for group_index, group in enumerate(_groups(source))
    ]


def _authorization(
    source: ObjectBongardPanelRubricCalibrationSource,
    *,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    if (
        isinstance(minutes, bool)
        or not isinstance(minutes, int)
        or not 1 <= minutes <= 120
        or not isinstance(executable, str)
        or not executable
        or not isinstance(expected_launcher_sha256, str)
        or _RAW_DIGEST.fullmatch(expected_launcher_sha256) is None
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination runtime selectors are invalid"
        )
    return _durable._record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_id": COMMAND_ID,
            "historical_source_digest": source.source_digest,
            "historical_source_plan_file_sha256": source.historical_plan_file_sha256,
            "historical_source_plan_record_digest": (
                source.historical_plan_record_digest
            ),
            "context_task_id": source.panels[0].task_id,
            "context_task_id_is_transport_metadata_only": True,
            "neutral_groups": _panel_commitments(source),
            "semantic_protocol_digest": (
                object_bongard_shared_witness_semantics_protocol_digest()
            ),
            "semantic_output_schema_digest": canonical_digest(
                object_bongard_shared_witness_semantics_output_schema()
            ),
            "source_identities": _source_identities(),
            "runtime_policy": {
                "model": NOMINATION_MODEL,
                "reasoning_effort": NOMINATION_REASONING_EFFORT,
                "minutes": minutes,
                "verbose": False,
                "executable": executable,
                "expected_launcher_sha256": expected_launcher_sha256,
            },
            "exact_historical_released_panels_only": True,
            "authorization_and_precommit_fsynced_before_inference": True,
            **_authority_data(),
        },
        "authorization_digest",
    )


def _create_runtime(
    authorization: Mapping[str, Any],
    *,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> tuple[ObjectBongardTurnRuntime, Mapping[str, str]]:
    policy = authorization["runtime_policy"]
    cache = cache_snapshotter()
    catalog = catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(
        catalog, CodexModelCatalogSnapshot
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "runtime snapshotter returned the wrong type"
        )
    fingerprint = launcher_fingerprinter(
        policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
    )
    attestation = runtime_attester(
        executable=policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "runtime attester returned the wrong type"
        )
    runtime = ObjectBongardTurnRuntime(
        model=policy["model"],
        reasoning_effort=policy["reasoning_effort"],
        minutes=policy["minutes"],
        verbose=policy["verbose"],
        executable=policy["executable"],
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=policy["expected_launcher_sha256"],
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    return runtime, fingerprint


def _precommit(
    authorization: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    fingerprint: Mapping[str, str],
) -> dict[str, Any]:
    return _durable._record(
        {
            "schema": PRECOMMIT_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "historical_source_digest": authorization[
                "historical_source_digest"
            ],
            "context_task_id": authorization["context_task_id"],
            "neutral_groups": authorization["neutral_groups"],
            "semantic_protocol_digest": authorization[
                "semantic_protocol_digest"
            ],
            "semantic_output_schema_digest": authorization[
                "semantic_output_schema_digest"
            ],
            "source_identities": authorization["source_identities"],
            "runtime_binding": runtime.binding,
            "cloud_policy_cache_snapshot_base64": _durable._encode_bytes(
                runtime.cloud_policy_cache_snapshot.data
                if runtime.cloud_policy_cache_snapshot is not None
                else None
            ),
            "model_catalog_snapshot_base64": _durable._encode_bytes(
                runtime.model_catalog_snapshot.data
            ),
            "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
            "launcher_fingerprint": dict(fingerprint),
            "precommit_fsynced_before_inference": True,
            **_authority_data(),
        },
        "precommit_digest",
    )


def _runtime_from_precommit(
    precommit: Mapping[str, Any], authorization: Mapping[str, Any]
) -> ObjectBongardTurnRuntime:
    raw = _durable._validate_record(
        precommit,
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="shared-witness nomination precommit",
    )
    expected_fields = {
        "schema",
        "command_id",
        "authorization_digest",
        "historical_source_digest",
        "context_task_id",
        "neutral_groups",
        "semantic_protocol_digest",
        "semantic_output_schema_digest",
        "source_identities",
        "runtime_binding",
        "cloud_policy_cache_snapshot_base64",
        "model_catalog_snapshot_base64",
        "no_tools_attestation",
        "launcher_fingerprint",
        "precommit_fsynced_before_inference",
        *_authority_data(),
        "precommit_digest",
    }
    if (
        set(raw) != expected_fields
        or raw["command_id"] != COMMAND_ID
        or raw["authorization_digest"] != authorization["authorization_digest"]
        or raw["historical_source_digest"]
        != authorization["historical_source_digest"]
        or raw["context_task_id"] != authorization["context_task_id"]
        or raw["neutral_groups"] != authorization["neutral_groups"]
        or raw["semantic_protocol_digest"]
        != object_bongard_shared_witness_semantics_protocol_digest()
        or raw["semantic_output_schema_digest"]
        != canonical_digest(object_bongard_shared_witness_semantics_output_schema())
        or raw["source_identities"] != _source_identities()
        or raw["precommit_fsynced_before_inference"] is not True
        or any(raw[key] != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "shared-witness runtime precommit differs"
        )
    policy = authorization["runtime_policy"]
    if raw["launcher_fingerprint"] != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": policy["expected_launcher_sha256"],
    }:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "launcher fingerprint differs"
        )
    binding = raw["runtime_binding"]
    cache = CloudPolicyCacheSnapshot(
        _durable._decode_bytes(
            raw["cloud_policy_cache_snapshot_base64"], "policy cache"
        )
    )
    catalog_bytes = _durable._decode_bytes(
        raw["model_catalog_snapshot_base64"], "model catalog"
    )
    if catalog_bytes is None:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "model catalog snapshot is absent"
        )
    runtime = ObjectBongardTurnRuntime(
        model=binding["model"],
        reasoning_effort=binding["reasoning_effort"],
        minutes=binding["minutes"],
        verbose=binding["verbose"],
        executable=binding["executable"],
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=CodexModelCatalogSnapshot(catalog_bytes),
        expected_launcher_digest=binding["expected_launcher_digest"],
        no_tools_attestation=CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        ),
        transport_source_digest=binding["transport_source_digest"],
    )
    if (
        runtime.binding != binding
        or policy
        != {
            "model": runtime.model,
            "reasoning_effort": runtime.reasoning_effort,
            "minutes": runtime.minutes,
            "verbose": runtime.verbose,
            "executable": runtime.executable,
            "expected_launcher_sha256": runtime.expected_launcher_digest,
        }
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "runtime binding differs from authorization"
        )
    return runtime


def _journal(
    root: Path,
    source: ObjectBongardPanelRubricCalibrationSource,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    transport: Callable[..., CodexStructuredResult],
) -> ObjectBongardNamedImageTurnJournalTransport:
    return ObjectBongardNamedImageTurnJournalTransport(
        root / JOURNAL_DIRECTORY / "semantic_nomination",
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id=authorization["context_task_id"],
        turn_kind="shared_witness_nomination",
        expected_prompt=object_bongard_shared_witness_semantics_prompt(),
        expected_images=_images(source),
        expected_output_schema=(
            object_bongard_shared_witness_semantics_output_schema()
        ),
        runtime=runtime,
        underlying_transport=transport,
    )


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold shared-witness replay attempted model transport")


def _contrast_commitments(
    artifact: ObjectBongardSharedWitnessSemanticArtifact,
) -> list[dict[str, object]]:
    if artifact.status is not PrototypeSceneObserverStatus.SUCCESS:
        return []
    specs = build_shared_witness_rubric_specs(
        artifact, expected_artifact_digest=artifact.artifact_digest
    )
    return [
        {
            "candidate_rank": item.candidate_rank,
            "contrast_digest": item.contrast.contrast_digest,
            "rubric_spec_digest": item.spec_digest,
            "shared_anchor": item.contrast.shared_anchor,
            "visual_axis": item.contrast.visual_axis,
            "group_0_endpoint": item.contrast.group_0_endpoint,
            "group_1_endpoint": item.contrast.group_1_endpoint,
        }
        for item in specs
    ]


def _verify_prefix(
    root: Path,
    source: ObjectBongardPanelRubricCalibrationSource,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    ObjectBongardTurnRuntime,
    ObjectBongardSharedWitnessSemanticArtifact,
    Mapping[str, object],
]:
    authorization = _durable._validate_record(
        _durable._read_record(root / AUTHORIZATION_FILENAME, "authorization"),
        schema=AUTHORIZATION_SCHEMA,
        digest_field="authorization_digest",
        label="shared-witness authorization",
    )
    policy = authorization.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "authorization runtime policy is malformed"
        )
    expected_authorization = _authorization(
        source,
        minutes=policy.get("minutes"),
        executable=policy.get("executable"),
        expected_launcher_sha256=policy.get("expected_launcher_sha256"),
    )
    if authorization != expected_authorization:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "authorization differs on replay"
        )
    precommit = _durable._validate_record(
        _durable._read_record(root / PRECOMMIT_FILENAME, "execution precommit"),
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="shared-witness execution precommit",
    )
    runtime = _runtime_from_precommit(precommit, authorization)
    artifact = ObjectBongardSharedWitnessSemanticArtifact.from_data(
        _durable._read_record(root / ARTIFACT_FILENAME, "semantic artifact")
    )
    verify_object_bongard_shared_witness_semantic_artifact(
        artifact,
        support_png_by_panel_id=_support(source),
        expected_task_id=authorization["context_task_id"],
        expected_observation_context_digest=precommit["precommit_digest"],
        expected_artifact_digest=artifact.artifact_digest,
    )
    journal_root = root / JOURNAL_DIRECTORY
    expected_journal = journal_root / "semantic_nomination"
    if (
        not journal_root.is_dir()
        or journal_root.is_symlink()
        or {item.name for item in journal_root.iterdir()} != {"semantic_nomination"}
        or not expected_journal.is_dir()
        or expected_journal.is_symlink()
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "semantic journal inventory differs"
        )
    journal = _journal(
        root, source, authorization, precommit, runtime, _forbidden_transport
    )
    replayed = describe_object_bongard_shared_witness_support(
        task_id=authorization["context_task_id"],
        group_0_panel_ids=_groups(source)[0],
        group_1_panel_ids=_groups(source)[1],
        support_png_by_panel_id=_support(source),
        observation_context_digest=precommit["precommit_digest"],
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=journal,
    )
    if (
        replayed != artifact
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardSharedWitnessNominationCommandError(
            "semantic journal cold replay differs"
        )
    summary = verify_object_bongard_turn_journal(journal).to_data()
    if summary["terminal_status"] not in {"success", "failure"}:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "semantic journal is not terminal"
        )
    return authorization, precommit, runtime, artifact, summary


def _replay_record(
    source: ObjectBongardPanelRubricCalibrationSource,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    artifact: ObjectBongardSharedWitnessSemanticArtifact,
    journal_summary: Mapping[str, object],
) -> dict[str, Any]:
    return _durable._record(
        {
            "schema": REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "historical_source_digest": source.source_digest,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "semantic_artifact_digest": artifact.artifact_digest,
            "semantic_status": artifact.status.value,
            "contrast_commitments": _contrast_commitments(artifact),
            "journal_summary": dict(journal_summary),
            "exact_historical_png_bytes_replayed": True,
            "semantic_parser_and_renderer_replayed": True,
            "model_calls_during_replay": 0,
            "source_identities": _source_identities(),
            **_authority_data(),
        },
        "replay_digest",
    )


def _result_record(
    source: ObjectBongardPanelRubricCalibrationSource,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    artifact: ObjectBongardSharedWitnessSemanticArtifact,
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    return _durable._record(
        {
            "schema": RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "accepted": artifact.status is PrototypeSceneObserverStatus.SUCCESS,
            "historical_source_digest": source.source_digest,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "semantic_artifact_digest": artifact.artifact_digest,
            "cold_replay_digest": replay["replay_digest"],
            "semantic_status": artifact.status.value,
            "contrast_commitments": _contrast_commitments(artifact),
            "model_calls_during_replay": 0,
            **_authority_data(),
        },
        "result_digest",
    )


def _verification(
    root: Path,
    source: ObjectBongardPanelRubricCalibrationSource,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    artifact: ObjectBongardSharedWitnessSemanticArtifact,
    replay: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardSharedWitnessNomination:
    return VerifiedObjectBongardSharedWitnessNomination(
        artifact,
        authorization["authorization_digest"],
        precommit["precommit_digest"],
        replay["replay_digest"],
        result["result_digest"],
        source.source_digest,
        result["accepted"],
        root,
    )


def _verify_loaded(
    output_root: str | os.PathLike[str],
    source: ObjectBongardPanelRubricCalibrationSource,
) -> VerifiedObjectBongardSharedWitnessNomination:
    root = _existing_root(output_root)
    expected = {
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        ARTIFACT_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
        JOURNAL_DIRECTORY,
    }
    if {item.name for item in root.iterdir()} != expected:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination root inventory differs"
        )
    authorization, precommit, _runtime, artifact, summary = _verify_prefix(
        root, source
    )
    expected_replay = _replay_record(
        source, authorization, precommit, artifact, summary
    )
    replay = _durable._validate_record(
        _durable._read_record(root / REPLAY_FILENAME, "cold replay"),
        schema=REPLAY_SCHEMA,
        digest_field="replay_digest",
        label="shared-witness cold replay",
    )
    if replay != expected_replay:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "cold replay record differs"
        )
    expected_result = _result_record(
        source, authorization, precommit, artifact, replay
    )
    result = _durable._validate_record(
        _durable._read_record(root / RESULT_FILENAME, "result"),
        schema=RESULT_SCHEMA,
        digest_field="result_digest",
        label="shared-witness result",
    )
    if result != expected_result:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "result record differs"
        )
    return _verification(
        root, source, authorization, precommit, artifact, replay, result
    )


def run_object_bongard_shared_witness_nomination(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
    minutes: int = DEFAULT_MINUTES,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: Callable[..., CodexNoToolsAttestation] = attest_codex_no_tools,
    visual_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> VerifiedObjectBongardSharedWitnessNomination:
    """Run exactly one structured nomination in a fresh immutable root."""

    root = _fresh_root(output_root)
    source = load_object_bongard_panel_rubric_calibration_source(source_root)
    authorization = _authorization(
        source,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    _durable._write_once(root / AUTHORIZATION_FILENAME, authorization, "authorization")
    stored_authorization = _durable._validate_record(
        _durable._read_record(root / AUTHORIZATION_FILENAME, "authorization"),
        schema=AUTHORIZATION_SCHEMA,
        digest_field="authorization_digest",
        label="shared-witness authorization",
    )
    if stored_authorization != authorization:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "persisted authorization differs"
        )
    runtime, fingerprint = _create_runtime(
        stored_authorization,
        cache_snapshotter=cache_snapshotter,
        catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    precommit = _precommit(stored_authorization, runtime, fingerprint)
    _durable._write_once(
        root / PRECOMMIT_FILENAME, precommit, "execution precommit"
    )
    stored_precommit = _durable._validate_record(
        _durable._read_record(root / PRECOMMIT_FILENAME, "execution precommit"),
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="shared-witness execution precommit",
    )
    # Exact persisted precommit replay is the last action before a transport-
    # capable journal exists.
    runtime = _runtime_from_precommit(stored_precommit, stored_authorization)
    journal = _journal(
        root,
        source,
        stored_authorization,
        stored_precommit,
        runtime,
        visual_transport,
    )
    artifact = describe_object_bongard_shared_witness_support(
        task_id=stored_authorization["context_task_id"],
        group_0_panel_ids=_groups(source)[0],
        group_1_panel_ids=_groups(source)[1],
        support_png_by_panel_id=_support(source),
        observation_context_digest=stored_precommit["precommit_digest"],
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=journal,
    )
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardSharedWitnessNominationCommandError(
            "nomination did not make exactly one fresh physical call"
        )
    verify_object_bongard_shared_witness_semantic_artifact(
        artifact,
        support_png_by_panel_id=_support(source),
        expected_task_id=stored_authorization["context_task_id"],
        expected_observation_context_digest=stored_precommit["precommit_digest"],
        expected_artifact_digest=artifact.artifact_digest,
    )
    _durable._write_once(
        root / ARTIFACT_FILENAME, artifact.to_data(), "semantic artifact"
    )
    authorization, precommit, _runtime, artifact, summary = _verify_prefix(
        root, source
    )
    replay = _replay_record(source, authorization, precommit, artifact, summary)
    _durable._write_once(root / REPLAY_FILENAME, replay, "cold replay")
    result = _result_record(source, authorization, precommit, artifact, replay)
    _durable._write_once(root / RESULT_FILENAME, result, "result")
    return _verify_loaded(root, source)


def verify_object_bongard_shared_witness_nomination(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> VerifiedObjectBongardSharedWitnessNomination:
    """Cold-verify one completed structured nomination without transport."""

    source = load_object_bongard_panel_rubric_calibration_source(source_root)
    return _verify_loaded(output_root, source)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m bongard.object_bongard_shared_witness_nomination_command",
        description="Launch or cold-verify one structured historical nomination",
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("launch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--source-root",
            type=Path,
            default=DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
        )
    launch = commands.choices["launch"]
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_EXPECTED_LAUNCHER_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        if args.operation == "launch":
            verified = run_object_bongard_shared_witness_nomination(
                args.output_root,
                source_root=args.source_root,
                minutes=args.minutes,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
            )
        else:
            verified = verify_object_bongard_shared_witness_nomination(
                args.output_root, source_root=args.source_root
            )
    except Exception as exc:
        try:
            prefix = str(exc).encode("utf-8", errors="replace")[:4096]
        except Exception:
            prefix = b""
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-shared-witness-nomination-error.v1",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message_prefix_sha256": (
                        None if not prefix else hashlib.sha256(prefix).hexdigest()
                    ),
                    "raw_message_persisted": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(verified.summary_data()).decode("utf-8"))
    return 0 if verified.accepted else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DEFAULT_EXPECTED_LAUNCHER_SHA256",
    "ObjectBongardSharedWitnessNominationCommandError",
    "VerifiedObjectBongardSharedWitnessNomination",
    "main",
    "object_bongard_shared_witness_nomination_command_source_digest",
    "run_object_bongard_shared_witness_nomination",
    "verify_object_bongard_shared_witness_nomination",
)
