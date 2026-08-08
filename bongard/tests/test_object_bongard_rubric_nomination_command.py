"""Sealed-boundary tests for the one-turn rubric cue nomination."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard import object_bongard_rubric_nomination_command as nomination_command
from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
)
from bongard.object_bongard_rubric_nomination_command import (
    ARTIFACT_FILENAME,
    AUTHORIZATION_FILENAME,
    PRECOMMIT_FILENAME,
    REPLAY_FILENAME,
    RESULT_FILENAME,
    DEFAULT_EXPECTED_LAUNCHER_SHA256,
    ObjectBongardRubricNominationCommandError,
    cold_verify_object_bongard_rubric_nomination,
    run_object_bongard_rubric_nomination,
)
from bongard.object_bongard_soft_cues import ObjectBongardSoftCuePair
from bongard.prototype_object_scene_observer import PrototypeSceneObserverStatus
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


def test_nomination_seals_one_turn_then_cold_replays_and_resumes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "nomination"
    catalog, attestation = canonical_no_tools_runtime(
        DEFAULT_EXPECTED_LAUNCHER_SHA256
    )
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        calls += 1
        assert (output_root / AUTHORIZATION_FILENAME).is_file()
        assert (output_root / PRECOMMIT_FILENAME).is_file()
        assert len(paths) == len(names) == 12
        payload = {
            "proposal_0": {
                "group_0_cue_text": "mismatched joined sector-like pieces",
                "group_1_cue_text": "rounded leaf-like contour arrangement",
            },
            "proposal_1": {
                "group_0_cue_text": "mismatched joined sector-like pieces",
                "group_1_cue_text": (
                    "three line-like spans forming a triangular arrangement"
                ),
            },
        }
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=DEFAULT_EXPECTED_LAUNCHER_SHA256,
            reasoning_effort="medium",
            names=names,
        )
        return CodexStructuredResult(payload, receipt)

    kwargs = {
        "source_root": DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
        "cache_snapshotter": lambda: CloudPolicyCacheSnapshot(None),
        "catalog_snapshotter": lambda: catalog,
        "launcher_fingerprinter": lambda _executable, **_kwargs: {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": DEFAULT_EXPECTED_LAUNCHER_SHA256,
        },
        "runtime_attester": lambda **_kwargs: attestation,
        "visual_transport": transport,
    }
    verified = run_object_bongard_rubric_nomination(output_root, **kwargs)
    assert calls == 1
    assert verified.accepted is True
    assert tuple(
        (
            item.group_0_cue.text,
            item.group_1_cue.text,
        )
        for item in verified.artifact.soft_cue_candidates
    ) == (
        (
            "mismatched joined sector-like pieces",
            "rounded leaf-like contour arrangement",
        ),
        (
            "mismatched joined sector-like pieces",
            "three line-like spans forming a triangular arrangement",
        ),
    )
    result = json.loads((output_root / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert "feature_ids_in_neutral_group_order" not in result
    assert "vision_prose_audit_only" not in result
    assert result["ranked_soft_cue_pairs"] == [
        {
            "candidate_rank": pair.candidate_rank,
            "pair_digest": pair.pair_digest,
            "group_0_cue_digest": pair.group_0_cue.cue_digest,
            "group_0_cue_text": pair.group_0_cue.text,
            "group_1_cue_digest": pair.group_1_cue.cue_digest,
            "group_1_cue_text": pair.group_1_cue.text,
        }
        for pair in verified.artifact.soft_cue_candidates
    ]
    authorization = json.loads(
        (output_root / AUTHORIZATION_FILENAME).read_text(encoding="utf-8")
    )
    assert authorization["feature_catalog_used"] is False
    assert authorization["ranked_soft_cue_pair_count"] == 2
    assert authorization["soft_cue_grammar_digest"] == (
        nomination_command.object_bongard_soft_cue_grammar_digest()
    )
    assert any(
        item["role"] == "soft_cue_source_sha256"
        for item in authorization["source_digests"]
    )
    assert cold_verify_object_bongard_rubric_nomination(
        output_root, source_root=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ) == verified
    for name in (
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        ARTIFACT_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
    ):
        assert (output_root / name).is_file()

    resumed = run_object_bongard_rubric_nomination(
        output_root,
        **{
            **kwargs,
            "visual_transport": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed journal called transport again")
            ),
        },
    )
    assert resumed == verified
    assert calls == 1


def test_self_digested_tampered_resume_precommit_cannot_call_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the causal gate without recomputing historical morphology."""

    root = tmp_path / "partial"
    root.mkdir()
    source_digests = [{"role": "fixture", "sha256": "1" * 64}]
    groups = [
        {
            "group_id": "group_0",
            "panel_ids": [f"panel-{index}" for index in range(6)],
            "png_sha256": ["2" * 64] * 6,
            "panel_binding_digests": ["3" * 64] * 6,
        },
        {
            "group_id": "group_1",
            "panel_ids": [f"foil-{index}" for index in range(6)],
            "png_sha256": ["4" * 64] * 6,
            "panel_binding_digests": ["5" * 64] * 6,
        },
    ]
    policy = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "minutes": 15,
        "verbose": False,
        "executable": "codex",
        "expected_launcher_sha256": DEFAULT_EXPECTED_LAUNCHER_SHA256,
    }
    semantic_protocol_digest = (
        nomination_command.object_bongard_semantics_protocol_digest()
    )
    semantic_output_schema_digest = nomination_command.canonical_digest(
        nomination_command.object_bongard_semantics_output_schema()
    )
    soft_cue_grammar_digest = (
        nomination_command.object_bongard_soft_cue_grammar_digest()
    )
    authorization = nomination_command._record(
        {
            "schema": nomination_command.AUTHORIZATION_SCHEMA,
            "runtime_policy": policy,
            "source_digest": "6" * 64,
            "source_digests": source_digests,
            "context_task_id": "bd_fixture_0000",
            "groups": groups,
            "semantic_protocol_digest": semantic_protocol_digest,
            "semantic_output_schema_digest": semantic_output_schema_digest,
            "soft_cue_grammar_digest": soft_cue_grammar_digest,
            "ranked_soft_cue_pair_count": 2,
        },
        "authorization_digest",
    )
    precommit = nomination_command._record(
        {
            "schema": nomination_command.PRECOMMIT_SCHEMA,
            "command_id": nomination_command.COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "source_digest": authorization["source_digest"],
            "source_digests": source_digests,
            "context_task_id": authorization["context_task_id"],
            "groups": groups,
            "semantic_protocol_digest": semantic_protocol_digest,
            "semantic_output_schema_digest": semantic_output_schema_digest,
            "soft_cue_grammar_digest": soft_cue_grammar_digest,
            "ranked_soft_cue_pair_count": 2,
            "runtime_binding": {},
            "cloud_policy_cache_snapshot_base64": None,
            "model_catalog_snapshot_base64": "",
            "no_tools_attestation": {},
            "launcher_fingerprint": {
                "version": "codex-cli 0.146.0",
                "launcher_digest": DEFAULT_EXPECTED_LAUNCHER_SHA256,
            },
            "precommit_fsynced_before_inference": True,
            "physical_model_call_count": 1,
            **nomination_command._authority_data(),
        },
        "precommit_digest",
    )
    nomination_command._write_once(
        root / AUTHORIZATION_FILENAME, authorization, "fixture authorization"
    )
    nomination_command._write_once(
        root / PRECOMMIT_FILENAME, precommit, "fixture precommit"
    )
    monkeypatch.setattr(
        nomination_command, "_load_calibration_source", lambda _root: object()
    )
    monkeypatch.setattr(
        nomination_command, "_authorization", lambda *_args, **_kwargs: authorization
    )
    monkeypatch.setattr(
        nomination_command, "_source_digests", lambda: source_digests
    )
    calls = 0

    def must_not_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("tampered precommit reached transport")

    with pytest.raises(
        ObjectBongardRubricNominationCommandError,
        match="launcher fingerprint differs",
    ):
        run_object_bongard_rubric_nomination(
            root,
            source_root="unused-by-fixture",
            visual_transport=must_not_call,
        )
    assert calls == 0
    assert not (root / "journals").exists()


def test_result_binds_ranked_soft_cue_text_and_no_feature_id_summary() -> None:
    pairs = (
        ObjectBongardSoftCuePair.create(
            0,
            "mismatched joined sector-like pieces",
            "rounded leaf-like contour arrangement",
        ),
        ObjectBongardSoftCuePair.create(
            1,
            "mismatched joined sector-like pieces",
            "three line-like spans forming a triangular arrangement",
        ),
    )
    artifact = SimpleNamespace(
        status=PrototypeSceneObserverStatus.SUCCESS,
        artifact_digest="a" * 64,
        soft_cue_candidates=pairs,
    )
    result = nomination_command._result_record(
        SimpleNamespace(source_digest="b" * 64),
        {"authorization_digest": "sha256:" + "c" * 64},
        {"precommit_digest": "sha256:" + "d" * 64},
        artifact,
        {"replay_digest": "sha256:" + "e" * 64},
    )

    assert result["schema"] == nomination_command.RESULT_SCHEMA
    assert result["ranked_soft_cue_pairs"] == (
        nomination_command._soft_cue_pair_commitments(artifact)
    )
    assert "feature_ids_in_neutral_group_order" not in result
    assert "vision_prose_audit_only" not in result
    assert result["soft_cue_text_defines_identity"] is True
    assert result["feature_catalog_used"] is False


def test_nomination_command_source_has_no_lean_import() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "object_bongard_rubric_nomination_command.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any("lean" in name.lower() for name in imports)
