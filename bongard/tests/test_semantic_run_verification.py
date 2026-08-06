from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bongard import cli
from bongard.artifacts import canonical_digest, canonical_json
from bongard.benchmark import (
    EpisodeStatus,
    SupportGatePolicy,
    VISUAL_SEMANTIC_PREDICATE_MODE,
    prepare_episode,
    run_episode,
)
from bongard.semantic_calibration import (
    CalibrationLabelJoinReceipt,
    CalibrationPanelSelection,
    SemanticCalibrationArtifact,
    SemanticCalibrationMeasurement,
    SemanticCalibrationPlan,
)
from bongard.semantic_episode import VisualSemanticEpisode
from bongard.semantic_protocol import (
    build_prospective_soft_scorer_protocol,
    build_visual_semantic_policy,
)
from bongard.semantic_run_verification import (
    VisualSemanticCalibrationCampaignAnchor,
    VisualSemanticRunVerificationError,
    build_visual_semantic_run_record,
    verify_visual_semantic_run_bytes,
    verify_visual_semantic_run_data,
)
from bongard.soft_predicates import SoftFamilyDevelopmentUnit, SoftScorerFamily
from bongard.tests.test_semantic_episode import (
    _corpus,
    _mixed_proposal_payload,
    _proposal_payload,
)
from bongard.tests.test_semantic_observation import _receipt as _scorer_receipt
from bongard.tests.test_cli import (
    MANIFEST_DIGEST,
    MODEL,
    TASK_ID,
    _FakeCorpus,
    _FakeResult,
)
from bongard.tests.test_typed_visual_transport import _receipt
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)


_STAGE_A_LAUNCHER_DIGEST = "b" * 64
_STAGE_A_CACHE_BINDING = "absent"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _receipt_environment(
    receipt: CodexReceipt,
    *,
    launcher_digest: str,
    cache_binding: str,
) -> CodexReceipt:
    data = receipt.to_dict()
    data["codex_launcher_digest"] = launcher_digest
    data["cloud_config_bundle_cache_binding"] = cache_binding
    data.pop("receipt_digest")
    data["receipt_digest"] = canonical_digest(data)
    return CodexReceipt(
        **{
            **data,
            "event_types": tuple(data["event_types"]),
            "item_types": tuple(data["item_types"]),
        }
    )


def _calibration(
    corpus, *, proposer_model_id: str = "fixture-proposer"
) -> SemanticCalibrationArtifact:
    protocol = build_prospective_soft_scorer_protocol(
        proposer_model_id=proposer_model_id,
        proposer_reasoning_effort="medium",
        scorer_model_id="fixture-scorer",
        scorer_reasoning_effort="medium",
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=8,
    )
    selections = tuple(
        sorted(
            (
                CalibrationPanelSelection(
                    observation_id=f"{prefix}-{index:02d}",
                    task_id=f"development-{prefix}-{index:02d}",
                    panel_id=f"panel-{prefix}-{index:02d}",
                    panel_digest=_digest(f"panel bytes {prefix} {index}"),
                    split="train",
                    dependence_cluster_id=f"cluster-{prefix}-{index:02d}",
                )
                for prefix in ("low", "high")
                for index in range(8)
            ),
            key=lambda item: item.observation_id,
        )
    )
    manifest = corpus.build_manifest()
    plan = SemanticCalibrationPlan(
        protocol_digest=protocol.digest(),
        corpus_manifest_digest=manifest.digest,
        development_manifest_digest=manifest.digest,
        split_source_digest=corpus.split.source_digest,
        split_manifest_digest=canonical_digest(corpus.split.to_manifest_dict()),
        label_reveal_protocol_digest=_digest("label reveal protocol"),
        selections=selections,
    )
    measurements = []
    for selection in selections:
        positive = selection.observation_id.startswith("high-")
        score = 1.0 if positive else 0.0
        score_artifact_digest = _digest(
            f"score artifact {selection.observation_id}"
        )
        score_record_digest = _digest(f"score record {selection.observation_id}")
        scorer_receipt_digest = _digest(
            f"scorer receipt {selection.observation_id}"
        )
        label_receipt_digest = _digest(
            f"label receipt {selection.observation_id}"
        )
        join = CalibrationLabelJoinReceipt(
            plan_digest=plan.digest,
            selection_digest=selection.digest,
            score_artifact_digest=score_artifact_digest,
            score_record_digest=score_record_digest,
            scorer_receipt_digest=scorer_receipt_digest,
            label_reveal_protocol_digest=plan.label_reveal_protocol_digest,
            label_reveal_receipt_digest=label_receipt_digest,
            affirmative_label=positive,
        )
        unit = SoftFamilyDevelopmentUnit(
            observation_id=selection.observation_id,
            task_id=selection.task_id,
            panel_digest=selection.panel_digest,
            claim_digest=_digest(f"claim {selection.observation_id}"),
            scorer_protocol_digest=protocol.digest(),
            proposer_call_id=f"proposer-{selection.observation_id}",
            scorer_call_id=f"scorer-{selection.observation_id}",
            dependence_cluster_id=selection.dependence_cluster_id,
            score_record_digest=score_record_digest,
            annotation_receipt_digest=join.digest,
            score=score,
            affirmative_label=positive,
            score_bin_index=1 if positive else 0,
        )
        measurements.append(
            SemanticCalibrationMeasurement(
                plan_digest=plan.digest,
                selection=selection,
                score_artifact_digest=score_artifact_digest,
                label_reveal_receipt_digest=label_receipt_digest,
                join_receipt=join,
                development_unit=unit,
            )
        )
    units = tuple(item.development_unit for item in measurements)
    family = SoftScorerFamily.fit(
        protocol, units, expected_protocol_digest=protocol.digest()
    )
    return SemanticCalibrationArtifact(
        plan=plan,
        protocol=protocol,
        measurements=tuple(measurements),
        family=family,
    )


def _record_for_payload(
    tmp_path: Path,
    *,
    payload: dict[str, object] | None = None,
    mixed: bool = False,
    seed: str = "strict-semantic-outer-record",
    proposer_launcher_digest: str = _STAGE_A_LAUNCHER_DIGEST,
    proposer_cache_binding: str = _STAGE_A_CACHE_BINDING,
    scorer_launcher_digest: str = _STAGE_A_LAUNCHER_DIGEST,
    scorer_cache_binding: str = _STAGE_A_CACHE_BINDING,
    anchor_launcher_digest: str = _STAGE_A_LAUNCHER_DIGEST,
    anchor_cache_binding: str = _STAGE_A_CACHE_BINDING,
    enforce_live_environment: bool = True,
):
    corpus, task_id = _corpus(tmp_path)
    calibration = _calibration(corpus)
    campaign_anchor = VisualSemanticCalibrationCampaignAnchor(
        _digest("verified full Stage-A campaign"),
        calibration,
        anchor_launcher_digest,
        anchor_cache_binding,
    )
    policy = build_visual_semantic_policy(
        calibration.family,
        prospective_protocol=calibration.protocol,
    )
    manifest = corpus.build_manifest()
    plan = prepare_episode(
        corpus,
        task_id,
        seed=seed,
        corpus_manifest=manifest,
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    if payload is None:
        payload = _mixed_proposal_payload() if mixed else _proposal_payload()

    def proposer_transport(prompt, paths, schema, **kwargs):
        receipt = _receipt(
            prompt,
            paths,
            schema,
            payload,
            model=kwargs["model"],
            effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt_environment(
                receipt,
                launcher_digest=proposer_launcher_digest,
                cache_binding=proposer_cache_binding,
            ),
        )

    semantic_episode: VisualSemanticEpisode

    def scorer_transport(prompt, paths, names, schema, **kwargs):
        if not mixed:
            raise AssertionError("direct-only fixture attempted soft scoring")
        with Image.open(paths[0]) as source_image:
            two_components = sum(source_image.convert("L").histogram()[:128]) > 450
        assert semantic_episode.compiled is not None
        lowering = semantic_episode.compiled.lowering_archive.soft_lowering
        assert lowering is not None
        score_payload = {
            "cue_judgments": [
                {
                    "cue_id": cue.cue_id,
                    "judgment": "supported" if two_components else "unsupported",
                    "witness_ids": ["panel:geometry"] if two_components else [],
                }
                for cue in lowering.claim.cues
            ]
        }
        receipt = _scorer_receipt(
            prompt, paths, names, schema, score_payload
        )
        return CodexStructuredResult(
            payload=score_payload,
            receipt=_receipt_environment(
                receipt,
                launcher_digest=scorer_launcher_digest,
                cache_binding=scorer_cache_binding,
            ),
        )

    semantic_episode = VisualSemanticEpisode(
        task_id=task_id,
        support_commitment=plan.support,
        policy=policy,
        family=calibration.family,
        protocol=calibration.protocol,
        proposer_transport=proposer_transport,
        scorer_transport=scorer_transport,
        expected_codex_launcher_digest=(
            anchor_launcher_digest if enforce_live_environment else None
        ),
        expected_cloud_policy_cache_binding=(
            anchor_cache_binding if enforce_live_environment else None
        ),
        cloud_policy_cache_snapshot=(
            CloudPolicyCacheSnapshot(None) if enforce_live_environment else None
        ),
    )
    result = run_episode(
        plan,
        semantic_episode,
        semantic_episode,
        support_gate_policy=SupportGatePolicy.visual_semantic(),
    )
    record = build_visual_semantic_run_record(
        corpus_manifest_digest=manifest.digest,
        split_source_digest=corpus.split.source_digest,
        official_release=None,
        calibration_campaign=campaign_anchor,
        plan=plan,
        result=result,
        episode=semantic_episode,
        exposure=None,
    )
    sources = (
        (*plan._support_sources, *plan._query_sources)
        if result.bundle is not None
        else plan._support_sources
    )
    preimages = {
        source.panel.blob_id: source.read_verified() for source in sources
    }
    return record, preimages, calibration, result


def _campaign_anchor(
    calibration: SemanticCalibrationArtifact,
    *,
    campaign_digest: str | None = None,
    launcher_digest: str = _STAGE_A_LAUNCHER_DIGEST,
    cache_binding: str = _STAGE_A_CACHE_BINDING,
) -> VisualSemanticCalibrationCampaignAnchor:
    return VisualSemanticCalibrationCampaignAnchor(
        campaign_digest or _digest("verified full Stage-A campaign"),
        calibration,
        launcher_digest,
        cache_binding,
    )


def _completed_record(tmp_path: Path, *, mixed: bool = False):
    fixture = _record_for_payload(tmp_path, mixed=mixed)
    assert fixture[3].status is EpisodeStatus.COMPLETE
    return fixture


def _reseal_outer_plan(record: dict[str, object]) -> None:
    """Rehash all public envelopes after an adversarial plan substitution."""

    plan = record["plan"]
    episode = record["episode"]
    assert isinstance(plan, dict)
    assert isinstance(episode, dict)
    episode["plan_digest"] = canonical_digest(plan)
    content = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = canonical_digest(content)


def test_visual_semantic_outer_record_reconstructs_python_pipeline_and_pixels(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, result = _completed_record(tmp_path)
    verified = verify_visual_semantic_run_data(
        record,
        blob_bytes_by_id=preimages,
        calibration_campaign=_campaign_anchor(calibration),
    )

    assert verified.status == EpisodeStatus.COMPLETE.value
    assert verified.calibration_digest == calibration.digest
    assert verified.calibration_campaign_digest == record[
        "calibration_campaign_digest"
    ]
    assert verified.proposal_transport_digest is not None
    assert verified.pre_observation_commitment_digest is not None
    assert verified.lowering_archive_digest is not None
    assert verified.support_gate_digest == result.support_gate.digest
    assert verified.proposal_freeze_digest == result.proposal_freeze.digest()
    assert verified.prediction_commitment_digest == (
        result.bundle.predictions.digest()
    )
    assert len(verified.support_observation_digests) == 12
    assert len(verified.query_observation_digests) == 2
    assert verified.registered_atom_replays == 14
    assert verified.optional_checker_required is False
    assert set(verified.verified_blob_ids) == set(preimages)

    encoded = canonical_json(record)
    assert verify_visual_semantic_run_bytes(
        encoded,
        blob_bytes_by_id=preimages,
        calibration_campaign=_campaign_anchor(calibration),
    ).record_digest == record["record_digest"]


@pytest.mark.parametrize(
    ("field", "message"),
    (
        (
            "latent_query_digest",
            "latent query commitment differs from archive release",
        ),
        (
            "label_commitment_digest",
            "label seal differs from archive label reveal",
        ),
    ),
)
def test_visual_semantic_outer_verifier_opens_public_query_and_label_commitments(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    record, preimages, calibration, _result = _completed_record(tmp_path / field)
    changed = copy.deepcopy(record)
    assert isinstance(changed["plan"], dict)
    changed["plan"][field] = "0" * 64
    _reseal_outer_plan(changed)

    with pytest.raises(VisualSemanticRunVerificationError, match=message):
        verify_visual_semantic_run_data(
            changed,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(calibration),
        )


def test_visual_semantic_outer_verifier_joins_external_task_manifest_authority(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, _result = _completed_record(tmp_path)
    assert isinstance(record["plan"], dict)
    trusted_digest = record["plan"]["task_manifest_digest"]
    assert isinstance(trusted_digest, str)
    assert verify_visual_semantic_run_data(
        record,
        blob_bytes_by_id=preimages,
        calibration_campaign=_campaign_anchor(calibration),
        expected_task_manifest_digest=trusted_digest,
    ).status == EpisodeStatus.COMPLETE.value

    changed = copy.deepcopy(record)
    assert isinstance(changed["plan"], dict)
    changed["plan"]["task_manifest_digest"] = "0" * 64
    _reseal_outer_plan(changed)
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="task manifest digest differs from external authority",
    ):
        verify_visual_semantic_run_data(
            changed,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(calibration),
            expected_task_manifest_digest=trusted_digest,
        )


def test_visual_semantic_outer_verifier_rejects_pixel_or_sidecar_tamper(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, _result = _completed_record(tmp_path)
    changed_pixels = dict(preimages)
    first_id = sorted(changed_pixels)[0]
    changed_pixels[first_id] = changed_pixels[first_id] + b"tamper"
    with pytest.raises(
        VisualSemanticRunVerificationError, match="differs from its BlobRef"
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=changed_pixels,
            calibration_campaign=_campaign_anchor(calibration),
        )

    changed_record = copy.deepcopy(record)
    changed_record["visual_semantic"]["checker_sidecar"] = {
        "status": "agreed"
    }
    content = {
        key: value for key, value in changed_record.items() if key != "record_digest"
    }
    changed_record["record_digest"] = canonical_digest(content)
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="visual-semantic artifact fields differ",
    ):
        verify_visual_semantic_run_data(
            changed_record,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(calibration),
        )


def test_visual_semantic_outer_verifier_replays_soft_score_receipts_and_atoms(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, _result = _completed_record(
        tmp_path, mixed=True
    )
    verified = verify_visual_semantic_run_data(
        record,
        blob_bytes_by_id=preimages,
        calibration_campaign=_campaign_anchor(calibration),
    )
    assert verified.status == EpisodeStatus.COMPLETE.value
    assert verified.registered_atom_replays == 28
    assert len(verified.support_observation_digests) == 12
    assert len(verified.query_observation_digests) == 2


def test_visual_semantic_outer_verifier_replays_proposal_and_gate_rejections(
    tmp_path: Path,
) -> None:
    invalid = _proposal_payload()
    invalid["deterministic_atoms"] = []
    invalid["formula"] = {"kind": "all", "atom_indices": []}
    rejected_record, rejected_pixels, rejected_calibration, rejected_result = (
        _record_for_payload(
            tmp_path / "proposal-error",
            payload=invalid,
            seed="semantic-proposal-error",
        )
    )
    assert rejected_result.status is EpisodeStatus.PROPOSAL_ERROR
    rejected = verify_visual_semantic_run_data(
        rejected_record,
        blob_bytes_by_id=rejected_pixels,
        calibration_campaign=_campaign_anchor(rejected_calibration),
    )
    assert rejected.status == EpisodeStatus.PROPOSAL_ERROR.value
    assert rejected.proposal_transport_digest is not None
    assert rejected.pre_observation_commitment_digest is None
    assert rejected.support_observation_digests == ()

    unsupported = _proposal_payload()
    unsupported["deterministic_atoms"] = [
        {
            "catalog_key": "component.count",
            "comparison": "equal",
            "arguments": {"target_count": 3},
        }
    ]
    gate_record, gate_pixels, gate_calibration, gate_result = (
        _record_for_payload(
            tmp_path / "support-rejected",
            payload=unsupported,
            seed="semantic-support-rejected",
        )
    )
    assert gate_result.status is EpisodeStatus.SUPPORT_REJECTED
    verified_gate = verify_visual_semantic_run_data(
        gate_record,
        blob_bytes_by_id=gate_pixels,
        calibration_campaign=_campaign_anchor(gate_calibration),
    )
    assert verified_gate.status == EpisodeStatus.SUPPORT_REJECTED.value
    assert verified_gate.support_gate_digest is not None
    assert len(verified_gate.support_observation_digests) == 12
    assert verified_gate.query_observation_digests == ()


def test_outer_verifier_requires_campaign_authority_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, _result = _completed_record(tmp_path)

    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="bare SemanticCalibrationArtifact",
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=preimages,
            calibration_campaign=calibration,
        )

    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="campaign digest differs",
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(
                calibration, campaign_digest=_digest("another campaign")
            ),
        )

    other_calibration = _calibration(
        _corpus(tmp_path / "other-calibration")[0],
        proposer_model_id="another-proposer",
    )
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="embedded semantic calibration differs",
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(other_calibration),
        )


def test_campaign_anchor_and_cold_replay_bind_stage_a_transport_environment(
    tmp_path: Path,
) -> None:
    record, preimages, calibration, _result = _completed_record(tmp_path)

    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="expected Codex launcher digest",
    ):
        VisualSemanticCalibrationCampaignAnchor(
            _digest("campaign"),
            calibration,
            "not-a-digest",
            "absent",
        )
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="cloud-policy cache binding",
    ):
        VisualSemanticCalibrationCampaignAnchor(
            _digest("campaign"),
            calibration,
            _STAGE_A_LAUNCHER_DIGEST,
            "unbound-cache",
        )

    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="accepted typed proposer receipt Codex launcher differs",
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(
                calibration,
                launcher_digest="c" * 64,
            ),
        )
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="accepted typed proposer receipt cloud-policy cache differs",
    ):
        verify_visual_semantic_run_data(
            record,
            blob_bytes_by_id=preimages,
            calibration_campaign=_campaign_anchor(
                calibration,
                cache_binding="sha256:" + "d" * 64,
            ),
        )


def test_cold_replay_checks_rejected_proposer_and_every_codex_scorer_environment(
    tmp_path: Path,
) -> None:
    invalid = _proposal_payload()
    invalid["deterministic_atoms"] = []
    invalid["formula"] = {"kind": "all", "atom_indices": []}
    rejected_record, rejected_pixels, rejected_calibration, rejected_result = (
        _record_for_payload(
            tmp_path / "rejected-proposer-environment",
            payload=invalid,
            seed="semantic-rejected-environment",
            proposer_launcher_digest="c" * 64,
            enforce_live_environment=False,
        )
    )
    assert rejected_result.status is EpisodeStatus.PROPOSAL_ERROR
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="rejected typed proposer receipt Codex launcher differs",
    ):
        verify_visual_semantic_run_data(
            rejected_record,
            blob_bytes_by_id=rejected_pixels,
            calibration_campaign=_campaign_anchor(rejected_calibration),
        )

    scorer_record, scorer_pixels, scorer_calibration, scorer_result = (
        _record_for_payload(
            tmp_path / "scorer-environment",
            mixed=True,
            seed="semantic-scorer-environment",
            scorer_launcher_digest="c" * 64,
            enforce_live_environment=False,
        )
    )
    assert scorer_result.status is EpisodeStatus.COMPLETE
    with pytest.raises(
        VisualSemanticRunVerificationError,
        match="blind scorer receipt Codex launcher differs",
    ):
        verify_visual_semantic_run_data(
            scorer_record,
            blob_bytes_by_id=scorer_pixels,
            calibration_campaign=_campaign_anchor(scorer_calibration),
        )


def test_cli_loads_exact_full_campaign_and_exposes_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bongard.semantic_calibration_campaign import (
        SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA,
    )

    corpus, _task_id = _corpus(tmp_path / "corpus")
    calibration = _calibration(corpus)
    campaign_digest = _digest("campaign loader fixture")
    campaign_data = {
        "schema": SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA,
        "campaign_digest": campaign_digest,
    }
    campaign = SimpleNamespace(
        digest=campaign_digest,
        calibration=calibration,
    )
    monkeypatch.setattr(
        "bongard.semantic_calibration_campaign.verify_semantic_campaign_against_corpus",
        lambda *args, **kwargs: (campaign, {}),
        raising=False,
    )
    source = tmp_path / "semantic-calibration-campaign.json"
    source.write_bytes(canonical_json(campaign_data) + b"\n")
    loaded = cli._load_semantic_calibration_campaign(
        source,
        expected_campaign_digest=campaign_digest,
        corpus=corpus,
        corpus_manifest=corpus.build_manifest(),
    )
    assert loaded.digest == campaign_digest

    with pytest.raises(cli.CliError, match="expected content digest"):
        cli._load_semantic_calibration_campaign(
            source,
            expected_campaign_digest=_digest("wrong campaign"),
            corpus=corpus,
            corpus_manifest=corpus.build_manifest(),
        )

    noncanonical = tmp_path / "noncanonical-campaign.json"
    noncanonical.write_bytes(canonical_json(campaign_data))
    with pytest.raises(cli.CliError, match="canonical JSON plus one newline"):
        cli._load_semantic_calibration_campaign(
            noncanonical,
            expected_campaign_digest=campaign_digest,
            corpus=corpus,
            corpus_manifest=corpus.build_manifest(),
        )

    bare = tmp_path / "bare-calibration.json"
    bare.write_bytes(canonical_json(calibration.to_data()) + b"\n")
    with pytest.raises(cli.CliError, match="bare semantic calibration"):
        cli._load_semantic_calibration_campaign(
            bare,
            expected_campaign_digest=campaign_digest,
            corpus=corpus,
            corpus_manifest=corpus.build_manifest(),
        )

    args = cli.build_parser().parse_args(
        [
            "run",
            "--corpus",
            "corpus",
            "--task-id",
            TASK_ID,
            "--seed",
            "seed",
            "--out",
            "run.json",
            "--exposure-dir",
            "exposure",
            "--predicate-mode",
            "visual-semantic",
            "--semantic-calibration-campaign",
            str(source),
            "--expected-semantic-calibration-campaign-digest",
            campaign_digest,
        ]
    )
    assert args.predicate_mode == "visual-semantic"
    assert args.semantic_calibration_campaign == str(source)
    assert args.expected_semantic_calibration_campaign_digest == campaign_digest

    verify_args = cli.build_parser().parse_args(
        [
            "verify",
            "--run",
            "run.json",
            "--corpus",
            "corpus",
            "--archive",
            "ShapeBongard_V2.zip",
            "--expected-sha256",
            "0" * 64,
            "--semantic-calibration-campaign",
            str(source),
            "--expected-semantic-calibration-campaign-digest",
            campaign_digest,
        ]
    )
    assert verify_args.semantic_calibration_campaign == str(source)
    assert (
        verify_args.expected_semantic_calibration_campaign_digest
        == campaign_digest
    )


def test_run_record_checks_campaign_corpus_anchor_before_episode() -> None:
    with pytest.raises(cli.CliError, match="cold-verified calibration campaign"):
        cli._run_record(
            corpus=_FakeCorpus(),
            task_id=TASK_ID,
            seed="seed",
            session=SimpleNamespace(artifact_data=lambda: {}),
            sealed_test=False,
            exposure_dir="unused-exposure",
            ledger_in=None,
            require_unseen=False,
            model=MODEL,
            expected_corpus_manifest_digest="sha256:" + "0" * 64,
        )


def test_cli_wires_visual_semantic_profile_and_immediate_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exposure = {
        "ledger_before_digest": "sha256:" + "a" * 64,
        "ledger_after_digest": "sha256:" + "b" * 64,
        "event_digest": "sha256:" + "c" * 64,
        "successor_filename": "b" * 64 + ".exposure.json",
    }
    record = {"exposure": exposure, "official_release": None}
    result = _FakeResult()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / exposure["successor_filename"]).write_text(
        "{}", encoding="utf-8"
    )

    protocol = SimpleNamespace(
        proposer_model_id=MODEL,
        proposer_reasoning_effort="medium",
    )
    calibration = SimpleNamespace(
        family=object(),
        protocol=protocol,
        plan=SimpleNamespace(
            corpus_manifest_digest=MANIFEST_DIGEST,
            split_source_digest=None,
        ),
        to_data=lambda: {"calibration_artifact_digest": "e" * 64},
    )
    campaign = SimpleNamespace(
        digest="f" * 64,
        calibration=calibration,
        score_batch=SimpleNamespace(
            commitment_batch=SimpleNamespace(
                proposal_archive=SimpleNamespace(
                    execution_config=SimpleNamespace(
                        expected_codex_launcher_digest=(
                            _STAGE_A_LAUNCHER_DIGEST
                        ),
                        cloud_policy_cache_binding=_STAGE_A_CACHE_BINDING,
                    )
                )
            )
        ),
    )
    policy = SimpleNamespace(digest=lambda: "d" * 64)
    captured: dict[str, object] = {}
    captured_episode: dict[str, object] = {}

    def fake_run_record(**kwargs):
        captured.update(kwargs)
        return record, result

    monkeypatch.setattr(cli, "_load_corpus", lambda args: _FakeCorpus())
    monkeypatch.setattr(
        cli,
        "_load_semantic_calibration_campaign",
        lambda *args, **kwargs: campaign,
    )
    monkeypatch.setattr(
        cli, "build_visual_semantic_policy", lambda *args, **kwargs: policy
    )
    monkeypatch.setattr(
        cli,
        "codex_cli_fingerprint",
        lambda executable: {
            "launcher_digest": _STAGE_A_LAUNCHER_DIGEST,
            "version": "codex-cli fixture",
        },
    )
    frozen_cache = CloudPolicyCacheSnapshot(None)
    monkeypatch.setattr(
        cli, "snapshot_cloud_policy_cache", lambda: frozen_cache
    )
    monkeypatch.setattr(cli, "_run_record", fake_run_record)
    monkeypatch.setattr(
        cli,
        "VisualSemanticEpisode",
        lambda **kwargs: captured_episode.update(kwargs) or SimpleNamespace(),
    )
    args = SimpleNamespace(
        exposure_dir=str(ledger_dir),
        ledger_in=None,
        require_unseen=False,
        sealed_test=False,
        official_release=False,
        archive=None,
        release_descriptor="unused.json",
        model=MODEL,
        reasoning_effort="medium",
        predicate_mode="visual-semantic",
        prototype_calibration=None,
        semantic_calibration_campaign="semantic-calibration-campaign.json",
        expected_semantic_calibration_campaign_digest="f" * 64,
        expected_codex_launcher_sha256=_STAGE_A_LAUNCHER_DIGEST,
        proposer_minutes=2,
        observer_minutes=3,
        verbose=False,
        task_id=TASK_ID,
        seed="seed",
        out=str(tmp_path / "run.json"),
    )

    assert cli._run(args) == 2
    assert captured["session"] is None
    assert callable(captured["session_factory"])
    assert captured["support_gate_policy"].mode.value == (
        "visual_semantic_replay"
    )
    assert captured["predicate_mode"] == VISUAL_SEMANTIC_PREDICATE_MODE
    assert captured["predicate_policy_digest"] == "d" * 64
    assert captured["artifact_field"] == "visual_semantic"
    assert captured["record_additions"] == {
        "calibration_campaign_digest": "f" * 64,
        "calibration": {"calibration_artifact_digest": "e" * 64}
    }
    assert captured["expected_corpus_manifest_digest"] == (
        calibration.plan.corpus_manifest_digest
    )
    assert captured["expected_split_source_digest"] == (
        calibration.plan.split_source_digest
    )
    assert callable(captured["record_verifier"])
    captured["session_factory"](
        SimpleNamespace(task_id=TASK_ID, support=object())
    )
    assert captured_episode["cloud_policy_cache_snapshot"] is frozen_cache
    assert captured_episode["expected_codex_launcher_digest"] == (
        _STAGE_A_LAUNCHER_DIGEST
    )
    assert captured_episode["expected_cloud_policy_cache_binding"] == (
        _STAGE_A_CACHE_BINDING
    )
    assert json.loads(capsys.readouterr().out)["status"] == result.status.value


def test_cli_semantic_environment_requires_external_launcher_and_exact_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expected_codex_launcher_digest=_STAGE_A_LAUNCHER_DIGEST,
        cloud_policy_cache_binding=_STAGE_A_CACHE_BINDING,
    )
    with pytest.raises(cli.CliError, match="visual-semantic requires"):
        cli._freeze_semantic_execution_environment(
            externally_expected_launcher_digest=None,
            stage_a_execution_config=config,
            official_release=False,
        )
    with pytest.raises(cli.CliError, match="differs from the cold-verified"):
        cli._freeze_semantic_execution_environment(
            externally_expected_launcher_digest="c" * 64,
            stage_a_execution_config=config,
            official_release=False,
        )

    monkeypatch.setattr(
        cli,
        "codex_cli_fingerprint",
        lambda executable: {
            "launcher_digest": _STAGE_A_LAUNCHER_DIGEST,
            "version": "codex-cli fixture",
        },
    )
    monkeypatch.setattr(
        cli,
        "snapshot_cloud_policy_cache",
        lambda: SimpleNamespace(binding="sha256:" + "d" * 64),
    )
    with pytest.raises(cli.CliError, match="current cloud-policy cache differs"):
        cli._freeze_semantic_execution_environment(
            externally_expected_launcher_digest=_STAGE_A_LAUNCHER_DIGEST,
            stage_a_execution_config=config,
            official_release=False,
        )
