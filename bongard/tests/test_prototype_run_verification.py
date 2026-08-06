from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest, canonical_json
from bongard.benchmark import (
    EpisodePlan,
    EpisodeStatus,
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    SupportGatePolicy,
    prepare_episode,
    run_episode,
)
from bongard.corpus import ShapeBongardCorpus
from bongard.prototype_calibration import (
    calibrate_prototype_margins,
)
from bongard.prototype_episode import HeadlessPrototypeEpisode
from bongard.prototype_artifacts import (
    FeatureExtractionPreimage,
    PrototypePreQueryFreeze,
)
from bongard.prototype_run_verification import (
    PROTOTYPE_OUTER_RUN_SCHEMA,
    PrototypeRunVerificationError,
    build_prototype_run_record,
    verify_prototype_run_bytes,
    verify_prototype_run_data,
)
from bongard.support_prototypes import (
    PositivePrototypeFormula,
    fit_support_prototypes,
)
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    CodexStructuredResult,
    semantic_panel_set_digest,
    validate_codex_receipt,
)


def _draw_panel(path: Path, *, positive: bool, index: int) -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    offset = index
    if positive:
        draw.rectangle((25 - offset, 26, 69 + offset, 68), fill="black")
    else:
        draw.rectangle((12, 27 - offset, 34, 66 + offset), fill="black")
        draw.rectangle((61, 27 + offset, 83, 66 - offset), fill="black")
    image.save(path, format="PNG", optimize=False)


def _corpus(tmp_path: Path) -> tuple[ShapeBongardCorpus, str]:
    root = tmp_path / "ShapeBongard_V2"
    task_id = "ff_nact2_5_0000"
    for positive, label in ((True, "1"), (False, "0")):
        directory = root / "ff" / "images" / task_id / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            _draw_panel(directory / f"{index}.png", positive=positive, index=index)
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({"train": [task_id]}), encoding="utf-8"
    )
    return ShapeBongardCorpus.from_root(root), task_id


def _proposal_payload(observable_id: str = "prototype.topology") -> dict[str, Any]:
    return {
        "positive_description": "one connected foreground component",
        "panel_descriptions": {
            **{f"pos_{index}": "one compact foreground block" for index in range(6)},
            **{
                f"neg_{index}": "two separated foreground blocks"
                for index in range(6)
            },
        },
        "view": "literal_ink",
        "observable_requests": [
            {
                "observable_id": observable_id,
                "affirmative_interpretation": (
                    "one connected foreground component is present"
                ),
                "arguments": {},
            }
        ],
        "formula_template": {"kind": "all", "atoms": [observable_id]},
        "hybrid_claim": None,
        "confidence": "high",
    }


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(
    prompt: str,
    paths: tuple[str, ...],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
) -> CodexReceipt:
    identities = [
        {
            "name": Path(path).name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": _sha_bytes(Path(path).read_bytes()),
        }
        for path in paths
    ]
    prompt_digest = _sha_bytes(prompt.encode("utf-8"))
    schema_digest = _sha_bytes(canonical_json(dict(schema)))
    panel_view_digest = canonical_digest(identities)
    panel_set_digest = semantic_panel_set_digest(paths)
    envelope = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_panel_identities": identities,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": model,
        "model_identity_evidence": "jsonl-reported-model",
        "requested_reasoning_effort": reasoning_effort,
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 1,
        "reasoning_output_tokens": 0,
        "thread_id": "00000000-0000-4000-8000-000000000001",
        "codex_cli_version": "test",
        "codex_launcher_digest": "a" * 64,
        "cloud_config_bundle_cache_binding": "absent",
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "input_digest": canonical_digest(envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "structured_output_digest": _sha_bytes(canonical_json(dict(payload))),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "b" * 64,
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    validate_codex_receipt(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _transport_for(payload: Mapping[str, Any]):
    def fake_transport(prompt, paths, schema, **kwargs):
        canonical_paths = tuple(paths)
        assert tuple(Path(path).name for path in canonical_paths) == tuple(
            [f"pos_{index}.png" for index in range(6)]
            + [f"neg_{index}.png" for index in range(6)]
        )
        return CodexStructuredResult(
            payload=dict(payload),
            receipt=_receipt(
                prompt,
                canonical_paths,
                schema,
                payload,
                model=kwargs["model"],
                reasoning_effort=kwargs["reasoning_effort"],
            ),
        )

    return fake_transport


def _blob_bytes(plan: EpisodePlan, *, include_queries: bool) -> dict[str, bytes]:
    sources = plan._support_sources + (  # noqa: SLF001 - verifier fixture.
        plan._query_sources if include_queries else ()  # noqa: SLF001
    )
    return {source.panel.blob_id: source.read_verified() for source in sources}


def _reseal_outer(record: dict[str, Any]) -> None:
    record["record_digest"] = canonical_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )


def _reseal_archive(archive: dict[str, Any]) -> None:
    archive["archive_digest"] = canonical_digest(
        {key: value for key, value in archive.items() if key != "archive_digest"}
    )


def _run(
    tmp_path: Path,
    *,
    margin: float,
    payload: Mapping[str, Any],
):
    corpus, task_id = _corpus(tmp_path)
    manifest = corpus.build_manifest()
    calibration = calibrate_prototype_margins(
        corpus,
        [task_id],
        seed="prototype-verification-calibration",
        candidate_margins=[margin],
    )
    policy = calibration.to_freeze_policy()
    plan = prepare_episode(
        corpus,
        task_id,
        seed="prototype-verification-episode",
        corpus_manifest=manifest,
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    adapter = HeadlessPrototypeEpisode(
        support_commitment=plan.support,
        policy=policy,
        proposer_transport=_transport_for(payload),
    )
    result = run_episode(
        plan,
        adapter,
        adapter,
        support_gate_policy=SupportGatePolicy.prototype(),
    )
    record = build_prototype_run_record(
        corpus_manifest_digest=manifest.digest,
        split_source_digest=corpus.split.source_digest,
        official_release=None,
        calibration=calibration,
        plan=plan,
        result=result,
        episode=adapter,
        exposure=None,
    )
    return plan, result, calibration, record


def test_complete_record_exact_shape_and_cold_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, result, calibration, record = _run(
        tmp_path,
        margin=1e-6,
        payload=_proposal_payload(),
    )
    assert result.status is EpisodeStatus.COMPLETE
    assert set(record) == {
        "schema",
        "corpus_manifest_digest",
        "split_source_digest",
        "official_release",
        "calibration",
        "plan",
        "support_commitment",
        "episode",
        "prototype",
        "run_archive",
        "exposure",
        "record_digest",
    }
    assert record["schema"] == PROTOTYPE_OUTER_RUN_SCHEMA
    assert "support_gate" in record["prototype"]
    assert "proposal_freeze" in record["prototype"]

    import bongard.prototype_run_verification as verification_module

    original = verification_module.extract_neutral_features
    calls = 0

    def counting_extractor(panel_bytes: bytes):
        nonlocal calls
        calls += 1
        return original(panel_bytes)

    monkeypatch.setattr(
        verification_module,
        "extract_neutral_features",
        counting_extractor,
    )
    blobs = _blob_bytes(plan, include_queries=True)
    report = verify_prototype_run_data(record, blob_bytes_by_id=blobs)
    assert report.status == EpisodeStatus.COMPLETE.value
    assert report.calibration_digest == calibration.digest()
    assert report.neutral_extraction_replays == 26
    assert calls == 26
    assert report.verified_blob_ids == tuple(sorted(blobs))
    assert report.missing_blob_ids == ()
    assert report.archive is not None
    assert verify_prototype_run_bytes(
        canonical_json(record), blob_bytes_by_id=blobs
    ).to_data() == report.to_data()

    wrong = dict(blobs)
    wrong["query-panel-0"] = wrong["query-panel-0"] + b"tamper"
    with pytest.raises(PrototypeRunVerificationError, match="outer BlobRef"):
        verify_prototype_run_data(record, blob_bytes_by_id=wrong)


def test_support_rejected_record_is_reproducible_without_query_bytes(
    tmp_path: Path,
) -> None:
    plan, result, _calibration, record = _run(
        tmp_path,
        margin=1_000_000.0,
        payload=_proposal_payload(),
    )
    assert result.status is EpisodeStatus.SUPPORT_REJECTED
    assert record["run_archive"] is None
    assert record["prototype"]["observations"] == {}
    blobs = _blob_bytes(plan, include_queries=False)
    report = verify_prototype_run_data(record, blob_bytes_by_id=blobs)
    assert report.status == EpisodeStatus.SUPPORT_REJECTED.value
    assert report.neutral_extraction_replays == 24
    assert report.archive_digest is None
    assert report.query_artifact_digests == ()
    assert set(report.verified_blob_ids) == set(blobs)


def test_proposal_error_record_replays_rejected_structured_attempt(
    tmp_path: Path,
) -> None:
    plan, result, _calibration, record = _run(
        tmp_path,
        margin=1e-6,
        payload=_proposal_payload("prototype.unknown"),
    )
    assert result.status is EpisodeStatus.PROPOSAL_ERROR
    assert record["prototype"]["rejected_proposal_attempt"] is not None
    blobs = _blob_bytes(plan, include_queries=False)
    report = verify_prototype_run_data(record, blob_bytes_by_id=blobs)
    assert report.status == EpisodeStatus.PROPOSAL_ERROR.value
    assert report.proposal_digest is not None
    assert report.proposal_receipt_digest is not None
    assert report.pre_query_freeze_digest is None
    assert report.neutral_extraction_replays == 12


def test_external_blob_map_must_cover_exactly_released_panels(tmp_path: Path) -> None:
    plan, result, _calibration, record = _run(
        tmp_path,
        margin=1_000_000.0,
        payload=_proposal_payload(),
    )
    assert result.status is EpisodeStatus.SUPPORT_REJECTED
    blobs = _blob_bytes(plan, include_queries=False)
    blobs.pop(next(iter(blobs)))
    with pytest.raises(PrototypeRunVerificationError, match="missing="):
        verify_prototype_run_data(record, blob_bytes_by_id=blobs)


def test_independently_resealed_semantic_mutations_are_rejected(
    tmp_path: Path,
) -> None:
    plan, result, _calibration, record = _run(
        tmp_path,
        margin=1e-6,
        payload=_proposal_payload(),
    )
    assert result.status is EpisodeStatus.COMPLETE
    blobs = _blob_bytes(plan, include_queries=True)

    mutations: list[tuple[str, dict[str, Any], str]] = []

    # A second internally valid calibration record cannot silently change the
    # selected margins committed by the episode's prototype policy.
    source_root = plan._support_sources[0].path.parents[4]  # noqa: SLF001
    corpus = ShapeBongardCorpus.from_root(source_root)
    other_calibration = calibrate_prototype_margins(
        corpus,
        [plan.task_id],
        seed="prototype-verification-calibration",
        candidate_margins=[0.5],
    )
    changed = copy.deepcopy(record)
    changed["calibration"] = other_calibration.to_data()
    _reseal_outer(changed)
    mutations.append(("calibration-policy", changed, "calibration and policy differ"))

    # Forge one internally self-consistent archived feature measurement, refit
    # every support-derived object, and reseal it.  Only a fresh extraction of
    # the exact embedded PNG exposes that the numeric interval was fabricated.
    prequery = PrototypePreQueryFreeze.from_committed_data(
        record["prototype"]["pre_query_commitment"],
        support_commitment=plan.support,
    )
    original = prequery.positive_support[0]
    packet = original.feature_packet
    commitment = original.receipt.packet_commitment
    assert packet is not None and commitment is not None
    altered_interval = replace(
        commitment.values[0],
        lower=commitment.values[0].lower + 0.125,
        upper=commitment.values[0].upper + 0.125,
    )
    altered_values = (altered_interval, *commitment.values[1:])
    altered_commitment = replace(commitment, values=altered_values)
    altered_receipt = replace(
        original.receipt,
        packet_commitment=altered_commitment,
        packet_commitment_digest=altered_commitment.digest(),
    )
    altered_packet = replace(
        packet,
        extractor_receipt_digest=altered_receipt.digest(),
        values=altered_values,
    )
    altered_extraction = FeatureExtractionPreimage(
        original.panel_bytes,
        altered_receipt,
        altered_packet,
    )
    altered_positive = tuple(
        altered_extraction if item == original else item
        for item in prequery.positive_support
    )
    altered_prototypes = fit_support_prototypes(
        prequery.fit_plan,
        prequery.feature_space,
        tuple(item.require_present() for item in altered_positive),
        tuple(item.require_present() for item in prequery.negative_support),
        expected_plan_digest=prequery.fit_plan_digest,
    )
    altered_formula = PositivePrototypeFormula(
        claim=prequery.positive_formula.claim,
        feature_space_digest=prequery.feature_space_digest,
        prototype_digest=altered_prototypes.digest(),
        support_assignment_digest=prequery.support_assignment_digest,
        decision_margin=prequery.fixed_decision_margin,
    )
    altered_prequery = PrototypePreQueryFreeze.create(
        support_commitment=plan.support,
        policy=prequery.policy,
        selected_feature_group_id=prequery.selected_feature_group_id,
        feature_space=prequery.feature_space,
        positive_support=altered_positive,
        negative_support=prequery.negative_support,
        fit_plan=prequery.fit_plan,
        prototypes=altered_prototypes,
        positive_formula=altered_formula,
        semantic_proposal_digest=prequery.semantic_proposal_digest,
    )
    changed = copy.deepcopy(record)
    changed["prototype"]["pre_query_commitment"] = (
        altered_prequery.committed_data()
    )
    _reseal_outer(changed)
    mutations.append(
        (
            "fresh-feature-replay",
            changed,
            "archived neutral extraction differs from fresh Python replay",
        )
    )

    changed = copy.deepcopy(record)
    gate = changed["prototype"]["support_gate"]
    gate["ordered_entries"][0]["evidence"]["provenance"]["method"] += ".forged"
    gate["gate_digest"] = canonical_digest(
        {key: value for key, value in gate.items() if key != "gate_digest"}
    )
    _reseal_outer(changed)
    mutations.append(
        ("support-gate-evidence", changed, "differs from prototype replay")
    )

    changed = copy.deepcopy(record)
    observations = changed["prototype"]["observations"]
    query = observations[sorted(observations)[0]]
    query["evidence"]["provenance"]["method"] += ".forged"
    query["evidence_digest"] = canonical_digest(query["evidence"])
    _reseal_outer(changed)
    mutations.append(("query-evidence", changed, "query artifact .* is invalid"))

    changed = copy.deepcopy(record)
    archive = changed["run_archive"]
    formula = archive["proposal_freeze"]["formula"]
    formula["call"]["leg"]["name"] = "forged_support_prototype_match"
    archive["proposal_freeze"]["formula_digest"] = canonical_digest(formula)
    _reseal_archive(archive)
    _reseal_outer(changed)
    mutations.append(("generic-formula", changed, "generic run archive is invalid"))

    changed = copy.deepcopy(record)
    archive = changed["run_archive"]
    archive["attachment_contract"]["boundary_types"][0][1]["name"] = (
        "forged_features"
    )
    _reseal_archive(archive)
    _reseal_outer(changed)
    mutations.append(("generic-attachment", changed, "generic run archive is invalid"))

    for name, mutated, expected in mutations:
        with pytest.raises(
            PrototypeRunVerificationError,
            match=expected,
        ) as caught:
            verify_prototype_run_data(mutated, blob_bytes_by_id=blobs)
        assert str(caught.value), name
