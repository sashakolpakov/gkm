from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest, canonical_json
from bongard.corpus import BongardTask, ShapeBongardCorpus, SplitIndex
from bongard.exposure import ExposureLedger
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_CANDIDATE_SCHEMA_V1,
    CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1,
    CAMPAIGN_SELECTION_ALGORITHM_V1,
    DIRECT_ONLY,
    SOFT_ACCEPTED,
    TRANSPORT_FAILED,
    TYPED_REJECTED,
    SemanticCalibrationCampaignError,
    SemanticCalibrationCampaignFitFailed,
    SemanticCalibrationCampaignNoSoftClaims,
    SemanticCalibrationCampaignProposalPhaseFailed,
    SemanticCalibrationCampaignScoringFailed,
    SemanticCalibrationCandidate,
    SemanticCalibrationProposalArchive,
    SemanticCalibrationScoreBatch,
    resolve_semantic_campaign_panels,
    run_semantic_calibration_campaign,
    select_semantic_calibration_tasks,
    semantic_generator_cluster_id,
    verify_semantic_campaign_against_corpus,
)
from bongard.semantic_protocol import build_prospective_soft_scorer_protocol
from bongard.tests.test_semantic_observation import _receipt as _scorer_receipt
from bongard.tests.test_typed_visual_transport import _receipt as _proposer_receipt
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)
from bongard.typed_visual_proposal import PANEL_DESCRIPTION_KEYS


SOURCE_MANIFEST = "sha256:" + hashlib.sha256(
    b"trusted synthetic full release"
).hexdigest()
SPLIT_SOURCE = "sha256:" + hashlib.sha256(
    b"authenticated synthetic split"
).hexdigest()


_DRILL_IDS = (
    "bd_trapez_parallelogram_0000",
    "bd_thin_parallel_bridge_0000",
    "bd_open_uneven_band_four_arcs3_0000",
    "bd_open_uneven_band_four_arcs2_0000",
    "bd_asymmetric_arrow_0000",
    "hd_has_seven_straight_lines-exist_triangle_0000",
    "hd_exist_regular-exist_quadrangle_0000",
    "hd_has_six_straight_lines-exist_triangle_0000",
    "hd_has_obtuse_angle-exist_sector_0000",
)
_SEALED_TEST_ID = "hd_balanced_two-exist_quadrangle_0000"


def _metadata_only_corpus() -> ShapeBongardCorpus:
    task_ids = _DRILL_IDS + (_SEALED_TEST_ID,)
    tasks = tuple(
        BongardTask(
            task_id=task_id,
            family=task_id[:2],
            root=Path("/must-not-open") / task_id,
            positive=(),
            negative=(),
        )
        for task_id in task_ids
    )
    return ShapeBongardCorpus(
        Path("/must-not-open"),
        tasks,
        layout="archive",
        split=SplitIndex(
            groups=(
                ("test_hd_comb", (_SEALED_TEST_ID,)),
                ("train", tuple(sorted(_DRILL_IDS))),
            ),
            source_digest=SPLIT_SOURCE,
        ),
    )


def _as_legacy_v1_proposal_archive(
    archive: SemanticCalibrationProposalArchive,
) -> dict[str, Any]:
    data = deepcopy(archive.to_data())
    for record in data["records"]:
        candidate = record["candidate"]
        candidate["schema"] = CAMPAIGN_CANDIDATE_SCHEMA_V1
        candidate["selection_algorithm"] = CAMPAIGN_SELECTION_ALGORITHM_V1
        candidate["candidate_digest"] = canonical_digest(
            {
                key: value
                for key, value in candidate.items()
                if key != "candidate_digest"
            }
        )
        record["candidate_digest"] = candidate["candidate_digest"]
        record["record_digest"] = canonical_digest(
            {
                key: value
                for key, value in record.items()
                if key != "record_digest"
            }
        )
    data["schema"] = CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1
    data["selection_algorithm"] = CAMPAIGN_SELECTION_ALGORITHM_V1
    data["dependence_unit_semantics"] = (
        "bd-morphology-cluster-and-hd-exact-ordered-pair/v1"
    )
    data["hd_cross_pair_independence_status"] = (
        "assumed-for-fit-not-established-by-corpus-metadata/v1"
    )
    data["proposal_archive_digest"] = canonical_digest(
        {
            key: value
            for key, value in data.items()
            if key != "proposal_archive_digest"
        }
    )
    return data


def test_selection_is_metadata_only_deterministic_and_morphology_independent() -> None:
    corpus = _metadata_only_corpus()
    already_exposed = _DRILL_IDS[0]
    ledger = ExposureLedger.create(SOURCE_MANIFEST).record(
        phase="prior-analysis",
        actor="fixture",
        purpose="exclude one exact task",
        task_ids=(already_exposed,),
        observed_at="2026-01-01T00:00:00Z",
        known_task_ids=corpus.task_ids,
    )

    selected = select_semantic_calibration_tasks(
        corpus,
        candidate_count=6,
        seed="public-selection-seed",
        exposure_ledger=ledger,
        expected_exposure_ledger_digest=ledger.digest,
    )
    repeated = select_semantic_calibration_tasks(
        corpus,
        candidate_count=6,
        seed="public-selection-seed",
        exposure_ledger=ledger,
        expected_exposure_ledger_digest=ledger.digest,
    )

    assert selected == repeated
    assert already_exposed not in {item.task_id for item in selected}
    assert _SEALED_TEST_ID not in {item.task_id for item in selected}
    clusters = tuple(
        semantic_generator_cluster_id(item.family, item.concepts)
        for item in selected
    )
    assert len(clusters) == len(set(clusters)) == 6
    morphology_siblings = {
        "bd_open_uneven_band_four_arcs3_0000",
        "bd_open_uneven_band_four_arcs2_0000",
    }
    assert len(morphology_siblings & {item.task_id for item in selected}) <= 1

    with pytest.raises(
        SemanticCalibrationCampaignError,
        match="precommitted digest",
    ):
        select_semantic_calibration_tasks(
            corpus,
            candidate_count=1,
            seed="public-selection-seed",
            exposure_ledger=ledger,
            expected_exposure_ledger_digest="sha256:" + "0" * 64,
        )


def test_hd_selection_rejects_constituent_reuse_within_batch_and_prior_ledger() -> None:
    corpus = _metadata_only_corpus()
    exposed = "hd_has_seven_straight_lines-exist_triangle_0000"
    ledger = ExposureLedger.create(SOURCE_MANIFEST).record(
        phase="prior-analysis",
        actor="fixture",
        purpose="adversarial constituent exposure",
        task_ids=(exposed,),
        observed_at="2026-01-01T00:00:00Z",
        known_task_ids=corpus.task_ids,
    )

    selected = select_semantic_calibration_tasks(
        corpus,
        candidate_count=2,
        seed="constituent-disjoint-seed",
        exposure_ledger=ledger,
        expected_exposure_ledger_digest=ledger.digest,
        families=("hd",),
    )
    attributes = [concept for item in selected for concept in item.concepts]
    assert len(attributes) == len(set(attributes))
    assert "exist_triangle" not in attributes
    assert exposed not in {item.task_id for item in selected}

    with pytest.raises(
        SemanticCalibrationCampaignError,
        match="permit only 2",
    ):
        select_semantic_calibration_tasks(
            corpus,
            candidate_count=3,
            seed="constituent-disjoint-seed",
            exposure_ledger=ledger,
            expected_exposure_ledger_digest=ledger.digest,
            families=("hd",),
        )


_CAMPAIGN_DRILL_IDS = (
    "bd_trapez_parallelogram_0000",
    "bd_thin_parallel_bridge_0000",
    "bd_asymmetric_arrow_0000",
    "hd_has_seven_straight_lines-exist_triangle_0000",
    "hd_exist_regular-exist_quadrangle_0000",
    "hd_has_six_straight_lines-exist_triangle_0000",
    "hd_has_obtuse_angle-exist_sector_0000",
)
_LAUNCHER_DIGEST = "b" * 64


def _draw_unique_panel(
    path: Path,
    *,
    task_index: int,
    positive: bool,
    panel_index: int,
) -> None:
    image = Image.new(
        "L",
        (38 + task_index, 38 + panel_index),
        color=255,
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            3,
            4,
            10 + panel_index + (2 if positive else 0),
            13 + task_index + (1 if positive else 0),
        ),
        fill=0,
    )
    image.save(path, format="PNG")


def _campaign_corpus(tmp_path: Path):
    root = tmp_path / "ShapeBongard_V2"
    task_ids = _CAMPAIGN_DRILL_IDS + (_SEALED_TEST_ID,)
    for task_index, task_id in enumerate(task_ids):
        family = task_id[:2]
        for label, positive in (("1", True), ("0", False)):
            directory = root / family / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for panel_index in range(7):
                _draw_unique_panel(
                    directory / f"{panel_index}.png",
                    task_index=task_index,
                    positive=positive,
                    panel_index=panel_index,
                )
    split_path = root / "ShapeBongard_V2_split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": list(_CAMPAIGN_DRILL_IDS),
                "test_hd_comb": [_SEALED_TEST_ID],
            }
        ),
        encoding="utf-8",
    )
    corpus = ShapeBongardCorpus.from_root(root)
    full_manifest = corpus.build_manifest()
    test_paths = corpus.task(_SEALED_TEST_ID).panels
    for path in test_paths:
        path.write_bytes(b"deliberately invalid sealed test bytes")
    return corpus, full_manifest, test_paths


def _proposal_payload(kind: str, variant: int = 0) -> dict[str, object]:
    base: dict[str, object] = {
        "positive_description": "a compact articulated ink arrangement",
        "panel_descriptions": {
            name: "a compact literal ink arrangement is clearly visible"
            for name in PANEL_DESCRIPTION_KEYS
        },
        "view": "carrier_shape",
    }
    if kind == "soft":
        motif = ("avian", "crystalline", "braided", "radial")[variant]
        return {
            **base,
            "deterministic_atoms": [],
            "soft_claim": {
                "positive_description": (
                    f"a {motif} articulated ink silhouette"
                ),
                "cue_descriptions": [
                    f"a compact {motif} body has lateral extensions",
                    f"the {motif} outer contour has an oblique angular rhythm",
                ],
            },
            "formula": {"kind": "all", "atom_indices": [0]},
        }
    if kind == "direct":
        return {
            **base,
            "deterministic_atoms": [
                {
                    "catalog_key": "component.count",
                    "comparison": "equal",
                    "arguments": {"target_count": 1},
                }
            ],
            "soft_claim": None,
            "formula": {"kind": "all", "atom_indices": [0]},
        }
    if kind == "rejected":
        return {
            **base,
            "deterministic_atoms": [],
            "soft_claim": None,
            "formula": {"kind": "all", "atom_indices": []},
        }
    raise ValueError(kind)


def _unique_receipt(receipt: CodexReceipt, ordinal: int) -> CodexReceipt:
    data = receipt.to_dict()
    data["thread_id"] = f"00000000-0000-4000-8000-{ordinal + 1:012d}"
    data["event_stream_digest"] = hashlib.sha256(
        f"fixture-event-stream-{ordinal}".encode("utf-8")
    ).hexdigest()
    data.pop("receipt_digest")
    data["receipt_digest"] = canonical_digest(data)
    return CodexReceipt(
        **{
            **data,
            "event_types": tuple(data["event_types"]),
            "item_types": tuple(data["item_types"]),
        }
    )


def _campaign_protocol():
    return build_prospective_soft_scorer_protocol(
        proposer_model_id="fixture-proposer",
        proposer_reasoning_effort="medium",
        scorer_model_id="fixture-scorer",
        scorer_reasoning_effort="medium",
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=2,
    )


def test_full_campaign_attrition_phase_barriers_and_corpus_cold_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, full_manifest, corrupted_test_paths = _campaign_corpus(tmp_path)
    protocol = _campaign_protocol()
    ledger = ExposureLedger.create(full_manifest.digest)
    events: list[str] = []
    proposer_calls = 0
    scorer_calls = 0

    def on_exposure_precommit(successor: ExposureLedger) -> None:
        assert proposer_calls == scorer_calls == 0
        assert len(successor.events) == 6
        events.append("exposure-precommitted")

    def proposer_transport(prompt, paths, schema, **kwargs):
        nonlocal proposer_calls
        call_index = proposer_calls
        proposer_calls += 1
        events.append(f"proposer-{call_index}")
        assert len(paths) == 12
        assert tuple(Path(path).name for path in paths) == tuple(
            [f"pos_{index}.png" for index in range(6)]
            + [f"neg_{index}.png" for index in range(6)]
        )
        kind = "soft" if call_index < 4 else (
            "direct" if call_index == 4 else "rejected"
        )
        payload = _proposal_payload(kind, call_index)
        receipt = _proposer_receipt(
            prompt,
            paths,
            schema,
            payload,
            model=kwargs["model"],
            effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(
            payload=payload,
            receipt=_unique_receipt(receipt, call_index),
        )

    def scorer_transport(
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **kwargs,
    ):
        nonlocal scorer_calls
        call_index = scorer_calls
        scorer_calls += 1
        events.append(f"scorer-{call_index}")
        assert proposer_calls == 6
        assert tuple(names) == ("query.png",)
        assert tuple(Path(path).name for path in paths) == ("query.png",)
        cue_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["cue_id"]["enum"]
        )
        witness_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["witness_ids"]["items"]["enum"]
        )
        judgment = "supported" if call_index < 2 else "unsupported"
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue_id,
                    "judgment": judgment,
                    "witness_ids": (
                        [witness_ids[0]] if judgment == "supported" else []
                    ),
                }
                for cue_id in cue_ids
            ]
        }
        receipt = _scorer_receipt(prompt, paths, names, schema, payload)
        return CodexStructuredResult(
            payload=payload,
            receipt=_unique_receipt(receipt, 100 + call_index),
        )

    campaign = run_semantic_calibration_campaign(
        corpus,
        protocol,
        candidate_count=6,
        seed="end-to-end-public-seed",
        source_corpus_manifest_digest=full_manifest.digest,
        expected_codex_launcher_digest=_LAUNCHER_DIGEST,
        exposure_ledger=ledger,
        expected_exposure_ledger_digest=ledger.digest,
        label_nonce_root=hashlib.sha256(b"fixture label nonce root").hexdigest(),
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        proposer_transport=proposer_transport,
        scorer_transport=scorer_transport,
        on_exposure_precommit=on_exposure_precommit,
    )

    archive = campaign.score_batch.commitment_batch.proposal_archive
    assert proposer_calls == 6
    assert scorer_calls == 4
    assert events == [
        "exposure-precommitted",
        *(f"proposer-{index}" for index in range(6)),
        *(f"scorer-{index}" for index in range(4)),
    ]
    assert tuple(item.status for item in archive.records) == (
        SOFT_ACCEPTED,
        SOFT_ACCEPTED,
        SOFT_ACCEPTED,
        SOFT_ACCEPTED,
        DIRECT_ONLY,
        TYPED_REJECTED,
    )
    assert archive.selection_seed == "end-to-end-public-seed"
    assert archive.candidate_count == 6
    assert archive.execution_config.proposer_max_workers == 1
    assert archive.execution_config.scorer_max_workers == 1
    assert archive.execution_config.expected_codex_launcher_digest == (
        _LAUNCHER_DIGEST
    )
    assert archive.execution_config.cloud_policy_cache_binding == "absent"
    assert archive.source_corpus_manifest_digest == full_manifest.digest
    assert archive.development_manifest_digest != full_manifest.digest
    assert archive.exposure_predecessor == ledger
    assert len(archive.exposure_successor.events) == 6
    legacy_archive_data = _as_legacy_v1_proposal_archive(archive)
    legacy_archive = SemanticCalibrationProposalArchive.from_data(
        legacy_archive_data,
        expected_digest=legacy_archive_data["proposal_archive_digest"],
    )
    assert legacy_archive.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM_V1
    assert all(
        isinstance(record.candidate, SemanticCalibrationCandidate)
        and record.candidate.selection_algorithm
        == CAMPAIGN_SELECTION_ALGORITHM_V1
        for record in legacy_archive.records
    )
    assert legacy_archive.to_data() == legacy_archive_data
    assert all(
        path.read_bytes().startswith(b"deliberately invalid")
        for path in corrupted_test_paths
    )

    commitment_batch = campaign.score_batch.commitment_batch
    assert len(commitment_batch.commitments) == 4
    assert len(campaign.label_reveals) == 4
    assert len(campaign.calibration.measurements) == 4
    assert campaign.to_data()["calibration_scope"] == (
        "stage-a-conditional-on-soft-claim-emission-not-support-gate-pass/v1"
    )
    label_free = canonical_json(campaign.score_batch.to_data()).decode("utf-8")
    assert '"affirmative_label"' not in label_free
    serialized = canonical_json(campaign.to_data()).decode("utf-8").lower()
    assert '"lean"' not in serialized
    assert "lean_backend" not in serialized
    assert "proof_backend" not in serialized

    panels = resolve_semantic_campaign_panels(
        campaign.to_data(),
        corpus=corpus,
        corpus_manifest=full_manifest,
    )
    assert set(panels) == {
        item.selection.observation_id for item in commitment_batch.commitments
    }
    campaign_data = campaign.to_data()
    score_batch_checks = 0
    score_batch_serializations = 0
    original_score_batch_check = SemanticCalibrationScoreBatch.assert_untampered
    original_score_batch_content = SemanticCalibrationScoreBatch.content_data

    def counted_score_batch_check(self) -> None:
        nonlocal score_batch_checks
        score_batch_checks += 1
        original_score_batch_check(self)

    def counted_score_batch_content(self):
        nonlocal score_batch_serializations
        score_batch_serializations += 1
        return original_score_batch_content(self)

    monkeypatch.setattr(
        SemanticCalibrationScoreBatch,
        "assert_untampered",
        counted_score_batch_check,
    )
    monkeypatch.setattr(
        SemanticCalibrationScoreBatch,
        "content_data",
        counted_score_batch_content,
    )
    for record, reveal in zip(
        commitment_batch.proposal_archive.soft_records,
        campaign.label_reveals,
        strict=True,
    ):
        reveal._assert_matches_verified_parents(record.candidate, campaign.score_batch)
    # A reveal join uses the batch identity established at its verified parent
    # boundary; it must not serialize the complete shared batch itself.
    assert score_batch_serializations == 0
    verified, verified_panels = verify_semantic_campaign_against_corpus(
        campaign_data,
        corpus=corpus,
        corpus_manifest=full_manifest,
    )
    # The shared batch is checked once at the campaign boundary, not once per
    # label reveal.  Its complete content is serialized a fixed number of
    # times by the enclosing cold decoder, never once per reveal.  This removes
    # the per-reveal multiplier; the legacy nested batch schema itself remains
    # quadratic in the number of soft claims.
    assert score_batch_checks == 1
    assert score_batch_serializations < 20
    assert verified.digest == campaign.digest
    assert verified.calibration.family.digest() == (
        campaign.calibration.family.digest()
    )
    assert verified_panels == panels

    flipped = deepcopy(campaign.to_data())
    flipped_label = flipped["label_reveals"][0]["labels"][0]
    flipped_label["positive"] = not flipped_label["positive"]
    with pytest.raises(Exception):
        verify_semantic_campaign_against_corpus(
            flipped,
            corpus=corpus,
            corpus_manifest=full_manifest,
        )

    swapped = deepcopy(campaign.to_data())
    swapped_queries = swapped["score_batch"]["commitment_batch"][
        "proposal_archive"
    ]["records"][0]["candidate"]["query_panels"]
    swapped_queries.reverse()
    with pytest.raises(Exception):
        verify_semantic_campaign_against_corpus(
            swapped,
            corpus=corpus,
            corpus_manifest=full_manifest,
        )


@pytest.mark.parametrize("failure_mode", ("transport", "no-soft"))
def test_proposal_terminal_failures_retain_complete_archive(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    corpus, full_manifest, _ = _campaign_corpus(tmp_path / failure_mode)
    protocol = _campaign_protocol()
    ledger = ExposureLedger.create(full_manifest.digest)
    calls = 0

    def proposer_transport(prompt, paths, schema, **kwargs):
        nonlocal calls
        call_index = calls
        calls += 1
        if failure_mode == "transport" and call_index == 2:
            raise RuntimeError("synthetic proposer transport failure")
        kind = (
            "soft"
            if failure_mode == "transport"
            else ("rejected" if call_index == 5 else "direct")
        )
        payload = _proposal_payload(kind, call_index % 4)
        receipt = _proposer_receipt(
            prompt,
            paths,
            schema,
            payload,
            model=kwargs["model"],
            effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(
            payload=payload,
            receipt=_unique_receipt(receipt, 200 + call_index),
        )

    expected_error = (
        SemanticCalibrationCampaignProposalPhaseFailed
        if failure_mode == "transport"
        else SemanticCalibrationCampaignNoSoftClaims
    )
    with pytest.raises(expected_error) as caught:
        run_semantic_calibration_campaign(
            corpus,
            protocol,
            candidate_count=6,
            seed=f"proposal-terminal-{failure_mode}",
            source_corpus_manifest_digest=full_manifest.digest,
            expected_codex_launcher_digest=_LAUNCHER_DIGEST,
            exposure_ledger=ledger,
            expected_exposure_ledger_digest=ledger.digest,
            label_nonce_root=hashlib.sha256(
                f"label-{failure_mode}".encode("utf-8")
            ).hexdigest(),
            cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
            proposer_transport=proposer_transport,
            scorer_transport=lambda *args, **kwargs: pytest.fail(
                "proposal terminal failure reached scorer"
            ),
        )

    failure = caught.value
    archive = failure.proposal_archive
    assert calls == 6
    assert len(archive.records) == 6
    assert len(archive.exposure_successor.events) == 6
    assert archive.exposure_predecessor == ledger
    assert failure.to_data()["failure_digest"] == failure.digest
    if failure_mode == "transport":
        assert tuple(item.status for item in archive.records).count(
            TRANSPORT_FAILED
        ) == 1
        assert failure.to_data()["family_fit_authorized"] is False
    else:
        assert not archive.soft_records


@pytest.mark.parametrize("failure_mode", ("scorer", "sparse-fit"))
def test_post_proposal_failures_retain_scores_reveals_and_never_impute_zero(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    corpus, full_manifest, _ = _campaign_corpus(tmp_path / failure_mode)
    protocol = _campaign_protocol()
    ledger = ExposureLedger.create(full_manifest.digest)
    proposer_calls = 0
    scorer_calls = 0

    def proposer_transport(prompt, paths, schema, **kwargs):
        nonlocal proposer_calls
        call_index = proposer_calls
        proposer_calls += 1
        payload = _proposal_payload("soft", call_index)
        receipt = _proposer_receipt(
            prompt,
            paths,
            schema,
            payload,
            model=kwargs["model"],
            effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(
            payload=payload,
            receipt=_unique_receipt(receipt, 300 + call_index),
        )

    def scorer_transport(prompt, paths, names, schema, **kwargs):
        nonlocal scorer_calls
        call_index = scorer_calls
        scorer_calls += 1
        if failure_mode == "scorer" and call_index == 1:
            raise RuntimeError("synthetic blind scorer failure")
        cue_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["cue_id"]["enum"]
        )
        witness_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["witness_ids"]["items"]["enum"]
        )
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue_id,
                    "judgment": "supported",
                    "witness_ids": [witness_ids[0]],
                }
                for cue_id in cue_ids
            ]
        }
        receipt = _scorer_receipt(prompt, paths, names, schema, payload)
        return CodexStructuredResult(
            payload=payload,
            receipt=_unique_receipt(receipt, 400 + call_index),
        )

    expected_error = (
        SemanticCalibrationCampaignScoringFailed
        if failure_mode == "scorer"
        else SemanticCalibrationCampaignFitFailed
    )
    with pytest.raises(expected_error) as caught:
        run_semantic_calibration_campaign(
            corpus,
            protocol,
            candidate_count=4,
            seed=f"post-proposal-{failure_mode}",
            source_corpus_manifest_digest=full_manifest.digest,
            expected_codex_launcher_digest=_LAUNCHER_DIGEST,
            exposure_ledger=ledger,
            expected_exposure_ledger_digest=ledger.digest,
            label_nonce_root=hashlib.sha256(
                f"post-label-{failure_mode}".encode("utf-8")
            ).hexdigest(),
            cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
            proposer_transport=proposer_transport,
            scorer_transport=scorer_transport,
        )

    failure = caught.value
    assert proposer_calls == 4
    assert scorer_calls == 4
    assert len(failure.score_batch.attempts) == 4
    assert len(
        failure.score_batch.commitment_batch.proposal_archive.exposure_successor.events
    ) == 4
    assert failure.to_data()["failure_digest"] == failure.digest
    if failure_mode == "scorer":
        failed_records = tuple(
            item.score_artifact.record
            for item in failure.score_batch.attempts
            if item.score_artifact.record.outcome != "present"
        )
        assert len(failed_records) == 1
        assert failed_records[0].score is None
        assert failure.to_data()["label_state"] == "withheld"
        assert '"affirmative_label"' not in canonical_json(
            failure.to_data()
        ).decode("utf-8")
    else:
        assert len(failure.label_reveals) == 4
        assert len(failure.measurements) == 4
