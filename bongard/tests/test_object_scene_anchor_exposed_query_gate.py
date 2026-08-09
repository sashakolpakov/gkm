from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import zlib

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_scene_anchor_exposed_query_gate import (
    EXPECTED_EXPOSED_QUERY_ORDINALS,
    EXPOSED_QUERY_PANEL_COUNT,
    ObjectSceneAnchorExposedQueryGateError,
    ObjectSceneAnchorExposedQueryPlan,
    ObjectSceneAnchorExposedQueryRelease,
    ObjectSceneAnchorExposedQueryScore,
    ObjectSceneAnchorHistoricalMetadata,
    ObjectSceneAnchorPredicateDurableFreezeCommitment,
    ObjectSceneAnchorQueryPrediction,
    ObjectSceneAnchorQueryReleasedRecordLocator,
    bind_caller_durable_object_scene_anchor_python_predicate,
    build_object_scene_anchor_exposed_query_plan,
    build_object_scene_anchor_historical_metadata,
    release_object_scene_anchor_exposed_queries,
    released_record_directory_loader,
    reveal_and_score_object_scene_anchor_exposed_queries,
    verify_object_scene_anchor_exposed_query_plan,
    verify_object_scene_anchor_exposed_query_release,
    verify_object_scene_anchor_exposed_query_score,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorSelectionCommitment,
    freeze_object_scene_anchor_python_predicate,
)
from bongard.object_scene_anchor_version_space import ObjectSceneAnchorOrientation
from bongard.official_panel_archive import (
    OfficialPanelReceipt,
    ReleasedOfficialPanel,
    _released_panel_content,
)
from bongard.tests.test_object_scene_anchor_python_predicate import _version_fixture


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _address(text: str) -> str:
    return "sha256:" + _sha(text)


def _png(index: int) -> bytes:
    image = Image.new("RGB", (48 + index, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.line((6, 30, 18, 7, 31, 30, 6, 30), fill="black", width=2)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _seal_record(
    panel_id: str,
    payload: bytes,
) -> ReleasedOfficialPanel:
    _, tail = panel_id.split("/", 1)
    receipt = OfficialPanelReceipt.seal(
        panel_id=panel_id,
        payload=payload,
        archive_member=f"ShapeBongard_V2/bd/images/{tail}",
        zip_crc32=zlib.crc32(payload),
        release_descriptor_digest=_address("release-descriptor"),
        archive_digest=_address("archive"),
        central_directory_digest=_address("central-directory"),
    )
    values = {
        "panel_id": panel_id,
        "exact_png_bytes": payload,
        "exact_png_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "release_receipt": receipt,
        "execution_precommit_digest": _address("execution-precommit"),
        "exposure_successor_digest": _address("exposure-successor"),
    }
    provisional = object.__new__(ReleasedOfficialPanel)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ReleasedOfficialPanel(
        **values,
        record_digest="sha256:" + canonical_digest(
            _released_panel_content(provisional)
        ),
    )


@pytest.fixture(scope="module")
def historical_fixture():
    payloads: dict[int, bytes] = {}
    plan_scenes = []
    runtime_scenes = []
    for ordinal in range(28):
        payload = _png(ordinal)
        payloads[ordinal] = payload
        task_id = f"bd_task_{ordinal:02d}"
        panel_id = f"bd/{task_id}/0/{ordinal % 7}.png"
        states = (
            ("present", "absent")
            if ordinal < 14
            else ("absent", "present")
        )
        plan_scenes.append(
            {
                "schema": "gkm.bongard-prototype-scene-calibration-scene.v1",
                "ordinal": ordinal,
                "task_id": task_id,
                "panel_id": panel_id,
                "record_digest": _address(f"plan-scene-{ordinal}"),
                "expected_tag_states": [
                    {"tag_id": "opaque_visual_tag_0", "state": states[0]},
                    {"tag_id": "opaque_visual_tag_1", "state": states[1]},
                ],
            }
        )
        runtime_scenes.append(
            {
                "schema": "gkm.bongard-prototype-scene-runtime-archive-entry.v1",
                "panel_id": panel_id,
                "scene_task_id": task_id,
                "scene_png_byte_count": len(payload),
                "scene_png_sha256": hashlib.sha256(payload).hexdigest(),
                "observation_context_digest": "",
            }
        )
    plan = {
        "schema": "gkm.bongard-prototype-scene-calibration-plan.v1",
        "scenes": plan_scenes,
    }
    plan["record_digest"] = "sha256:" + canonical_digest(plan)
    for item in runtime_scenes:
        item["observation_context_digest"] = plan["record_digest"]
    runtime = {
        "schema": "gkm.bongard-prototype-scene-runtime-archive.v1",
        "scenes": runtime_scenes,
    }
    runtime["record_digest"] = "sha256:" + canonical_digest(runtime)
    metadata = build_object_scene_anchor_historical_metadata(
        plan=plan,
        plan_file_sha256=hashlib.sha256(
            canonical_json(plan) + b"\n"
        ).hexdigest(),
        runtime_archive=runtime,
        runtime_archive_file_sha256=hashlib.sha256(
            canonical_json(runtime) + b"\n"
        ).hexdigest(),
    )
    return metadata, payloads


@pytest.fixture(scope="module")
def predicate_fixture():
    version, _, _ = _version_fixture(
        lambda *_: Disposition.CERTIFIED_ABSENT
    )
    selected = next(
        item for item in version.candidates if len(item.witness_digests) == 2
    )
    selection = ObjectSceneAnchorSelectionCommitment.create(
        version,
        selected_candidate_digest=selected.candidate_digest,
        selection_kind="external_exact_selection",
        selector_record_digest=_sha("query-gate-selector"),
    )
    predicate = freeze_object_scene_anchor_python_predicate(version, selection)
    assert (
        predicate.candidate.orientation
        is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    predicate_payload = canonical_json(predicate.to_data())
    durable = bind_caller_durable_object_scene_anchor_python_predicate(
        predicate,
        persisted_predicate_payload_sha256=hashlib.sha256(
            predicate_payload
        ).hexdigest(),
        persisted_predicate_payload_byte_count=len(predicate_payload),
        persistence_receipt_digest=_sha("durable-persistence-receipt"),
    )
    return predicate, durable


@pytest.fixture(scope="module")
def release_fixture(historical_fixture, predicate_fixture):
    metadata, payloads = historical_fixture
    predicate, durable = predicate_fixture
    plan = build_object_scene_anchor_exposed_query_plan(metadata)
    records = {}
    locators = []
    for item in plan.items:
        released = _seal_record(item.panel_id, payloads[item.ordinal])
        raw = released.to_data()
        file_bytes = canonical_json(raw) + b"\n"
        file_sha = hashlib.sha256(file_bytes).hexdigest()
        locator = ObjectSceneAnchorQueryReleasedRecordLocator.create(
            item.query_alias,
            released_record_file_sha256=file_sha,
            released_record_digest=released.record_digest,
        )
        records[file_sha] = raw
        locators.append(locator)
    calls = []

    def loader(locator):
        calls.append(locator.query_alias)
        return deepcopy(records[locator.released_record_file_sha256])

    bundle = release_object_scene_anchor_exposed_queries(
        plan=plan,
        predicate=predicate,
        durable_freeze=durable,
        expected_durable_freeze_commitment_digest=durable.commitment_digest,
        locators=tuple(locators),
        load_released_record=loader,
    )
    assert calls == [f"query_{index:03d}" for index in range(16)]
    return metadata, plan, predicate, durable, tuple(locators), records, bundle


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_metadata_plan_is_exact_set_difference_without_pixels_or_labels(
    historical_fixture,
) -> None:
    metadata, _ = historical_fixture
    plan = build_object_scene_anchor_exposed_query_plan(metadata)
    data = plan.to_data()

    assert ObjectSceneAnchorExposedQueryPlan.from_data(data) == plan
    assert verify_object_scene_anchor_exposed_query_plan(plan, metadata) == plan
    assert plan.all_plan_ordinals == tuple(range(28))
    assert plan.derived_query_ordinals == EXPECTED_EXPOSED_QUERY_ORDINALS
    assert plan.derived_query_ordinals == tuple(
        ordinal
        for ordinal in plan.all_plan_ordinals
        if ordinal not in set(plan.support_ordinals)
    )
    assert plan.query_count == EXPOSED_QUERY_PANEL_COUNT == 16
    assert tuple(item.query_alias for item in plan.items) == tuple(
        f"query_{index:03d}" for index in range(16)
    )
    assert not any("state" in key or "disposition" in key for key in _all_keys(data))
    assert not any(isinstance(item, bytes) for item in data.values())
    assert data["historical_pixels_previously_exposed"] is True
    assert data["engineering_ablation_only"] is True
    assert data["blindness_claimed"] is False


def test_bad_durable_commitment_and_incomplete_locators_open_zero_records(
    historical_fixture,
    predicate_fixture,
) -> None:
    metadata, _ = historical_fixture
    predicate, durable = predicate_fixture
    plan = build_object_scene_anchor_exposed_query_plan(metadata)
    calls = []

    with pytest.raises(ObjectSceneAnchorExposedQueryGateError, match="durable"):
        release_object_scene_anchor_exposed_queries(
            plan=plan,
            predicate=predicate,
            durable_freeze=durable,
            expected_durable_freeze_commitment_digest=_sha("wrong-freeze"),
            locators=(),
            load_released_record=lambda locator: calls.append(locator),
        )
    assert calls == []

    with pytest.raises(ObjectSceneAnchorExposedQueryGateError, match="locator"):
        release_object_scene_anchor_exposed_queries(
            plan=plan,
            predicate=predicate,
            durable_freeze=durable,
            expected_durable_freeze_commitment_digest=durable.commitment_digest,
            locators=(),
            load_released_record=lambda locator: calls.append(locator),
        )
    assert calls == []


def test_release_opens_exactly_sixteen_matching_records_and_stays_neutral(
    release_fixture,
) -> None:
    _, plan, predicate, durable, _, _, bundle = release_fixture
    data = bundle.release.to_data()

    assert ObjectSceneAnchorExposedQueryRelease.from_data(data) == bundle.release
    assert bundle.release.query_plan_digest == plan.plan_digest
    assert bundle.release.predicate_digest == predicate.predicate_digest
    assert (
        bundle.release.durable_freeze_commitment_digest
        == durable.commitment_digest
    )
    assert bundle.release.exact_loader_call_count == 16
    assert data["labels_revealed"] is False
    assert not any("expected" in key for key in _all_keys(data))
    neutral = bundle.neutral_panel_inputs
    assert tuple(item.query_alias for item in neutral) == bundle.release.query_aliases
    assert all(
        set(item.__slots__)
        == {"query_alias", "exact_png_bytes", "png_sha256", "release_item_digest"}
        for item in neutral
    )
    assert all(not hasattr(item, "panel_id") for item in neutral)
    assert all(not hasattr(item, "ordinal") for item in neutral)


def test_release_cold_replay_is_model_free_and_exact(release_fixture) -> None:
    _, plan, predicate, durable, locators, records, bundle = release_fixture
    calls = []

    def loader(locator):
        calls.append(locator.query_alias)
        return deepcopy(records[locator.released_record_file_sha256])

    assert (
        verify_object_scene_anchor_exposed_query_release(
            bundle,
            plan=plan,
            predicate=predicate,
            durable_freeze=durable,
            expected_durable_freeze_commitment_digest=durable.commitment_digest,
            locators=locators,
            load_released_record=loader,
        )
        == bundle
    )
    assert calls == [f"query_{index:03d}" for index in range(16)]


def test_directory_loader_reads_only_explicit_synthetic_locator(tmp_path) -> None:
    payload = _png(1)
    panel = _seal_record("bd/bd_loader_test/0/0.png", payload)
    encoded = canonical_json(panel.to_data()) + b"\n"
    file_sha = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / f"{file_sha}.json"
    path.write_bytes(encoded)
    (tmp_path / ("f" * 64 + ".json")).write_text("not opened", encoding="utf-8")
    locator = ObjectSceneAnchorQueryReleasedRecordLocator.create(
        "query_000",
        released_record_file_sha256=file_sha,
        released_record_digest=panel.record_digest,
    )

    assert released_record_directory_loader(tmp_path)(locator) == panel.to_data()


def _predictions(release, predicate, *, include_all: bool = True):
    states = (
        (Disposition.PRESENT,) * 8
        + (Disposition.CERTIFIED_ABSENT,) * 8
    )
    rows = tuple(
        ObjectSceneAnchorQueryPrediction.create(
            query_alias=alias,
            query_release_digest=release.release_digest,
            predicate_digest=predicate.predicate_digest,
            disposition=state,
        )
        for alias, state in zip(release.query_aliases, states, strict=True)
    )
    return rows if include_all else rows[:-1]


def test_labels_remain_sealed_until_complete_prediction_digest_set(
    release_fixture,
) -> None:
    metadata, plan, predicate, _, _, _, bundle = release_fixture

    class PoisonLabel(str):
        def __eq__(self, other):
            raise AssertionError("expected label was accessed before predictions")

    poison_scenes = []
    for source in metadata.scenes:
        clone = object.__new__(type(source))
        for name in (
            "ordinal",
            "task_id",
            "panel_id",
            "plan_scene_record_digest",
            "png_byte_count",
            "png_sha256",
        ):
            object.__setattr__(clone, name, getattr(source, name))
        object.__setattr__(clone, "tag_0_state", PoisonLabel(source.tag_0_state))
        object.__setattr__(clone, "tag_1_state", PoisonLabel(source.tag_1_state))
        poison_scenes.append(clone)
    poison_metadata = ObjectSceneAnchorHistoricalMetadata(
        metadata.plan_file_sha256,
        metadata.plan_record_digest,
        metadata.runtime_archive_file_sha256,
        metadata.runtime_archive_record_digest,
        tuple(poison_scenes),
    )
    with pytest.raises(
        ObjectSceneAnchorExposedQueryGateError,
        match="labels remain sealed",
    ):
        reveal_and_score_object_scene_anchor_exposed_queries(
            metadata=poison_metadata,
            plan=plan,
            release=bundle.release,
            predicate=predicate,
            predictions=_predictions(
                bundle.release, predicate, include_all=False
            ),
        )


def test_post_prediction_reveal_scores_and_cold_replays(release_fixture) -> None:
    metadata, plan, predicate, _, _, _, bundle = release_fixture
    predictions = _predictions(bundle.release, predicate)
    score = reveal_and_score_object_scene_anchor_exposed_queries(
        metadata=metadata,
        plan=plan,
        release=bundle.release,
        predicate=predicate,
        predictions=predictions,
    )

    assert ObjectSceneAnchorExposedQueryScore.from_data(score.to_data()) == score
    assert score.prediction_digests == tuple(
        item.prediction_digest for item in predictions
    )
    assert score.query_count == score.determinate_count == score.correct_count == 16
    assert score.accuracy_ppm == 1_000_000
    assert score.to_data()[
        "all_prediction_digests_validated_before_label_access"
    ] is True
    assert (
        verify_object_scene_anchor_exposed_query_score(
            score,
            metadata=metadata,
            plan=plan,
            release=bundle.release,
            predicate=predicate,
            predictions=predictions,
        )
        == score
    )


def test_indeterminate_and_error_predictions_never_count_as_negative(
    release_fixture,
) -> None:
    metadata, plan, predicate, _, _, _, bundle = release_fixture
    predictions = list(_predictions(bundle.release, predicate))
    predictions[0] = ObjectSceneAnchorQueryPrediction.create(
        query_alias="query_000",
        query_release_digest=bundle.release.release_digest,
        predicate_digest=predicate.predicate_digest,
        disposition=Disposition.INDETERMINATE,
    )
    predictions[15] = ObjectSceneAnchorQueryPrediction.create(
        query_alias="query_015",
        query_release_digest=bundle.release.release_digest,
        predicate_digest=predicate.predicate_digest,
        disposition=Disposition.ERROR,
    )
    score = reveal_and_score_object_scene_anchor_exposed_queries(
        metadata=metadata,
        plan=plan,
        release=bundle.release,
        predicate=predicate,
        predictions=tuple(predictions),
    )

    assert score.determinate_count == 14
    assert score.correct_count == 14
    assert score.rows[0].correct is False
    assert score.rows[-1].correct is False


def test_tampered_plan_release_prediction_and_score_fail_roundtrip(
    release_fixture,
) -> None:
    metadata, plan, predicate, _, _, _, bundle = release_fixture
    plan_data = deepcopy(plan.to_data())
    plan_data["derived_query_ordinals"][0] = 3
    plan_data["plan_digest"] = canonical_digest(
        {key: item for key, item in plan_data.items() if key != "plan_digest"}
    )
    with pytest.raises(ObjectSceneAnchorExposedQueryGateError):
        ObjectSceneAnchorExposedQueryPlan.from_data(plan_data)

    release_data = deepcopy(bundle.release.to_data())
    release_data["exact_loader_call_count"] = 15
    release_data["release_digest"] = canonical_digest(
        {key: item for key, item in release_data.items() if key != "release_digest"}
    )
    with pytest.raises(ObjectSceneAnchorExposedQueryGateError):
        ObjectSceneAnchorExposedQueryRelease.from_data(release_data)

    prediction = _predictions(bundle.release, predicate)[0]
    prediction_data = deepcopy(prediction.to_data())
    prediction_data["created_without_expected_label"] = False
    prediction_data["prediction_digest"] = canonical_digest(
        {
            key: item
            for key, item in prediction_data.items()
            if key != "prediction_digest"
        }
    )
    with pytest.raises(ObjectSceneAnchorExposedQueryGateError):
        ObjectSceneAnchorQueryPrediction.from_data(prediction_data)


def test_durable_freeze_commitment_rejects_forged_persistence_bytes(
    predicate_fixture,
) -> None:
    predicate, durable = predicate_fixture
    assert (
        ObjectSceneAnchorPredicateDurableFreezeCommitment.from_data(
            durable.to_data()
        )
        == durable
    )
    with pytest.raises(ObjectSceneAnchorExposedQueryGateError, match="payload"):
        bind_caller_durable_object_scene_anchor_python_predicate(
            predicate,
            persisted_predicate_payload_sha256=_sha("wrong-predicate-bytes"),
            persisted_predicate_payload_byte_count=1,
            persistence_receipt_digest=_sha("receipt"),
        )


def test_synthetic_metadata_type_roundtrip_is_runtime_only(
    historical_fixture,
) -> None:
    metadata, _ = historical_fixture
    assert type(metadata) is ObjectSceneAnchorHistoricalMetadata
    assert metadata.scenes[2].tag_0_state == "present"
    assert metadata.scenes[22].tag_1_state == "present"
