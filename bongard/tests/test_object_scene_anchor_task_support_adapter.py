from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, PngImagePlugin
import pytest

from bongard.canonical import canonical_digest
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    FAMILIES,
    ObjectBongardTaskPlan,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    create_object_bongard_execution_precommit,
    prepare_object_bongard_release,
    release_object_bongard_support_panel,
)
from bongard.object_scene_anchor_task_support_adapter import (
    ObjectSceneAnchorTaskSupportAdapter,
    ObjectSceneAnchorTaskSupportAdapterError,
    build_object_scene_anchor_task_support_corpus,
    verify_object_scene_anchor_task_support_corpus,
)
from bongard.official_panel_archive import (
    OfficialPanelArchive,
    ReleasedOfficialPanel,
    _released_panel_content,
)
from bongard.release import OfficialReleaseDescriptor


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _scene(token: str) -> bytes:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((8, 30, 20, 8, 32, 30, 8, 30), fill="black", width=3)
    draw.line((58, 42, 70, 18, 84, 42, 58, 42), fill="black", width=3)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("synthetic_panel_identity", token)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, pnginfo=metadata)
    return output.getvalue()


@pytest.fixture(scope="module")
def prepared_source(tmp_path_factory):
    root: Path = tmp_path_factory.mktemp("task-support-adapter")
    inventory = tuple(
        sorted(
            f"{family}_task{index:02d}"
            for family in FAMILIES
            for index in range(3)
        )
    )
    used = tuple(sorted(f"{family}_task00" for family in FAMILIES))
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    corpus_digest = _address({"synthetic": "task-support-adapter"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = root / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for task_id in inventory:
            family = task_id.split("_", 1)[0]
            for side in ("0", "1"):
                for index in range(7):
                    bundle.writestr(
                        (
                            f"ShapeBongard_V2/{family}/images/"
                            f"{task_id}/{side}/{index}.png"
                        ),
                        _scene(f"{task_id}-{side}-{index}"),
                    )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-task-support-adapter-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=_address({"split": "train"}),
        split_size_bytes=1,
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=tuple((family, 3) for family in FAMILIES),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=corpus_digest,
    )
    plan = plan_object_bongard_batch(
        task_ids=inventory,
        train_task_ids=inventory,
        exact_used_task_ids=used,
        selection_seed="task-support-adapter-test",
        requested_per_family=1,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
        exposure_predecessor_digest=predecessor.digest,
        historical_exposure_digest=_address({"historical": []}),
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=inventory,
        train_task_ids=inventory,
        exact_used_task_ids=used,
        runtime_source_bindings={"adapter_source": _address({"adapter": 1})},
        configuration={"headless": True, "pipeline": "anchor"},
        exposure_observed_at="2026-08-09T12:00:00Z",
    )
    store = ObjectBongardReleaseStore((root / "release-store").absolute())
    prepared = prepare_object_bongard_release(
        store=store,
        plan=plan,
        precommit=precommit,
        predecessor=predecessor,
    )
    task = plan.tasks[0]
    expected_ids = (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    released = tuple(
        release_object_bongard_support_panel(
            prepared=prepared,
            archive=archive,
            panel_id=panel_id,
        )[0]
        for panel_id in expected_ids
    )
    return prepared, archive, task, released


@pytest.fixture(scope="module")
def adapted(prepared_source):
    prepared, _, task, released = prepared_source
    return build_object_scene_anchor_task_support_corpus(
        task=task,
        prepared=prepared,
        released_panels=released,
    )


def _has_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_has_bytes(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_bytes(item) for item in value)
    return False


def test_adapter_is_canonical_byte_free_and_binds_exact_task_release(
    prepared_source,
    adapted,
) -> None:
    prepared, _, task, released = prepared_source
    artifact = adapted.adapter
    data = artifact.to_data()

    assert ObjectSceneAnchorTaskSupportAdapter.from_data(data) == artifact
    assert not _has_bytes(data)
    assert artifact.task_id == task.task_id
    assert artifact.task_plan_digest == task.record_digest
    assert artifact.prepared_batch_plan_digest == prepared.plan.record_digest
    assert artifact.execution_precommit_digest == prepared.precommit.record_digest
    assert artifact.exposure_successor_digest == prepared.successor.digest
    assert artifact.expected_support_panel_ids == tuple(
        item.panel_id for item in released
    )
    assert artifact.support_corpus_freeze == adapted.support_corpus.freeze
    assert artifact.complete_object_count == 24
    assert (
        len(
            {
                item.source_panel_binding_digest
                for item in artifact.panel_bindings
            }
        )
        == 12
    )


def test_adapter_derives_exact_six_then_six_inputs_and_label_free_geometry(
    prepared_source,
    adapted,
) -> None:
    _, _, task, _ = prepared_source
    bindings = adapted.adapter.panel_bindings

    assert tuple(item.panel_id for item in bindings) == (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    assert tuple(item.opaque_support_bucket_index for item in bindings) == (
        (0,) * 6 + (1,) * 6
    )
    geometry = adapted.geometry_panel_inputs
    assert tuple(item.panel_alias for item in geometry) == tuple(
        f"panel_{index:03d}" for index in range(12)
    )
    assert all(
        set(item.__slots__)
        == {
            "panel_alias",
            "exact_png_bytes",
            "png_sha256",
            "source_panel_binding_digest",
        }
        for item in geometry
    )
    assert all(not hasattr(item, "task_id") for item in geometry)
    assert all(not hasattr(item, "panel_id") for item in geometry)
    assert all(not hasattr(item, "support_bucket_index") for item in geometry)
    assert all(not hasattr(item, "side") for item in geometry)


@pytest.mark.parametrize("mutation", ("swap", "duplicate"))
def test_adapter_rejects_wrong_order_and_duplicate_before_geometry(
    prepared_source,
    monkeypatch,
    mutation,
) -> None:
    prepared, _, task, released = prepared_source
    rows = list(released)
    if mutation == "swap":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[1] = rows[0]
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("geometry ran before adapter validation")

    monkeypatch.setattr(
        "bongard.object_scene_anchor_task_support_adapter."
        "build_object_scene_anchor_support_panel",
        forbidden,
    )
    with pytest.raises(
        ObjectSceneAnchorTaskSupportAdapterError,
        match="inventory/order",
    ):
        build_object_scene_anchor_task_support_corpus(
            task=task,
            prepared=prepared,
            released_panels=tuple(rows),
        )
    assert called is False


def test_adapter_rejects_query_and_foreign_task_panels_before_geometry(
    prepared_source,
    monkeypatch,
) -> None:
    prepared, archive, task, released = prepared_source
    query = ReleasedOfficialPanel.release(
        archive,
        task.side_0_query_panel_id,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        expected_execution_precommit_digest=prepared.precommit.record_digest,
        expected_exposure_successor_digest=prepared.successor.digest,
    )
    other_task = prepared.plan.tasks[1]
    foreign = release_object_bongard_support_panel(
        prepared=prepared,
        archive=archive,
        panel_id=other_task.side_0_support_panel_ids[0],
    )[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("geometry ran for forbidden task panel")

    monkeypatch.setattr(
        "bongard.object_scene_anchor_task_support_adapter."
        "build_object_scene_anchor_support_panel",
        forbidden,
    )
    for replacement in (query, foreign):
        rows = (replacement, *released[1:])
        with pytest.raises(
            ObjectSceneAnchorTaskSupportAdapterError,
            match="inventory/order",
        ):
            build_object_scene_anchor_task_support_corpus(
                task=task,
                prepared=prepared,
                released_panels=rows,
            )


def test_adapter_rejects_forged_release_parent_before_geometry(
    prepared_source,
    monkeypatch,
) -> None:
    prepared, _, task, released = prepared_source
    source = released[0]
    values = {
        "panel_id": source.panel_id,
        "exact_png_bytes": source.exact_png_bytes,
        "exact_png_digest": source.exact_png_digest,
        "release_receipt": source.release_receipt,
        "execution_precommit_digest": source.execution_precommit_digest,
        "exposure_successor_digest": _address({"forged": "successor"}),
    }
    provisional = object.__new__(ReleasedOfficialPanel)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    forged = ReleasedOfficialPanel(
        **values,
        record_digest="sha256:" + canonical_digest(
            _released_panel_content(provisional)
        ),
    )
    monkeypatch.setattr(
        "bongard.object_scene_anchor_task_support_adapter."
        "build_object_scene_anchor_support_panel",
        lambda *args, **kwargs: pytest.fail("geometry ran for forged custody"),
    )
    with pytest.raises(
        ObjectSceneAnchorTaskSupportAdapterError,
        match="custody",
    ):
        build_object_scene_anchor_task_support_corpus(
            task=task,
            prepared=prepared,
            released_panels=(forged, *released[1:]),
        )


def test_adapter_rejects_foreign_task_plan(prepared_source) -> None:
    prepared, _, _, released = prepared_source
    foreign = ObjectBongardTaskPlan.create(
        "bd_foreign_task",
        seed_digest=_address({"foreign": "seed"}),
    )
    with pytest.raises(
        ObjectSceneAnchorTaskSupportAdapterError,
        match="inventory/order",
    ):
        build_object_scene_anchor_task_support_corpus(
            task=foreign,
            prepared=prepared,
            released_panels=released,
        )


def test_adapter_cold_replays_all_twelve_geometry_stacks(
    prepared_source,
    adapted,
) -> None:
    prepared, _, task, _ = prepared_source
    assert (
        verify_object_scene_anchor_task_support_corpus(
            adapted,
            task=task,
            prepared=prepared,
            expected_adapter_digest=adapted.adapter.adapter_digest,
        )
        == adapted
    )


def test_adapter_tamper_fails_even_when_outer_digest_is_resealed(adapted) -> None:
    data = deepcopy(adapted.adapter.to_data())
    data["panel_bindings"][0]["png_byte_count"] += 1
    data["panel_bindings"][0]["binding_digest"] = canonical_digest(
        {
            key: item
            for key, item in data["panel_bindings"][0].items()
            if key != "binding_digest"
        }
    )
    data["adapter_digest"] = canonical_digest(
        {key: item for key, item in data.items() if key != "adapter_digest"}
    )
    with pytest.raises(ObjectSceneAnchorTaskSupportAdapterError):
        ObjectSceneAnchorTaskSupportAdapter.from_data(data)
