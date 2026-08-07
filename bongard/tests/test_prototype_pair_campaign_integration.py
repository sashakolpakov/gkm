from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
import zipfile

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

from bongard.canonical import canonical_json
from bongard.exposure import ExposureLedger
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.prototype_pair_campaign import (
    PrototypePairCampaignArtifact,
    PrototypePairCampaignConfiguration,
    PrototypePairCampaignError,
    PrototypePairCampaignStatus,
    cold_replay_prototype_pair_campaign,
    prototype_pair_campaign_runtime_source_digests,
    run_prototype_pair_campaign,
)
from bongard.prototype_pair_campaign_store import (
    PrototypePairCallClaim,
    PrototypePairCampaignStore,
    PrototypePairCampaignStoreError,
)
from bongard.prototype_pair_cohort import (
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OPAQUE_TAG_IDS,
    plan_prototype_pair_cohort,
    task_id_inventory_digest,
)
from bongard.prototype_pair_execution_precommit import (
    PrototypePairExecutionIdentities,
    PrototypePairExecutionPrecommit,
    prepare_prototype_pair_execution_precommit,
)
from bongard.prototype_scene_calibration import (
    PrototypeSceneTagThreshold,
    calibration_algorithm_digest,
    threshold_commitment,
)
from bongard.prototype_scene_codex_ranker import (
    PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
    PrototypeSceneCodexRanker,
    prototype_scene_codex_ranker_environment_digest,
    prototype_scene_codex_ranker_model_identity_digest,
    prototype_scene_codex_ranker_protocol_digest,
    prototype_scene_codex_ranker_transport_source_digest,
)
from bongard.prototype_scene_headless_runner import (
    RUNNER_ID,
    prototype_scene_runner_source_digest,
)
from bongard.prototype_scene_observer import (
    PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
    prototype_rubric_description_protocol_digest,
    prototype_scene_observer_environment_digest,
    prototype_scene_observer_model_digest,
    prototype_scene_scoring_protocol_digest,
)
from bongard.release import OfficialReleaseDescriptor
from bongard.tests.test_prototype_pair_cohort import (
    SEED,
    _fixture as _cohort_fixture,
    _kwargs as _cohort_kwargs,
)
from bongard.tests.test_prototype_scene_codex_ranker import (
    _receipt as _rank_receipt,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    _description_payload,
    _receipt as _observer_receipt,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.transport import CloudPolicyCacheSnapshot, CodexStructuredResult


_CLI_VERSION = "codex-cli test"
_PYTHON_RUNTIME_ID = "cpython-campaign-integration-test"
_PYTHON_RUNTIME_DIGEST = hashlib.sha256(
    _PYTHON_RUNTIME_ID.encode("utf-8")
).hexdigest()
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(
    LAUNCHER_DIGEST
)


def _panel_png(panel_id: str) -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    metadata = PngInfo()
    metadata.add_text("panel_id", panel_id)
    output = BytesIO()
    image.save(output, format="PNG", pnginfo=metadata, optimize=False)
    return output.getvalue()


class _Clock:
    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def now(self, phase: str, subject_id: str, event: str) -> str:
        del phase, subject_id, event
        with self._lock:
            value = self._counter
            self._counter += 1
        return f"2026-08-07T00:00:00.{value:06d}Z"


@dataclass(frozen=True, slots=True)
class _Fixture:
    root: Path
    plan: Any
    precommit: PrototypePairExecutionPrecommit
    predecessor: ExposureLedger
    descriptor: OfficialReleaseDescriptor
    archive: OfficialPanelArchive
    configuration: PrototypePairCampaignConfiguration
    panel_id_by_digest: Mapping[str, str]

    def run_kwargs(self, store: PrototypePairCampaignStore) -> dict[str, object]:
        return {
            "cohort_plan": self.plan,
            "precommit": self.precommit,
            "exposure_predecessor": self.predecessor,
            "release_descriptor": self.descriptor,
            "official_archive": self.archive,
            "store": store,
            "clock": _Clock(),
            "configuration": self.configuration,
            "cloud_policy_cache_snapshot": None,
            "model_catalog_snapshot": MODEL_CATALOG,
            "no_tools_attestation": NO_TOOLS_ATTESTATION,
            "observed_codex_cli_version": _CLI_VERSION,
            "observed_codex_launcher_sha256": LAUNCHER_DIGEST,
            "observed_python_runtime_id": _PYTHON_RUNTIME_ID,
            "observed_python_runtime_identity_digest": _PYTHON_RUNTIME_DIGEST,
            "expected_precommit_digest": self.precommit.record_digest,
            "expected_cohort_plan_digest": self.plan.record_digest,
            "expected_identity_bundle_digest": self.precommit.identities.record_digest,
            "expected_exposure_predecessor_digest": self.predecessor.digest,
        }


@pytest.fixture(scope="module")
def campaign_fixture(tmp_path_factory: pytest.TempPathFactory) -> _Fixture:
    root = tmp_path_factory.mktemp("prototype-pair-campaign-integration")
    historical, _old_release, split, inventory, _old_exposure, _candidate_ids = (
        _cohort_fixture()
    )
    archive_path = root / "ShapeBongard_V2.zip"
    panel_id_by_digest: dict[str, str] = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for task_id in inventory:
            for side in ("0", "1"):
                for index in range(7):
                    panel_id = f"bd/{task_id}/{side}/{index}.png"
                    payload = _panel_png(panel_id)
                    digest = hashlib.sha256(payload).hexdigest()
                    assert digest not in panel_id_by_digest
                    panel_id_by_digest[digest] = panel_id
                    bundle.writestr(
                        f"ShapeBongard_V2/bd/images/{task_id}/{side}/{index}.png",
                        payload,
                    )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-campaign-integration-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + hashlib.sha256(split).hexdigest(),
        split_size_bytes=len(split),
        upstream_repository=OFFICIAL_UPSTREAM_REPOSITORY,
        upstream_commit=OFFICIAL_UPSTREAM_COMMIT,
        family_counts=(("bd", len(inventory)),),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=task_id_inventory_digest(inventory),
        corpus_manifest_sha256=(
            "sha256:" + hashlib.sha256(b"campaign integration corpus").hexdigest()
        ),
    )
    predecessor = ExposureLedger.create(descriptor.corpus_manifest_sha256)
    plan = plan_prototype_pair_cohort(
        **_cohort_kwargs(
            historical,
            descriptor,
            split,
            inventory,
            predecessor,
            seed=SEED,
        )
    )
    configuration = PrototypePairCampaignConfiguration(
        actor="campaign-integration-test",
        parallel_workers=4,
        observer_minutes=1,
        observer_verbose=False,
        observer_executable="codex-test",
        ranker_minutes=1,
        ranker_verbose=False,
        ranker_executable="codex-test",
        runtime_archive_source_id="campaign-integration-archive",
        runtime_verifier_id="campaign-integration-verifier",
    )
    thresholds = tuple(
        PrototypeSceneTagThreshold(tag_id, 250_000, 750_000)
        for tag_id in OPAQUE_TAG_IDS
    )
    identities = PrototypePairExecutionIdentities.create(
        exposure_predecessor_digest=predecessor.digest,
        execution_configuration_digest=configuration.record_digest,
        thresholds=thresholds,
        threshold_commitment=threshold_commitment(thresholds),
        calibration_algorithm_digest=calibration_algorithm_digest(),
        observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
        observer_description_protocol_digest=(
            prototype_rubric_description_protocol_digest()
        ),
        observer_scoring_protocol_digest=prototype_scene_scoring_protocol_digest(),
        observer_environment_digest=prototype_scene_observer_environment_digest(
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_binding="absent",
            model_catalog_digest=MODEL_CATALOG.raw_digest,
            no_tools_attestation_digest=(
                NO_TOOLS_ATTESTATION.attestation_digest
            ),
        ),
        observer_model_id=MODEL,
        observer_reasoning_effort=EFFORT,
        observer_model_identity_digest=prototype_scene_observer_model_digest(
            MODEL, EFFORT
        ),
        ranker_model_id=MODEL,
        ranker_reasoning_effort=EFFORT,
        ranker_model_identity_digest=(
            prototype_scene_codex_ranker_model_identity_digest(MODEL, EFFORT)
            .removeprefix("sha256:")
        ),
        ranker_protocol_id=PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
        ranker_protocol_digest=prototype_scene_codex_ranker_protocol_digest(),
        ranker_environment_digest=prototype_scene_codex_ranker_environment_digest(
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            expected_cloud_policy_cache_binding="absent",
            expected_transport_source_digest=(
                prototype_scene_codex_ranker_transport_source_digest()
            ),
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
        ),
        runner_protocol_id=RUNNER_ID,
        runner_algorithm_digest=prototype_scene_runner_source_digest(),
        codex_cli_version=_CLI_VERSION,
        codex_launcher_sha256=LAUNCHER_DIGEST,
        cloud_policy_cache_binding="absent",
        codex_model_catalog_snapshot=MODEL_CATALOG,
        codex_no_tools_attestation=NO_TOOLS_ATTESTATION,
        python_runtime_id=_PYTHON_RUNTIME_ID,
        python_runtime_identity_digest=_PYTHON_RUNTIME_DIGEST,
        runtime_source_digests=prototype_pair_campaign_runtime_source_digests(),
    )
    precommit = prepare_prototype_pair_execution_precommit(
        cohort_plan=plan,
        identities=identities,
        expected_cohort_plan_digest=plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=predecessor.digest,
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    return _Fixture(
        root=root,
        plan=plan,
        precommit=precommit,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        configuration=configuration,
        panel_id_by_digest=panel_id_by_digest,
    )


def _description_transport(counter: list[int], *, fail: bool = False):
    def transport(
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **_kwargs: object,
    ) -> CodexStructuredResult:
        counter[0] += 1
        assert _kwargs["model_catalog_snapshot"] is MODEL_CATALOG
        assert _kwargs["tool_surface_attestation"] is NO_TOOLS_ATTESTATION
        if fail:
            raise RuntimeError("deliberate description transport failure")
        payload = _description_payload()
        return CodexStructuredResult(
            payload, _observer_receipt(prompt, paths, names, schema, payload)
        )

    return transport


def _scene_transport(
    fixture: _Fixture, counter: list[int], *, mode: str = "complete"
):
    calibration_states = {
        item.panel_id: dict(item.expected_tag_states)
        for item in fixture.plan.calibration_clusters
    }
    positive = set(fixture.plan.drill.positive_panel_ids)
    negative = set(fixture.plan.drill.negative_panel_ids)
    witness_panel = next(
        role.source_panel_id
        for role in fixture.precommit.support_roles
        if role.opaque_side_id == "side_1"
    )
    lock = threading.Lock()

    def transport(
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **_kwargs: object,
    ) -> CodexStructuredResult:
        with lock:
            counter[0] += 1
        assert _kwargs["model_catalog_snapshot"] is MODEL_CATALOG
        assert _kwargs["tool_surface_attestation"] is NO_TOOLS_ATTESTATION
        digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel_id = fixture.panel_id_by_digest[digest]
        if panel_id in calibration_states:
            states = calibration_states[panel_id]
        elif panel_id in positive:
            states = {tag_id: "present" for tag_id in OPAQUE_TAG_IDS}
        elif panel_id in negative:
            drill_state = "present" if mode == "support_gap" else "absent"
            states = {tag_id: drill_state for tag_id in OPAQUE_TAG_IDS}
        else:  # pragma: no cover - catches a campaign release-schedule regression
            raise AssertionError(f"unexpected scene panel: {panel_id}")
        cells = []
        for index, tag_id in enumerate(OPAQUE_TAG_IDS):
            if (
                mode == "calibration_gap"
                and panel_id == next(iter(calibration_states))
                and index == 0
            ) or (
                mode == "support_witness_gap" and panel_id == witness_panel
            ):
                cells.append(
                    {
                        "group_id": f"group_{index}",
                        "state": "indeterminate",
                        "lower_ppm": None,
                        "upper_ppm": None,
                        "reason_code": "ambiguous_visible_match",
                    }
                )
                continue
            lower, upper = (
                (800_000, 900_000)
                if states[tag_id] == "present"
                else (100_000, 200_000)
            )
            cells.append(
                {
                    "group_id": f"group_{index}",
                    "state": "scored",
                    "lower_ppm": lower,
                    "upper_ppm": upper,
                    "reason_code": None,
                }
            )
        payload = {
            "description": "A compact angular drawing with oblique strokes.",
            "cells": cells,
        }
        return CodexStructuredResult(
            payload, _observer_receipt(prompt, paths, names, schema, payload)
        )

    return transport


def _ranker(
    counter: list[int], *, fail: bool = False
) -> PrototypeSceneCodexRanker:
    def transport(
        prompt: str,
        schema: Mapping[str, Any],
        **_kwargs: object,
    ) -> CodexStructuredResult:
        counter[0] += 1
        assert _kwargs["model_catalog_snapshot"] is MODEL_CATALOG
        assert _kwargs["tool_surface_attestation"] is NO_TOOLS_ATTESTATION
        if fail:
            raise RuntimeError("deliberate rank transport failure")
        survivors = schema["properties"]["ordered_candidate_ids"]["items"]["enum"]
        payload = {"ordered_candidate_ids": list(reversed(survivors))}
        return CodexStructuredResult(payload, _rank_receipt(prompt, schema, payload))

    return PrototypeSceneCodexRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            prototype_scene_codex_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        minutes=1,
        verbose=False,
        executable="codex-test",
        transport=transport,
    )


def test_complete_campaign_runs_through_real_archive_and_disk_store(
    campaign_fixture: _Fixture,
) -> None:
    store = PrototypePairCampaignStore.open(campaign_fixture.root / "complete-store")
    description_calls = [0]
    scene_calls = [0]
    rank_calls = [0]
    artifact = run_prototype_pair_campaign(
        **campaign_fixture.run_kwargs(store),
        description_transport=_description_transport(description_calls),
        scene_transport=_scene_transport(campaign_fixture, scene_calls),
        ranker=_ranker(rank_calls),
    )

    assert artifact.status is PrototypePairCampaignStatus.COMPLETE
    assert artifact.model_calls_made == 44
    assert len(artifact.released_panels) == 48
    assert len(artifact.call_terminals) == 44
    assert description_calls == [1]
    assert scene_calls == [42]
    assert rank_calls == [1]
    assert artifact.headless_archive is not None
    assert artifact.headless_archive.query_source_calls_made == 1
    assert artifact.headless_archive.rank_calls_made == 1
    assert artifact.release_authorization["selected_task_ids"] == list(
        campaign_fixture.plan.selected_task_ids
    )
    assert canonical_json(artifact.to_data())

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a terminal replay attempted a fresh transport")

    replay_rank_calls = [0]
    replayed = run_prototype_pair_campaign(
        **campaign_fixture.run_kwargs(store),
        description_transport=forbidden,
        scene_transport=forbidden,
        ranker=_ranker(replay_rank_calls, fail=True),
    )
    assert replayed == artifact
    assert replay_rank_calls == [0]

    replay_kwargs = {
        "cohort_plan": campaign_fixture.plan,
        "precommit": campaign_fixture.precommit,
        "release_descriptor": campaign_fixture.descriptor,
        "official_archive_path": campaign_fixture.archive.archive_path,
        "store": store,
        "expected_campaign_digest": artifact.record_digest,
        "expected_precommit_digest": campaign_fixture.precommit.record_digest,
        "expected_cohort_plan_digest": campaign_fixture.plan.record_digest,
        "expected_identity_bundle_digest": (
            campaign_fixture.precommit.identities.record_digest
        ),
        "expected_exposure_predecessor_digest": campaign_fixture.predecessor.digest,
    }
    assert cold_replay_prototype_pair_campaign(artifact, **replay_kwargs) == artifact

    with pytest.raises(PrototypePairCampaignStoreError, match="already authorized"):
        store.authorize_release(
            campaign_fixture.plan,
            campaign_fixture.predecessor,
            artifact.precommit_receipt,
            expected_plan_digest=campaign_fixture.plan.record_digest,
            expected_execution_precommit_digest=(
                campaign_fixture.precommit.record_digest
            ),
            expected_exposure_predecessor_digest=campaign_fixture.predecessor.digest,
            actor="different-campaign-actor",
            observed_at="2026-08-07T01:00:00Z",
        )

    def assert_changed_precommit_rejected(
        changed_identities: PrototypePairExecutionIdentities,
    ) -> None:
        changed = prepare_prototype_pair_execution_precommit(
            cohort_plan=campaign_fixture.plan,
            identities=changed_identities,
            expected_cohort_plan_digest=campaign_fixture.plan.record_digest,
            expected_identity_bundle_digest=changed_identities.record_digest,
            expected_exposure_predecessor_digest=campaign_fixture.predecessor.digest,
        )
        receipt = store.persist_execution_precommit(
            canonical_json(changed.to_data()) + b"\n", changed.record_digest
        )
        with pytest.raises(
            PrototypePairCampaignStoreError, match="already authorized"
        ):
            store.authorize_release(
                campaign_fixture.plan,
                campaign_fixture.predecessor,
                receipt,
                expected_plan_digest=campaign_fixture.plan.record_digest,
                expected_execution_precommit_digest=changed.record_digest,
                expected_exposure_predecessor_digest=(
                    campaign_fixture.predecessor.digest
                ),
                actor=campaign_fixture.configuration.actor,
                observed_at="2026-08-07T01:00:01Z",
            )

    assert_changed_precommit_rejected(
        replace(
            campaign_fixture.precommit.identities,
            python_runtime_id="cpython-different-campaign-integration-test",
        )
    )
    changed_configuration = replace(
        campaign_fixture.configuration, observer_minutes=2
    )
    assert_changed_precommit_rejected(
        replace(
            campaign_fixture.precommit.identities,
            execution_configuration_digest=changed_configuration.record_digest,
        )
    )
    successor_directory = store.root / "objects" / "exposure_successor"
    assert len(list(successor_directory.glob("*.json"))) == 1
    assert len(
        [
            path
            for path in (store.root / "authorizations").glob("*.json")
            if not path.name.endswith((".claim.json", ".complete.json"))
        ]
    ) == 1

    serialized_tamper = deepcopy(artifact.to_data())
    serialized_tamper["model_calls_made"] = 43
    with pytest.raises(PrototypePairCampaignError):
        PrototypePairCampaignArtifact.from_data(serialized_tamper)

    catalog_binding = next(
        item for item in artifact.stored_objects if item.kind == "reference_catalog"
    )
    tamper_path = store.root / catalog_binding.storage_receipt["relative_path"]
    tamper_path.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError):
        cold_replay_prototype_pair_campaign(artifact, **replay_kwargs)


def test_changed_execution_configuration_is_rejected_before_any_model_call(
    campaign_fixture: _Fixture,
) -> None:
    store = PrototypePairCampaignStore.open(
        campaign_fixture.root / "configuration-mismatch-store"
    )
    calls = [0]

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls[0] += 1
        raise AssertionError("configuration mismatch reached a model transport")

    changed = replace(
        campaign_fixture.configuration,
        runtime_verifier_id="campaign-integration-verifier-reroll",
    )
    kwargs = campaign_fixture.run_kwargs(store)
    kwargs["configuration"] = changed
    with pytest.raises(
        PrototypePairCampaignError,
        match="execution configuration differs from precommit",
    ):
        run_prototype_pair_campaign(
            **kwargs,
            description_transport=forbidden,
            scene_transport=forbidden,
            ranker=_ranker(calls, fail=True),
        )

    assert calls == [0]
    assert not list((store.root / "authorizations").glob("*.json"))


@pytest.mark.parametrize(
    (
        "mode",
        "description_fails",
        "rank_fails",
        "expected_status",
        "expected_calls",
        "released",
    ),
    (
        (
            "complete",
            True,
            False,
            PrototypePairCampaignStatus.DESCRIPTION_GAP,
            (1, 0, 0),
            6,
        ),
        (
            "calibration_gap",
            False,
            False,
            PrototypePairCampaignStatus.CALIBRATION_GAP,
            (1, 28, 0),
            34,
        ),
        (
            "support_gap",
            False,
            False,
            PrototypePairCampaignStatus.SUPPORT_LANGUAGE_GAP,
            (1, 40, 0),
            46,
        ),
        (
            "support_witness_gap",
            False,
            False,
            PrototypePairCampaignStatus.SUPPORT_WITNESS_GAP,
            (1, 40, 0),
            46,
        ),
        (
            "complete",
            False,
            True,
            PrototypePairCampaignStatus.RANKER_ERROR,
            (1, 40, 1),
            46,
        ),
    ),
)
def test_terminal_gap_and_ranker_error_branches_are_durable(
    campaign_fixture: _Fixture,
    mode: str,
    description_fails: bool,
    rank_fails: bool,
    expected_status: PrototypePairCampaignStatus,
    expected_calls: tuple[int, int, int],
    released: int,
) -> None:
    store = PrototypePairCampaignStore.open(
        campaign_fixture.root / f"{expected_status.value}-store"
    )
    description_calls = [0]
    scene_calls = [0]
    rank_calls = [0]
    artifact = run_prototype_pair_campaign(
        **campaign_fixture.run_kwargs(store),
        description_transport=_description_transport(
            description_calls, fail=description_fails
        ),
        scene_transport=_scene_transport(campaign_fixture, scene_calls, mode=mode),
        ranker=_ranker(rank_calls, fail=rank_fails),
    )

    assert artifact.status is expected_status
    assert (
        description_calls[0],
        scene_calls[0],
        rank_calls[0],
    ) == expected_calls
    assert artifact.model_calls_made == sum(expected_calls)
    assert len(artifact.call_terminals) == artifact.model_calls_made
    assert len(artifact.released_panels) == released
    replay_kwargs = {
        "cohort_plan": campaign_fixture.plan,
        "precommit": campaign_fixture.precommit,
        "release_descriptor": campaign_fixture.descriptor,
        "official_archive_path": campaign_fixture.archive.archive_path,
        "store": store,
        "expected_campaign_digest": artifact.record_digest,
        "expected_precommit_digest": campaign_fixture.precommit.record_digest,
        "expected_cohort_plan_digest": campaign_fixture.plan.record_digest,
        "expected_identity_bundle_digest": (
            campaign_fixture.precommit.identities.record_digest
        ),
        "expected_exposure_predecessor_digest": campaign_fixture.predecessor.digest,
    }
    assert cold_replay_prototype_pair_campaign(artifact, **replay_kwargs) == artifact

    if expected_status is PrototypePairCampaignStatus.DESCRIPTION_GAP:
        authorization = store.load_release_authorization(
            artifact.release_authorization["record_digest"]
        )
        extra_context = "sha256:" + "e" * 64
        with pytest.raises(PrototypePairCampaignStoreError, match="sealed"):
            store.claim_call(
                authorization,
                phase="headless_codex_candidate_ranked",
                subject_id=campaign_fixture.precommit.drill_task_id,
                context_digest=extra_context,
                claimed_at="2026-08-07T02:00:00Z",
            )

        forged = PrototypePairCallClaim.seal(
            authorization_digest=authorization.record_digest,
            phase="headless_codex_candidate_ranked",
            subject_id=campaign_fixture.precommit.drill_task_id,
            context_digest=extra_context,
            claimed_at="2026-08-07T02:00:00Z",
        )
        forged_path = store.root / "claims" / (
            forged.key_digest.removeprefix("sha256:") + ".claim.json"
        )
        forged_path.write_bytes(canonical_json(forged.to_data()) + b"\n")
        with pytest.raises(RuntimeError, match="journal"):
            cold_replay_prototype_pair_campaign(artifact, **replay_kwargs)
