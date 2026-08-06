from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.artifacts import canonical_digest
import bongard.atomic_smoke_precommit as P
from bongard.atomic_smoke_precommit import (
    ATOMIC_SMOKE_SELECTION_POLICY,
    OFFICIAL_SUCCESSOR_UNIVERSE_COUNT,
    OFFICIAL_SUCCESSOR_UNIVERSE_DIGEST,
    AtomicSmokePrecommit,
    AtomicSmokePrecommitError,
    AtomicSmokeSelection,
    cold_decode_and_replay_atomic_smoke_precommit,
    prepare_atomic_smoke_precommit,
    replay_atomic_smoke_selection,
    select_atomic_smoke_task,
)
from bongard.corpus import (
    BongardTask,
    CorpusManifest,
    ShapeBongardCorpus,
    SplitIndex,
)
from bongard.exposure import ExposureLedger


SOURCE_DEPENDENCIES = hashlib.sha256(
    b"synthetic authoritative sources"
).hexdigest()
SPLIT_SOURCE = "sha256:" + hashlib.sha256(b"synthetic split").hexdigest()
LABEL_NONCE = hashlib.sha256(b"private synthetic label nonce").hexdigest()
EPISODE_SEED = hashlib.sha256(b"private synthetic episode seed").hexdigest()
SEED = "post-freeze synthetic selection seed"
OBSERVED_AT = "2026-08-06T12:00:00Z"

PREDECESSOR_CONSUMED_TASK = "bd_mismatch_triangle_rec4_0000"
UNIVERSE = (
    "bd_mismatch_triangle_rec1_0000",
    "bd_mismatch_triangle_rec2_0000",
    "bd_mismatch_triangle_rec5_0000",
    "bd_mismatch_triangle_rec6_0000",
    "bd_open_equil_obtuse_triangle1_0000",
    "bd_open_uneven_band_four_arcs1_0000",
    "bd_open_uneven_band_four_arcs3_0000",
    "bd_thin_three_sides2_0000",
    "bd_thin_three_sides3_0000",
)
A3_SUCCESSOR_UNIVERSE = tuple(sorted((*UNIVERSE, PREDECESSOR_CONSUMED_TASK)))
EXPOSED_SIBLINGS = (
    "bd_mismatch_triangle_rec3_0000",
    "bd_open_equil_obtuse_triangle2_0000",
    "bd_open_uneven_band_four_arcs2_0000",
    "bd_thin_three_sides1_0000",
)
VAL_REPEATED = "bd_trapezoid_parallel1_0000"
TEST_REPEATED = "bd_thin_seven_lines1_0000"
NON_DRILL_REPEATED = "bd_open_obtuse_triangle1_0000"
NEW_GENERATOR = "bd_asymmetric_arrow_0000"
DECOY_EXPOSURES = (
    "bd_trapezoid_parallel2_0000",
    "bd_thin_seven_lines4_0000",
    "bd_open_obtuse_triangle2_0000",
)


def _task(corpus_root: Path, task_id: str) -> BongardTask:
    root = corpus_root / "bd" / "images" / task_id
    sides: dict[str, tuple[Path, ...]] = {}
    for label in ("1", "0"):
        directory = root / label
        directory.mkdir(parents=True)
        paths: list[Path] = []
        for index in range(7):
            path = directory / f"{index}.png"
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + f"{task_id}:{label}:{index}".encode("utf-8")
            )
            paths.append(path)
        sides[label] = tuple(paths)
    return BongardTask(
        task_id=task_id,
        family="bd",
        root=root,
        positive=sides["1"],
        negative=sides["0"],
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drop: str | None = None,
) -> tuple[ShapeBongardCorpus, CorpusManifest, ExposureLedger, Path]:
    train = tuple(
        task_id
        for task_id in (
            *A3_SUCCESSOR_UNIVERSE,
            *EXPOSED_SIBLINGS,
            NON_DRILL_REPEATED,
            NEW_GENERATOR,
            *DECOY_EXPOSURES,
        )
        if task_id != drop
    )
    val = () if drop == VAL_REPEATED else (VAL_REPEATED,)
    test = () if drop == TEST_REPEATED else (TEST_REPEATED,)
    task_ids = tuple(sorted((*train, *val, *test)))
    corpus = ShapeBongardCorpus(
        tmp_path,
        tuple(_task(tmp_path, task_id) for task_id in task_ids),
        layout="archive",
        split=SplitIndex(
            groups=(("test", test), ("train", train), ("val", val)),
            source_digest=SPLIT_SOURCE,
        ),
    )
    manifest = corpus.build_manifest()
    historical_parent = ExposureLedger.create(manifest.digest).record(
        phase="stage-a3",
        actor="fixture",
        purpose="freeze already disclosed BD generators",
        task_ids=(*EXPOSED_SIBLINGS, *DECOY_EXPOSURES),
        observed_at="2026-08-06T10:00:00Z",
        known_task_ids=corpus.task_ids,
    )
    predecessor = historical_parent.record(
        phase=P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PHASE,
        actor="fixture-verifier",
        purpose=P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PURPOSE,
        task_ids=(PREDECESSOR_CONSUMED_TASK,),
        source="atomic-smoke-selection:sha256:" + "1" * 64,
        observed_at="2026-08-06T11:00:00Z",
        known_task_ids=corpus.task_ids,
        require_unseen=True,
    )
    monkeypatch.setattr(P, "OFFICIAL_CORPUS_MANIFEST_DIGEST", manifest.digest)
    monkeypatch.setattr(P, "OFFICIAL_A3_LEDGER_DIGEST", historical_parent.digest)
    monkeypatch.setattr(
        P, "OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST", predecessor.digest
    )
    monkeypatch.setattr(
        P,
        "OFFICIAL_SUCCESSOR_PREDECESSOR_APPEND_SEQUENCE",
        predecessor.events[-1].sequence,
    )
    monkeypatch.setattr(
        P,
        "OFFICIAL_SUCCESSOR_PREDECESSOR_APPEND_ACTOR",
        predecessor.events[-1].actor,
    )
    monkeypatch.setattr(P, "OFFICIAL_SPLIT_SOURCE_DIGEST", SPLIT_SOURCE)
    monkeypatch.setattr(P, "OFFICIAL_TASK_COUNT", len(task_ids))
    monkeypatch.setattr(
        P,
        "OFFICIAL_FAMILY_COUNTS",
        (("ff", 0), ("bd", len(task_ids)), ("hd", 0)),
    )
    monkeypatch.setattr(
        P,
        "OFFICIAL_SPLIT_COUNTS",
        (("train", len(train)), ("val", len(val)), ("test", len(test))),
    )
    monkeypatch.setattr(P, "OFFICIAL_REGIME_COUNTS", ())
    store = tmp_path / "exposure-store"
    store.mkdir()
    return corpus, manifest, predecessor, store


def _prepare(
    corpus: ShapeBongardCorpus,
    manifest: CorpusManifest,
    predecessor: ExposureLedger,
    store: Path,
) -> AtomicSmokePrecommit:
    return prepare_atomic_smoke_precommit(
        corpus,
        seed=SEED,
        episode_seed=EPISODE_SEED,
        full_corpus_manifest=manifest,
        source_corpus_manifest_digest=manifest.digest,
        source_dependency_digest=SOURCE_DEPENDENCIES,
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
        label_seal_nonce=LABEL_NONCE,
        exposure_store_dir=store,
        verifier_id="fixture-verifier",
        observed_at=OBSERVED_AT,
    )


def _cold_kwargs(
    precommit: AtomicSmokePrecommit,
    corpus: ShapeBongardCorpus,
    manifest: CorpusManifest,
    predecessor: ExposureLedger,
    store: Path,
) -> dict[str, object]:
    return {
        "value": precommit.to_data(),
        "expected_precommit_digest": precommit.digest,
        "corpus": corpus,
        "seed": SEED,
        "episode_seed": EPISODE_SEED,
        "full_corpus_manifest": manifest,
        "source_corpus_manifest_digest": manifest.digest,
        "source_dependency_digest": SOURCE_DEPENDENCIES,
        "exposure_predecessor": predecessor,
        "exposure_successor": precommit.exposure_successor,
        "exposure_store_dir": store,
        "label_seal_nonce": LABEL_NONCE,
    }


def test_production_official_anchors_are_exact() -> None:
    assert P.OFFICIAL_CORPUS_MANIFEST_DIGEST.endswith("51dce138")
    assert P.OFFICIAL_A3_LEDGER_DIGEST == (
        "sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4"
    )
    assert P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST == (
        "sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a"
    )
    assert P.OFFICIAL_SUCCESSOR_PREDECESSOR_APPEND_SEQUENCE == 147
    assert (
        P.OFFICIAL_SUCCESSOR_PREDECESSOR_APPEND_ACTOR
        == "canonical-bongard-verifier"
    )
    assert P.OFFICIAL_SPLIT_SOURCE_DIGEST.endswith("ed7230")
    assert P.OFFICIAL_HISTORICAL_SEED_DIGEST.endswith("e02ebf")
    assert P.OFFICIAL_RESOLVER_POLICY_DIGEST.endswith("47af9a")
    assert P.OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST.endswith("c5f6b8")
    assert P.OFFICIAL_RELEASE_DESCRIPTOR_DIGEST.endswith("56cd2b")


def test_historical_a3_is_lineage_only_and_rejected_as_active_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, _store = _fixture(tmp_path, monkeypatch)
    historical_parent = ExposureLedger(
        corpus_digest=predecessor.corpus_digest,
        events=predecessor.events[:-1],
    )
    assert historical_parent.digest == P.OFFICIAL_A3_LEDGER_DIGEST
    assert predecessor.digest == P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
    with pytest.raises(
        AtomicSmokePrecommitError,
        match="not the official successor pin",
    ):
        select_atomic_smoke_task(
            corpus,
            seed=SEED,
            full_corpus_manifest=manifest,
            source_corpus_manifest_digest=manifest.digest,
            exposure_ledger=historical_parent,
            expected_exposure_ledger_digest=historical_parent.digest,
        )


def test_metadata_selection_authenticates_exact_universe_without_pixel_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, _store = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        BongardTask,
        "build_manifest",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("metadata selection opened a task")
        ),
    )
    kwargs = {
        "seed": SEED,
        "full_corpus_manifest": manifest,
        "source_corpus_manifest_digest": manifest.digest,
        "exposure_ledger": predecessor,
        "expected_exposure_ledger_digest": predecessor.digest,
    }
    first = select_atomic_smoke_task(corpus, **kwargs)
    second = select_atomic_smoke_task(corpus, **kwargs)
    assert first == second
    assert first.selected_task_id in UNIVERSE
    assert first.selection_policy == ATOMIC_SMOKE_SELECTION_POLICY
    assert first.universe_count == OFFICIAL_SUCCESSOR_UNIVERSE_COUNT == 9
    assert first.universe_task_ids_digest == OFFICIAL_SUCCESSOR_UNIVERSE_DIGEST
    assert "sha256:" + canonical_digest(list(sorted(UNIVERSE))) == (
        OFFICIAL_SUCCESSOR_UNIVERSE_DIGEST
    )
    assert PREDECESSOR_CONSUMED_TASK not in UNIVERSE
    assert not set(UNIVERSE) & set(predecessor.exposed_task_ids)
    assert all(
        getattr(first, name) is False
        for name in (
            "dependence_design_authorized",
            "calibration_authorized",
            "benchmark_claim_authorized",
            "official_test_authorized",
        )
    )
    assert AtomicSmokeSelection.from_data(first.to_data()) == first
    assert replay_atomic_smoke_selection(
        first.to_data(),
        expected_selection_digest=first.digest,
        corpus=corpus,
        seed=SEED,
        full_corpus_manifest=manifest,
        source_corpus_manifest_digest=manifest.digest,
        exposure_ledger=predecessor,
    ) == first


def test_every_seed_excludes_all_predecessor_exposed_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, _store = _fixture(tmp_path, monkeypatch)
    selected: set[str] = set()
    for index in range(64):
        selection = select_atomic_smoke_task(
            corpus,
            seed=f"post-freeze-successor-seed-{index:02d}",
            full_corpus_manifest=manifest,
            source_corpus_manifest_digest=manifest.digest,
            exposure_ledger=predecessor,
            expected_exposure_ledger_digest=predecessor.digest,
        )
        selected.add(selection.selected_task_id)
        assert selection.selected_task_id in UNIVERSE
        assert selection.selected_task_id not in predecessor.exposed_task_ids
    assert PREDECESSOR_CONSUMED_TASK not in selected


def test_active_predecessor_requires_one_task_only_atomic_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, _store = _fixture(tmp_path, monkeypatch)
    historical_parent = ExposureLedger(
        corpus_digest=predecessor.corpus_digest,
        events=predecessor.events[:-1],
    )
    common = {
        "phase": P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PHASE,
        "actor": "fixture-verifier",
        "purpose": P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PURPOSE,
        "task_ids": (PREDECESSOR_CONSUMED_TASK,),
        "source": "atomic-smoke-selection:sha256:" + "3" * 64,
        "observed_at": "2026-08-06T11:00:00Z",
    }
    malformed = (
        historical_parent.record(**{**common, "actor": "model-proposer"}),
        historical_parent.record(**{**common, "phase": "model-run"}),
        historical_parent.record(**{**common, "purpose": "retry"}),
        historical_parent.record(**{**common, "source": "model:gpt"}),
        historical_parent.record(
            **common,
            panel_ids=(f"bd/{PREDECESSOR_CONSUMED_TASK}/1/0.png",),
        ),
    )
    for ledger in malformed:
        monkeypatch.setattr(
            P, "OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST", ledger.digest
        )
        with pytest.raises(AtomicSmokePrecommitError, match="atomic smoke protocol"):
            select_atomic_smoke_task(
                corpus,
                seed=SEED,
                full_corpus_manifest=manifest,
                source_corpus_manifest_digest=manifest.digest,
                exposure_ledger=ledger,
                expected_exposure_ledger_digest=ledger.digest,
            )

    extra = predecessor.record(
        phase=P.ATOMIC_SMOKE_EXPOSURE_PHASE,
        actor="fixture-verifier",
        purpose=P.ATOMIC_SMOKE_EXPOSURE_PURPOSE,
        task_ids=(UNIVERSE[0],),
        source="atomic-smoke-selection:sha256:" + "4" * 64,
        observed_at="2026-08-06T12:00:00Z",
        require_unseen=True,
    )
    monkeypatch.setattr(
        P, "OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST", extra.digest
    )
    with pytest.raises(AtomicSmokePrecommitError, match="exactly one append"):
        select_atomic_smoke_task(
            corpus,
            seed=SEED,
            full_corpus_manifest=manifest,
            source_corpus_manifest_digest=manifest.digest,
            exposure_ledger=extra,
            expected_exposure_ledger_digest=extra.digest,
        )


def test_owned_fsync_reload_precedes_all_selected_pixel_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    original = BongardTask.build_manifest
    calls: list[str] = []

    def guarded_build(task: BongardTask):
        files = tuple(store.glob("*.exposure.json"))
        assert len(files) == 1
        persisted = ExposureLedger.load(files[0])
        assert len(persisted.events) == len(predecessor.events) + 1
        calls.append(task.task_id)
        return original(task)

    monkeypatch.setattr(BongardTask, "build_manifest", guarded_build)
    precommit = _prepare(corpus, manifest, predecessor, store)
    assert calls == [precommit.selection.selected_task_id] * 2
    successor = precommit.exposure_successor
    assert successor.events[:-1] == predecessor.events
    assert len(successor.events) == len(predecessor.events) + 1
    assert successor.exposed_task_ids - predecessor.exposed_task_ids == frozenset(
        {precommit.selection.selected_task_id}
    )
    assert successor.events[-1].task_ids == (
        precommit.selection.selected_task_id,
    )
    assert not successor.events[-1].panel_ids
    persisted_path = store / precommit.exposure_persistence_receipt.filename
    assert persisted_path.is_file()
    assert precommit.exposure_successor.digest == precommit.exposure_successor_digest
    assert "on_exposure_precommit" not in inspect.signature(
        prepare_atomic_smoke_precommit
    ).parameters


def test_invalid_secrets_or_store_fail_before_pixel_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        BongardTask,
        "build_manifest",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("invalid precommit reached selected pixels")
        ),
    )
    common = {
        "corpus": corpus,
        "seed": SEED,
        "episode_seed": EPISODE_SEED,
        "full_corpus_manifest": manifest,
        "source_corpus_manifest_digest": manifest.digest,
        "source_dependency_digest": SOURCE_DEPENDENCIES,
        "exposure_ledger": predecessor,
        "expected_exposure_ledger_digest": predecessor.digest,
        "label_seal_nonce": LABEL_NONCE,
        "exposure_store_dir": store,
    }
    for changes in (
        {"label_seal_nonce": "sha256:" + LABEL_NONCE},
        {"episode_seed": "not-a-digest"},
        {"episode_seed": SEED},
    ):
        with pytest.raises(AtomicSmokePrecommitError):
            prepare_atomic_smoke_precommit(**{**common, **changes})
    assert not tuple(store.iterdir())
    bad_store = tmp_path / "not-a-directory"
    bad_store.write_text("x", encoding="utf-8")
    with pytest.raises(AtomicSmokePrecommitError, match="store"):
        prepare_atomic_smoke_precommit(
            **{**common, "exposure_store_dir": bad_store}
        )


def test_serialization_hides_query_commitment_and_cold_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    precommit = _prepare(corpus, manifest, predecessor, store)
    data = precommit.to_data()
    encoded = json.dumps(data, sort_keys=True)
    assert AtomicSmokePrecommit.from_data(data) == precommit
    detached = AtomicSmokePrecommit.from_data(data)
    with pytest.raises(AtomicSmokePrecommitError, match="no live private"):
        _ = detached.episode_plan
    assert set(data["episode_public_data"]) == {
        "version",
        "task_id",
        "family",
        "split",
        "regime",
        "run_id",
        "verifier_id",
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "label_commitment_digest",
    }
    for forbidden in (
        "latent_query_digest",
        '"queries"',
        '"query_id"',
        '"labels"',
        LABEL_NONCE,
        EPISODE_SEED,
    ):
        assert forbidden not in encoded
    replayed = cold_decode_and_replay_atomic_smoke_precommit(
        **_cold_kwargs(precommit, corpus, manifest, predecessor, store)
    )
    assert replayed.to_data() == data
    projected = replayed.episode_plan.to_data()
    projected.pop("latent_query_digest")
    assert projected == data["episode_public_data"]


def test_mutated_selected_pixels_are_rejected_after_durable_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    selection = select_atomic_smoke_task(
        corpus,
        seed=SEED,
        full_corpus_manifest=manifest,
        source_corpus_manifest_digest=manifest.digest,
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
    )
    corpus.task(selection.selected_task_id).positive[0].write_bytes(
        b"\x89PNG\r\n\x1a\nmutated-after-trusted-manifest"
    )
    with pytest.raises(AtomicSmokePrecommitError, match="trusted full-manifest"):
        _prepare(corpus, manifest, predecessor, store)
    files = tuple(store.glob("*.exposure.json"))
    assert len(files) == 1
    assert selection.selected_task_id in ExposureLedger.load(files[0]).exposed_task_ids


def test_redirected_task_binding_and_unknown_predecessor_fail_pixel_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    selection = select_atomic_smoke_task(
        corpus,
        seed=SEED,
        full_corpus_manifest=manifest,
        source_corpus_manifest_digest=manifest.digest,
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
    )
    other = next(task for task in corpus.tasks if task.task_id != selection.selected_task_id)
    corpus._by_id[selection.selected_task_id] = BongardTask(
        task_id=selection.selected_task_id,
        family="bd",
        root=other.root,
        positive=other.positive,
        negative=other.negative,
    )
    with pytest.raises(AtomicSmokePrecommitError, match="ownership"):
        _prepare(corpus, manifest, predecessor, store)
    assert not tuple(store.iterdir())

    corpus._by_id[selection.selected_task_id] = next(
        task for task in corpus.tasks if task.task_id == selection.selected_task_id
    )
    historical_parent = ExposureLedger(
        corpus_digest=predecessor.corpus_digest,
        events=predecessor.events[:-1],
    )
    forged = historical_parent.record(
        phase=P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PHASE,
        actor=P.OFFICIAL_SUCCESSOR_PREDECESSOR_APPEND_ACTOR,
        purpose=P.ATOMIC_SMOKE_PREDECESSOR_EXPOSURE_PURPOSE,
        task_ids=("bd_unknown_generator_0000",),
        source="atomic-smoke-selection:sha256:" + "2" * 64,
        observed_at="2026-08-06T11:00:00Z",
    )
    monkeypatch.setattr(
        P, "OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST", forged.digest
    )
    with pytest.raises(AtomicSmokePrecommitError, match="outside the official inventory"):
        select_atomic_smoke_task(
            corpus,
            seed=SEED,
            full_corpus_manifest=manifest,
            source_corpus_manifest_digest=manifest.digest,
            exposure_ledger=forged,
            expected_exposure_ledger_digest=forged.digest,
        )


def test_symlink_persistence_and_bool_numeric_laundering_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    selection = select_atomic_smoke_task(
        corpus,
        seed=SEED,
        full_corpus_manifest=manifest,
        source_corpus_manifest_digest=manifest.digest,
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
    )
    successor = predecessor.record(
        phase=P.ATOMIC_SMOKE_EXPOSURE_PHASE,
        actor="fixture-verifier",
        purpose=P.ATOMIC_SMOKE_EXPOSURE_PURPOSE,
        task_ids=(selection.selected_task_id,),
        source="atomic-smoke-selection:" + selection.digest,
        observed_at=OBSERVED_AT,
        known_task_ids=corpus.task_ids,
        require_unseen=True,
    )
    target = tmp_path / "target-ledger.json"
    target.write_text(successor.to_json(), encoding="utf-8")
    (store / (successor.digest.removeprefix("sha256:") + ".exposure.json")).symlink_to(
        target
    )
    with pytest.raises(AtomicSmokePrecommitError, match="no-follow"):
        _prepare(corpus, manifest, predecessor, store)

    (store / (successor.digest.removeprefix("sha256:") + ".exposure.json")).unlink()
    precommit = _prepare(corpus, manifest, predecessor, store)
    data = precommit.to_data()
    bad_selection = deepcopy(data["selection"])
    bad_selection["sample_size"] = True
    with pytest.raises(AtomicSmokePrecommitError, match="exact integer"):
        AtomicSmokeSelection.from_data(bad_selection)
    bad_manifest = deepcopy(data)
    bad_manifest["development_manifest_data"]["family_counts"]["bd"] = True
    with pytest.raises(AtomicSmokePrecommitError, match="scope"):
        AtomicSmokePrecommit.from_data(bad_manifest)


def test_cold_replay_rejects_rehashed_source_seed_episode_and_store_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, manifest, predecessor, store = _fixture(tmp_path, monkeypatch)
    precommit = _prepare(corpus, manifest, predecessor, store)
    common = _cold_kwargs(precommit, corpus, manifest, predecessor, store)
    rehashed = deepcopy(precommit.to_data())
    rehashed["source_dependency_digest"] = "0" * 64
    rehashed["precommit_digest"] = "sha256:" + canonical_digest(
        {key: value for key, value in rehashed.items() if key != "precommit_digest"}
    )
    with pytest.raises(AtomicSmokePrecommitError, match="source dependencies"):
        cold_decode_and_replay_atomic_smoke_precommit(
            **{
                **common,
                "value": rehashed,
                "expected_precommit_digest": rehashed["precommit_digest"],
            }
        )
    with pytest.raises(AtomicSmokePrecommitError, match="independent replay"):
        cold_decode_and_replay_atomic_smoke_precommit(
            **{**common, "seed": "different frozen seed"}
        )
    with pytest.raises(AtomicSmokePrecommitError, match="independent.*replay"):
        cold_decode_and_replay_atomic_smoke_precommit(
            **{**common, "episode_seed": "1" * 64}
        )
    persisted = store / precommit.exposure_persistence_receipt.filename
    persisted.write_text("{}\n", encoding="utf-8")
    with pytest.raises(AtomicSmokePrecommitError, match="persisted exposure"):
        cold_decode_and_replay_atomic_smoke_precommit(**common)
