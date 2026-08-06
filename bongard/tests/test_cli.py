from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard import cli
from bongard.artifacts import BlobRef, canonical_digest, canonical_json
from bongard.corpus import ShapeBongardCorpus
from bongard.exposure import (
    ExposureIntegrityError,
    ExposureLedger,
    ExposureViolation,
    semantic_resolver_policy_digest,
    semantic_policy_blocked_keys,
)
from bongard.historical_exposure import load_historical_exposure


TASK_ID = "bd_fixture_0000"
MODEL = "headless-codex-test"
MANIFEST_DIGEST = "sha256:" + "1" * 64
SUPPORT_DATA = {
    "run_id": "run-fixture",
    "issued_by": "canonical-bongard-verifier",
    "corpus_digest": "1" * 64,
    "support": [
        {
            "panel": {
                "blob_id": "support-negative-0",
                "sha256": "7" * 64,
                "byte_count": 1,
                "media_type": "image/png",
            },
            "positive": False,
        },
        {
            "panel": {
                "blob_id": "support-positive-0",
                "sha256": "8" * 64,
                "byte_count": 1,
                "media_type": "image/png",
            },
            "positive": True,
        },
    ],
    "verifier_nonce": "9" * 64,
    "version": "support-commitment/v1",
}
PLAN_DATA = {
    "version": "official-two-query-benchmark/v3",
    "task_id": TASK_ID,
    "family": "bd",
    "split": "train",
    "regime": None,
    "run_id": "run-fixture",
    "verifier_id": "canonical-bongard-verifier",
    "seed_digest": "2" * 64,
    "corpus_digest": "1" * 64,
    "task_manifest_digest": "3" * 64,
    "support_commitment_digest": canonical_digest(SUPPORT_DATA),
    "latent_query_digest": "5" * 64,
    "label_commitment_digest": "6" * 64,
}
PLAN_DIGEST = canonical_digest(PLAN_DATA)


@dataclass(frozen=True)
class _FakeManifest:
    digest: str = MANIFEST_DIGEST
    tasks: tuple[object, ...] = (
        SimpleNamespace(task_id=TASK_ID, panels=()),
    )


class _FakeCorpus:
    task_ids = (TASK_ID,)

    def __init__(self) -> None:
        self.manifest = _FakeManifest()
        self.split = SimpleNamespace(source_digest=None)

    def build_manifest(self) -> _FakeManifest:
        return self.manifest


class _FakeSession:
    def artifact_data(self) -> dict[str, object]:
        return {"schema": "fake-visual-artifact/v1"}


class _FakeScore:
    def to_data(self) -> dict[str, object]:
        return {
            "image_correct": 0,
            "image_total": 2,
            "image_accuracy": 0.0,
            "puzzle_correct": False,
            "puzzle_accuracy": 0.0,
            "determinate": 0,
            "abstentions": 2,
            "errors": 2,
        }


class _FakeResult:
    bundle = None
    support_gate = None
    proposal_freeze = None
    status = SimpleNamespace(value="proposal_error")
    score = _FakeScore()

    def __init__(self, split: str = "train", plan_digest: str = PLAN_DIGEST) -> None:
        self.split = split
        self.plan_digest = plan_digest

    def to_data(self) -> dict[str, object]:
        return {
            "version": "fake-protocol/v1",
            "task_id": TASK_ID,
            "family": "bd",
            "split": self.split,
            "regime": None,
            "run_id": "run-fixture",
            "plan_digest": self.plan_digest,
            "status": self.status.value,
            "score": self.score.to_data(),
            "phases": ["plan_committed", "support_released", "proposal_failed"],
            "artifact_chain": None,
            "failure": {
                "stage": "proposal",
                "error_type": "RuntimeError",
                "reason": "synthetic proposer failure",
            },
        }


def _plan(split: str = "train") -> SimpleNamespace:
    data = {**PLAN_DATA, "split": split}
    digest = canonical_digest(data)
    return SimpleNamespace(
        task_id=TASK_ID,
        split=split,
        regime=None,
        digest=digest,
        support=SimpleNamespace(to_data=lambda: dict(SUPPORT_DATA)),
        to_data=lambda: dict(data),
    )


def _install_fake_episode(
    monkeypatch: pytest.MonkeyPatch,
    exposure_dir: Path,
    *,
    split: str = "train",
    calls: list[str] | None = None,
) -> None:
    monkeypatch.setattr(cli, "prepare_episode", lambda *args, **kwargs: _plan(split))

    def fake_run_episode(
        plan, proposer, observer, *, support_gate_policy, sealed_guard=None
    ):
        assert support_gate_policy.mode.value == "empirical_replay"
        paths = tuple(exposure_dir.glob("*.exposure.json"))
        assert len(paths) == 1, "successor ledger must exist before support release"
        ledger = ExposureLedger.load(paths[0])
        assert ledger.events[-1].phase == "support_release_precommit"
        if calls is not None:
            calls.append("run_episode")
        return _FakeResult(split, plan.digest)

    monkeypatch.setattr(cli, "run_episode", fake_run_episode)


def _run_record(
    corpus: _FakeCorpus,
    exposure_dir: Path,
    *,
    ledger_in: Path | None = None,
    require_unseen: bool = False,
    sealed_test: bool = False,
):
    return cli._run_record(
        corpus=corpus,
        task_id=TASK_ID,
        seed="fixture-seed",
        session=_FakeSession(),
        sealed_test=sealed_test,
        exposure_dir=exposure_dir,
        ledger_in=ledger_in,
        require_unseen=require_unseen,
        model=MODEL,
    )


def test_run_parser_requires_exposure_directory():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
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
            ]
        )


def test_run_parser_rejects_arbitrary_codex_executable():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
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
                "ledger",
                "--codex",
                "/tmp/decoy",
            ]
        )


def test_require_unseen_and_sealed_test_require_input_ledger(tmp_path: Path):
    corpus = _FakeCorpus()
    with pytest.raises(cli.CliError, match="--require-unseen requires --ledger-in"):
        _run_record(corpus, tmp_path / "a", require_unseen=True)
    with pytest.raises(cli.CliError, match="--sealed-test requires --ledger-in"):
        _run_record(corpus, tmp_path / "b", sealed_test=True)


def test_precommit_is_persisted_before_proposal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = _FakeCorpus()
    exposure_dir = tmp_path / "exposure"
    calls: list[str] = []
    _install_fake_episode(monkeypatch, exposure_dir, calls=calls)

    record, result = _run_record(corpus, exposure_dir)
    successor_path = exposure_dir / record["exposure"]["successor_filename"]

    assert calls == ["run_episode"]
    assert result.bundle is None
    assert successor_path.is_file()
    successor = ExposureLedger.load(successor_path)
    assert len(successor.events) == 1
    event = successor.events[0]
    assert event.task_ids == (TASK_ID,)
    assert event.actor == MODEL
    assert event.phase == "support_release_precommit"
    assert TASK_ID in event.purpose
    assert MODEL in event.purpose
    assert PLAN_DIGEST in event.purpose

    exposure = record["exposure"]
    assert record["schema"] == "gkm.bongard-episode-run.v5"
    assert "complete-run" not in record["schema"]
    assert record["plan"] == _plan().to_data()
    assert record["support_commitment"] == SUPPORT_DATA
    assert canonical_digest(record["plan"]) == record["episode"]["plan_digest"]
    assert exposure["ledger_after_digest"] == successor.digest
    assert exposure["event_digest"] == event.digest
    assert exposure["successor_filename"] == successor_path.name
    assert exposure["external_anchor"] is None
    assert exposure["ledger_input_supplied"] is False
    assert exposure["unseen_required"] is False
    assert cli._verify_exposure_object(
        exposure,
        corpus_manifest_digest=record["corpus_manifest_digest"],
        episode=record["episode"],
    ) == exposure


def test_reusing_successor_with_require_unseen_fails_before_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = _FakeCorpus()
    first_dir = tmp_path / "first"
    calls: list[str] = []
    _install_fake_episode(monkeypatch, first_dir, calls=calls)
    first_record, _result = _run_record(corpus, first_dir)
    successor_path = first_dir / first_record["exposure"]["successor_filename"]
    assert calls == ["run_episode"]

    with pytest.raises(ExposureViolation, match="not unseen"):
        _run_record(
            corpus,
            tmp_path / "second",
            ledger_in=successor_path,
            require_unseen=True,
        )
    assert calls == ["run_episode"]
    assert not (tmp_path / "second").exists()


def test_sealed_test_forces_complete_guard_and_unseen_precommit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = _FakeCorpus()
    exposure_dir = tmp_path / "sealed"
    blank_path = ExposureLedger.create(MANIFEST_DIGEST).write_content_addressed(
        tmp_path / "input"
    )
    _install_fake_episode(monkeypatch, exposure_dir, split="test")

    class FakeGuard:
        captures: list[bool] = []

        @classmethod
        def capture(cls, corpus_arg, *, corpus_manifest, require_complete):
            assert corpus_arg is corpus
            assert corpus_manifest is corpus.manifest
            cls.captures.append(require_complete)
            return cls()

        def verify_all(self):
            return None

    monkeypatch.setattr(cli, "SealedTestGuard", FakeGuard)
    sealed_record, _result = _run_record(
        corpus,
        exposure_dir,
        ledger_in=blank_path,
        sealed_test=True,
    )
    successor_path = exposure_dir / sealed_record["exposure"]["successor_filename"]
    assert FakeGuard.captures == [True]
    assert sealed_record["exposure"]["ledger_input_supplied"] is True
    assert sealed_record["exposure"]["unseen_required"] is True

    with pytest.raises(ExposureViolation, match="not unseen"):
        _run_record(
            corpus,
            tmp_path / "sealed-again",
            ledger_in=successor_path,
            sealed_test=True,
        )


def test_input_ledger_must_be_bound_to_manifest_and_known_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = _FakeCorpus()
    _install_fake_episode(monkeypatch, tmp_path / "out")

    wrong_corpus = ExposureLedger.create("sha256:" + "9" * 64)
    wrong_path = wrong_corpus.write_content_addressed(tmp_path / "wrong")
    with pytest.raises(ExposureViolation, match="ledger belongs"):
        _run_record(corpus, tmp_path / "out", ledger_in=wrong_path)

    unknown = ExposureLedger.create(MANIFEST_DIGEST).record(
        phase="historical",
        actor="fixture",
        purpose="unknown task fixture",
        task_ids=("bd_unknown_0000",),
    )
    unknown_path = unknown.write_content_addressed(tmp_path / "unknown")
    with pytest.raises(cli.CliError, match="outside the corpus manifest"):
        _run_record(corpus, tmp_path / "out", ledger_in=unknown_path)


def test_verify_rejects_tampered_embedded_exposure_before_archive_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = _FakeCorpus()
    exposure_dir = tmp_path / "exposure"
    _install_fake_episode(monkeypatch, exposure_dir)
    record, _result = _run_record(corpus, exposure_dir)
    record["exposure"]["event"]["purpose"] = "tampered"
    content = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = canonical_digest(content)
    run_path = tmp_path / "run.json"
    run_path.write_bytes(canonical_json(record))
    expected_sha256 = hashlib.sha256(run_path.read_bytes()).hexdigest()

    with pytest.raises(cli.CliError, match="purpose does not bind"):
        cli._verify(
            SimpleNamespace(
                run=str(run_path),
                expected_sha256=expected_sha256,
            )
        )


def test_verify_requires_exact_external_root_hash(tmp_path: Path):
    path = tmp_path / "run.json"
    path.write_bytes(canonical_json({"schema": "irrelevant"}))
    with pytest.raises(cli.CliError, match="expected run SHA-256"):
        cli._verify(SimpleNamespace(run=str(path), expected_sha256="BAD"))
    with pytest.raises(cli.CliError, match="run file SHA-256"):
        cli._verify(SimpleNamespace(run=str(path), expected_sha256="0" * 64))


def _task_mapping_fixture(tmp_path: Path):
    root = tmp_path / "ShapeBongard_V2"
    task_id = "bd_mapping_fixture_0000"
    for label, offset in (("1", 0), ("0", 7)):
        directory = root / "bd" / "images" / task_id / label
        directory.mkdir(parents=True)
        for index in range(7):
            (directory / f"{index}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + bytes([offset + index])
            )
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({"train": [task_id]}), encoding="utf-8"
    )
    corpus = ShapeBongardCorpus.from_root(root)
    task_manifest = corpus.build_manifest().tasks[0]
    positive = tuple(
        sorted(
            (panel for panel in task_manifest.panels if panel.polarity == "positive"),
            key=lambda panel: panel.index,
        )
    )
    negative = tuple(
        sorted(
            (panel for panel in task_manifest.panels if panel.polarity == "negative"),
            key=lambda panel: panel.index,
        )
    )

    def ref_for(panel, blob_id):
        return BlobRef(
            blob_id=blob_id,
            sha256=panel.sha256.removeprefix("sha256:"),
            byte_count=panel.size_bytes,
            media_type="image/png",
        )

    refs = tuple(
        [
            ref_for(panel, f"support-positive-{slot}")
            for slot, panel in enumerate(positive[:6])
        ]
        + [
            ref_for(panel, f"support-negative-{slot}")
            for slot, panel in enumerate(negative[:6])
        ]
        + [
            ref_for(positive[6], "query-panel-0"),
            ref_for(negative[6], "query-panel-1"),
        ]
    )

    def archive_for(candidate_refs, *, query_labels=(True, False)):
        return SimpleNamespace(
            bundle=SimpleNamespace(
                support=SimpleNamespace(
                    support=tuple(
                        SimpleNamespace(
                            panel=item,
                            positive=item.blob_id.startswith("support-positive-"),
                        )
                        for item in candidate_refs[:12]
                    )
                ),
                release=SimpleNamespace(
                    queries=tuple(
                        SimpleNamespace(query_id=f"query-{index}", panel=item)
                        for index, item in enumerate(candidate_refs[12:])
                    )
                ),
                labels=SimpleNamespace(
                    labels=tuple(
                        SimpleNamespace(
                            query_id=f"query-{index}", positive=positive_label
                        )
                        for index, positive_label in enumerate(query_labels)
                    )
                ),
            )
        )

    return corpus, task_manifest, refs, archive_for


def test_official_task_byte_mapping_is_exact_bijective_and_rejects_missing(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    mapped = cli._map_official_task_blob_bytes(task_manifest, archive_for(refs))
    assert set(mapped) == {item.blob_id for item in refs}
    assert len(mapped) == 14
    missing_ref = BlobRef(
        refs[-1].blob_id,
        "f" * 64,
        refs[-1].byte_count,
        "image/png",
    )
    with pytest.raises(cli.CliError, match="missing=.*extras="):
        cli._map_official_task_blob_bytes(
            task_manifest, archive_for((*refs[:-1], missing_ref))
        )


def test_official_rejected_support_mapping_preserves_names_and_polarity(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, _archive_for = _task_mapping_fixture(tmp_path)
    support_refs = refs[:12]
    names = tuple(
        [f"pos_{index}.png" for index in range(6)]
        + [f"neg_{index}.png" for index in range(6)]
    )
    attempt = {
        "support_presentation": [
            {
                "name": name,
                "byte_count": ref.byte_count,
                "content_digest": ref.sha256,
            }
            for name, ref in zip(names, support_refs, strict=True)
        ]
    }
    mapped = cli._map_official_rejected_support_bytes(task_manifest, attempt)
    assert tuple(mapped) == names
    assert all(
        hashlib.sha256(payload).hexdigest()
        == attempt["support_presentation"][index]["content_digest"]
        for index, payload in enumerate(mapped.values())
    )

    swapped = json.loads(json.dumps(attempt))
    swapped["support_presentation"][0]["content_digest"] = support_refs[6].sha256
    swapped["support_presentation"][0]["byte_count"] = support_refs[6].byte_count
    with pytest.raises(cli.CliError, match="absent from the official task"):
        cli._map_official_rejected_support_bytes(task_manifest, swapped)

def test_official_task_byte_mapping_rejects_duplicate_identity(tmp_path: Path) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    duplicate_ref = BlobRef(
        refs[1].blob_id,
        refs[0].sha256,
        refs[0].byte_count,
        "image/png",
    )
    with pytest.raises(cli.CliError, match=r"ambiguous digest\+size"):
        cli._map_official_task_blob_bytes(
            task_manifest,
            archive_for((refs[0], duplicate_ref, *refs[2:])),
        )


def _ref_with_identity(role: BlobRef, source: BlobRef) -> BlobRef:
    return BlobRef(
        role.blob_id,
        source.sha256,
        source.byte_count,
        source.media_type,
    )


def test_official_task_mapping_rejects_resealed_polarity_permutation(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    swapped = list(refs)
    swapped[0] = _ref_with_identity(refs[0], refs[6])
    swapped[6] = _ref_with_identity(refs[6], refs[0])

    with pytest.raises(cli.CliError, match="official polarity"):
        cli._map_official_task_blob_bytes(task_manifest, archive_for(tuple(swapped)))


def test_official_task_mapping_rejects_resealed_support_index_permutation(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    swapped = list(refs)
    swapped[0] = _ref_with_identity(refs[0], refs[1])
    swapped[1] = _ref_with_identity(refs[1], refs[0])

    with pytest.raises(cli.CliError, match="official panel index/identity"):
        cli._map_official_task_blob_bytes(task_manifest, archive_for(tuple(swapped)))


def test_official_task_mapping_rejects_resealed_query_label(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    with pytest.raises(cli.CliError, match="official polarity"):
        cli._map_official_task_blob_bytes(
            task_manifest,
            archive_for(refs, query_labels=(False, False)),
        )


def test_official_mapping_does_not_claim_seed_selected_query_order(
    tmp_path: Path,
) -> None:
    _corpus, task_manifest, refs, archive_for = _task_mapping_fixture(tmp_path)
    swapped = list(refs)
    swapped[12] = _ref_with_identity(refs[12], refs[13])
    swapped[13] = _ref_with_identity(refs[13], refs[12])

    # The v3 run commits only the seed digest.  Exact official verification
    # binds both submitted query roles and their labels, but cannot reproduce
    # seed-selected order without a future schema carrying the seed preimage.
    mapped = cli._map_official_task_blob_bytes(
        task_manifest,
        archive_for(tuple(swapped), query_labels=(False, True)),
    )
    assert set(mapped) == {item.blob_id for item in refs}


def test_official_corpus_binding_rejects_swapped_task_metadata(tmp_path: Path) -> None:
    corpus, task_manifest, _refs, _archive_for = _task_mapping_fixture(tmp_path)
    manifest = corpus.build_manifest()
    record = {
        "corpus_manifest_digest": manifest.digest,
        "split_source_digest": corpus.split.source_digest,
        "episode": {
            "task_id": task_manifest.task_id,
            "family": task_manifest.family,
            "split": "train",
            "regime": None,
        },
        "plan": {
            "task_id": "bd_swapped_fixture_0000",
            "family": task_manifest.family,
            "split": "train",
            "regime": None,
            "task_manifest_digest": task_manifest.digest.removeprefix("sha256:"),
        },
    }
    with pytest.raises(cli.CliError, match="task identity differs"):
        cli._bind_record_to_official_corpus(record, corpus, manifest)


def test_verify_parser_requires_official_corpus_and_archive() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["verify", "--run", "run.json", "--expected-sha256", "0" * 64]
        )


def test_cli_summary_includes_all_precommit_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
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
    successor = ledger_dir / exposure["successor_filename"]
    successor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_corpus", lambda args: _FakeCorpus())
    monkeypatch.setattr(cli, "HeadlessCodexEpisode", lambda **kwargs: _FakeSession())
    monkeypatch.setattr(
        cli,
        "_run_record",
        lambda **kwargs: (record, result),
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
        proposer_minutes=1,
        observer_minutes=1,
        verbose=False,
        task_id=TASK_ID,
        seed="seed",
        out=str(tmp_path / "run.json"),
    )

    assert cli._run(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["exposure_ledger_before_digest"] == exposure["ledger_before_digest"]
    assert summary["exposure_ledger_after_digest"] == exposure["ledger_after_digest"]
    assert summary["exposure_event_digest"] == exposure["event_digest"]
    assert summary["exposure_ledger_out"] == str(successor.resolve())
    assert summary["official_release_digest"] is None
    assert summary["codex_launcher_sha256"] is None
    assert summary["codex_cli_version"] is None


def test_sealed_cli_requires_exact_official_archive_identity():
    with pytest.raises(cli.CliError, match="exact --official-release"):
        cli._validate_release_args(
            official_release=False,
            archive=None,
            sealed_test=True,
        )
    with pytest.raises(cli.CliError, match="requires --archive"):
        cli._validate_release_args(
            official_release=True,
            archive=None,
            sealed_test=False,
        )


def test_official_run_requires_externally_pinned_codex_launcher():
    with pytest.raises(cli.CliError, match="expected-codex-launcher-sha256"):
        cli._validate_codex_launcher(expected_sha256=None, official_release=True)


def test_codex_launcher_pin_is_checked_before_run(
    monkeypatch: pytest.MonkeyPatch,
):
    actual = "a" * 64
    monkeypatch.setattr(
        cli,
        "codex_cli_fingerprint",
        lambda executable: {
            "version": "codex-cli test",
            "launcher_digest": actual,
        },
    )
    with pytest.raises(cli.CliError, match="fixed Codex launcher SHA-256"):
        cli._validate_codex_launcher(
            expected_sha256="b" * 64,
            official_release=True,
        )
    assert cli._validate_codex_launcher(
        expected_sha256=actual,
        official_release=True,
    ) == {
        "version": "codex-cli test",
        "launcher_digest": actual,
    }


def _cohort_cli_corpus(
    tmp_path: Path,
) -> tuple[ShapeBongardCorpus, tuple[str, ...], str]:
    historical = load_historical_exposure()
    exposed_pair = historical.abstract_partition.sealed[0]
    eligible_pair = historical.abstract_partition.sealed[1]
    sibling_ids = tuple(
        f"hd_{exposed_pair[0]}-{exposed_pair[1]}_{index:04d}"
        for index in range(3)
    )
    eligible_id = f"hd_{eligible_pair[0]}-{eligible_pair[1]}_0000"
    task_ids = sibling_ids + (eligible_id,)
    root = tmp_path / "ShapeBongard_V2"
    for task_ordinal, task_id in enumerate(task_ids):
        for label_ordinal, label in enumerate(("1", "0")):
            directory = root / "hd" / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for panel_index in range(7):
                directory.joinpath(f"{panel_index}.png").write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                    + bytes((task_ordinal, label_ordinal, panel_index))
                )
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({"train": list(task_ids)}), encoding="utf-8"
    )
    return ShapeBongardCorpus.from_root(root), sibling_ids, eligible_id


def _cohort_args(
    corpus: ShapeBongardCorpus,
    *,
    ledger_in: Path | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        corpus=str(corpus.root),
        split_file=None,
        require_complete=False,
        split=None,
        family="hd",
        cohort="sealed",
        limit=20,
        ledger_in=str(ledger_in) if ledger_in is not None else None,
        out=None,
    )


def _membership_digest(task_ids: tuple[str, ...]) -> str:
    payload = "".join(f"{task_id}\n" for task_id in sorted(task_ids)).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_cohorts_without_live_ledger_preserves_frozen_report_and_stays_metadata_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, sibling_ids, eligible_id = _cohort_cli_corpus(tmp_path)
    historical = load_historical_exposure()
    frozen = cli.build_cohort_report(
        corpus, historical, family="hd", cohort="sealed"
    )
    monkeypatch.setattr(cli, "_load_corpus", lambda args: corpus)
    monkeypatch.setattr(
        corpus,
        "build_manifest",
        lambda: (_ for _ in ()).throw(
            AssertionError("no live ledger should trigger no panel hashing")
        ),
    )

    assert cli._cohorts(_cohort_args(corpus, ledger_in=None)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["report_digest"] == frozen.digest
    assert output["counts"] == dict(frozen.counts)
    assert output["membership_digests"] == dict(frozen.membership_digests)
    assert output["selected_task_ids"] == sorted((*sibling_ids, eligible_id))
    assert "live_eligibility" not in output


def test_cohorts_live_ledger_filters_exact_and_semantic_siblings_with_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, sibling_ids, eligible_id = _cohort_cli_corpus(tmp_path)
    historical = load_historical_exposure()
    frozen = cli.build_cohort_report(
        corpus, historical, family="hd", cohort="sealed"
    )
    manifest = corpus.build_manifest()
    panel_only_id = f"hd/{sibling_ids[0]}/1/0.png"
    ledger = ExposureLedger.create(manifest.digest).record(
        phase="support_release_precommit",
        actor="fixture",
        purpose="live cohort exclusion fixture",
        panel_ids=(panel_only_id,),
        source="test",
        observed_at="2026-08-06T00:00:00Z",
    )
    ledger_path = ledger.write_once(tmp_path / "live-ledger.json")
    ledger_bytes = ledger_path.read_bytes()
    real_build_manifest = corpus.build_manifest
    manifest_builds: list[str] = []

    def freshly_build_manifest():
        manifest_builds.append("built")
        return real_build_manifest()

    monkeypatch.setattr(cli, "_load_corpus", lambda args: corpus)
    monkeypatch.setattr(corpus, "build_manifest", freshly_build_manifest)

    assert cli._cohorts(_cohort_args(corpus, ledger_in=ledger_path)) == 0
    output = json.loads(capsys.readouterr().out)
    live = output["live_eligibility"]
    semantic_ids = tuple(sorted(sibling_ids))
    exact_ids = (sibling_ids[0],)

    # The frozen historical report is unchanged; only selected_task_ids gains
    # the separately named live overlay.
    assert output["report_digest"] == frozen.digest
    assert output["counts"] == dict(frozen.counts)
    assert output["membership_digests"] == dict(frozen.membership_digests)
    assert output["selected_task_ids"] == [eligible_id]
    assert live["corpus_manifest_digest"] == manifest.digest
    assert live["ledger_digest"] == ledger.digest
    assert live["historical_seed_digest"] == historical.seed_digest
    assert live["semantic_resolver_policy_digest"] == (
        semantic_resolver_policy_digest(historical)
    )
    blocked_count = len(semantic_policy_blocked_keys(historical))
    assert live["counts"] == {
        "historical_scope": 4,
        "ledger_recorded_tasks_total": 1,
        "ledger_recorded_semantic_keys_total": 1,
        "policy_blocked_semantic_keys_total": blocked_count,
        "effective_exposed_semantic_keys_total": blocked_count + 1,
        "exact_task_collision": 1,
        "semantic_key_collision": 3,
        "exact_and_semantic_collision": 1,
        "live_excluded_union": 3,
        "live_eligible": 1,
    }
    assert live["policy_blocked_semantic_keys_digest"].startswith("sha256:")
    assert live["effective_exposed_semantic_keys_digest"].startswith("sha256:")
    assert live["membership_digests"] == {
        "exact_task_collision": _membership_digest(exact_ids),
        "semantic_key_collision": _membership_digest(semantic_ids),
        "exact_and_semantic_collision": _membership_digest(exact_ids),
        "live_excluded_union": _membership_digest(semantic_ids),
        "live_eligible": _membership_digest((eligible_id,)),
    }
    live_content = {key: value for key, value in live.items() if key != "digest"}
    assert live["digest"] == "sha256:" + canonical_digest(live_content)
    assert manifest_builds == ["built"]
    assert ledger_path.read_bytes() == ledger_bytes


def test_cohorts_live_ledger_rejects_wrong_corpus_and_malformed_recorded_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, sibling_ids, _eligible_id = _cohort_cli_corpus(tmp_path)
    manifest = corpus.build_manifest()
    monkeypatch.setattr(cli, "_load_corpus", lambda args: corpus)

    wrong = ExposureLedger.create("sha256:" + "9" * 64).record(
        phase="historical",
        actor="fixture",
        purpose="wrong corpus",
        task_ids=(sibling_ids[0],),
        observed_at="2026-08-06T00:00:00Z",
    )
    wrong_path = wrong.write_once(tmp_path / "wrong-corpus.json")
    with pytest.raises(ExposureViolation, match="ledger belongs"):
        cli._cohorts(_cohort_args(corpus, ledger_in=wrong_path))

    malformed = ExposureLedger.create(manifest.digest).record(
        phase="historical",
        actor="fixture",
        purpose="malformed recorded id",
        task_ids=("not-an-official-task-id",),
        observed_at="2026-08-06T00:00:00Z",
    )
    malformed_path = malformed.write_once(tmp_path / "malformed-id.json")
    with pytest.raises(cli.CliError, match="outside the corpus manifest"):
        cli._cohorts(_cohort_args(corpus, ledger_in=malformed_path))

    valid = ExposureLedger.create(manifest.digest).record(
        phase="historical",
        actor="fixture",
        purpose="serialized malformed id",
        task_ids=(sibling_ids[0],),
        observed_at="2026-08-06T00:00:00Z",
    ).to_dict()
    valid["events"][0]["task_ids"] = [7]
    malformed_serialized = tmp_path / "malformed-serialized-id.json"
    malformed_serialized.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ExposureIntegrityError, match="task_ids must contain"):
        cli._cohorts(_cohort_args(corpus, ledger_in=malformed_serialized))


def test_cohorts_parser_accepts_optional_live_ledger() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "cohorts",
            "--corpus",
            "corpus",
            "--ledger-in",
            "ledger.json",
        ]
    )
    assert args.handler is cli._cohorts
    assert args.ledger_in == "ledger.json"
