from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

import bongard.semantic_calibration_command as command_module
from bongard.artifacts import canonical_digest, canonical_json
from bongard.corpus import CorpusManifest, ShapeBongardCorpus, SplitIndex
from bongard.exposure import ExposureLedger
from bongard.release import OfficialReleaseDescriptor
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_SELECTION_ALGORITHM,
    CAMPAIGN_SELECTION_ALGORITHM_V1,
    TRANSPORT_FAILED,
    SemanticCalibrationCampaignFitFailed,
    SemanticCalibrationCampaignNoSoftClaims,
    SemanticCalibrationCampaignProposalPhaseFailed,
    SemanticCalibrationCampaignScoringFailed,
)
from bongard.semantic_calibration_command import (
    DESCRIPTIVE_STAGE_A_DESIGN,
    STAGE_A_COMMAND_RECEIPT_SCHEMA_V1,
    STAGE_A_OPERATIONAL_FAILURE_SCHEMA,
    STAGE_A_SCOPE,
    STAGE_A_SOURCE_DEPENDENCY_SCOPE,
    StageACalibrationCommandConfig,
    StageACalibrationCommandError,
    StageACommandReceipt,
    StageAPersistenceConfig,
    StageATrustedCorpus,
    execute_stage_a_calibration,
    freeze_stage_a_source_dependencies,
    load_stage_a_cache_snapshot,
    load_stage_a_command_receipt,
    persist_stage_a_cache_snapshot,
    persist_stage_a_outcome,
    run_stage_a_calibration_command,
)
from bongard.transport import CloudPolicyCacheSnapshot


LAUNCHER = hashlib.sha256(b"externally pinned Codex launcher").hexdigest()
MANIFEST = "sha256:" + hashlib.sha256(b"trusted corpus manifest").hexdigest()
SELECTION_SEED = hashlib.sha256(
    b"fresh external selection beacon after protocol freeze"
).hexdigest()
SELECTION_PROVENANCE = "fixture-os.urandom-after-population-and-protocol-freeze"


def _trusted(tmp_path: Path) -> StageATrustedCorpus:
    split = SplitIndex.empty()
    corpus = ShapeBongardCorpus(
        tmp_path / "trusted-corpus",
        (),
        layout="archive",
        split=split,
    )
    manifest = CorpusManifest(
        layout="archive",
        family_counts=tuple(corpus.family_counts.items()),
        tasks=(),
        split=split,
        digest=MANIFEST,
    )
    return StageATrustedCorpus.from_trusted_objects(
        corpus=corpus,
        full_manifest=manifest,
        trust_authority="synthetic-focused-test-boundary/v1",
    )


def _config(ledger: ExposureLedger, **changes: Any) -> StageACalibrationCommandConfig:
    values: dict[str, Any] = {
        "expected_codex_launcher_digest": LAUNCHER,
        "expected_exposure_ledger_digest": ledger.digest,
        "design_mode": DESCRIPTIVE_STAGE_A_DESIGN,
        "selection_seed": SELECTION_SEED,
        "selection_seed_provenance": SELECTION_PROVENANCE,
        "candidate_count": 2,
        "families": ("bd",),
        "minimum_clusters_per_bin": 2,
    }
    values.update(changes)
    return StageACalibrationCommandConfig(**values)


def _successor(ledger: ExposureLedger, count: int) -> ExposureLedger:
    result = ledger
    for index in range(count):
        result = result.record(
            phase="semantic-calibration",
            actor="fixture",
            purpose="stage-a-soft-scorer-calibration-candidate",
            task_ids=(f"bd_fixture_{index:04d}",),
            observed_at=f"2026-08-06T00:00:{index:02d}Z",
            require_unseen=True,
        )
    return result


@dataclass
class _FakeArchive:
    protocol: Any
    selection_seed: str
    candidate_count: int
    families: tuple[str, ...]
    semantic_cohort: str
    source_corpus_manifest_digest: str
    exposure_predecessor: ExposureLedger
    exposure_successor: ExposureLedger
    execution_config: Any
    selection_algorithm: str = CAMPAIGN_SELECTION_ALGORITHM
    records: tuple[Any, ...] = ()

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "fixture-stage-a-proposal-archive/v1",
            "candidate_count": self.candidate_count,
            "selection_seed": self.selection_seed,
            "exposure_successor_digest": self.exposure_successor.digest,
        }


class _FakeCampaign:
    def __init__(self, archive: _FakeArchive) -> None:
        self.score_batch = SimpleNamespace(
            commitment_batch=SimpleNamespace(proposal_archive=archive)
        )
        self._content = {
            "schema": "fixture-stage-a-campaign/v1",
            "reference_execution": "python-only/v1",
        }
        self.digest = canonical_digest(self._content)

    def to_data(self) -> dict[str, object]:
        return {**self._content, "campaign_digest": self.digest}


def _archive(
    *,
    protocol: Any,
    kwargs: dict[str, Any],
    ledger: ExposureLedger,
    snapshot: CloudPolicyCacheSnapshot,
) -> _FakeArchive:
    config = SimpleNamespace(
        proposer_minutes=kwargs["proposer_minutes"],
        scorer_minutes=kwargs["scorer_minutes"],
        proposer_max_workers=kwargs["proposer_max_workers"],
        scorer_max_workers=kwargs["scorer_max_workers"],
        executable=kwargs["executable"],
        expected_codex_launcher_digest=kwargs[
            "expected_codex_launcher_digest"
        ],
        cloud_policy_cache_binding=snapshot.binding,
    )
    return _FakeArchive(
        protocol=protocol,
        selection_seed=kwargs["seed"],
        candidate_count=kwargs["candidate_count"],
        families=kwargs["families"],
        semantic_cohort=kwargs["semantic_cohort"],
        source_corpus_manifest_digest=kwargs["source_corpus_manifest_digest"],
        exposure_predecessor=ledger,
        exposure_successor=_successor(ledger, kwargs["candidate_count"]),
        execution_config=config,
    )


def _fake_success_outcome(
    tmp_path: Path,
    *,
    source_dependency_root: Path | None = None,
):
    trusted = _trusted(tmp_path)
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)
    campaigns: list[_FakeCampaign] = []

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        campaign = _FakeCampaign(archive)
        campaigns.append(campaign)
        return campaign

    def verifier(raw, **kwargs):
        assert raw == campaigns[0].to_data()
        return campaigns[0], {}

    outcome = execute_stage_a_calibration(
        trusted,
        ledger,
        config,
        on_exposure_precommit=lambda successor, frozen_snapshot: None,
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
        campaign_verifier=verifier,
        source_dependency_root=source_dependency_root,
    )
    return outcome, trusted, ledger, config, snapshot


def test_requested_defaults_are_frozen_but_selection_scope_is_configurable() -> None:
    ledger = ExposureLedger.create(MANIFEST)
    config = StageACalibrationCommandConfig(
        expected_codex_launcher_digest=LAUNCHER,
        expected_exposure_ledger_digest=ledger.digest,
        design_mode=DESCRIPTIVE_STAGE_A_DESIGN,
        selection_seed=SELECTION_SEED,
        selection_seed_provenance=SELECTION_PROVENANCE,
    )

    assert config.candidate_count == 48
    assert config.selection_seed == SELECTION_SEED
    assert config.selection_seed_provenance == SELECTION_PROVENANCE
    assert config.design_mode == DESCRIPTIVE_STAGE_A_DESIGN
    assert config.semantic_cohort == "drill"
    assert config.families == ("bd", "hd")
    assert config.score_bin_edges == (0.0, 0.75, 1.0)
    assert config.affirmative_boundary == 0.5
    assert config.confidence_level == 0.90
    assert config.minimum_clusters_per_bin == 12
    assert config.proposer_model_id == config.scorer_model_id == "gpt-5.6-sol"
    assert config.proposer_reasoning_effort == config.scorer_reasoning_effort == "medium"
    assert (config.proposer_max_workers, config.scorer_max_workers) == (4, 4)
    assert (config.proposer_minutes, config.scorer_minutes) == (15, 10)
    assert config.to_data()["stage_a_scope"] == STAGE_A_SCOPE
    assert config.to_data()["reference_execution"] == "python-only/v1"

    bd_only = _config(ledger, candidate_count=7, families=("bd",))
    hd_only = _config(ledger, candidate_count=9, families=("hd",))
    assert bd_only.families == ("bd",)
    assert hd_only.families == ("hd",)
    assert bd_only.candidate_count == 7
    assert hd_only.candidate_count == 9

    with pytest.raises(
        StageACalibrationCommandError,
        match="no inferential design is authorized",
    ):
        _config(ledger, design_mode="pooled-independent-hoeffding/v1")
    with pytest.raises(
        StageACalibrationCommandError,
        match="256-bit selection seed",
    ):
        _config(
            ledger,
            selection_seed=(
                "shape-bongard-v2-clean-drill-soft-calibration-stage-a-v1"
            ),
        )


def test_source_identity_excludes_only_the_detached_checker_sidecar(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-identity"
    source_root.mkdir()
    authority = source_root / "authority.py"
    checker = source_root / "semantic_checker.py"
    nested_checker = source_root / "optional" / "semantic_checker.py"
    nested_checker.parent.mkdir()
    authority.write_text("VALUE = 1\n", encoding="utf-8")
    checker.write_text("CHECKER = 'one'\n", encoding="utf-8")
    nested_checker.write_text("NOT_THE_EXCLUDED_FILE = 1\n", encoding="utf-8")

    baseline = freeze_stage_a_source_dependencies(source_root)
    assert baseline.content_data()["scope"] == STAGE_A_SOURCE_DEPENDENCY_SCOPE
    assert tuple(path for path, _, _ in baseline.entries) == (
        "authority.py",
        "optional/semantic_checker.py",
    )

    checker.write_text("CHECKER = 'changed'\n", encoding="utf-8")
    assert freeze_stage_a_source_dependencies(source_root) == baseline
    checker.unlink()
    assert freeze_stage_a_source_dependencies(source_root) == baseline

    authority.write_text("VALUE = 2\n", encoding="utf-8")
    assert freeze_stage_a_source_dependencies(source_root) != baseline


def test_checker_edit_or_removal_cannot_change_stage_a_receipt_authority(
    tmp_path: Path,
) -> None:
    work = tmp_path / "checker-neutral-receipt"
    source_root = work / "sources"
    source_root.mkdir(parents=True)
    (source_root / "authority.py").write_text("VALUE = 1\n", encoding="utf-8")
    checker = source_root / "semantic_checker.py"
    checker.write_text("CHECKER = 'one'\n", encoding="utf-8")
    outcome, _, _, _, _ = _fake_success_outcome(
        work,
        source_dependency_root=source_root,
    )
    persistence = StageAPersistenceConfig(
        artifact_directory=work / "artifacts",
        exposure_directory=work / "exposure",
        cache_snapshot_directory=work / "cache",
    )

    baseline = persist_stage_a_outcome(outcome, persistence)
    checker.write_text("CHECKER = 'changed'\n", encoding="utf-8")
    after_edit = persist_stage_a_outcome(outcome, persistence)
    checker.unlink()
    after_removal = persist_stage_a_outcome(outcome, persistence)

    assert baseline.status == "succeeded"
    assert after_edit == baseline
    assert after_removal == baseline
    receipt = load_stage_a_command_receipt(
        baseline.command_receipt_path,
        baseline.command_receipt_digest,
    )
    assert receipt.source_dependencies == outcome.source_dependencies
    assert receipt.source_dependencies.content_data()["scope"] == (
        STAGE_A_SOURCE_DEPENDENCY_SCOPE
    )


def test_official_input_constructor_invokes_archive_and_fresh_corpus_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = _trusted(tmp_path)
    release = OfficialReleaseDescriptor(
        release_id="fixture-official-release",
        archive_filename="ShapeBongard_V2.zip",
        archive_sha256="sha256:" + "1" * 64,
        archive_size_bytes=1,
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + "2" * 64,
        split_size_bytes=1,
        upstream_repository="fixture/repository",
        upstream_commit="3" * 40,
        family_counts=tuple(sorted(trusted.full_manifest.family_counts)),
        primary_split_counts=(("test", 0), ("train", 0), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256="sha256:" + "4" * 64,
        corpus_manifest_sha256=trusted.full_manifest.digest,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        OfficialReleaseDescriptor,
        "verify_archive",
        lambda self, path: calls.append(("archive", Path(path))),
    )

    def verify_corpus(self, corpus, *, manifest=None):
        calls.append(("corpus", corpus, manifest))
        return trusted.full_manifest

    monkeypatch.setattr(OfficialReleaseDescriptor, "verify_corpus", verify_corpus)
    result = StageATrustedCorpus.from_official_release(
        corpus=trusted.corpus,
        release=release,
        archive_path=tmp_path / "ShapeBongard_V2.zip",
        supplied_manifest=trusted.full_manifest,
    )

    assert [item[0] for item in calls] == ["archive", "corpus"]
    assert result.authentication_mode == "official-release-archive-and-corpus/v1"
    assert result.release_descriptor_digest == release.digest
    assert result.archive_sha256 == release.archive_sha256


def test_launcher_preflight_and_one_cache_snapshot_precede_campaign_and_cold_replay(
    tmp_path: Path,
) -> None:
    trusted = _trusted(tmp_path)
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    events: list[str] = []
    snapshots: list[CloudPolicyCacheSnapshot] = []
    campaigns: list[_FakeCampaign] = []

    def fingerprint(executable: str, *, expected_launcher_digest: str):
        events.append("fingerprint")
        assert executable == "codex"
        assert expected_launcher_digest == LAUNCHER
        return {"version": "codex-cli fixture", "launcher_digest": LAUNCHER}

    def snapshotter():
        events.append("snapshot")
        snapshot = CloudPolicyCacheSnapshot(
            canonical_json(
                {
                    "signed_payload": {
                        "account_id": "fixture-account-identifier",
                        "chatgpt_user_id": "fixture-user-identifier",
                        "bundle": {"config_toml": {}, "requirements_toml": {}},
                    },
                    "signature": "fixture-signature",
                }
            )
        )
        snapshots.append(snapshot)
        return snapshot

    def runner(corpus, protocol, **kwargs):
        events.append("campaign")
        assert corpus is trusted.corpus
        assert kwargs["cloud_policy_cache_snapshot"] is snapshots[0]
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshots[0],
        )
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        campaign = _FakeCampaign(archive)
        campaigns.append(campaign)
        return campaign

    def verifier(raw, *, corpus, corpus_manifest):
        events.append("cold-verify")
        assert raw == campaigns[0].to_data()
        assert corpus is trusted.corpus
        assert corpus_manifest is trusted.full_manifest
        return campaigns[0], {"development-000000": tmp_path / "query.png"}

    outcome = execute_stage_a_calibration(
        trusted,
        ledger,
        config,
        on_exposure_precommit=lambda successor, snapshot: events.append(
            "precommit"
        ),
        launcher_fingerprinter=fingerprint,
        cache_snapshotter=snapshotter,
        campaign_runner=runner,
        campaign_verifier=verifier,
    )

    assert events == [
        "fingerprint",
        "snapshot",
        "campaign",
        "precommit",
        "cold-verify",
    ]
    assert len(snapshots) == 1
    assert outcome.status == "succeeded"
    assert outcome.cold_verified is True
    assert outcome.cloud_policy_cache_snapshot is snapshots[0]
    assert outcome.cloud_policy_cache_binding == snapshots[0].binding
    assert outcome.launcher_digest == LAUNCHER
    assert outcome.terminal_payload == canonical_json(campaigns[0].to_data()) + b"\n"

    result = persist_stage_a_outcome(
        outcome,
        StageAPersistenceConfig(
            artifact_directory=tmp_path / "artifacts",
            exposure_directory=tmp_path / "exposure",
            cache_snapshot_directory=tmp_path / "private-cache",
        ),
    )
    assert result.status == "succeeded"
    assert result.internal_digest == outcome.internal_digest
    assert result.artifact_path.read_bytes() == outcome.terminal_payload
    assert result.artifact_file_sha256 == (
        "sha256:" + hashlib.sha256(outcome.terminal_payload).hexdigest()
    )
    assert ExposureLedger.load(result.exposure_ledger_path) == outcome.exposure_successor
    assert result.cloud_policy_cache_snapshot_path.read_bytes() == snapshots[0].data
    assert stat.S_IMODE(
        result.cloud_policy_cache_snapshot_path.stat().st_mode
    ) == 0o600
    loaded_snapshot = load_stage_a_cache_snapshot(
        result.cloud_policy_cache_snapshot_path,
        expected_binding=result.cloud_policy_cache_binding,
        expected_file_sha256=(
            result.cloud_policy_cache_snapshot_file_sha256
        ),
    )
    assert loaded_snapshot == snapshots[0]
    # The scientific/summary JSON binds the private handoff but never embeds it.
    summary_json = canonical_json(result.to_data())
    assert b"fixture-account-identifier" not in summary_json
    assert b"fixture-user-identifier" not in summary_json
    assert result.command_config["design_mode"] == DESCRIPTIVE_STAGE_A_DESIGN
    assert result.command_config["selection_seed"] == SELECTION_SEED
    assert result.command_config["selection_seed_provenance"] == (
        SELECTION_PROVENANCE
    )
    command_receipt = json.loads(result.command_receipt_path.read_bytes())
    assert command_receipt["command_config"] == config.to_data()
    assert command_receipt["command_receipt_digest"] == (
        result.command_receipt_digest
    )
    assert command_receipt["cloud_policy_cache_snapshot_bytes_embedded"] is False
    assert result.command_receipt_file_sha256 == (
        "sha256:"
        + hashlib.sha256(result.command_receipt_path.read_bytes()).hexdigest()
    )
    loaded_receipt = load_stage_a_command_receipt(
        result.command_receipt_path,
        result.command_receipt_digest,
    )
    assert loaded_receipt.command_config == config
    assert loaded_receipt.receipt_digest == result.command_receipt_digest
    assert loaded_receipt.load_cache_snapshot() == snapshots[0]
    with pytest.raises(
        StageACalibrationCommandError,
        match="receipt digest differs",
    ):
        load_stage_a_command_receipt(
            result.command_receipt_path,
            "0" * 64,
        )
    with pytest.raises(
        StageACalibrationCommandError,
        match="bytes or fields are not canonical",
    ):
        StageACommandReceipt.from_bytes(
            result.command_receipt_path.read_bytes() + b"\n",
            expected_receipt_digest=result.command_receipt_digest,
        )
    forged_metadata = deepcopy(command_receipt)
    forged_metadata["cloud_policy_cache_snapshot_byte_count"] += 1
    forged_content = {
        key: value
        for key, value in forged_metadata.items()
        if key != "command_receipt_digest"
    }
    forged_digest = canonical_digest(forged_content)
    forged_metadata["command_receipt_digest"] = forged_digest
    forged_path = tmp_path / "forged-command-receipt.json"
    forged_path.write_bytes(canonical_json(forged_metadata) + b"\n")
    with pytest.raises(
        StageACalibrationCommandError,
        match="byte count differs",
    ):
        load_stage_a_command_receipt(
            forged_path,
            forged_digest,
        )
    # Identical content-addressed persistence is safely idempotent.
    repeated = persist_stage_a_outcome(
        outcome,
        StageAPersistenceConfig(
            artifact_directory=tmp_path / "artifacts",
            exposure_directory=tmp_path / "exposure",
            cache_snapshot_directory=tmp_path / "private-cache",
        ),
    )
    assert repeated == result

    result.artifact_path.write_bytes(b"different bytes")
    collision = persist_stage_a_outcome(
        outcome,
        StageAPersistenceConfig(
            artifact_directory=tmp_path / "artifacts",
            exposure_directory=tmp_path / "exposure",
            cache_snapshot_directory=tmp_path / "private-cache",
        ),
    )
    assert collision.status == "failed"
    collision_terminal = json.loads(collision.artifact_path.read_bytes())
    assert collision_terminal["failure"]["error_type"] == (
        "StageACalibrationCommandError"
    )
    assert result.artifact_path.read_bytes() == b"different bytes"


def test_launcher_mismatch_stops_before_snapshot_or_campaign(tmp_path: Path) -> None:
    trusted = _trusted(tmp_path)
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    called: list[str] = []

    with pytest.raises(
        StageACalibrationCommandError,
        match="actual Codex launcher differs",
    ):
        execute_stage_a_calibration(
            trusted,
            ledger,
            _config(ledger),
            on_exposure_precommit=lambda successor, snapshot: called.append(
                "precommit"
            ),
            launcher_fingerprinter=lambda executable, **kwargs: {
                "version": "wrong fixture",
                "launcher_digest": "0" * 64,
            },
            cache_snapshotter=lambda: called.append("snapshot"),  # type: ignore[arg-type]
            campaign_runner=lambda *args, **kwargs: called.append(
                "campaign"
            ),  # type: ignore[arg-type]
        )
    assert called == []


def test_private_cache_handoff_rejects_credential_like_payload(
    tmp_path: Path,
) -> None:
    snapshot = CloudPolicyCacheSnapshot(
        canonical_json(
            {
                "signed_payload": {
                    "access_token": "not-a-real-token-but-forbidden-by-schema"
                },
                "signature": "fixture-signature",
            }
        )
    )
    with pytest.raises(
        StageACalibrationCommandError,
        match="credential-like",
    ):
        persist_stage_a_cache_snapshot(snapshot, tmp_path / "private-cache")
    assert not (tmp_path / "private-cache").exists()


def test_absolute_repo_local_cache_path_is_accepted_only_under_ignored_downloads() -> None:
    repository = Path(__file__).resolve().parents[2]
    downloads = repository / "downloads"
    assert downloads.is_dir()
    with tempfile.TemporaryDirectory(
        prefix="stage-a-cache-command-test-",
        dir=downloads,
    ) as temporary:
        cache_directory = (Path(temporary) / "private-cache").resolve()
        path, file_sha256, byte_count = persist_stage_a_cache_snapshot(
            CloudPolicyCacheSnapshot(None),
            cache_directory,
        )
        assert path.is_absolute()
        assert cache_directory in path.parents
        assert file_sha256 == (
            "sha256:" + hashlib.sha256(b"").hexdigest()
        )
        assert byte_count == 0
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    nonignored = repository / "bongard" / ".stage-a-private-cache-probe"
    assert not nonignored.exists()
    with pytest.raises(StageACalibrationCommandError, match="Git-ignored"):
        persist_stage_a_cache_snapshot(
            CloudPolicyCacheSnapshot(None),
            nonignored,
        )
    assert not nonignored.exists()


def test_ledger_digest_mismatch_stops_before_environment_or_campaign(
    tmp_path: Path,
) -> None:
    trusted = _trusted(tmp_path)
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    wrong_expected = "sha256:" + "0" * 64
    called: list[str] = []

    with pytest.raises(
        StageACalibrationCommandError,
        match="externally expected digest",
    ):
        execute_stage_a_calibration(
            trusted,
            ledger,
            _config(
                ledger,
                expected_exposure_ledger_digest=wrong_expected,
            ),
            on_exposure_precommit=lambda successor, snapshot: called.append(
                "precommit"
            ),
            launcher_fingerprinter=lambda executable, **kwargs: called.append(
                "fingerprint"
            ),  # type: ignore[arg-type]
            cache_snapshotter=lambda: called.append(
                "snapshot"
            ),  # type: ignore[arg-type]
            campaign_runner=lambda *args, **kwargs: called.append(
                "campaign"
            ),  # type: ignore[arg-type]
        )
    assert called == []


def _terminal_error(kind: str, archive: _FakeArchive) -> BaseException:
    if kind == "proposal":
        archive.records = (  # type: ignore[misc]
            SimpleNamespace(
                status=TRANSPORT_FAILED,
                candidate=SimpleNamespace(
                    selection=SimpleNamespace(observation_id="development-000000")
                ),
            ),
        )
        return SemanticCalibrationCampaignProposalPhaseFailed(archive)  # type: ignore[arg-type]
    if kind == "no-soft":
        return SemanticCalibrationCampaignNoSoftClaims(archive)  # type: ignore[arg-type]

    record = SimpleNamespace(outcome="error")
    attempt = SimpleNamespace(
        commitment=SimpleNamespace(
            selection=SimpleNamespace(observation_id="development-000000")
        ),
        score_artifact=SimpleNamespace(record=record),
    )

    class ScoreBatch:
        commitment_batch = SimpleNamespace(proposal_archive=archive)
        attempts = (attempt,)
        digest = canonical_digest({"schema": "fixture-score-batch/v1"})

        @staticmethod
        def to_data():
            return {
                "schema": "fixture-score-batch/v1",
                "score_batch_digest": ScoreBatch.digest,
            }

    score_batch = ScoreBatch()
    if kind == "scoring":
        return SemanticCalibrationCampaignScoringFailed(score_batch)  # type: ignore[arg-type]
    if kind == "fit":
        return SemanticCalibrationCampaignFitFailed(
            score_batch,  # type: ignore[arg-type]
            (),
            (),
            RuntimeError("synthetic fit failure"),
        )
    raise AssertionError(kind)


def test_interrupt_after_precommit_persists_typed_failure_and_receipt(
    tmp_path: Path,
) -> None:
    trusted = _trusted(tmp_path / "crash")
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)
    persistence = StageAPersistenceConfig(
        artifact_directory=tmp_path / "crash" / "artifacts",
        exposure_directory=tmp_path / "crash" / "exposure",
        cache_snapshot_directory=tmp_path / "crash" / "private-cache",
    )
    expected: list[ExposureLedger] = []

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        expected.append(archive.exposure_successor)
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        # The hook cannot return until both files are fsynced and cold-loaded.
        assert len(tuple(persistence.exposure_directory.glob("*.exposure.json"))) == 1
        assert len(tuple(persistence.cache_snapshot_directory.iterdir())) == 1
        raise KeyboardInterrupt("synthetic process death after precommit")

    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        persistence,
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
    )

    ledgers = tuple(persistence.exposure_directory.glob("*.exposure.json"))
    caches = tuple(persistence.cache_snapshot_directory.iterdir())
    assert len(ledgers) == len(caches) == 1
    assert ExposureLedger.load(ledgers[0]) == expected[0]
    assert len(expected[0].events) - len(ledger.events) == config.candidate_count
    assert load_stage_a_cache_snapshot(
        caches[0],
        expected_binding="absent",
        expected_file_sha256="sha256:" + hashlib.sha256(b"").hexdigest(),
    ) == snapshot
    assert result.status == "failed"
    assert result.terminal_schema == STAGE_A_OPERATIONAL_FAILURE_SCHEMA
    assert result.command_receipt_path.is_file()
    terminal = json.loads(result.artifact_path.read_bytes())
    assert terminal["failure"]["error_type"] == "KeyboardInterrupt"
    assert terminal["label_state"] == "withheld"
    assert terminal["campaign_result_state"] == "absent"
    receipt = load_stage_a_command_receipt(
        result.command_receipt_path,
        result.command_receipt_digest,
    )
    assert receipt.status == "failed"
    assert receipt.cold_verified is False


def test_source_mutation_during_mocked_proposer_fails_closed_and_persists(
    tmp_path: Path,
) -> None:
    trusted = _trusted(tmp_path / "source-mutation")
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)
    persistence = StageAPersistenceConfig(
        artifact_directory=tmp_path / "source-mutation" / "artifacts",
        exposure_directory=tmp_path / "source-mutation" / "exposure",
        cache_snapshot_directory=tmp_path / "source-mutation" / "private-cache",
    )
    source_root = tmp_path / "source-mutation" / "source-boundary"
    source_root.mkdir(parents=True)
    dependency = source_root / "stage_a_dependency.py"
    dependency.write_text("FROZEN_VALUE = 1\n", encoding="utf-8")

    def mutating_proposer(*args, **kwargs):
        dependency.write_text("FROZEN_VALUE = 2\n", encoding="utf-8")
        return SimpleNamespace(payload={}, receipt=None)

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        # The command wraps this mock exactly as it wraps the live proposer.
        # Its after-proposer check must observe the real file mutation.
        kwargs["proposer_transport"]()
        pytest.fail("mutated proposer transport returned to the campaign")

    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        persistence,
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
        campaign_verifier=lambda *args, **kwargs: pytest.fail(
            "source-mutated campaign reached success verification"
        ),
        proposer_transport=mutating_proposer,
        source_dependency_root=source_root,
    )

    assert result.status == "failed"
    assert result.terminal_schema == STAGE_A_OPERATIONAL_FAILURE_SCHEMA
    raw = result.artifact_path.read_bytes()
    failure = json.loads(raw)
    assert failure["failure_phase"] == "after-proposer"
    assert failure["failure"]["error_type"] == (
        "StageASourceDependencyMutationError"
    )
    assert failure["source_dependency_state"] == "mutated"
    assert failure["source_dependency_digest"] != (
        failure["observed_source_dependency_digest"]
    )
    assert failure["label_state"] == "withheld"
    assert failure["campaign_result_state"] == "absent"
    assert failure["fit_authorized"] is False
    assert "campaign_digest" not in failure
    assert b"label_reveals" not in raw

    receipt = load_stage_a_command_receipt(
        result.command_receipt_path,
        result.command_receipt_digest,
    )
    assert receipt.status == "failed"
    assert receipt.cold_verified is False
    assert receipt.source_dependencies.digest == (
        failure["source_dependency_digest"]
    )
    assert result.command_receipt_path.is_file()
    assert result.artifact_path.is_file()

    # Historical A1-shaped v1 receipts remain strict, canonical audit inputs.
    # They intentionally load without inventing a source identity; a Stage-B
    # authorization boundary can then reject this one for its failed status.
    historical = json.loads(result.command_receipt_path.read_bytes())
    historical["schema"] = STAGE_A_COMMAND_RECEIPT_SCHEMA_V1
    historical.pop("source_dependencies")
    historical.pop("source_dependency_digest")
    historical_content = {
        key: value
        for key, value in historical.items()
        if key != "command_receipt_digest"
    }
    historical_digest = canonical_digest(historical_content)
    historical["command_receipt_digest"] = historical_digest
    historical_payload = canonical_json(historical) + b"\n"
    historical_path = persistence.artifact_directory / "historical-a1-v1.json"
    historical_path.write_bytes(historical_payload)
    loaded_historical = load_stage_a_command_receipt(
        historical_path,
        historical_digest,
    )
    assert loaded_historical.status == "failed"
    assert loaded_historical.source_dependencies is None
    assert loaded_historical.receipt_payload == historical_payload


@pytest.mark.parametrize("kind", ("proposal", "no-soft", "scoring", "fit"))
def test_every_canonical_terminal_failure_persists_artifact_and_successor(
    tmp_path: Path,
    kind: str,
) -> None:
    trusted = _trusted(tmp_path / kind)
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        raise _terminal_error(kind, archive)

    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        StageAPersistenceConfig(
            artifact_directory=tmp_path / kind / "artifacts",
            exposure_directory=tmp_path / kind / "exposure",
            cache_snapshot_directory=tmp_path / kind / "private-cache",
        ),
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
        campaign_verifier=lambda *args, **kwargs: pytest.fail(
            "terminal failure reached success verifier"
        ),
    )

    assert result.status == "failed"
    assert result.cold_verified is False
    raw = result.artifact_path.read_bytes()
    decoded = json.loads(raw)
    assert raw == canonical_json(decoded) + b"\n"
    assert decoded["failure_digest"] == result.internal_digest
    assert result.artifact_file_sha256 == (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    successor = ExposureLedger.load(result.exposure_ledger_path)
    assert successor.digest == result.exposure_ledger_digest
    assert len(successor.events) - len(ledger.events) == config.candidate_count
    assert result.cloud_policy_cache_snapshot_byte_count == 0
    assert result.cloud_policy_cache_snapshot_path.read_bytes() == b""
    assert load_stage_a_cache_snapshot(
        result.cloud_policy_cache_snapshot_path,
        expected_binding="absent",
        expected_file_sha256=(
            result.cloud_policy_cache_snapshot_file_sha256
        ),
    ) == snapshot


def test_source_change_between_execute_and_persist_becomes_failed_receipt(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-persistence" / "sources"
    source_root.mkdir(parents=True)
    dependency = source_root / "authority.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    outcome, _, _, _, _ = _fake_success_outcome(
        tmp_path / "source-persistence",
        source_dependency_root=source_root,
    )
    original_source_digest = outcome.source_dependencies.digest
    dependency.write_text("VALUE = 2\n", encoding="utf-8")

    result = persist_stage_a_outcome(
        outcome,
        StageAPersistenceConfig(
            artifact_directory=tmp_path / "source-persistence" / "artifacts",
            exposure_directory=tmp_path / "source-persistence" / "exposure",
            cache_snapshot_directory=tmp_path / "source-persistence" / "cache",
        ),
    )

    assert result.status == "failed"
    assert result.terminal_schema == STAGE_A_OPERATIONAL_FAILURE_SCHEMA
    terminal = json.loads(result.artifact_path.read_bytes())
    assert terminal["failure_phase"] == "before-terminal-artifact-persistence"
    assert terminal["failure"]["error_type"] == (
        "StageASourceDependencyMutationError"
    )
    assert terminal["source_dependency_state"] == "mutated"
    assert terminal["source_dependency_digest"] == original_source_digest
    assert terminal["observed_source_dependency_digest"] != original_source_digest
    assert not tuple(
        result.artifact_path.parent.glob(
            "*.semantic-calibration-campaign.json"
        )
    )
    receipt = load_stage_a_command_receipt(
        result.command_receipt_path,
        result.command_receipt_digest,
    )
    assert receipt.status == "failed"


def test_unreadable_source_root_between_execute_and_persist_is_terminalized(
    tmp_path: Path,
) -> None:
    work = tmp_path / "unreadable-source-persistence"
    source_root = work / "sources"
    source_root.mkdir(parents=True)
    (source_root / "authority.py").write_text("VALUE = 1\n", encoding="utf-8")
    outcome, _, _, _, _ = _fake_success_outcome(
        work,
        source_dependency_root=source_root,
    )
    source_root.rename(work / "sources-moved-away")

    result = persist_stage_a_outcome(
        outcome,
        StageAPersistenceConfig(
            artifact_directory=work / "artifacts",
            exposure_directory=work / "exposure",
            cache_snapshot_directory=work / "cache",
        ),
    )

    assert result.status == "failed"
    terminal = json.loads(result.artifact_path.read_bytes())
    assert terminal["failure"]["error_type"] == (
        "StageASourceDependencyMutationError"
    )
    assert terminal["source_dependency_state"] == "unreadable"
    assert terminal["observed_source_dependencies"] is None
    assert terminal["source_observation_error_digest"] is not None
    assert result.command_receipt_path.is_file()


def test_terminal_persistence_interrupt_becomes_failed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, _, _, _, _ = _fake_success_outcome(tmp_path / "persist-interrupt")
    persistence = StageAPersistenceConfig(
        artifact_directory=tmp_path / "persist-interrupt" / "artifacts",
        exposure_directory=tmp_path / "persist-interrupt" / "exposure",
        cache_snapshot_directory=tmp_path / "persist-interrupt" / "cache",
    )
    original_write = command_module._write_once_or_identical
    calls = 0

    def interrupt_once(path: Path, payload: bytes) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("synthetic terminal-persistence interrupt")
        return original_write(path, payload)

    monkeypatch.setattr(
        command_module,
        "_write_once_or_identical",
        interrupt_once,
    )
    result = persist_stage_a_outcome(outcome, persistence)

    assert result.status == "failed"
    terminal = json.loads(result.artifact_path.read_bytes())
    assert terminal["failure_phase"] == "terminal-persistence"
    assert terminal["failure"]["error_type"] == "KeyboardInterrupt"
    assert result.command_receipt_path.is_file()
    assert not tuple(
        persistence.artifact_directory.glob(
            "*.semantic-calibration-campaign.json"
        )
    )


def test_interrupt_after_ledger_write_before_precommit_return_is_terminalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = _trusted(tmp_path / "precommit-interrupt")
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)
    persistence = StageAPersistenceConfig(
        artifact_directory=tmp_path / "precommit-interrupt" / "artifacts",
        exposure_directory=tmp_path / "precommit-interrupt" / "exposure",
        cache_snapshot_directory=tmp_path / "precommit-interrupt" / "cache",
    )

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        pytest.fail("interrupted precommit returned to the campaign")

    original_hash = command_module._file_sha256
    calls = 0

    def interrupt_first_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert tuple(persistence.exposure_directory.glob("*.exposure.json"))
            raise KeyboardInterrupt("synthetic interrupt after ledger write")
        return original_hash(path)

    monkeypatch.setattr(command_module, "_file_sha256", interrupt_first_hash)
    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        persistence,
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
    )

    assert result.status == "failed"
    terminal = json.loads(result.artifact_path.read_bytes())
    assert terminal["failure"]["error_type"] == "KeyboardInterrupt"
    assert len(tuple(persistence.exposure_directory.glob("*.exposure.json"))) == 1
    assert result.command_receipt_path.is_file()


def test_returned_outcome_recovers_empty_runner_precommit_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, trusted, ledger, config, _ = _fake_success_outcome(
        tmp_path / "empty-precommit-window"
    )
    persistence = StageAPersistenceConfig(
        artifact_directory=tmp_path / "empty-precommit-window" / "artifacts",
        exposure_directory=tmp_path / "empty-precommit-window" / "exposure",
        cache_snapshot_directory=tmp_path / "empty-precommit-window" / "cache",
    )
    monkeypatch.setattr(
        command_module,
        "execute_stage_a_calibration",
        lambda *args, **kwargs: outcome,
    )

    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        persistence,
    )

    assert result.status == "succeeded"
    assert ExposureLedger.load(result.exposure_ledger_path) == (
        outcome.exposure_successor
    )
    assert result.command_receipt_path.is_file()


def test_current_command_rejects_legacy_v1_campaign_selection(
    tmp_path: Path,
) -> None:
    trusted = _trusted(tmp_path / "legacy-selection")
    ledger = ExposureLedger.create(trusted.full_manifest.digest)
    config = _config(ledger)
    snapshot = CloudPolicyCacheSnapshot(None)
    campaigns: list[_FakeCampaign] = []

    def runner(corpus, protocol, **kwargs):
        archive = _archive(
            protocol=protocol,
            kwargs=kwargs,
            ledger=ledger,
            snapshot=snapshot,
        )
        archive.selection_algorithm = CAMPAIGN_SELECTION_ALGORITHM_V1
        kwargs["on_exposure_precommit"](archive.exposure_successor)
        campaign = _FakeCampaign(archive)
        campaigns.append(campaign)
        return campaign

    outcome = execute_stage_a_calibration(
        trusted,
        ledger,
        config,
        on_exposure_precommit=lambda successor, frozen_snapshot: None,
        launcher_fingerprinter=lambda executable, **kwargs: {
            "version": "codex-cli fixture",
            "launcher_digest": LAUNCHER,
        },
        cache_snapshotter=lambda: snapshot,
        campaign_runner=runner,
        campaign_verifier=lambda raw, **kwargs: (campaigns[0], {}),
    )

    assert outcome.status == "failed"
    terminal = outcome.terminal_data()
    assert terminal["failure_phase"] == "cold-verification"
    assert terminal["failure"]["error_type"] == "StageACalibrationCommandError"
