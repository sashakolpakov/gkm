from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from bongard import benchmark
import bongard.atomic_smoke_command as C
import bongard.atomic_smoke_precommit as P
from bongard.cohorts import parse_official_task_id
from bongard.corpus import BongardTask, ShapeBongardCorpus, SplitIndex
from bongard.exposure import ExposureLedger
from bongard.semantic_calibration_command import StageASourceDependencyIdentity
from bongard.semantic_calibration_campaign import semantic_generator_cluster_id
from bongard.transport import CloudPolicyCacheSnapshot, StagedCodexLauncher


HEX_A = hashlib.sha256(b"a").hexdigest()
HEX_B = hashlib.sha256(b"b").hexdigest()
HEX_C = hashlib.sha256(b"c").hexdigest()
HEX_D = hashlib.sha256(b"d").hexdigest()
SECRET_A = hashlib.sha256(b"fixture-secret-a").hexdigest()
SECRET_B = hashlib.sha256(b"fixture-secret-b").hexdigest()
SECRET_C = hashlib.sha256(b"fixture-secret-c").hexdigest()


def _sources() -> StageASourceDependencyIdentity:
    return StageASourceDependencyIdentity((("atomic_smoke_command.py", 1, HEX_A),))


def _preflight(
    *,
    launcher_path: str = "/private/fake/staged-codex",
    launcher_version: str = "codex-cli fixture",
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
) -> C.AtomicSmokeTransportPreflightReceipt:
    return C.AtomicSmokeTransportPreflightReceipt.create(
        launcher_path=launcher_path,
        launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
        launcher_version=launcher_version,
        model=model,
        reasoning_effort=reasoning_effort,
        proposal_transport_receipt_digest=HEX_C,
        scoring_transport_receipt_digest=HEX_D,
    )


def _config() -> C.AtomicSmokeCommandConfig:
    preflight = _preflight()
    return C.AtomicSmokeCommandConfig(
        input_authentication_digest="sha256:" + HEX_B,
        source_dependencies=_sources(),
        cache_binding="absent",
        cache_file_sha256="sha256:" + hashlib.sha256(b"").hexdigest(),
        cache_byte_count=0,
        expected_launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
        staged_launcher_path=preflight.launcher_path,
        launcher_version=preflight.launcher_version,
        preflight_receipt=preflight,
        preflight_receipt_file_sha256="sha256:" + HEX_A,
        preflight_receipt_filename=(
            preflight.receipt_digest.removeprefix("sha256:")
            + ".atomic-smoke-preflight.json"
        ),
        preflight_receipt_byte_count=1,
        run_protocol_digest=C.atomic_smoke_run_protocol_digest(),
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        minutes=15,
        verifier_id="fixture-verifier",
    )


def _claim(config: C.AtomicSmokeCommandConfig, root: Path) -> C.AtomicSmokeAttemptClaim:
    predecessor_home = root / "private-predecessor"
    predecessor_home.mkdir(mode=0o700, parents=True)
    predecessor_home.chmod(0o700)
    predecessor_path = predecessor_home / "bfd.exposure.json"
    predecessor_payload = b"fixture predecessor ledger"
    predecessor_path.write_bytes(predecessor_payload)
    inputs = SimpleNamespace(
        digest=config.input_authentication_digest,
        predecessor_path=predecessor_path.resolve(),
        predecessor_file_sha256=hashlib.sha256(predecessor_payload).hexdigest(),
        predecessor=SimpleNamespace(
            digest=P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
        ),
        prior_attempt=C.AtomicSmokePriorAttemptRecord.load(
            C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH
        ),
    )
    persistence = C.AtomicSmokeDurabilityReceipt(
        "transport-preflight",
        (root / config.preflight_receipt_filename).resolve(),
        config.preflight_receipt.receipt_digest,
        config.preflight_receipt_file_sha256,
        config.preflight_receipt_byte_count,
    )
    return C.AtomicSmokeAttemptClaim.create(
        inputs=inputs,
        config=config,
        preflight_persistence=persistence,
    )


def _fake_command_inputs(root: Path) -> SimpleNamespace:
    predecessor_home = root / "canonical-predecessor"
    predecessor_home.mkdir(mode=0o700)
    predecessor_home.chmod(0o700)
    predecessor_path = predecessor_home / "bfd.exposure.json"
    payload = b"fixture bfd predecessor bytes"
    predecessor_path.write_bytes(payload)
    return SimpleNamespace(
        digest="sha256:" + HEX_B,
        trusted=SimpleNamespace(corpus=object(), full_manifest=object()),
        predecessor=SimpleNamespace(
            digest=P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
        ),
        predecessor_path=predecessor_path.resolve(),
        predecessor_file_sha256=hashlib.sha256(payload).hexdigest(),
        prior_attempt=C.AtomicSmokePriorAttemptRecord.load(
            C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH
        ),
    )


def _persist_empty_cache(
    _snapshot: CloudPolicyCacheSnapshot, directory: str | Path
) -> tuple[Path, str, int]:
    path = Path(directory) / "empty.cache"
    path.write_bytes(b"")
    return path, "sha256:" + hashlib.sha256(b"").hexdigest(), 0


def _genuine_precommit(root: Path, *, source_digest: str) -> P.AtomicSmokePrecommit:
    task_id = "bd_mismatch_triangle_rec1_0000"
    task_root = root / "bd" / "images" / task_id
    sides: dict[str, tuple[Path, ...]] = {}
    for label in ("1", "0"):
        directory = task_root / label
        directory.mkdir(parents=True)
        paths: list[Path] = []
        for index in range(7):
            path = directory / f"{index}.png"
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + f"{task_id}:{label}:{index}".encode("ascii")
            )
            paths.append(path)
        sides[label] = tuple(paths)
    task = BongardTask(
        task_id=task_id,
        family="bd",
        root=task_root,
        positive=sides["1"],
        negative=sides["0"],
    )
    corpus = ShapeBongardCorpus(
        root,
        (task,),
        layout="archive",
        split=SplitIndex(
            groups=(("test", ()), ("train", (task_id,)), ("val", ())),
            source_digest=P.OFFICIAL_SPLIT_SOURCE_DIGEST,
        ),
    )
    manifest = corpus.build_manifest()
    label_nonce = "a" * 64
    plan = benchmark.prepare_episode(
        corpus,
        task_id,
        seed="synthetic episode seed",
        corpus_manifest=manifest,
        verifier_id="offline-fixture",
        label_seal_nonce=label_nonce,
    )
    parsed = parse_official_task_id(task_id)
    selection = P.AtomicSmokeSelection.create(
        source_corpus_manifest_digest=P.OFFICIAL_CORPUS_MANIFEST_DIGEST,
        split_source_digest=P.OFFICIAL_SPLIT_SOURCE_DIGEST,
        exposure_predecessor_digest=(
            P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
        ),
        historical_seed_digest=P.OFFICIAL_HISTORICAL_SEED_DIGEST,
        resolver_policy_digest=P.OFFICIAL_RESOLVER_POLICY_DIGEST,
        blocked_morphology_policy_digest=(
            P.OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST
        ),
        seed="synthetic post-freeze selection seed",
        selected_task_id=task_id,
        selected_generator_cluster_id=semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        ),
    )
    successor = ExposureLedger.create(P.OFFICIAL_CORPUS_MANIFEST_DIGEST).record(
        phase=P.ATOMIC_SMOKE_EXPOSURE_PHASE,
        actor="offline-fixture",
        purpose=P.ATOMIC_SMOKE_EXPOSURE_PURPOSE,
        task_ids=(task_id,),
        source="atomic-smoke-selection:" + selection.digest,
        observed_at="2026-08-06T12:00:00Z",
        known_task_ids=(task_id,),
    )
    exposure_payload = successor.to_json().encode("utf-8")
    receipt = P.ExposurePersistenceReceipt.create(
        ledger=successor,
        filename=(successor.digest.removeprefix("sha256:") + ".exposure.json"),
        payload=exposure_payload,
    )
    return P.AtomicSmokePrecommit.create(
        selection=selection,
        exposure_successor=successor,
        exposure_persistence_receipt=receipt,
        source_dependency_digest=source_digest,
        development_manifest=manifest,
        episode_plan=plan,
    )


def test_production_pins_and_secret_free_config() -> None:
    assert C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST == (
        "ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02"
    )
    config = _config()
    assert C.AtomicSmokeCommandConfig.from_data(config.to_data()) == config
    assert config.to_data()["schema"] == "gkm.bongard-atomic-smoke-command-config.v5"
    assert config.to_data()["attempt_ordinal"] == 3
    assert config.to_data()["scope"] == C.ATOMIC_SMOKE_COMMAND_SCOPE
    assert config.to_data()["reference_execution"] == "python-canonical/v1"
    assert config.to_data()["official_active_exposure_predecessor_digest"] == (
        P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
    )
    assert config.to_data()["official_immediate_b053_parent_ledger_digest"] == (
        P.OFFICIAL_B053_LEDGER_DIGEST
    )
    assert config.to_data()["official_historical_a3_ancestor_ledger_digest"] == (
        P.OFFICIAL_A3_LEDGER_DIGEST
    )
    prior = config.to_data()["prior_attempt_record"]
    assert isinstance(prior, dict)
    assert "record_file_sha256" not in prior
    assert set(prior["source_snapshot"]) == {
        "scope", "source_dependency_digest"
    }
    assert prior["artifacts"]["command_config_content_address"] == (
        C.ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
    )
    assert prior["failure"]["reason_digest"] == C.ATOMIC_SMOKE_PRIOR_REASON_DIGEST
    display = config.to_data()["lineage_display_metadata"]
    assert display["record_file_sha256"] == C.ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256
    assert display["source_snapshot"] == {
        "commit": C.ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT,
        "tag": C.ATOMIC_SMOKE_PRIOR_SOURCE_TAG,
    }
    encoded = json.dumps(config.to_data(), sort_keys=True)
    encoded_authority = json.dumps(config.content_data(), sort_keys=True)
    assert "official_a3_successor_ledger_digest" not in encoded
    assert "official_active_exposure_predecessor_digest" in encoded
    assert C.ATOMIC_SMOKE_PRIOR_REASON_DIGEST in encoded
    assert "lean-optional" not in encoded_authority
    assert C.ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT not in encoded_authority
    assert C.ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256 not in encoded_authority
    assert config.to_data()["transport_preflight"]["bongard_call_count"] == 0
    assert config.to_data()["transport_preflight"]["secret_count"] == 0
    assert config.digest.startswith("sha256:")
    assert "selection_seed" not in encoded
    assert "episode_seed" not in encoded
    assert "label_nonce" not in encoded
    assert config.to_data()["secrets_generated_after_persistence"] is True
    assert all(
        config.to_data()[name] is False
        for name in (
            "dependence_design_authorized",
            "calibration_authorized",
            "benchmark_claim_authorized",
            "official_test_authorized",
        )
    )


def test_prior_attempt_file_and_consumed_lineage_are_exact(tmp_path: Path) -> None:
    prior = C.AtomicSmokePriorAttemptRecord.load(
        C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH
    )
    assert prior.file_sha256 == (
        "242ebc5914020a683a6f34a0b50688bf3190f4c4cbd6d345d15ebb5e775eb6b3"
    )
    assert prior.record_schema == (
        "gkm.bongard-atomic-smoke-attempt2-proposal-contract-failure.v1"
    )
    assert prior.predecessor_digest == P.OFFICIAL_B053_LEDGER_DIGEST
    assert prior.successor_digest == (
        P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
    )
    assert prior.command_config_digest == C.ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
    assert prior.precommit_digest == C.ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST
    assert prior.run_digest == C.ATOMIC_SMOKE_PRIOR_RUN_DIGEST
    assert prior.terminal_digest == C.ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST
    assert prior.evidence_digest == C.ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST
    assert prior.journal_header_digest == C.ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST
    assert prior.journal_receipt_digest == (
        C.ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST
    )
    assert prior.failure_reason_digest == C.ATOMIC_SMOKE_PRIOR_REASON_DIGEST
    assert prior.legacy_digest == (
        "sha256:42a41e4cf53a1f109e469fe99a79f2aeebb751e3c8f638e43bc36080cac7211e"
    )
    assert prior.remaining_universe_digest == (
        "sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9"
    )
    binding = prior.to_data()
    assert binding["journal"] == {
        "header_digest": C.ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST,
        "receipt_digest": C.ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST,
        "intent_count": 13,
        "result_count": 13,
        "state": "result-closed",
        "terminal_persisted": True,
    }
    calls = binding["call_boundary"]
    assert calls["neutral_support_description_calls"] == 12
    assert calls["atom_proposal_calls"] == 1
    assert calls["support_scoring_calls"] == 0
    assert calls["query_calls"] == 0
    assert calls["formula_frozen"] is False
    assert calls["selection_archive_persisted"] is False
    assert calls["prediction_persisted"] is False
    assert calls["query_labels_materialized"] is False
    assert calls["query_labels_revealed"] is False
    assert calls["score"] is None
    assert binding["failure"] == {
        "result_class": "implementation_contract_failure",
        "phase": "atom-proposal",
        "error_type": "AtomicSemanticSynthesisError",
        "exact_error": C.ATOMIC_SMOKE_PRIOR_EXACT_ERROR,
        "reason_digest": C.ATOMIC_SMOKE_PRIOR_REASON_DIGEST,
        "cold_replay_passed": True,
    }
    assert set(binding["claim_authority"].values()) == {False}
    assert binding["consumption"] == {
        "selected_task_consumed": True,
        "selected_task_may_be_rerolled": False,
    }

    tampered = tmp_path / "attempt2.json"
    tampered.write_bytes(C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH.read_bytes() + b"\n")
    with pytest.raises(C.AtomicSmokeCommandError, match="exact pin"):
        C.AtomicSmokePriorAttemptRecord.load(tampered)

    malformed = json.loads(C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH.read_bytes())
    malformed["call_boundary"]["support_scoring_calls"] = 1
    with pytest.raises(C.AtomicSmokeCommandError, match="causal facts"):
        C.AtomicSmokePriorAttemptRecord._from_data(
            malformed,
            file_sha256=C.ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256,
        )


def test_display_lineage_bytes_cannot_change_v5_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    claim = _claim(config, tmp_path / "claim")
    baseline_prior_authority_digest = C._prior_attempt_authority_digest()
    baseline_prior_legacy_digest = C._prior_attempt_legacy_digest()
    baseline_config_content = config.content_data()
    baseline_config_digest = config.digest
    baseline_claim_content = claim.content_data()
    baseline_claim_digest = claim.claim_digest
    baseline_display = config.to_data()["lineage_display_metadata"]

    monkeypatch.setattr(C, "ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT", "e" * 40)
    monkeypatch.setattr(C, "ATOMIC_SMOKE_PRIOR_SOURCE_TAG", "display-only-tag")
    monkeypatch.setattr(C, "ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256", "f" * 64)

    assert C._prior_attempt_authority_digest() == baseline_prior_authority_digest
    assert config.content_data() == baseline_config_content
    assert config.digest == baseline_config_digest
    assert claim.content_data() == baseline_claim_content
    assert claim.claim_digest == baseline_claim_digest
    assert config.to_data()["lineage_display_metadata"] != baseline_display
    assert C._prior_attempt_legacy_digest() != baseline_prior_legacy_digest


def test_historical_v4_config_v1_claim_and_v4_terminal_remain_exactly_decodable(
    tmp_path: Path,
) -> None:
    legacy_config = replace(_config(), protocol_version="v4")
    config_data = legacy_config.to_data()
    assert config_data["schema"] == C.ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA_V4
    assert config_data["reference_execution"] == (
        C.LEGACY_LEAN_OPTIONAL_REFERENCE_EXECUTION
    )
    assert "lineage_display_metadata" not in config_data
    assert C.AtomicSmokeCommandConfig.from_data(config_data) == legacy_config

    legacy_claim = _claim(legacy_config, tmp_path / "legacy-claim")
    claim_data = legacy_claim.to_data()
    assert claim_data["schema"] == C.ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA_V1
    assert claim_data["prior_attempt_record_raw_file_sha256"] == (
        C.ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256
    )
    assert C.AtomicSmokeAttemptClaim.from_data(claim_data) == legacy_claim

    legacy_terminal = C.AtomicSmokeCommandTerminal.failure(
        RuntimeError("historical fixture failure"),
        phase="fixture",
        config=legacy_config,
        precommit=None,
        launcher_version=legacy_config.launcher_version,
    )
    terminal_data = legacy_terminal.to_data()
    assert terminal_data["schema"] == C.ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA_V4
    assert C.AtomicSmokeCommandTerminal.from_data(terminal_data) == legacy_terminal


def test_content_addressed_config_and_terminal_are_reloaded_and_tamper_fails(
    tmp_path: Path,
) -> None:
    config_store = tmp_path / "config"
    terminal_store = tmp_path / "terminal"
    config_store.mkdir()
    terminal_store.mkdir()
    config = _config()
    first = C._persist_config(config, config_store)
    second = C._persist_config(config, config_store)
    assert first.path == second.path
    assert json.loads(first.path.read_bytes()) == config.to_data()

    terminal = C.AtomicSmokeCommandTerminal.failure(
        RuntimeError("bounded fixture failure"),
        phase="fixture",
        config=config,
        precommit=None,
        launcher_version=config.launcher_version,
    )
    receipt = C._persist_terminal(terminal, terminal_store)
    assert C.AtomicSmokeCommandTerminal.from_data(
        json.loads(receipt.path.read_bytes())
    ) == terminal
    receipt.path.write_bytes(b"{}")
    with pytest.raises(C.AtomicSmokeCommandError, match="different|reloaded"):
        C._persist_terminal(terminal, terminal_store)


def test_source_guard_checks_before_and_after_and_preserves_transport_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    frozen = C.freeze_stage_a_source_dependencies(tmp_path)
    guard = C._SourceGuard(tmp_path, frozen)
    sentinel = object()
    calls: list[str] = []

    def transport() -> object:
        calls.append("transport")
        return sentinel

    assert guard.wrap("fixture", transport)() is sentinel
    assert calls == ["transport"]

    def mutating_transport() -> object:
        source.write_text("VALUE = 2\n", encoding="utf-8")
        return sentinel

    with pytest.raises(C.AtomicSmokeCommandError, match="sources changed"):
        guard.wrap("fixture", mutating_transport)()


@pytest.mark.parametrize(
    "persistence_mode", ("ordinary", "primary-fails", "store-unavailable")
)
def test_command_persists_config_before_secrets_and_uses_guarded_test_seams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistence_mode: str,
) -> None:
    stores = {}
    for name in (
        "config", "exposure", "journal", "prediction", "terminal", "cache",
        "preflight",
    ):
        stores[name] = tmp_path / name
        stores[name].mkdir(mode=0o700)
        stores[name].chmod(0o700)
    events: list[str] = []
    sources = _sources()
    fake_inputs = _fake_command_inputs(tmp_path)
    fake_precommit = SimpleNamespace(
        digest="sha256:" + HEX_D,
        episode_plan=SimpleNamespace(split="train"),
    )

    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs)
    monkeypatch.setattr(C, "freeze_stage_a_source_dependencies", lambda _root: sources)
    monkeypatch.setattr(
        C,
        "persist_stage_a_cache_snapshot",
        _persist_empty_cache,
    )
    monkeypatch.setattr(
        C,
        "load_stage_a_cache_snapshot",
        lambda _path, **_kw: CloudPolicyCacheSnapshot(None),
    )

    generated = iter((SECRET_A, SECRET_B, SECRET_C))

    def secret_factory(_bytes: int) -> str:
        assert list(stores["config"].glob("*.atomic-smoke-command.json"))
        assert list(stores["preflight"].glob("*.atomic-smoke-preflight.json"))
        assert C._attempt_claim_path(fake_inputs.predecessor_path).is_file()
        assert not tuple(stores["exposure"].iterdir())
        assert not tuple(stores["journal"].iterdir())
        assert not tuple(stores["prediction"].iterdir())
        events.append("secret")
        return next(generated)

    def prepare(_corpus: object, **kwargs: object) -> object:
        assert kwargs["seed"] == SECRET_A
        assert kwargs["episode_seed"] == SECRET_B
        assert kwargs["label_seal_nonce"] == SECRET_C
        assert kwargs["expected_exposure_ledger_digest"] == (
            P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
        )
        events.append("precommit")
        return fake_precommit

    monkeypatch.setattr(C, "prepare_atomic_smoke_precommit", prepare)

    @contextmanager
    def stage(_executable: str, *, expected_launcher_digest: str):
        assert not tuple(stores["config"].iterdir())
        events.append("stage")
        yield StagedCodexLauncher(
            executable="/private/fake/staged-codex",
            launcher_digest=expected_launcher_digest,
            version="codex-cli fixture",
        )

    def preflight_runner(**kwargs: object) -> C.AtomicSmokeTransportPreflightReceipt:
        staged = kwargs["staged"]
        assert isinstance(staged, StagedCodexLauncher)
        assert not tuple(stores["config"].iterdir())
        events.append("preflight")
        return _preflight(
            launcher_path=staged.executable,
            launcher_version=staged.version,
            model=kwargs["model"],  # type: ignore[arg-type]
            reasoning_effort=kwargs["reasoning_effort"],  # type: ignore[arg-type]
        )

    named_sentinel = object()
    text_sentinel = object()

    def named_transport(*_args: object, **_kwargs: object) -> object:
        events.append("named")
        return named_sentinel

    def text_transport(*_args: object, **_kwargs: object) -> object:
        events.append("text")
        return text_sentinel

    def fake_runner(_precommit: object, **kwargs: object) -> object:
        assert _precommit is fake_precommit
        assert kwargs["source_dependency_digest"] == sources.digest
        assert kwargs["expected_protocol_digest"] == C.atomic_smoke_run_protocol_digest()
        config_data = json.loads(
            next(stores["config"].glob("*.atomic-smoke-command.json")).read_bytes()
        )
        assert kwargs["command_config_digest"] == config_data["config_digest"]
        assert kwargs["journal_store_dir"] == stores["journal"]
        assert kwargs["named_image_transport"]() is named_sentinel
        assert kwargs["text_transport"]() is text_sentinel
        assert kwargs["expected_launcher_digest"] == C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST
        assert kwargs["executable"] == "/private/fake/staged-codex"
        assert kwargs["prediction_store_dir"] == stores["prediction"]
        assert kwargs["cloud_policy_cache_snapshot"] == CloudPolicyCacheSnapshot(None)
        raise RuntimeError("stop before any live call")

    monkeypatch.setattr(C, "run_atomic_smoke", fake_runner)
    persist_terminal = C._persist_terminal
    persistence_attempts = 0

    def fail_primary_terminal_persistence(
        terminal: C.AtomicSmokeCommandTerminal,
        directory: str | Path,
        *,
        source_guard: object | None = None,
    ) -> C.AtomicSmokeDurabilityReceipt:
        nonlocal persistence_attempts
        persistence_attempts += 1
        if persistence_attempts == 1:
            raise OSError("bounded primary terminal persistence failure")
        if persistence_mode == "store-unavailable":
            raise OSError("bounded fallback terminal store unavailable")
        return persist_terminal(
            terminal,
            directory,
            source_guard=source_guard,  # type: ignore[arg-type]
        )

    if persistence_mode != "ordinary":
        monkeypatch.setattr(C, "_persist_terminal", fail_primary_terminal_persistence)
    command_kwargs = dict(
        corpus_path=tmp_path,
        archive_path=tmp_path / "archive.zip",
        predecessor_ledger_path=tmp_path / "ledger.json",
        config_store_dir=stores["config"],
        exposure_store_dir=stores["exposure"],
        journal_store_dir=stores["journal"],
        prediction_store_dir=stores["prediction"],
        terminal_store_dir=stores["terminal"],
        cache_store_dir=stores["cache"],
        preflight_store_dir=stores["preflight"],
        secret_factory=secret_factory,
        cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        launcher_stager=stage,
        preflight_runner=preflight_runner,
        named_image_transport=named_transport,
        text_transport=text_transport,
    )
    if persistence_mode == "store-unavailable":
        with pytest.raises(
            OSError, match="bounded fallback terminal store unavailable"
        ):
            C.run_atomic_smoke_command(**command_kwargs)
        assert persistence_attempts == 2
        return
    result = C.run_atomic_smoke_command(**command_kwargs)
    assert events == [
        "stage", "preflight", "secret", "secret", "secret", "precommit",
        "named", "text",
    ]
    assert result.terminal.status == "failed"
    if persistence_mode == "primary-fails":
        assert result.terminal.failure_type == "OSError"
        assert result.terminal.failure_reason_digest == hashlib.sha256(
            b"bounded primary terminal persistence failure"
        ).hexdigest()
        assert persistence_attempts == 2
    else:
        assert result.terminal.failure_type == "RuntimeError"
        assert result.terminal.failure_reason_digest == hashlib.sha256(
            b"stop before any live call"
        ).hexdigest()
        assert persistence_attempts == 0
    assert result.terminal.run_data is None
    assert result.run_receipt is None
    assert result.config_receipt.path.is_file()
    assert result.terminal_receipt.path.is_file()
    terminal_bytes = result.terminal_receipt.path.read_text(encoding="utf-8")
    assert SECRET_A not in terminal_bytes
    assert SECRET_B not in terminal_bytes
    assert SECRET_C not in terminal_bytes


def test_launcher_failure_aborts_before_config_claim_secrets_or_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores: dict[str, Path] = {}
    for name in (
        "config", "exposure", "journal", "prediction", "terminal", "cache",
        "preflight",
    ):
        stores[name] = tmp_path / name
        stores[name].mkdir(mode=0o700)
        stores[name].chmod(0o700)
    sources = _sources()
    fake_inputs = _fake_command_inputs(tmp_path)
    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs)
    monkeypatch.setattr(C, "freeze_stage_a_source_dependencies", lambda _root: sources)
    monkeypatch.setattr(
        C,
        "persist_stage_a_cache_snapshot",
        _persist_empty_cache,
    )
    monkeypatch.setattr(
        C,
        "load_stage_a_cache_snapshot",
        lambda _path, **_kw: CloudPolicyCacheSnapshot(None),
    )
    events: list[str] = []

    def secret_factory(_bytes: int) -> str:
        events.append("secret")
        return HEX_A

    def prepare(*_args: object, **_kwargs: object) -> object:
        events.append("precommit")
        raise AssertionError("precommit must be unreachable")

    monkeypatch.setattr(C, "prepare_atomic_smoke_precommit", prepare)

    @contextmanager
    def failing_stage(_executable: str, *, expected_launcher_digest: str):
        assert expected_launcher_digest == C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST
        assert not tuple(stores["config"].iterdir())
        events.append("launcher")
        raise RuntimeError("bounded launcher authentication failure")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="launcher authentication failure"):
        C.run_atomic_smoke_command(
            corpus_path=tmp_path,
            archive_path=tmp_path / "archive.zip",
            predecessor_ledger_path=tmp_path / "predecessor.json",
            config_store_dir=stores["config"],
            exposure_store_dir=stores["exposure"],
            journal_store_dir=stores["journal"],
            prediction_store_dir=stores["prediction"],
            terminal_store_dir=stores["terminal"],
            cache_store_dir=stores["cache"],
            preflight_store_dir=stores["preflight"],
            secret_factory=secret_factory,
            cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
            launcher_stager=failing_stage,
        )
    assert events == ["launcher"]
    assert not tuple(stores["config"].iterdir())
    assert not tuple(stores["preflight"].iterdir())
    assert not tuple(stores["exposure"].iterdir())
    assert not tuple(stores["journal"].iterdir())
    assert not tuple(stores["prediction"].iterdir())
    assert not tuple(stores["terminal"].iterdir())


def test_source_freeze_brackets_complete_release_authentication_before_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _sources()
    observed = StageASourceDependencyIdentity(
        (("atomic_smoke_command.py", 2, HEX_B),)
    )
    events: list[str] = []
    freeze_count = 0

    def freeze(_root: Path) -> StageASourceDependencyIdentity:
        nonlocal freeze_count
        freeze_count += 1
        events.append(f"freeze-{freeze_count}")
        return expected if freeze_count <= 2 else observed

    def authenticate(**_kwargs: object) -> object:
        events.append("authenticate")
        return _fake_command_inputs(tmp_path)

    monkeypatch.setattr(C, "freeze_stage_a_source_dependencies", freeze)
    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", authenticate)
    with pytest.raises(C.AtomicSmokeSourceMutationError) as captured:
        C.run_atomic_smoke_command(
            corpus_path="/unused",
            archive_path="/unused",
            predecessor_ledger_path="/unused",
            config_store_dir="/unreached",
            exposure_store_dir="/unreached",
            journal_store_dir="/unreached",
            prediction_store_dir="/unreached",
            terminal_store_dir="/unreached",
            cache_store_dir="/unreached",
            preflight_store_dir="/unreached",
        )
    assert captured.value.phase == "after-input-authentication"
    assert events == ["freeze-1", "freeze-2", "authenticate", "freeze-3"]


@pytest.mark.parametrize(
    "dirty_store",
    ("config", "exposure", "journal", "prediction", "terminal", "cache", "preflight"),
)
def test_every_attempt_owned_store_must_be_pristine_before_launcher_or_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_store: str,
) -> None:
    stores: dict[str, Path] = {}
    for name in (
        "config", "exposure", "journal", "prediction", "terminal", "cache",
        "preflight",
    ):
        stores[name] = tmp_path / name
        stores[name].mkdir(mode=0o700)
        stores[name].chmod(0o700)
    (stores[dirty_store] / "existing").write_bytes(b"occupied")
    fake_inputs = _fake_command_inputs(tmp_path)
    monkeypatch.setattr(
        C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs
    )
    monkeypatch.setattr(
        C, "freeze_stage_a_source_dependencies", lambda _root: _sources()
    )
    events: list[str] = []

    @contextmanager
    def stage(*_args: object, **_kwargs: object):
        events.append("launcher")
        yield  # pragma: no cover

    with pytest.raises(C.AtomicSmokeCommandError, match="fresh and pristine"):
        C.run_atomic_smoke_command(
            corpus_path="/unused",
            archive_path="/unused",
            predecessor_ledger_path="/unused",
            config_store_dir=stores["config"],
            exposure_store_dir=stores["exposure"],
            journal_store_dir=stores["journal"],
            prediction_store_dir=stores["prediction"],
            terminal_store_dir=stores["terminal"],
            cache_store_dir=stores["cache"],
            preflight_store_dir=stores["preflight"],
            launcher_stager=stage,
            secret_factory=lambda _count: events.append("secret") or HEX_A,
        )
    assert events == []
    assert not tuple(stores["exposure"].glob("*.exposure.json"))


def test_same_stores_and_fresh_stores_cannot_rerun_same_predecessor_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources()
    fake_inputs = _fake_command_inputs(tmp_path)
    monkeypatch.setattr(
        C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs
    )
    monkeypatch.setattr(
        C, "freeze_stage_a_source_dependencies", lambda _root: sources
    )
    events: list[str] = []

    @contextmanager
    def stage(_executable: str, *, expected_launcher_digest: str):
        events.append("launcher")
        yield StagedCodexLauncher(
            executable="/private/fake/staged-codex",
            launcher_digest=expected_launcher_digest,
            version="codex-cli fixture",
        )

    def preflight_runner(**kwargs: object) -> C.AtomicSmokeTransportPreflightReceipt:
        staged = kwargs["staged"]
        assert isinstance(staged, StagedCodexLauncher)
        events.append("preflight")
        return _preflight(
            launcher_path=staged.executable,
            launcher_version=staged.version,
            model=kwargs["model"],  # type: ignore[arg-type]
            reasoning_effort=kwargs["reasoning_effort"],  # type: ignore[arg-type]
        )

    secrets_seen: list[str] = []
    generated = iter((SECRET_A, SECRET_B, SECRET_C))

    def secret_factory(_count: int) -> str:
        value = next(generated)
        secrets_seen.append(value)
        events.append("secret")
        return value

    def stop_before_exposure(*_args: object, **_kwargs: object) -> object:
        events.append("precommit")
        raise RuntimeError("bounded stop after claim")

    monkeypatch.setattr(C, "prepare_atomic_smoke_precommit", stop_before_exposure)

    def make_stores(name: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for label in (
            "config", "exposure", "journal", "prediction", "terminal",
            "cache", "preflight",
        ):
            path = tmp_path / name / label
            path.mkdir(mode=0o700, parents=True)
            path.chmod(0o700)
            result[label] = path
        return result

    def kwargs(stores: Mapping[str, Path]) -> dict[str, object]:
        return {
            "corpus_path": "/unused",
            "archive_path": "/unused",
            "predecessor_ledger_path": "/unused",
            "config_store_dir": stores["config"],
            "exposure_store_dir": stores["exposure"],
            "journal_store_dir": stores["journal"],
            "prediction_store_dir": stores["prediction"],
            "terminal_store_dir": stores["terminal"],
            "cache_store_dir": stores["cache"],
            "preflight_store_dir": stores["preflight"],
            "secret_factory": secret_factory,
            "cache_snapshotter": lambda: CloudPolicyCacheSnapshot(None),
            "launcher_stager": stage,
            "preflight_runner": preflight_runner,
        }

    first_stores = make_stores("first")
    first = C.run_atomic_smoke_command(**kwargs(first_stores))
    assert first.terminal.phase == "precommit"
    assert first.attempt_claim_path is not None
    original_claim_bytes = first.attempt_claim_path.read_bytes()
    assert len(secrets_seen) == 3
    first_event_count = len(events)

    with pytest.raises(C.AtomicSmokeCommandError, match="fresh and pristine"):
        C.run_atomic_smoke_command(**kwargs(first_stores))
    assert len(events) == first_event_count
    assert len(secrets_seen) == 3

    second_stores = make_stores("second")
    second = C.run_atomic_smoke_command(**kwargs(second_stores))
    assert second.terminal.status == "failed"
    assert second.terminal.phase == "attempt-claim"
    assert second.attempt_claim is None
    assert second.attempt_claim_path is None
    assert second.terminal.attempt_claim_digest is None
    assert len(secrets_seen) == 3
    assert not tuple(second_stores["exposure"].iterdir())
    assert not tuple(second_stores["journal"].iterdir())
    assert not tuple(second_stores["prediction"].iterdir())
    assert first.attempt_claim_path.read_bytes() == original_claim_bytes


def test_claim_rejects_identical_recreate_but_copy_path_is_outside_protection(
    tmp_path: Path,
) -> None:
    config = _config()
    first = _claim(config, tmp_path / "first")
    first_path = C._persist_attempt_claim(first)
    claim_data = first.to_data()
    assert claim_data["attempt_ordinal"] == 3
    assert claim_data["schema"] == "gkm.bongard-atomic-smoke-attempt-claim.v2"
    assert claim_data["reference_execution"] == "python-canonical/v1"
    assert claim_data["predecessor_content_address"] == (
        P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
    )
    assert claim_data["predecessor_raw_file_sha256"] == (
        first.predecessor_file_sha256
    )
    assert claim_data["prior_attempt_record_content_address"] == (
        first.prior_attempt_digest
    )
    assert "prior_attempt_record_raw_file_sha256" not in claim_data
    assert claim_data["lineage_display_metadata"]["record_file_sha256"] == (
        C.ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256
    )
    assert claim_data["prior_attempt_record_content_address"] == (
        C._prior_attempt_authority_digest()
    )
    assert claim_data["command_config_content_address"] == config.digest
    assert claim_data["source_snapshot_digest"] == config.source_dependencies.digest
    assert claim_data["launcher"]["canonical_staged_path"] == (
        config.staged_launcher_path
    )
    assert claim_data["transport_preflight_receipt_digest"] == (
        config.preflight_receipt.receipt_digest
    )
    assert claim_data["secret_count_at_claim"] == 0
    assert claim_data["exposure_created_at_claim"] is False
    first_bytes = first_path.read_bytes()
    with pytest.raises(C.AtomicSmokeCommandError, match="already exists"):
        C._persist_attempt_claim(first)
    assert first_path.read_bytes() == first_bytes

    copied_ledger = first.predecessor_path.with_name("copied-bfd.exposure.json")
    copied_ledger.write_bytes(first.predecessor_path.read_bytes())
    copied_inputs = SimpleNamespace(
        digest=config.input_authentication_digest,
        predecessor_path=copied_ledger.resolve(),
        predecessor_file_sha256=first.predecessor_file_sha256,
        predecessor=SimpleNamespace(digest=first.predecessor_digest),
        prior_attempt=C.AtomicSmokePriorAttemptRecord.load(
            C.DEFAULT_PRIOR_ATTEMPT_RECORD_PATH
        ),
    )
    persistence = C.AtomicSmokeDurabilityReceipt(
        "transport-preflight",
        (tmp_path / config.preflight_receipt_filename).resolve(),
        config.preflight_receipt.receipt_digest,
        config.preflight_receipt_file_sha256,
        config.preflight_receipt_byte_count,
    )
    copied_claim = C.AtomicSmokeAttemptClaim.create(
        inputs=copied_inputs,
        config=config,
        preflight_persistence=persistence,
    )
    copied_claim_path = C._persist_attempt_claim(copied_claim)
    assert copied_claim_path != first_path
    assert "copied-ledger-paths-require-external-copy-control" in (
        copied_claim.to_data()["protection_scope"]
    )


def test_preflight_is_fixed_non_task_zero_secret_and_uses_production_parsers(
    tmp_path: Path,
) -> None:
    protocol = C.atomic_smoke_preflight_protocol_data()
    assert protocol["transport_call_count"] == 2
    assert protocol["bongard_call_count"] == 0
    assert protocol["secret_count"] == 0
    assert protocol["contains_images"] is False
    assert protocol["contains_bongard_material"] is False
    assert protocol["contains_task_material"] is False
    transport_material = json.dumps(
        {
            "proposal_prompt": C._PREFLIGHT_PROPOSAL_PROMPT,
            "proposal_payload": C._PREFLIGHT_PROPOSAL_PAYLOAD,
            "scorer_prompt": C._PREFLIGHT_SCORER_PROMPT,
            "scorer_payload": C._PREFLIGHT_SCORER_PAYLOAD,
        },
        sort_keys=True,
    ).lower()
    assert "bongard" not in transport_material
    assert "task" not in transport_material
    assert C.validate_atomic_smoke_proposal_payload(
        C._PREFLIGHT_PROPOSAL_PAYLOAD
    ) == ("Is one closed triangular outline visible?",)
    assert C.validate_atomic_smoke_scorer_payload(
        C._PREFLIGHT_SCORER_PAYLOAD,
        expected_atom_ids=(C._PREFLIGHT_ATOM_ID,),
    )[0][1] == "present"

    receipt = _preflight()
    assert C.AtomicSmokeTransportPreflightReceipt.from_data(
        receipt.to_data()
    ) == receipt
    malformed = receipt.to_data()
    malformed["secret_count"] = False
    with pytest.raises(C.AtomicSmokeCommandError, match="authority"):
        C.AtomicSmokeTransportPreflightReceipt.from_data(malformed)
    store = tmp_path / "preflight-store"
    store.mkdir(mode=0o700)
    store.chmod(0o700)
    persistence = C._persist_preflight(receipt, store)
    assert persistence.kind == "transport-preflight"
    assert persistence.content_address == receipt.receipt_digest


def test_no_bool_as_int_for_command_fields() -> None:
    values = _config().__dict__ if hasattr(_config(), "__dict__") else None
    assert values is None  # slots prevent mutable field injection.
    preflight = _preflight()
    with pytest.raises(C.AtomicSmokeCommandError, match="minutes"):
        C.AtomicSmokeCommandConfig(
            input_authentication_digest="sha256:" + HEX_B,
            source_dependencies=_sources(),
            cache_binding="absent",
            cache_file_sha256="sha256:" + hashlib.sha256(b"").hexdigest(),
            cache_byte_count=0,
            expected_launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
            staged_launcher_path=preflight.launcher_path,
            launcher_version=preflight.launcher_version,
            preflight_receipt=preflight,
            preflight_receipt_file_sha256="sha256:" + HEX_A,
            preflight_receipt_filename=(
                preflight.receipt_digest.removeprefix("sha256:")
                + ".atomic-smoke-preflight.json"
            ),
            preflight_receipt_byte_count=1,
            run_protocol_digest=C.atomic_smoke_run_protocol_digest(),
            model="gpt-5.6-sol",
            reasoning_effort="medium",
            minutes=True,
            verifier_id="fixture-verifier",
        )


def test_official_test_precommit_is_rejected_before_runner() -> None:
    with pytest.raises(C.AtomicSmokeCommandError, match="official test"):
        C._assert_non_test_precommit(
            SimpleNamespace(episode_plan=SimpleNamespace(split="test"))
        )


def test_source_mutation_is_explicit_in_terminal() -> None:
    expected = _sources()
    observed = StageASourceDependencyIdentity(
        (("atomic_smoke_command.py", 2, HEX_B),)
    )
    error = C.AtomicSmokeSourceMutationError(
        "after-terminal-serialization", expected, observed
    )
    terminal = C.AtomicSmokeCommandTerminal.failure(
        error,
        phase="terminal-persistence",
        config=_config(),
        precommit=None,
        launcher_version=_config().launcher_version,
    )
    assert terminal.status == "failed"
    assert terminal.source_dependency_state == "mutated"
    assert terminal.observed_source_dependency_digest == observed.digest
    assert terminal.failure_type == "AtomicSmokeSourceMutationError"
    assert C.AtomicSmokeCommandTerminal.from_data(terminal.to_data()) == terminal


def test_genuine_failed_run_mapping_proxy_wraps_and_round_trips(
    tmp_path: Path,
) -> None:
    sources = _sources()
    precommit = _genuine_precommit(tmp_path, source_digest=sources.digest)
    prediction_store = tmp_path / "prediction-store"
    prediction_store.mkdir(mode=0o700)
    prediction_store.chmod(0o700)
    journal_store = tmp_path / "journal-store"
    journal_store.mkdir(mode=0o700)
    journal_store.chmod(0o700)

    def offline_failure(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bounded offline observer failure")

    config = _config()
    claim = _claim(config, tmp_path / "claim-fixture")
    run = C.run_atomic_smoke(
        precommit,
        source_dependency_digest=sources.digest,
        expected_protocol_digest=C.atomic_smoke_run_protocol_digest(),
        expected_launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
        command_config_digest=config.digest,
        journal_store_dir=journal_store,
        prediction_store_dir=prediction_store,
        model="gpt-test",
        reasoning_effort="medium",
        named_image_transport=offline_failure,
        text_transport=offline_failure,
    )
    assert run.status == "failed"
    assert run.command_config_digest == config.digest
    assert isinstance(run.journal_receipt, C.AtomicSmokeJournalReceipt)
    assert run.journal_receipt.intent_count == 1
    assert run.journal_receipt.result_count == 0
    with pytest.raises(TypeError):
        json.dumps(run.precommit_public_data)
    with pytest.raises(C.AtomicSmokeCommandError, match="exact command config"):
        C.AtomicSmokeCommandTerminal.from_run(
            run,
            config=replace(config, verifier_id="different-fixture-verifier"),
            attempt_claim=claim,
            launcher_version=config.launcher_version,
        )
    assert hashlib.sha256(b"run precommit is not canonical JSON").hexdigest() == (
        "59fe3bfbe008279f711357dca206fd16afde61586ae9d160dec46d732da879e9"
    )
    assert hashlib.sha256(
        b"failed run precommit is not canonical JSON"
    ).hexdigest() == (
        "2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d"
    )

    terminal = C.AtomicSmokeCommandTerminal.from_run(
        run,
        config=config,
        attempt_claim=claim,
        launcher_version=config.launcher_version,
    )
    assert terminal.status == "failed"
    assert terminal.precommit_data == run.to_data()["precommit_public_data"]
    assert terminal.run_data == run.to_data()
    assert terminal.journal_receipt_data == run.journal_receipt.to_data()
    assert terminal.journal_receipt_digest == run.journal_receipt.receipt_digest
    assert C.AtomicSmokeCommandTerminal.from_data(terminal.to_data()) == terminal

    outer = C.AtomicSmokeCommandTerminal.failure(
        RuntimeError("bounded outer persistence failure"),
        phase="terminal-persistence",
        config=config,
        precommit=precommit,
        launcher_version=config.launcher_version,
        attempt_claim=claim,
        run=run,
    )
    assert outer.failure_type == "RuntimeError"
    assert outer.run_data == run.to_data()
    assert C.AtomicSmokeCommandTerminal.from_data(outer.to_data()) == outer


def test_command_persists_genuine_failed_run_before_context_exit_and_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources()
    precommit = _genuine_precommit(
        tmp_path / "runner-fixture", source_digest=sources.digest
    )

    def offline_failure(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bounded offline observer failure")

    stores: dict[str, Path] = {}
    for name in (
        "config", "exposure", "journal", "prediction", "terminal", "cache",
        "preflight",
    ):
        stores[name] = tmp_path / ("command-" + name)
        stores[name].mkdir(mode=0o700)
        stores[name].chmod(0o700)
    fake_inputs = _fake_command_inputs(tmp_path)
    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs)
    monkeypatch.setattr(C, "freeze_stage_a_source_dependencies", lambda _root: sources)
    monkeypatch.setattr(
        C,
        "persist_stage_a_cache_snapshot",
        _persist_empty_cache,
    )
    monkeypatch.setattr(
        C,
        "load_stage_a_cache_snapshot",
        lambda _path, **_kw: CloudPolicyCacheSnapshot(None),
    )
    monkeypatch.setattr(C, "prepare_atomic_smoke_precommit", lambda *_a, **_kw: precommit)
    context_exited: list[bool] = []
    persisted_runs: list[dict[str, object]] = []

    @contextmanager
    def stage(_executable: str, *, expected_launcher_digest: str):
        try:
            yield StagedCodexLauncher(
                executable="/private/fake/staged-codex",
                launcher_digest=expected_launcher_digest,
                version="codex-cli offline-test",
            )
        finally:
            persisted = tuple(
                stores["terminal"].glob("*.atomic-smoke-run.json")
            )
            assert len(persisted) == 1
            persisted_runs.append(json.loads(persisted[0].read_bytes()))
            context_exited.append(True)

    def preflight_runner(**kwargs: object) -> C.AtomicSmokeTransportPreflightReceipt:
        staged = kwargs["staged"]
        assert isinstance(staged, StagedCodexLauncher)
        return _preflight(
            launcher_path=staged.executable,
            launcher_version=staged.version,
            model=kwargs["model"],  # type: ignore[arg-type]
            reasoning_effort=kwargs["reasoning_effort"],  # type: ignore[arg-type]
        )

    generated = iter((HEX_A, HEX_B, HEX_C))
    result = C.run_atomic_smoke_command(
        corpus_path=tmp_path,
        archive_path=tmp_path / "archive.zip",
        predecessor_ledger_path=tmp_path / "ledger.json",
        config_store_dir=stores["config"],
        exposure_store_dir=stores["exposure"],
        journal_store_dir=stores["journal"],
        prediction_store_dir=stores["prediction"],
        terminal_store_dir=stores["terminal"],
        cache_store_dir=stores["cache"],
        preflight_store_dir=stores["preflight"],
        secret_factory=lambda _bytes: next(generated),
        cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        launcher_stager=stage,
        preflight_runner=preflight_runner,
        verifier_id="fixture-verifier",
        named_image_transport=offline_failure,
        text_transport=offline_failure,
    )
    assert context_exited == [True]
    assert result.run_receipt is not None
    assert result.run_receipt.kind == "atomic-smoke-run"
    assert result.run_receipt.content_address == (
        "sha256:" + result.terminal.run_digest
    )
    assert result.run_receipt.path.is_file()
    assert persisted_runs == [result.terminal.run_data]
    assert result.terminal.failure_type is None
    assert result.terminal.journal_receipt_data is not None
    assert result.terminal.journal_receipt_digest == (
        result.terminal.run_data["journal_receipt"]["receipt_digest"]
    )


def test_command_stores_reject_symlink_and_changed_identical_address(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    real.chmod(0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(C.AtomicSmokeCommandError, match="canonical directory"):
        C._persist_config(_config(), linked)

    receipt = C._persist_config(_config(), real)
    receipt.path.write_bytes(b"{}")
    with pytest.raises(C.AtomicSmokeCommandError, match="different"):
        C._persist_config(_config(), real)

    bound = C._StoreBinding.freeze("fixture", real)
    moved = tmp_path / "moved"
    real.rename(moved)
    real.mkdir(mode=0o700)
    real.chmod(0o700)
    with pytest.raises(C.AtomicSmokeCommandError, match="store changed"):
        bound.check("after replacement")


def test_command_store_binding_requires_current_owner_and_exact_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir(mode=0o755)
    store.chmod(0o755)
    with pytest.raises(C.AtomicSmokeCommandError, match="owner-only 0700"):
        C._StoreBinding.freeze("fixture", store)

    store.chmod(0o700)
    actual_uid = C.os.getuid()
    monkeypatch.setattr(C.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(C.AtomicSmokeCommandError, match="owner-only 0700"):
        C._StoreBinding.freeze("fixture", store)


def test_terminal_persistence_is_source_guarded_through_reload(tmp_path: Path) -> None:
    terminal_store = tmp_path / "terminal"
    terminal_store.mkdir()
    terminal = C.AtomicSmokeCommandTerminal.failure(
        RuntimeError("bounded fixture failure"),
        phase="fixture",
        config=_config(),
        precommit=None,
        launcher_version=_config().launcher_version,
    )
    phases: list[str] = []

    class Guard:
        def check(self, phase: str) -> None:
            phases.append(phase)

    receipt = C._persist_terminal(terminal, terminal_store, source_guard=Guard())
    assert receipt.path.is_file()
    assert phases == [
        "before-terminal-serialization",
        "after-terminal-serialization",
        "after-terminal-persistence-reload",
    ]


def test_cli_output_is_machine_readable_and_never_contains_selected_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = SimpleNamespace(
        to_data=lambda: {
            "schema": C.ATOMIC_SMOKE_COMMAND_RECEIPT_SCHEMA,
            "filename": "fixture.json",
        }
    )
    result = SimpleNamespace(
        config=SimpleNamespace(digest="sha256:" + HEX_A),
        config_receipt=receipt,
        preflight=SimpleNamespace(receipt_digest="sha256:" + HEX_B),
        attempt_claim=SimpleNamespace(claim_digest="sha256:" + HEX_C),
        attempt_claim_path=Path("/fixture/attempt.claim.json"),
        run_receipt=receipt,
        terminal=SimpleNamespace(
            status="complete",
            terminal_digest="sha256:" + HEX_B,
            precommit_digest="sha256:" + HEX_C,
            run_digest=HEX_D,
            journal_receipt_digest=HEX_A,
            journal_receipt_data={"intent_count": 29, "result_count": 29},
        ),
        terminal_receipt=receipt,
    )
    monkeypatch.setattr(C, "run_atomic_smoke_command", lambda **_kw: result)
    exit_code = C.main(
        [
            "--corpus", "/fixture/corpus",
            "--archive", "/fixture/archive.zip",
            "--predecessor-ledger", "/fixture/successor.json",
            "--config-store", "/fixture/config",
            "--exposure-store", "/fixture/exposure",
            "--journal-store", "/fixture/journal",
            "--prediction-store", "/fixture/prediction",
            "--terminal-store", "/fixture/terminal",
            "--cache-store", "/fixture/cache",
            "--preflight-store", "/fixture/preflight",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    decoded = json.loads(output)
    assert decoded["selected_task_id_included"] is False
    assert decoded["run_persistence"]["filename"] == "fixture.json"
    assert decoded["attempt_ordinal"] == 3
    assert decoded["schema"] == "gkm.bongard-atomic-smoke-cli-result.v5"
    assert decoded["transport_preflight_receipt_digest"] == "sha256:" + HEX_B
    assert decoded["attempt_claim_digest"] == "sha256:" + HEX_C
    assert decoded["attempt_claim_filename"] == "attempt.claim.json"
    assert decoded["claim_protection_scope"].startswith(
        "local-canonical-predecessor-path-only"
    )
    assert decoded["journal_intent_count"] == 29
    assert output.count("selected_task_id") == 1
    assert "bd_" not in output


def test_keyboard_interrupt_is_not_terminalized_or_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(**_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", interrupt)
    with pytest.raises(KeyboardInterrupt):
        C.run_atomic_smoke_command(
            corpus_path="/unused",
            archive_path="/unused",
            predecessor_ledger_path="/unused",
            config_store_dir="/unused",
            exposure_store_dir="/unused",
            journal_store_dir="/unused",
            prediction_store_dir="/unused",
            terminal_store_dir="/unused",
            cache_store_dir="/unused",
            preflight_store_dir="/unused",
        )
