from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bongard.atomic_smoke_command as C
from bongard.semantic_calibration_command import StageASourceDependencyIdentity
from bongard.transport import CloudPolicyCacheSnapshot, StagedCodexLauncher


HEX_A = hashlib.sha256(b"a").hexdigest()
HEX_B = hashlib.sha256(b"b").hexdigest()
HEX_C = hashlib.sha256(b"c").hexdigest()
HEX_D = hashlib.sha256(b"d").hexdigest()


def _sources() -> StageASourceDependencyIdentity:
    return StageASourceDependencyIdentity((("atomic_smoke_command.py", 1, HEX_A),))


def _config() -> C.AtomicSmokeCommandConfig:
    return C.AtomicSmokeCommandConfig(
        input_authentication_digest="sha256:" + HEX_B,
        source_dependencies=_sources(),
        cache_binding="absent",
        cache_file_sha256="sha256:" + hashlib.sha256(b"").hexdigest(),
        cache_byte_count=0,
        expected_launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
        run_protocol_digest=C.atomic_smoke_run_protocol_digest(),
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        minutes=15,
        verifier_id="fixture-verifier",
    )


def test_production_pins_and_secret_free_config() -> None:
    assert C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST == (
        "ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02"
    )
    config = _config()
    assert C.AtomicSmokeCommandConfig.from_data(config.to_data()) == config
    encoded = json.dumps(config.to_data(), sort_keys=True)
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
        launcher_version=None,
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


def test_command_persists_config_before_secrets_and_uses_guarded_test_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = {}
    for name in ("config", "exposure", "prediction", "terminal", "cache"):
        stores[name] = tmp_path / name
        stores[name].mkdir()
    events: list[str] = []
    sources = _sources()
    fake_inputs = SimpleNamespace(
        digest="sha256:" + HEX_B,
        trusted=SimpleNamespace(corpus=object(), full_manifest=object()),
        predecessor=object(),
    )
    fake_precommit = SimpleNamespace(
        digest="sha256:" + HEX_D,
        episode_plan=SimpleNamespace(split="train"),
    )

    monkeypatch.setattr(C, "authenticate_atomic_smoke_inputs", lambda **_kw: fake_inputs)
    monkeypatch.setattr(C, "freeze_stage_a_source_dependencies", lambda _root: sources)
    monkeypatch.setattr(
        C,
        "persist_stage_a_cache_snapshot",
        lambda _snapshot, directory: (
            Path(directory) / "empty.cache",
            "sha256:" + hashlib.sha256(b"").hexdigest(),
            0,
        ),
    )
    monkeypatch.setattr(
        C,
        "load_stage_a_cache_snapshot",
        lambda _path, **_kw: CloudPolicyCacheSnapshot(None),
    )

    generated = iter((HEX_A, HEX_B, HEX_C))

    def secret_factory(_bytes: int) -> str:
        assert list(stores["config"].glob("*.atomic-smoke-command.json"))
        events.append("secret")
        return next(generated)

    def prepare(_corpus: object, **kwargs: object) -> object:
        assert kwargs["seed"] == HEX_A
        assert kwargs["episode_seed"] == HEX_B
        assert kwargs["label_seal_nonce"] == HEX_C
        events.append("precommit")
        return fake_precommit

    monkeypatch.setattr(C, "prepare_atomic_smoke_precommit", prepare)

    @contextmanager
    def stage(_executable: str, *, expected_launcher_digest: str):
        events.append("stage")
        yield StagedCodexLauncher(
            executable="/private/fake/staged-codex",
            launcher_digest=expected_launcher_digest,
            version="codex-cli fixture",
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
        assert kwargs["named_image_transport"]() is named_sentinel
        assert kwargs["text_transport"]() is text_sentinel
        assert kwargs["expected_launcher_digest"] == C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST
        assert kwargs["executable"] == "/private/fake/staged-codex"
        assert kwargs["prediction_store_dir"] == stores["prediction"]
        assert kwargs["cloud_policy_cache_snapshot"] == CloudPolicyCacheSnapshot(None)
        raise RuntimeError("stop before any live call")

    monkeypatch.setattr(C, "run_atomic_smoke", fake_runner)
    result = C.run_atomic_smoke_command(
        corpus_path=tmp_path,
        archive_path=tmp_path / "archive.zip",
        exposure_ledger_path=tmp_path / "ledger.json",
        config_store_dir=stores["config"],
        exposure_store_dir=stores["exposure"],
        prediction_store_dir=stores["prediction"],
        terminal_store_dir=stores["terminal"],
        cache_store_dir=stores["cache"],
        secret_factory=secret_factory,
        cache_snapshotter=lambda: CloudPolicyCacheSnapshot(None),
        launcher_stager=stage,
        named_image_transport=named_transport,
        text_transport=text_transport,
    )
    assert events[:3] == ["secret", "secret", "secret"]
    assert events[3:] == ["precommit", "stage", "named", "text"]
    assert result.terminal.status == "failed"
    assert result.terminal.failure_type == "RuntimeError"
    assert result.terminal.run_data is None
    assert result.config_receipt.path.is_file()
    assert result.terminal_receipt.path.is_file()
    terminal_bytes = result.terminal_receipt.path.read_text(encoding="utf-8")
    assert HEX_A not in terminal_bytes
    assert HEX_B not in terminal_bytes
    assert HEX_C not in terminal_bytes


def test_no_bool_as_int_for_command_fields() -> None:
    values = _config().__dict__ if hasattr(_config(), "__dict__") else None
    assert values is None  # slots prevent mutable field injection.
    with pytest.raises(C.AtomicSmokeCommandError, match="minutes"):
        C.AtomicSmokeCommandConfig(
            input_authentication_digest="sha256:" + HEX_B,
            source_dependencies=_sources(),
            cache_binding="absent",
            cache_file_sha256="sha256:" + hashlib.sha256(b"").hexdigest(),
            cache_byte_count=0,
            expected_launcher_digest=C.ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
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
        launcher_version=None,
    )
    assert terminal.status == "failed"
    assert terminal.source_dependency_state == "mutated"
    assert terminal.observed_source_dependency_digest == observed.digest
    assert terminal.failure_type == "AtomicSmokeSourceMutationError"
    assert C.AtomicSmokeCommandTerminal.from_data(terminal.to_data()) == terminal


def test_command_stores_reject_symlink_and_changed_identical_address(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
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
    real.mkdir()
    with pytest.raises(C.AtomicSmokeCommandError, match="store changed"):
        bound.check("after replacement")


def test_terminal_persistence_is_source_guarded_through_reload(tmp_path: Path) -> None:
    terminal_store = tmp_path / "terminal"
    terminal_store.mkdir()
    terminal = C.AtomicSmokeCommandTerminal.failure(
        RuntimeError("bounded fixture failure"),
        phase="fixture",
        config=_config(),
        precommit=None,
        launcher_version=None,
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
        terminal=SimpleNamespace(
            status="complete",
            terminal_digest="sha256:" + HEX_B,
            precommit_digest="sha256:" + HEX_C,
            run_digest=HEX_D,
        ),
        terminal_receipt=receipt,
    )
    monkeypatch.setattr(C, "run_atomic_smoke_command", lambda **_kw: result)
    exit_code = C.main(
        [
            "--corpus", "/fixture/corpus",
            "--archive", "/fixture/archive.zip",
            "--exposure-ledger", "/fixture/a3.json",
            "--config-store", "/fixture/config",
            "--exposure-store", "/fixture/exposure",
            "--prediction-store", "/fixture/prediction",
            "--terminal-store", "/fixture/terminal",
            "--cache-store", "/fixture/cache",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    decoded = json.loads(output)
    assert decoded["selected_task_id_included"] is False
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
            exposure_ledger_path="/unused",
            config_store_dir="/unused",
            exposure_store_dir="/unused",
            prediction_store_dir="/unused",
            terminal_store_dir="/unused",
            cache_store_dir="/unused",
        )
