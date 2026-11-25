from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

import arc_agi3_container_worker as Worker


@pytest.fixture(autouse=True)
def _restore_attempt_solver_import_state():
    original_path = list(sys.path)
    missing = object()
    original_modules = {
        name: sys.modules.get(name, missing)
        for name in ("solve", "players", "legs")
    }
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name, module in original_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class FakeArena:
    def __init__(self):
        self.steps = []

    def step(self, action, x=None, y=None):
        self.steps.append((action, x, y))

    def clone(self):
        return FakeArena()


class FakeClient:
    instances = []

    def __init__(self, socket_path, token):
        self.socket_path = Path(socket_path)
        self.token = token
        self.root = FakeArena()
        self.closed = False
        self.__class__.instances.append(self)

    def close(self):
        self.closed = True


def make_config(tmp_path: Path, source: str) -> Worker.WorkerConfig:
    solve = tmp_path / "attempt" / "solve.py"
    solve.parent.mkdir()
    solve.write_text(source)
    token = tmp_path / "token"
    token.write_text("t" * 64)
    return Worker.WorkerConfig(
        socket_path=tmp_path / "arena.sock",
        token_file=token,
        solve_path=solve,
        outcome_path=tmp_path / "output" / "outcome.json",
    )


def valid_outcome(*, status: str = "completed") -> dict:
    return {
        "schema": Worker.WORKER_SCHEMA,
        "status": status,
        "solver_sha256": "0" * 64,
        "elapsed_ns": 1,
        "error": (
            None
            if status == "completed"
            else {
                "type": "SolverError",
                "message": "solver execution failed",
            }
        ),
        "authoritative": False,
    }


def test_worker_executes_solver_and_marks_outcome_non_authoritative(tmp_path: Path):
    FakeClient.instances.clear()
    config = make_config(
        tmp_path,
        "def solve(env):\n"
        "    env.step(1)\n"
        "    clone = env.clone()\n"
        "    clone.step(6, 2, 3)\n",
    )
    outcome = Worker.run_worker(config, client_factory=FakeClient)
    assert outcome["status"] == "completed"
    assert outcome["authoritative"] is False
    assert outcome["error"] is None
    assert len(outcome["solver_sha256"]) == 64
    assert FakeClient.instances[-1].root.steps == [(1, None, None)]
    assert FakeClient.instances[-1].closed is True
    assert json.loads(config.outcome_path.read_text()) == outcome


def test_solver_error_is_sanitized_without_traceback(tmp_path: Path):
    config = make_config(
        tmp_path,
        "def solve(env):\n"
        "    raise RuntimeError('expected failure\\nwith detail')\n",
    )
    outcome = Worker.run_worker(config, client_factory=FakeClient)
    assert outcome["status"] == "solver_error"
    assert outcome["error"] == {
        "type": "SolverError",
        "message": "solver execution failed",
    }
    raw = config.outcome_path.read_text()
    assert "Traceback" not in raw
    assert "expected failure" not in raw
    assert outcome["authoritative"] is False


@pytest.mark.parametrize(
    "argv,match",
    [
        ([], "missing"),
        (
            [
                "--socket=a",
                "--socket=b",
                "--token-file=c",
                "--solve=d",
                "--outcome=e",
            ],
            "duplicate",
        ),
        (
            [
                "--socket=a",
                "--token-file=c",
                "--solve=d",
                "--outcome=e",
                "--game=wa30",
            ],
            "unknown",
        ),
        (
            ["--socket", "--token-file=c", "--solve=d", "--outcome=e"],
            "exact",
        ),
        (
            ["--socket=", "--token-file=c", "--solve=d", "--outcome=e"],
            "cannot be empty",
        ),
    ],
)
def test_parser_is_strict_and_has_no_last_value_wins(argv, match):
    with pytest.raises(Worker.WorkerContractError, match=match):
        Worker.parse_args(argv)


def test_parser_accepts_exact_required_fields():
    config = Worker.parse_args(
        [
            "--socket=/rpc/arena.sock",
            "--token-file=/run/token",
            "--solve=/scratch/solve.py",
            "--outcome=/output/outcome.json",
        ]
    )
    assert config.socket_path == Path("/rpc/arena.sock")
    assert config.solve_path == Path("/scratch/solve.py")


def test_help_is_processed_without_touching_workspace():
    with pytest.raises(SystemExit, match="usage:"):
        Worker.parse_args(["--help"])


def test_token_symlink_is_rejected(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.write_text("t" * 64)
    link = tmp_path / "token"
    link.symlink_to(actual)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._read_token(link)


@pytest.mark.parametrize("suffix", ["\n", " ", "\t"])
def test_token_whitespace_aliases_are_rejected(tmp_path: Path, suffix: str):
    token = tmp_path / "token"
    token.write_text("t" * 64 + suffix)
    with pytest.raises(Worker.WorkerContractError, match="format"):
        Worker._read_token(token)


def test_token_hardlink_is_rejected(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.write_text("t" * 64)
    alias = tmp_path / "token"
    os.link(actual, alias)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._read_token(alias)


def test_solver_symlink_is_rejected(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("def solve(env): pass\n")
    link = tmp_path / "solve.py"
    link.symlink_to(source)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._load_solver(link)


def test_solver_hardlink_is_rejected(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("def solve(env): pass\n")
    alias = tmp_path / "solve.py"
    os.link(source, alias)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._load_solver(alias)


def test_solver_executes_the_exact_bytes_that_were_hashed(tmp_path: Path):
    source = tmp_path / "solve.py"
    admitted = (
        "from pathlib import Path\n"
        f"Path({str(source)!r}).write_text('def solve(env): env.step(7)\\n')\n"
        "def solve(env): env.step(1)\n"
    )
    source.write_text(admitted)
    solve, digest = Worker._load_solver(source)
    arena = FakeArena()
    solve(arena)
    assert arena.steps == [(1, None, None)]
    assert digest == hashlib.sha256(admitted.encode()).hexdigest()
    assert "env.step(7)" in source.read_text()


def test_missing_solve_callable_is_contract_error(tmp_path: Path):
    source = tmp_path / "solve.py"
    source.write_text("value = 1\n")
    with pytest.raises(Worker.WorkerContractError, match="solve"):
        Worker._load_solver(source)


def test_outcome_must_be_new_file(tmp_path: Path):
    outcome = tmp_path / "outcome.json"
    outcome.write_text("old")
    with pytest.raises(Worker.WorkerContractError, match="already exists"):
        Worker._write_outcome(
            outcome,
            valid_outcome(),
        )
    assert outcome.read_text() == "old"


def test_outcome_schema_cannot_claim_authority(tmp_path: Path):
    outcome = valid_outcome()
    outcome["authoritative"] = True
    with pytest.raises(Worker.WorkerContractError, match="invalid values"):
        Worker._write_outcome(tmp_path / "outcome.json", outcome)


def test_outcome_cannot_export_solver_controlled_error_text(tmp_path: Path):
    outcome = valid_outcome(status="solver_error")
    outcome["error"] = {
        "type": "RuntimeError",
        "message": "PRIVATE_TOKEN_OR_PATH",
    }
    with pytest.raises(Worker.WorkerContractError, match="malformed"):
        Worker._write_outcome(tmp_path / "outcome.json", outcome)


def test_outcome_symlink_and_hardlink_destinations_are_rejected(tmp_path: Path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._write_outcome(symlink, valid_outcome())
    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)
    with pytest.raises(Worker.WorkerContractError, match="unalias"):
        Worker._write_outcome(hardlink, valid_outcome())
    assert target.read_text() == "unchanged"


def test_outcome_symlinked_parent_is_rejected(tmp_path: Path):
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(Worker.WorkerContractError, match="symlinked"):
        Worker._write_outcome(
            linked_parent / "outcome.json",
            valid_outcome(),
        )
    assert not (actual_parent / "outcome.json").exists()


def test_outcome_publication_race_never_overwrites_forged_file(
    tmp_path: Path,
    monkeypatch,
):
    outcome = tmp_path / "output" / "outcome.json"
    original_link = Worker.os.link

    def race_link(src, dst, *, src_dir_fd, dst_dir_fd, follow_symlinks):
        del src, src_dir_fd, dst_dir_fd, follow_symlinks
        outcome.write_text("forged")
        raise FileExistsError(dst)

    monkeypatch.setattr(Worker.os, "link", race_link)
    with pytest.raises(Worker.WorkerContractError, match="already exists"):
        Worker._write_outcome(outcome, valid_outcome())
    monkeypatch.setattr(Worker.os, "link", original_link)
    assert outcome.read_text() == "forged"
    assert not list(outcome.parent.glob(".*.tmp"))


def test_failed_outcome_write_cleans_trusted_temporary(
    tmp_path: Path,
    monkeypatch,
):
    outcome = tmp_path / "output" / "outcome.json"
    original_write = Worker.os.write

    def fail_write(_descriptor, _view):
        raise OSError("injected write failure")

    monkeypatch.setattr(Worker.os, "write", fail_write)
    with pytest.raises(OSError, match="injected"):
        Worker._write_outcome(outcome, valid_outcome())
    monkeypatch.setattr(Worker.os, "write", original_write)
    assert not outcome.exists()
    assert not list(outcome.parent.glob(".*.tmp"))


def test_solver_cannot_replace_outcome_and_be_reported_as_success(tmp_path: Path):
    forged = {
        **valid_outcome(),
        "authoritative": True,
    }
    config = make_config(
        tmp_path,
        "from pathlib import Path\n"
        "import json\n"
        f"target = Path({str(tmp_path / 'output' / 'outcome.json')!r})\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        f"target.write_text(json.dumps({forged!r}))\n"
        "def solve(env): pass\n",
    )
    with pytest.raises(Worker.WorkerContractError, match="already exists"):
        Worker.run_worker(config, client_factory=FakeClient)
    assert json.loads(config.outcome_path.read_text())["authoritative"] is True


def test_top_level_exception_and_secret_are_not_exported(tmp_path: Path):
    secret = "s" * 64
    config = make_config(
        tmp_path,
        f"raise RuntimeError({(secret + ' ' + str(tmp_path))!r})\n",
    )
    config.token_file.write_text(secret)
    outcome = Worker.run_worker(config, client_factory=FakeClient)
    raw = config.outcome_path.read_text()
    assert outcome["status"] == "solver_error"
    assert secret not in raw
    assert str(tmp_path) not in raw
    assert "RuntimeError" not in raw


def test_close_failure_is_sanitized(tmp_path: Path):
    class FailingCloseClient(FakeClient):
        def close(self):
            raise RuntimeError("SECRET_CLOSE_DETAIL")

    config = make_config(tmp_path, "def solve(env): pass\n")
    outcome = Worker.run_worker(config, client_factory=FailingCloseClient)
    assert outcome["status"] == "solver_error"
    assert outcome["error"] == {
        "type": "SolverError",
        "message": "solver execution failed",
    }
    assert "SECRET_CLOSE_DETAIL" not in config.outcome_path.read_text()


def test_main_contract_error_stderr_is_fixed_and_secret_free(
    tmp_path: Path,
    capsys,
):
    config = make_config(tmp_path, "def solve(env): pass\n")
    secret = "PRIVATE_TOKEN_DETAIL"
    config.token_file.write_text(secret)
    code = Worker.main(
        [
            f"--socket={config.socket_path}",
            f"--token-file={config.token_file}",
            f"--solve={config.solve_path}",
            f"--outcome={config.outcome_path}",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.strip() == "WORKER_CONTRACT_ERROR"
    assert secret not in captured.err
    assert str(tmp_path) not in captured.err


def test_main_returns_nonzero_for_solver_error(tmp_path: Path, monkeypatch):
    config = make_config(
        tmp_path,
        "def solve(env):\n"
        "    raise ValueError('bad')\n",
    )
    monkeypatch.setattr(Worker, "ArenaRpcClient", FakeClient)
    # The default was bound when run_worker was defined, so supply a small
    # wrapper that preserves main's behavior while using the fake transport.
    original = Worker.run_worker

    def run_with_fake(parsed):
        return original(parsed, client_factory=FakeClient)

    monkeypatch.setattr(Worker, "run_worker", run_with_fake)
    code = Worker.main(
        [
            f"--socket={config.socket_path}",
            f"--token-file={config.token_file}",
            f"--solve={config.solve_path}",
            f"--outcome={config.outcome_path}",
        ]
    )
    assert code == 1
