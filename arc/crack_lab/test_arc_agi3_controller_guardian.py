from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import arc_agi3_controller_guardian as Guardian
import pytest


def test_controller_state_substrate_rejects_read_only_then_initializes_sqlite(
    tmp_path: Path,
):
    """Exercise the real filesystem distinction seen by Codex startup."""

    read_only = tmp_path / "read-only-state"
    read_only.mkdir(mode=0o500)
    os.chmod(read_only, 0o500)
    try:
        with pytest.raises(
            Guardian.GuardianError,
            match="not writable by its runtime identity",
        ):
            Guardian._probe_state_root_write(read_only)
        assert not (
            read_only / Guardian.STATE_WRITE_PROBE_NAME
        ).exists()
    finally:
        os.chmod(read_only, 0o700)

    isolated_state = tmp_path / "isolated-writable-state"
    isolated_state.mkdir(mode=0o700)
    os.chmod(isolated_state, 0o700)
    receipt = Guardian._probe_state_root_write(isolated_state)
    assert receipt["status"] == "PASS"
    assert receipt["runtime_uid"] == os.getuid()
    assert receipt["runtime_gid"] == os.getgid()
    assert receipt["probe_absent_after_fsync"] is True

    database = isolated_state / "state_5.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE substrate_probe (value INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO substrate_probe VALUES (1)"
        )
        connection.commit()
        assert connection.execute(
            "SELECT value FROM substrate_probe"
        ).fetchone() == (1,)
    finally:
        connection.close()
    assert database.read_bytes().startswith(b"SQLite format 3\x00")


def _sleeping_child(*, close_stdin: bool = False):
    source = (
        "import os,time; os.close(0); time.sleep(60)"
        if close_stdin
        else "import time; time.sleep(60)"
    )
    return subprocess.Popen(
        (sys.executable, "-c", source),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_control_pump_overflow_is_bounded_containment_failure(
    monkeypatch,
):
    monkeypatch.setattr(Guardian, "MAX_PENDING_CONTROL_BYTES", 1024)
    read_fd, write_fd = os.pipe()
    child = _sleeping_child()
    try:
        os.write(write_fd, b"x" * 4096)
        os.close(write_fd)
        write_fd = -1
        started = time.monotonic()
        outcome = Guardian._pump_control_input(
            child,
            child_group=child.pid,
            input_descriptor=read_fd,
            deadline=started + 10,
        )
        assert time.monotonic() - started < 5
        assert outcome.control_fault == "child_stdin_buffer_overflow"
        assert child.poll() is not None
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
        if child.poll() is None:
            child.kill()
            child.wait()


def test_control_pump_epipe_is_containment_failure():
    read_fd, write_fd = os.pipe()
    child = _sleeping_child(close_stdin=True)
    try:
        time.sleep(0.05)
        os.write(write_fd, b'{"id":1}\n')
        os.close(write_fd)
        write_fd = -1
        outcome = Guardian._pump_control_input(
            child,
            child_group=child.pid,
            input_descriptor=read_fd,
            deadline=time.monotonic() + 10,
        )
        assert outcome.control_fault == "child_stdin_epipe"
        assert child.poll() is not None
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
        if child.poll() is None:
            child.kill()
            child.wait()


def test_control_pump_stall_is_containment_failure(monkeypatch):
    monkeypatch.setattr(Guardian, "CONTROL_WRITE_STALL_SECONDS", 0.05)
    monkeypatch.setattr(
        Guardian, "MAX_PENDING_CONTROL_BYTES", 2 * 1024 * 1024
    )
    read_fd, write_fd = os.pipe()
    child = _sleeping_child()
    writer_error: list[BaseException] = []

    def feed() -> None:
        try:
            payload = b"x" * (512 * 1024)
            view = memoryview(payload)
            while view:
                written = os.write(write_fd, view)
                view = view[written:]
        except BaseException as exc:
            writer_error.append(exc)
        finally:
            os.close(write_fd)

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        started = time.monotonic()
        outcome = Guardian._pump_control_input(
            child,
            child_group=child.pid,
            input_descriptor=read_fd,
            deadline=started + 10,
        )
        assert time.monotonic() - started < 5
        assert outcome.control_fault == "child_stdin_stall"
        assert outcome.pending_input_peak_bytes > 0
        assert child.poll() is not None
    finally:
        os.close(read_fd)
        if child.poll() is None:
            child.kill()
            child.wait()
        writer.join(timeout=2)


def _git_environment(workspace: Path) -> dict[str, str]:
    return {
        "GIT_CEILING_DIRECTORIES": str(workspace),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }


def _git(
    workspace: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(workspace), *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        cwd="/",
        env=_git_environment(workspace),
        close_fds=True,
    )


def _assert_native_workspace_tamper_rejected(root: Path) -> None:
    root.mkdir(mode=0o700)
    for case in ("symlink", "hardlink", "alternates"):
        workspace = root / case
        workspace.mkdir(mode=0o700)
        os.chmod(workspace, 0o700)
        Guardian._initialize_native_workspace(workspace)
        if case == "symlink":
            (workspace / ".git" / "escape").symlink_to(root)
        elif case == "hardlink":
            os.link(
                workspace / ".git" / "HEAD",
                workspace / ".git" / "HEAD.alias",
            )
        else:
            info = workspace / ".git" / "objects" / "info"
            info.mkdir(mode=0o700)
            (info / "alternates").write_text(
                str(root / ".git" / "objects") + "\n",
                encoding="utf-8",
            )
        with pytest.raises(
            Guardian.GuardianError,
            match="escapes|aliases|allowlist|bytes",
        ):
            Guardian._validate_native_workspace(workspace)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_native_proposer_workspace_is_its_only_git_root_and_broad_git_is_local(
    tmp_path: Path,
):
    outer = tmp_path / "outer"
    workspace = outer / "native-proposer"
    workspace.mkdir(parents=True, mode=0o700)
    os.chmod(outer, 0o700)
    os.chmod(workspace, 0o700)
    # Material that must never become visible through parent-project
    # discovery.  The outer .git is intentionally not a valid repository:
    # success therefore proves the direct child .git is authoritative.
    (outer / ".git").mkdir(mode=0o700)
    for relative in (
        "ARC_AGI3_CAMPAIGN_PLAN.md",
        "quarantine-output.json",
        "manuscript.md",
        "comparator.md",
        "benchmark-scorecard.json",
    ):
        (outer / relative).write_text("forbidden\n", encoding="utf-8")

    receipt = Guardian._initialize_native_workspace(workspace)
    assert receipt["git_root_equals_workspace"] is True
    assert receipt["forbidden_classes_absent"] == list(
        Guardian.NATIVE_WORKSPACE_FORBIDDEN_CLASSES
    )

    top = _git(workspace, "rev-parse", "--show-toplevel")
    git_dir = _git(workspace, "rev-parse", "--git-dir")
    status = _git(workspace, "status", "--porcelain=v1")
    diff = _git(workspace, "diff", "--no-ext-diff")
    log = _git(workspace, "log", "-1", "--format=%H")
    assert top.returncode == 0 and Path(top.stdout.strip()) == workspace
    assert git_dir.returncode == 0 and git_dir.stdout.strip() == ".git"
    assert status.returncode == 0 and status.stdout == ""
    assert diff.returncode == 0 and diff.stdout == ""
    assert log.returncode == 0
    assert log.stdout.strip() == receipt["head_commit"]
    combined = top.stdout + status.stdout + diff.stdout + log.stdout
    assert not any(
        name in combined
        for name in (
            "ARC_AGI3_CAMPAIGN_PLAN",
            "quarantine",
            "manuscript",
            "comparator",
            "benchmark",
        )
    )
    _assert_native_workspace_tamper_rejected(tmp_path / "tamper")
    for index, relative in enumerate(
        (
            "ARC_AGI3_CAMPAIGN_PLAN.md",
            "quarantine/output.json",
            "manuscript/draft.md",
            "benchmark/scorecard.json",
            ".git",
        )
    ):
        rejected = tmp_path / f"preexisting-{index}"
        rejected.mkdir(mode=0o700)
        selected = rejected / relative
        if Path(relative).suffix:
            selected.parent.mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
            selected.write_text("forbidden\n", encoding="utf-8")
        else:
            selected.mkdir(mode=0o700)
        with pytest.raises(
            Guardian.GuardianError, match="not initially empty"
        ):
            Guardian._initialize_native_workspace(rejected)


@pytest.mark.parametrize(
    "relative",
    (
        "ARC_AGI3_CAMPAIGN_PLAN.md",
        "quarantine/output.json",
        "manuscript/draft.md",
        "benchmark/scorecard.json",
        ".git",
    ),
)
def test_native_proposer_workspace_rejects_preexisting_content(
    tmp_path: Path, relative: str
):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    os.chmod(workspace, 0o700)
    selected = workspace / relative
    if Path(relative).suffix:
        selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        selected.write_text("forbidden\n", encoding="utf-8")
    else:
        selected.mkdir(mode=0o700)
    with pytest.raises(
        Guardian.GuardianError, match="not initially empty"
    ):
        Guardian._initialize_native_workspace(workspace)


def test_native_proposer_workspace_rejects_symlink_hardlink_and_alternates(
    tmp_path: Path,
):
    _assert_native_workspace_tamper_rejected(tmp_path / "tamper")
