"""Offline unit tests for the enforced leg-library orchestration (gkm_legs.py).

No LLM, no game: the proposer and verifier are mocked so the control loop and the
MARGINAL free-energy accounting can be checked without credits. The point being
tested is the load-bearing property: reusing a leg is free, so later levels that add
no new legs have lower marginal novelty than early rule-learning levels.
"""
import os
import importlib.machinery
import inspect
import json
import py_compile
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import gkm_legs as L


def _minimal_arena_source(marker="SAFE"):
    return (
        f"PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER = {marker!r}\n"
        "DEFAULT_STEP_CAP = 600\n"
        "FRAME_SIDE = 64\n"
        "PRECONCEPTIONS = ''\n"
        "API = ''\n"
        "class Arena: pass\n"
        "def _compile(source): return (lambda env: None), None\n"
        "def free_energy(*args): return 0.0\n"
        "def run_program(*args): return (0, [], None)\n"
        "def validate(*args): return True\n"
    )


def test_clean_room_boundary_allows_only_exact_raw_arena_capability(tmp_path):
    arena_root = Path(L.__file__).resolve().parent
    clean = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {os.fspath(arena_root)!r})\n"
        "import gkm_arena as arena\n"
        "with open('checkpoint.json') as source:\n"
        "    checkpoint = json.load(source)\n"
        "limit = int(os.environ.get('PROBE_LIMIT', '10'))\n"
        "levels, path, error = arena.run_program('lf52', solve)\n"
    )
    assert L.APB.scan_python_source(
        clean,
        logical_path="probe.py",
        arena_module_root=arena_root,
    ) == ()

    unsafe = {
        "absolute_open": "open('/Users/sasha/gkm/README.md').read()\n",
        "parent_open": "open('../checkpoint.json').read()\n",
        "dynamic_open": "name='checkpoint.json'\nopen(name).read()\n",
        "pathlib_parent": (
            "from pathlib import Path\n"
            "Path('checkpoint.json').resolve().read_text()\n"
        ),
        "subprocess": (
            "import subprocess\n"
            "subprocess.run(['cat', '/etc/passwd'])\n"
        ),
        "shell": "import os\nos.system('cat /etc/passwd')\n",
        "introspection": "import sys\nprint(sys.modules)\n",
        "wrong_import": "import gkm_legs\n",
        "wrong_sys_path": "import sys\nsys.path.insert(0, '/tmp')\n",
        "arena_constructor": (
            f"import sys\nsys.path.insert(0, {os.fspath(arena_root)!r})\n"
            "import gkm_arena as arena\narena.Arena('lf52')\n"
        ),
        "arena_module_pass": (
            f"import sys\nsys.path.insert(0, {os.fspath(arena_root)!r})\n"
            "import gkm_arena as arena\nconsume(arena)\n"
        ),
        "arena_function_alias": (
            f"import sys\nsys.path.insert(0, {os.fspath(arena_root)!r})\n"
            "import gkm_arena as arena\nrun = arena.run_program\n"
        ),
        "arena_import_alias": (
            f"import sys\nsys.path.insert(0, {os.fspath(arena_root)!r})\n"
            "from gkm_arena import run_program\n"
        ),
    }
    for label, source in unsafe.items():
        assert L.APB.scan_python_source(
            source,
            logical_path=f"{label}.py",
            arena_module_root=arena_root,
        ), label


def test_authenticated_private_arena_is_exact_and_compatible():
    arena_root = Path(L.__file__).resolve().parent
    arena, digest = L._load_authenticated_arena(arena_root)

    assert digest == L.APB.arena_module_sha256(arena_root)
    assert arena.__file__ == os.fspath(arena_root / "gkm_arena.py")
    assert arena.__name__ not in sys.modules
    assert arena.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER == (
        L.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER
    )
    solve, error = arena._compile("def solve(env):\n    return None\n")
    assert solve is not None
    assert error is None
    assert L._compatibility_arena_control_reason() is None


@pytest.mark.parametrize("shadow", ["package", "extension"])
def test_physical_shadow_rejects_warmed_cache_postload_alternative(
    tmp_path,
    shadow,
):
    arena_root = tmp_path / "arena_root"
    arena_root.mkdir()
    (arena_root / "gkm_arena.py").write_text(_minimal_arena_source())
    arena, _digest = L._load_authenticated_arena(arena_root)
    assert arena.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER == "SAFE"
    # Warm the ordinary resolver before introducing the shadow.  The physical
    # inventory below must remain authoritative even if importer caches are
    # stale and without an invalidate_caches() call.
    assert importlib.machinery.PathFinder.find_spec(
        "gkm_arena", [os.fspath(arena_root)]
    ) is not None
    cached_times = os.stat(arena_root)
    if shadow == "package":
        package = arena_root / "gkm_arena"
        package.mkdir()
        (package / "__init__.py").write_text("SHADOW = True\n")
    else:
        extension = next(
            suffix
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            if suffix.startswith(".")
        )
        (arena_root / f"gkm_arena{extension}").write_bytes(b"")
    os.utime(
        arena_root,
        ns=(cached_times.st_atime_ns, cached_times.st_mtime_ns),
    )

    reason = L._compatibility_arena_host_shadow_reason(arena_root)

    assert reason is not None
    assert reason.startswith("arena_host_shadow:")
    with pytest.raises(RuntimeError, match="arena_host_shadow"):
        L._load_authenticated_arena(arena_root)


def test_authenticated_private_arena_ignores_forged_preload(monkeypatch):
    arena_root = Path(L.__file__).resolve().parent
    forged = types.ModuleType("gkm_arena")
    forged.__file__ = os.fspath(arena_root / "gkm_arena.py")
    forged.__spec__ = importlib.machinery.ModuleSpec(
        "gkm_arena",
        importlib.machinery.SourceFileLoader(
            "gkm_arena", os.fspath(arena_root / "gkm_arena.py")
        ),
        origin=os.fspath(arena_root / "gkm_arena.py"),
    )
    forged.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER = "POISON"
    monkeypatch.setitem(sys.modules, "gkm_arena", forged)

    arena, digest = L._load_authenticated_arena(arena_root)

    assert arena is not forged
    assert arena.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER != "POISON"
    assert digest == L.APB.arena_module_sha256(arena_root)


def test_import_time_authenticated_arena_ignores_forged_sys_modules():
    lab = Path(L.__file__).resolve().parent
    code = """
import sys, types
fake = types.ModuleType('gkm_arena')
fake.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER = 'POISON'
sys.modules['gkm_arena'] = fake
import gkm_legs as G
assert G.A is not fake
assert G.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER != 'POISON'
assert G.A.__name__ not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=lab,
        env={**os.environ, "PYTHONPATH": os.fspath(lab)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_authenticated_loader_ignores_and_preserves_pyc(tmp_path):
    arena_root = tmp_path / "arena_root"
    arena_root.mkdir()
    source = arena_root / "gkm_arena.py"
    source.write_text(_minimal_arena_source())
    legitimate = Path(py_compile.compile(os.fspath(source), doraise=True))
    poisoned = arena_root / "gkm_arena.pyc"
    poisoned.write_bytes(b"deliberately invalid ambient bytecode")
    before_legitimate = legitimate.read_bytes()
    before_poisoned = poisoned.read_bytes()

    arena, _digest = L._load_authenticated_arena(arena_root)

    assert arena.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER == "SAFE"
    assert legitimate.read_bytes() == before_legitimate
    assert poisoned.read_bytes() == before_poisoned
    assert L._compatibility_arena_host_shadow_reason(arena_root) is None


def test_workspace_boundary_fails_closed_on_arena_control_mismatch(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "gkm_legs_ws_lf52_origin"
    workspace.mkdir()
    monkeypatch.setattr(
        L,
        "_compatibility_arena_control_reason",
        lambda: "arena_module_control_drift: substituted",
    )

    assert L._workspace_boundary_reason(str(workspace)) == (
        "arena_module_control_drift: substituted"
    )


def test_boundary_checked_payload_rechecks_arena_before_admission(
    tmp_path, monkeypatch
):
    source = tmp_path / "solve.py"
    source.write_text("def solve(env):\n    return None\n")
    monkeypatch.setattr(
        L,
        "_compatibility_arena_control_reason",
        lambda: "arena_module_control_drift: admission",
    )

    with pytest.raises(L.WorkspaceTainted, match="admission"):
        L._boundary_checked_payload(os.fspath(source), "solve.py")


def test_gkm_solve_agent_defers_arena_and_uses_authenticated_tester():
    import gkm_solve_agent as solve_agent

    assert "A" not in solve_agent.__dict__
    assert "import gkm_arena" not in solve_agent.TESTER
    assert "A = G.A" in solve_agent.TESTER
    assert solve_agent._arena() is L.A
    assert "import gkm_arena" not in L.TESTER
    assert "A = G.A" in L.TESTER


def test_raw_arena_capability_rejects_host_sibling_modules_and_packages(
    tmp_path,
):
    arena_root = tmp_path / "arena_root"
    arena_root.mkdir()
    (arena_root / "gkm_arena.py").write_text("def run_program(*a): pass\n")
    (arena_root / "host_helper.py").write_text("SECRET = 1\n")
    (arena_root / "host_package").mkdir()
    prefix = (
        "import sys\n"
        f"sys.path.insert(0, {os.fspath(arena_root)!r})\n"
    )
    for module in ("host_helper", "host_package"):
        findings = L.APB.scan_python_source(
            prefix + f"import {module}\n",
            logical_path="probe.py",
            arena_module_root=arena_root,
        )
        assert any(item.code == "arena_sibling_import" for item in findings)


@pytest.mark.parametrize("shape", ["before", "conditional", "function"])
def test_raw_arena_requires_unconditional_insert_before_single_import(
    shape,
):
    arena_root = Path(L.__file__).resolve().parent
    insertion = f"sys.path.insert(0, {os.fspath(arena_root)!r})"
    sources = {
        "before": (
            "import sys\nimport gkm_arena as arena\n"
            f"{insertion}\n"
        ),
        "conditional": (
            "import sys\nif False:\n"
            f"    {insertion}\n"
            "import gkm_arena as arena\n"
        ),
        "function": (
            "import sys\ndef prepare():\n"
            f"    {insertion}\n"
            "import gkm_arena as arena\n"
        ),
    }

    findings = L.APB.scan_python_source(
        sources[shape],
        logical_path="probe.py",
        arena_module_root=arena_root,
    )

    assert any(item.code == "raw_arena_import_order" for item in findings)


def test_raw_arena_rejects_workspace_archive_loader_shadow():
    arena_root = Path(L.__file__).resolve().parent
    source = (
        "import sys, zipimport\n"
        "zipimport.zipimporter('payload.zip').load_module('gkm_arena')\n"
        f"sys.path.insert(0, {os.fspath(arena_root)!r})\n"
        "import gkm_arena as arena\n"
        "arena.run_program('lf52', solve)\n"
    )

    findings = L.APB.scan_python_source(
        source, logical_path="probe.py", arena_module_root=arena_root
    )

    assert any(
        item.code in {"dynamic_or_process_import", "runtime_introspection"}
        for item in findings
    )


@pytest.mark.parametrize("shadow", ["file", "package", "pyc", "extension"])
def test_workspace_rejects_reserved_raw_arena_shadow(tmp_path, shadow):
    workspace = tmp_path / "gkm_legs_ws_shadow"
    workspace.mkdir()
    if shadow == "file":
        (workspace / "gkm_arena.py").write_text("print('SHADOW')\n")
    elif shadow == "package":
        package = workspace / "gkm_arena"
        package.mkdir()
        (package / "__init__.py").write_text("print('PKGSHADOW')\n")
    elif shadow == "pyc":
        (workspace / "gkm_arena.pyc").write_bytes(b"shadow")
    else:
        (workspace / "gkm_arena.cpython-313-darwin.so").write_bytes(b"shadow")

    findings = L.APB.scan_workspace(
        workspace, arena_module_root=Path(L.__file__).resolve().parent
    )

    assert any(item.code == "reserved_arena_shadow" for item in findings)
    monitor = L.APB.LiveBoundaryMonitor(
        workspace, arena_module_root=Path(L.__file__).resolve().parent
    )
    assert any(
        item.code == "reserved_arena_shadow"
        for item in monitor.scan_workspace()
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ncopy = os.environ.copy()\n",
        "import io\nio.FileIO(chr(47) + 'etc/passwd').read()\n",
        "import urllib.request as net\nnet.urlopen('x')\n",
        "import os\nos.fork()\n",
        "def broken(:\n    pass\n",
        "import sys\nsys._getframe().f_globals\n",
        "import sys\nprint(sys.argv[0])\n",
        "getattr(object, '__sub' + 'classes__')()\n",
        "import _io\n_io.FileIO(chr(47) + 'etc/passwd')\n",
        "import _socket\n_socket.socket()\n",
        "import os\nos.symlink(chr(47) + 'etc/passwd', 'x')\nopen('x').read()\n",
        "from pathlib import Path\nPath('x').symlink_to(chr(47) + 'etc/passwd')\n",
        "import os\nos.link(chr(47) + 'etc/passwd', 'x')\n",
        "import shutil\nshutil.copy(chr(47) + 'etc/passwd', 'x')\n",
        "import numpy as np\nnp.genfromtxt(chr(47) + 'etc/passwd')\n",
        "import posix\nposix.listdir(chr(47))\n",
        "import platform\nplatform.os.listdir(chr(47))\n",
        "import os\nos.fdopen(3).read()\n",
        "f = object.__getattribute__\n",
    ],
)
def test_boundary_rejects_obfuscated_or_detached_python_capabilities(source):
    assert L.APB.scan_python_source(
        source,
        logical_path="probe.py",
        arena_module_root=Path(L.__file__).resolve().parent,
    )


def test_boundary_scans_extensionless_executables_and_cache_sources(tmp_path):
    workspace = tmp_path / "gkm_legs_ws_executables"
    workspace.mkdir()
    executable = workspace / "probe"
    executable.write_text(
        "#!/usr/bin/env python3\nopen('/etc/passwd').read()\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    cache = workspace / "__pycache__"
    cache.mkdir()
    (cache / "escape.py").write_text(
        "open('/etc/passwd').read()\n", encoding="utf-8"
    )
    findings = L.APB.scan_workspace(
        workspace, arena_module_root=Path(L.__file__).resolve().parent
    )
    paths = {item.path for item in findings}
    assert "probe" in paths
    assert "__pycache__/escape.py" in paths


def test_clean_room_boundary_rejects_symlink_and_immutable_command_escape(
    tmp_path,
):
    arena_root = Path(L.__file__).resolve().parent
    workspace = tmp_path / "gkm_legs_ws_test"
    workspace.mkdir()
    (workspace / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (workspace / "probe.py").symlink_to(workspace / "checkpoint.json")
    findings = L.APB.scan_workspace(
        workspace, arena_module_root=arena_root
    )
    assert any(item.code == "symlink_escape" for item in findings)

    transcript = tmp_path / "codex_turn_test.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "/bin/zsh -lc 'cat ../secret'",
            },
        }) + "\n",
        encoding="utf-8",
    )
    findings = L.APB.scan_codex_transcript(
        transcript,
        workspace_root=workspace,
        arena_module_root=arena_root,
    )
    assert any(item.code == "parent_path" for item in findings)


def test_historical_transcript_binds_exact_workspace_root_not_basename(
    tmp_path,
):
    workspace = tmp_path / "current" / "gkm_legs_ws_lf52_exact"
    workspace.mkdir(parents=True)
    accepted = tmp_path / "sealed" / "gkm_legs_ws_lf52_exact"
    accepted.mkdir(parents=True)
    transcript = tmp_path / "turn.jsonl"

    def write_change(path):
        transcript.write_text(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "change-1",
                "type": "file_change",
                "changes": [{"path": str(path)}],
            },
        }) + "\n", encoding="utf-8")

    write_change(accepted / "probe.py")
    assert not any(
        item.code == "file_change_escape"
        for item in L.APB.scan_codex_transcript(
            transcript,
            workspace_root=workspace,
            arena_module_root=Path(L.__file__).resolve().parent,
            accepted_workspace_root=str(accepted),
        )
    )

    decoy = tmp_path / "unsealed" / accepted.name / "probe.py"
    write_change(decoy)
    assert any(
        item.code == "file_change_escape"
        for item in L.APB.scan_codex_transcript(
            transcript,
            workspace_root=workspace,
            arena_module_root=Path(L.__file__).resolve().parent,
            accepted_workspace_root=str(accepted),
        )
    )


@pytest.mark.parametrize("invalid_binding", [7, [], {}])
def test_historical_transcript_rejects_nonstring_workspace_binding(
    tmp_path, invalid_binding
):
    workspace = tmp_path / "gkm_legs_ws_lf52_current"
    workspace.mkdir()
    transcript = tmp_path / "turn.jsonl"
    transcript.write_text(json.dumps({
        "type": "item.completed",
        "item": {"id": "message-1", "type": "agent_message", "text": "ok"},
    }) + "\n", encoding="utf-8")

    findings = L.APB.scan_codex_transcript(
        transcript,
        workspace_root=workspace,
        arena_module_root=Path(L.__file__).resolve().parent,
        accepted_workspace_root=invalid_binding,
    )

    assert any(item.code == "invalid_workspace_binding" for item in findings)


def test_workspace_boundary_precedes_marker_reads_and_transcript_aliases_fail(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "gkm_legs_ws_boundary_order"
    workspace.mkdir()
    target = tmp_path / "outside.py"
    target.write_text("open('/etc/passwd').read()\n", encoding="utf-8")
    (workspace / "probe.py").symlink_to(target)

    def marker_must_not_run(_ws):
        raise AssertionError("marker scan followed a path before lstat gate")

    monkeypatch.setattr(L, "_workspace_marker_taint_reason", marker_must_not_run)
    reason = L._workspace_taint_reason(str(workspace))
    assert reason is not None
    assert "symlink_escape" in reason

    transcript_target = tmp_path / "events.jsonl"
    transcript_target.write_text("{}\n", encoding="utf-8")
    transcript_link = tmp_path / "linked.jsonl"
    transcript_link.symlink_to(transcript_target)
    findings = L.APB.scan_codex_transcript(
        transcript_link,
        workspace_root=workspace,
        arena_module_root=Path(L.__file__).resolve().parent,
    )
    assert any(item.code == "symlink_transcript" for item in findings)

    transcript_hardlink = tmp_path / "hardlinked.jsonl"
    os.link(transcript_target, transcript_hardlink)
    findings = L.APB.scan_codex_transcript(
        transcript_hardlink,
        workspace_root=workspace,
        arena_module_root=Path(L.__file__).resolve().parent,
    )
    assert any(item.code == "aliased_transcript" for item in findings)


def test_live_boundary_monitor_rechecks_changed_source(tmp_path):
    arena_root = Path(L.__file__).resolve().parent
    workspace = tmp_path / "gkm_legs_ws_monitor"
    workspace.mkdir()
    probe = workspace / "probe.py"
    probe.write_text("open('checkpoint.json').read()\n", encoding="utf-8")
    monitor = L.APB.LiveBoundaryMonitor(
        workspace, arena_module_root=arena_root
    )
    assert monitor.scan_workspace() == ()
    probe.write_text("open('/etc/passwd').read()\n", encoding="utf-8")
    assert any(
        item.code == "absolute_path"
        for item in monitor.scan_workspace()
    )


def test_live_boundary_monitor_hashes_source_even_if_mtime_is_restored(tmp_path):
    arena_root = Path(L.__file__).resolve().parent
    workspace = tmp_path / "gkm_legs_ws_digest"
    workspace.mkdir()
    probe = workspace / "probe.py"
    unsafe = "open('/etc/passwd').read()\n"
    safe = "x = 1\n" + " " * (len(unsafe) - len("x = 1\n"))
    assert len(safe.encode()) == len(unsafe.encode())
    probe.write_text(safe, encoding="utf-8")
    before = probe.stat()
    monitor = L.APB.LiveBoundaryMonitor(
        workspace, arena_module_root=arena_root
    )
    assert monitor.scan_workspace() == ()
    probe.write_text(unsafe, encoding="utf-8")
    os.utime(probe, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert any(
        item.code == "absolute_path"
        for item in monitor.scan_workspace()
    )


def test_boundary_policy_source_drift_invalidates_loaded_monitor(
    tmp_path, monkeypatch
):
    policy = tmp_path / "policy.py"
    policy.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(L.APB, "_POLICY_SOURCE_PATH", policy)
    monkeypatch.setattr(
        L.APB,
        "_LOADED_POLICY_SHA256",
        __import__("hashlib").sha256(policy.read_bytes()).hexdigest(),
    )
    workspace = tmp_path / "gkm_legs_ws_policy_drift"
    workspace.mkdir()
    (workspace / "probe.py").write_text("x = 1\n", encoding="utf-8")
    monitor = L.APB.LiveBoundaryMonitor(workspace)
    assert monitor.scan_workspace() == ()
    policy.write_text("version = 2\n", encoding="utf-8")
    findings = monitor.scan_workspace()
    assert [finding.code for finding in findings] == ["policy_control_drift"]


def test_nonobject_wip_metadata_is_forensic_only_not_a_restore_crash(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]\n", encoding="utf-8")

    assert L._wip_uses_current_boundary_policy(metadata) is False


def test_shell_boundary_unwraps_and_scans_inline_interpreters():
    arena_root = Path(L.__file__).resolve().parent
    clean = (
        "/bin/zsh -lc \"python - <<'PY'\n"
        "import json\n"
        "for name in ('checkpoint.json',):\n"
        "    print(json.load(open(name)))\n"
        "PY\""
    )
    assert L.APB.scan_shell_command(
        clean,
        logical_path="turn.jsonl",
        line=1,
        arena_module_root=arena_root,
    ) == ()
    assert L.APB.scan_shell_command(
        "/bin/zsh -lc \"git diff -- legs.py players.py | sed -n '1,20p'\"",
        logical_path="turn.jsonl",
        line=1,
        arena_module_root=arena_root,
    ) == ()

    unsafe = {
        "computed_absolute": (
            "/bin/zsh -lc \"python -c \\\"import os; "
            "open(chr(47) + 'etc/passwd').read()\\\"\""
        ),
        "sys_path_read": (
            "/bin/zsh -lc \"python - <<'PY'\n"
            "import sys\nprint(sys.path[0])\nPY\""
        ),
        "pathlib_escape": (
            "/bin/zsh -lc \"python -c \\\"from pathlib import Path; "
            "print(Path.cwd().parent)\\\"\""
        ),
        "command_substitution": "/bin/zsh -lc 'value=$(pwd); echo $value'",
        "process_substitution": "/bin/zsh -lc 'cat <(cat checkpoint.json)'",
        "dynamic_redirection": "/bin/zsh -lc 'cat < \"$TARGET\"'",
        "root_cd": "/bin/zsh -lc 'cd /; python probe.py'",
        "source": "/bin/zsh -lc 'source probe.txt'",
        "detached": "/bin/zsh -lc 'python probe.py &'",
        "python_module": "/bin/zsh -lc 'python -m probe'",
        "extensionless": "/bin/zsh -lc 'python probe'",
        "hidden_cache": "/bin/zsh -lc 'python __pycache__/probe.py'",
        "perl": "/bin/zsh -lc 'perl -e print'",
        "ruby": "/bin/zsh -lc 'ruby -e puts'",
        "node": "/bin/zsh -lc 'node -e console.log(1)'",
        "awk_system": "/bin/zsh -lc \"awk 'BEGIN {system(\\\"id\\\")}'\"",
        "stdin_source": "/bin/zsh -lc 'python3 - < probe.txt'",
        "unsafe_substitution": "/bin/zsh -lc 'x=$(rg passwd / | cut -d: -f1)'",
        "direct_executable": "/bin/zsh -lc './extensionless'",
        "newline_source": "/bin/zsh -lc 'echo ok\nsource probe.txt'",
        "newline_cd": "/bin/zsh -lc 'echo ok\ncd /'",
        "newline_setsid": "/bin/zsh -lc 'printf ok\nsetsid python probe.py'",
        "source_write_execute_restore": (
            "/bin/zsh -lc \"printf 'import _io\\n' > probe.py; "
            "python3 probe.py; printf 'x=1\\n' > probe.py\""
        ),
    }
    for label, command in unsafe.items():
        assert L.APB.scan_shell_command(
            command,
            logical_path=f"{label}.jsonl",
            line=1,
            arena_module_root=arena_root,
        ), label


def test_transcript_unknown_or_malformed_lifecycle_cannot_hide_action(tmp_path):
    workspace = tmp_path / "gkm_legs_ws_schema"
    workspace.mkdir()
    transcript = tmp_path / "turn.jsonl"
    transcript.write_text(
        "\n".join([
            json.dumps({"type": "future.command", "command": "cat /etc/passwd"}),
            json.dumps({
                "type": "item.updated",
                "item": {
                    "id": "x", "type": "command_execution",
                    "command": "python probe.py",
                },
            }),
            json.dumps({
                "type": "item.completed",
                "item": {"id": "y", "type": "future_tool"},
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    codes = {
        finding.code
        for finding in L.APB.scan_codex_transcript(
            transcript,
            workspace_root=workspace,
            arena_module_root=Path(L.__file__).resolve().parent,
        )
    }
    assert {
        "unknown_transcript_event",
        "malformed_action_lifecycle",
        "unknown_item_type",
    } <= codes


def test_exact_host_scaffold_survives_wip_relocation_but_mutation_does_not(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "gkm_legs_ws_wa30_scaffold"
    workspace.mkdir()
    scaffold = L.TESTER.format(
        labdir=os.path.dirname(os.path.abspath(L.__file__)), game="wa30"
    )
    (workspace / "gkm_try.py").write_text(scaffold, encoding="utf-8")
    (workspace / "probe.py").write_text("print('safe')\n", encoding="utf-8")
    assert L._workspace_taint_reason(str(workspace)) is None
    snapshot = Path(L.snapshot_wip_context(
        "wa30", str(workspace), 1, "interrupted", reached=0, verbose=False
    ))
    destination = tmp_path / "destination"
    destination.mkdir()
    assert L._restore_wip_probes(
        "wa30", str(destination), 1, verbose=False
    ) == 1
    assert (destination / "probe.py").is_file()

    (snapshot / "files" / "gkm_try.py").write_text(
        scaffold + "\nimport gkm_legs\n", encoding="utf-8"
    )
    destination2 = tmp_path / "destination2"
    destination2.mkdir()
    assert L._restore_wip_probes(
        "wa30", str(destination2), 1, verbose=False
    ) == 0
    assert not (destination2 / "probe.py").exists()


def test_trusted_scaffold_literal_filter_is_exact_and_sealed(tmp_path):
    workspace = tmp_path / "gkm_legs_ws_wa30_scaffold_filter"
    workspace.mkdir()
    path = workspace / "gkm_try.py"
    exact = L.TESTER.format(
        labdir=os.path.dirname(os.path.abspath(L.__file__)), game="wa30"
    ).encode("utf-8")

    def filtered_codes(payload):
        trusted = L._trusted_host_scaffold_hashes(str(workspace))
        digest = L.hashlib.sha256(payload).hexdigest()
        findings = L.APB.scan_python_source(
            payload.decode("utf-8"),
            logical_path="gkm_try.py",
            arena_module_root=Path(L.__file__).resolve().parent,
            allow_host_scaffold=L.APB._trusted_digest(
                trusted, "gkm_try.py", digest
            ),
        )
        return {
            item.code
            for item in L._filter_trusted_scaffold_root_literal(
                workspace,
                findings,
                trusted=trusted,
                sealed_payloads={"gkm_try.py": payload},
            )
        }

    path.write_bytes(exact)
    assert filtered_codes(exact) == set()
    assert L._workspace_boundary_reason(str(workspace)) is None
    assert L._boundary_checked_payload(str(path), "gkm_try.py") == exact

    mutated = exact + b"\n"
    path.write_bytes(mutated)
    assert {
        "absolute_path",
        "dynamic_or_process_import",
        "private_harness_import",
    } <= filtered_codes(mutated)
    assert L._workspace_boundary_reason(str(workspace)) is not None
    with pytest.raises(L.WorkspaceTainted):
        L._boundary_checked_payload(str(path), "gkm_try.py")

    proposer_authored = b"EXTERNAL = '/etc/passwd'\n"
    path.write_bytes(proposer_authored)
    assert "absolute_path" in filtered_codes(proposer_authored)
    assert L._workspace_boundary_reason(str(workspace)).startswith(
        "absolute_path in gkm_try.py"
    )
    with pytest.raises(L.WorkspaceTainted, match="absolute_path"):
        L._boundary_checked_payload(str(path), "gkm_try.py")


@pytest.mark.parametrize(
    "template_name",
    ("TESTER", "_HOST_TESTER_POLICY_V1"),
)
def test_each_versioned_host_scaffold_is_exactly_digest_bound(
    tmp_path, template_name
):
    workspace = tmp_path / "gkm_legs_ws_lf52_versioned_scaffold"
    workspace.mkdir()
    path = workspace / "gkm_try.py"
    template = getattr(L, template_name)
    exact = template.format(
        labdir=os.path.dirname(os.path.abspath(L.__file__)), game="lf52"
    ).encode("utf-8")
    trusted = L._trusted_host_scaffold_hashes(str(workspace))

    assert L.hashlib.sha256(exact).hexdigest() in trusted["gkm_try.py"]
    path.write_bytes(exact)
    assert L._workspace_taint_reason(str(workspace)) is None

    # Even a one-byte, behavior-preserving proposer edit is a different
    # authority, not a compatible host scaffold.
    assert exact.endswith(b"\n")
    mutated = exact[:-1] + b" "
    assert mutated != exact
    assert len(mutated) == len(exact)
    assert sum(left != right for left, right in zip(exact, mutated)) == 1
    assert L.hashlib.sha256(mutated).hexdigest() not in trusted["gkm_try.py"]
    path.write_bytes(mutated)
    reason = L._workspace_taint_reason(str(workspace))
    assert reason is not None
    assert (
        "absolute_path" in reason
        or "dynamic_or_process_import" in reason
        or "private_harness_import" in reason
    )


def test_proposer_authored_scaffold_lookalike_has_no_host_authority(tmp_path):
    workspace = tmp_path / "gkm_legs_ws_lf52_scaffold_lookalike"
    workspace.mkdir()
    path = workspace / "gkm_try.py"
    lookalike = (
        "import importlib.util, os, sys\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(L.__file__))!r})\n"
        "import gkm_legs as G\n"
        "A = G.A\n"
        "spec = importlib.util.spec_from_file_location('solve', 'solve.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "levels, path, error = A.run_program('lf52', module.solve)\n"
    ).encode("utf-8")
    trusted = L._trusted_host_scaffold_hashes(str(workspace))

    assert L.hashlib.sha256(lookalike).hexdigest() not in trusted["gkm_try.py"]
    path.write_bytes(lookalike)
    reason = L._workspace_taint_reason(str(workspace))
    assert reason is not None
    assert (
        "absolute_path" in reason
        or "dynamic_or_process_import" in reason
        or "private_harness_import" in reason
    )


def test_lf52_live_policy_v1_scaffold_identity_remains_compatible(tmp_path):
    workspace = tmp_path / (
        "gkm_legs_ws_lf52_arc_agi3_n9_long_coherence_reset_pt7qyytw"
    )
    workspace.mkdir()
    assert L.hashlib.sha256(
        L._HOST_TESTER_POLICY_V1.encode("utf-8")
    ).hexdigest() == (
        "b79a64eaf5d13d1b4da4e87550c4d1292775d81801af1163105f63444e2701d2"
    )
    active_launch_bytes = L._HOST_TESTER_POLICY_V1.format(
        labdir="/Users/sasha/gkm/arc/crack_lab", game="lf52"
    ).encode("utf-8")
    assert L.hashlib.sha256(active_launch_bytes).hexdigest() == (
        "b753d2dcf5b44640e18e2f6a669f854222af9161b38da452f244c3a9d1d70aba"
    )
    exact = L._HOST_TESTER_POLICY_V1.format(
        labdir=os.path.dirname(os.path.abspath(L.__file__)), game="lf52"
    ).encode("utf-8")
    assert L.hashlib.sha256(exact).hexdigest() in (
        L._trusted_host_scaffold_hashes(str(workspace))["gkm_try.py"]
    )
    (workspace / "gkm_try.py").write_bytes(exact)

    assert L._workspace_taint_reason(str(workspace)) is None


def test_orchestrate_rejects_preexisting_boundary_fault_before_inspection(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "gkm_legs_ws_startup"
    workspace.mkdir()
    (workspace / "probe.py").write_text(
        "open('../secret').read()\n", encoding="utf-8"
    )
    monkeypatch.setattr(L, "setup_workspace", lambda *args, **kwargs: str(workspace))
    monkeypatch.setattr(
        L,
        "_load_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint inspected before startup boundary gate")
        ),
    )
    with pytest.raises(L.WorkspaceTainted):
        L.orchestrate(
            "startupboundary",
            max_level=1,
            propose_fn=lambda *_: None,
            verify_fn=lambda *_: (0, [], None),
            debrief_fn=lambda *_: None,
            seed_artifact=False,
            verbose=False,
        )


def test_phase_minus_one_verify_is_gated_after_last_moment_mutation(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "gkm_legs_ws_jitgate"
    workspace.mkdir()
    for name, body in {
        "legs.py": "# clean\n",
        "players.py": "# clean\n",
        "solve.py": "def solve(env):\n    pass\n",
    }.items():
        (workspace / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(L, "setup_workspace", lambda *args, **kwargs: str(workspace))
    calls = 0

    def verify(_game, _path):
        nonlocal calls
        calls += 1
        return 0, [], None

    def mutate_then_claim_source(*_args, **_kwargs):
        (workspace / "probe.py").write_text(
            "open('../secret').read()\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(
        L, "_workspace_has_unpromoted_solver_source", mutate_then_claim_source
    )
    with pytest.raises(L.WorkspaceTainted):
        L.orchestrate(
            "jitgate",
            max_level=1,
            propose_fn=lambda *_: None,
            verify_fn=verify,
            debrief_fn=lambda *_: None,
            seed_artifact=False,
            verbose=False,
        )
    # Only the clean startup replay ran.  The phase-minus-one source never did.
    assert calls == 1


def test_contiguous_workspace_write_path_is_boundary_checked():
    assert L.APB.dynamic_tool_boundary_hits(
        "workspace_write",
        {"path": "../escape.py", "text": "print('x')\n"},
    ) == ("workspace_write_path_escape",)


def test_boundary_lifecycle_rejects_turn_end_wip_and_orphan_recovery(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )

    wip_workspace = tmp_path / "wip_workspace"
    wip_workspace.mkdir()
    (wip_workspace / "probe.py").write_text(
        "open('../parent-secret').read()\n", encoding="utf-8"
    )
    with pytest.raises(L.WorkspaceTainted):
        L.snapshot_wip_context(
            "boundarywip", str(wip_workspace), 1, "interrupted",
            reached=0, verbose=False,
        )
    assert not (artifact_root / "boundarywip_legs" / "wip_context").exists()

    monkeypatch.setattr(
        L,
        "_candidate_paths_from_log",
        lambda _ws: (_ for _ in ()).throw(
            AssertionError("tainted orphan workspace was inspected")
        ),
    )
    with pytest.raises(L.WorkspaceTainted):
        L.recover_discovered_path_artifact(
            "boundaryorphan", str(wip_workspace), 1, [], verbose=False
        )
    monkeypatch.setattr(L, "_candidate_paths_from_log", lambda _ws: [])

    scratch = tmp_path / "turn_workspace"
    scratch.mkdir()
    monkeypatch.setattr(L, "setup_workspace", lambda game, tag="": str(scratch))

    def proposer(workspace, _level):
        Path(workspace, "probe.py").write_text(
            "open('../parent-secret').read()\n", encoding="utf-8"
        )

    with pytest.raises(L.WorkspaceTainted):
        L.orchestrate(
            "boundaryturn",
            max_level=1,
            propose_fn=proposer,
            verify_fn=lambda game, path: (0, [], None),
            debrief_fn=lambda workspace, level: None,
            seed_artifact=False,
            verbose=False,
        )
    assert not (artifact_root / "boundaryturn_legs").exists()


def test_wip_restore_reopens_boundary_and_skips_mutated_snapshot(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "probe.py").write_text("print('safe')\n", encoding="utf-8")
    snapshot = Path(L.snapshot_wip_context(
        "boundaryrestore", str(workspace), 1, "interrupted",
        reached=0, verbose=False,
    ))
    (snapshot / "files" / "probe.py").write_text(
        "open('/etc/passwd').read()\n", encoding="utf-8"
    )
    destination = tmp_path / "destination"
    destination.mkdir()
    assert L._restore_wip_probes(
        "boundaryrestore", str(destination), 1, verbose=False
    ) == 0
    assert not (destination / "probe.py").exists()


def test_unbound_legacy_wip_is_forensic_only_and_not_restored(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "probe.py").write_text("print('safe')\n", encoding="utf-8")
    snapshot = Path(L.snapshot_wip_context(
        "legacywip", str(workspace), 1, "interrupted",
        reached=0, verbose=False,
    ))
    metadata_path = snapshot / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("filesystem_boundary_policy_schema")
    metadata.pop("filesystem_boundary_policy_sha256")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    assert L._restore_wip_probes(
        "legacywip", str(destination), 1, verbose=False
    ) == 0
    assert not (destination / "probe.py").exists()


def test_promotion_rechecks_exact_bytes_after_prior_scan(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "promotion"
    workspace.mkdir()
    for name, source in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "def play_level_1(env):\n    pass\n",
        "solve.py": "def solve(env):\n    pass\n",
        "legs_log.md": "clean\n",
    }.items():
        (workspace / name).write_text(source, encoding="utf-8")
    report = L.Report(
        game="boundarypromotion",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    monkeypatch.setattr(L, "exact_level_boundary", lambda *args: [1])
    monkeypatch.setattr(L.A, "validate", lambda *args: True)
    original_assert = L.assert_workspace_not_tainted
    calls = 0

    def mutate_after_second_scan(selected):
        nonlocal calls
        original_assert(selected)
        calls += 1
        if calls == 2:
            (workspace / "legs.py").write_text(
                "open('/etc/passwd').read()\n", encoding="utf-8"
            )

    monkeypatch.setattr(L, "assert_workspace_not_tainted", mutate_after_second_scan)
    with pytest.raises(L.WorkspaceTainted):
        L.promote_verified_artifact(
            "boundarypromotion", str(workspace), report, verbose=False
        )
    assert not (artifact_root / "boundarypromotion_legs").exists()


def test_seed_rejects_boundary_violating_canonical_parent_before_copy(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    artifact = artifact_root / "boundaryseed_legs"
    artifact.mkdir(parents=True)
    report = L.Report(game="boundaryseed", reached=1, final_path=[1], validated=True)
    L._save_checkpoint(str(artifact), report)
    (artifact / "legs.py").write_text(
        "open('/etc/passwd').read()\n", encoding="utf-8"
    )
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(L.WorkspaceTainted):
        L.seed_workspace_from_artifact(
            "boundaryseed", str(destination), restore_wip=False, verbose=False
        )
    assert not (destination / "legs.py").exists()


def test_run_solve_file_isolates_generated_auxiliary_modules(
    tmp_path, monkeypatch
):
    workspaces = []
    for index in (1, 2):
        workspace = tmp_path / f"game_{index}"
        workspace.mkdir()
        (workspace / "perception.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8"
        )
        (workspace / "legs.py").write_text(
            "from perception import VALUE\n", encoding="utf-8"
        )
        (workspace / "players.py").write_text(
            "from legs import VALUE\n", encoding="utf-8"
        )
        (workspace / "solve.py").write_text(
            "import players\n\n"
            "def solve(env):\n"
            "    env.value = players.VALUE\n",
            encoding="utf-8",
        )
        workspaces.append(workspace)

    def fake_run_program(_game, solve, *, time_cap):
        assert time_cap == 7
        env = types.SimpleNamespace(value=None)
        solve(env)
        return env.value, [], None

    monkeypatch.setattr(L.A, "run_program", fake_run_program)
    sentinel = types.ModuleType("perception")
    sentinel.VALUE = 99
    monkeypatch.setitem(sys.modules, "perception", sentinel)

    observed = [
        L.run_solve_file(
            "test",
            str(workspace / "solve.py"),
            time_cap=7,
            resume_checkpoint=False,
        )[0]
        for workspace in (*workspaces, workspaces[0])
    ]

    assert observed == [1, 2, 1]
    assert sys.modules["perception"] is sentinel


def test_cli_help_and_unknown_options_never_dispatch_default_game(tmp_path):
    script = Path(L.__file__).resolve()
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))

    help_run = subprocess.run(
        [sys.executable, os.fspath(script), "--help"],
        cwd=script.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert help_run.returncode == 0
    assert help_run.stdout.startswith("usage: gkm_legs.py")
    assert "seeded workspace" not in help_run.stdout

    bad_run = subprocess.run(
        [sys.executable, os.fspath(script), "--definitely-unknown"],
        cwd=script.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert bad_run.returncode != 0
    assert "unknown argument: --definitely-unknown" in bad_run.stderr
    assert "seeded workspace" not in bad_run.stdout

    missing_run = subprocess.run(
        [sys.executable, os.fspath(script)],
        cwd=script.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert missing_run.returncode != 0
    assert "missing required arguments" in missing_run.stderr
    assert "seeded workspace" not in missing_run.stdout

    duplicate_run = subprocess.run(
        [
            sys.executable,
            os.fspath(script),
            "--game=wa30",
            "--game=bp35",
        ],
        cwd=script.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert duplicate_run.returncode != 0
    assert "duplicate argument: --game" in duplicate_run.stderr
    assert "seeded workspace" not in duplicate_run.stdout


def test_loc_ignores_blanks_and_comments():
    assert L._loc("def f():\n    return 1\n") == 2
    assert L._loc("\n# comment\n   \n") == 0


def test_authoritative_level_target_and_runner_reject_impossible_level(
    tmp_path, monkeypatch
):
    metadata = tmp_path / "re86" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"baseline_actions": list(range(8))}),
        encoding="utf-8",
    )
    assert L.authoritative_level_target("re86", tmp_path) == 8

    monkeypatch.setattr(L, "authoritative_level_target", lambda _game: 8)
    monkeypatch.setattr(
        L,
        "setup_workspace",
        lambda *_args, **_kwargs: pytest.fail(
            "workspace creation must not occur for an impossible target"
        ),
    )
    with pytest.raises(ValueError, match="authoritative_target=8"):
        L.orchestrate("re86", max_level=9, propose_fn=None, verbose=False)


def test_marginal_complexity_reuse_is_free():
    legs = "def a(env):\n    pass\n"
    # reused leg (legs unchanged) + a 1-line player -> marginal C counts only the player
    assert L.marginal_complexity(legs, legs, "", "play(env)\n") == 1
    # adding a new leg is paid for
    legs2 = legs + "def b(env):\n    pass\n"
    assert L.marginal_complexity(legs, legs2, "", "") == 2


def test_auto_solve_caps_general_leg_candidates_and_restores_players(tmp_path):
    legs = "\n".join(
        f"def solve_variant_{index}(env):\n    return None\n"
        for index in range(L.AUTO_SOLVE_MAX_CANDIDATES + 4)
    )
    players = "def play_level_1(env):\n    return None\n"
    players_path = tmp_path / "players.py"
    solve_path = tmp_path / "solve.py"
    players_path.write_text(players)
    solve_path.write_text("def solve(env):\n    return None\n")
    calls = []

    result = L._try_auto_solve(
        2, legs, players, str(players_path), str(solve_path), "fake",
        lambda game, path: calls.append((game, path)) or (1, [], None),
    )

    assert result is None
    assert len(calls) == L.AUTO_SOLVE_MAX_CANDIDATES
    assert players_path.read_text() == players


def test_marginal_complexity_nets_replacement_within_each_file():
    before = "old_call(env)\n"
    after = "new_call(env)\n"
    assert L.description_complexity(before) == L.description_complexity(after)
    assert L.marginal_complexity(before, after, "", "") == 0


def test_free_energy_rewards_levels_and_penalises_novelty():
    assert L.free_energy(3, 0) == -3.0
    assert L.free_energy(3, 100, lam=0.02) == -3.0 + 2.0
    # more levels for the same novelty is always lower F
    assert L.free_energy(4, 50) < L.free_energy(3, 50)


def test_level_record_upsert_and_legacy_checkpoint_deduplication(tmp_path):
    rep = L.Report(
        game="duptest",
        reached=3,
        records=[
            L.LevelRecord(level=1, marginal_C=10, reached=True),
            L.LevelRecord(level=3, marginal_C=14, reached=True),
            L.LevelRecord(level=3, marginal_C=184, reached=True),
        ],
        total_marginal_C=208,
    )
    L._save_checkpoint(str(tmp_path), rep)
    assert [(r.level, r.marginal_C) for r in rep.records] == [(1, 10), (3, 184)]
    assert rep.total_marginal_C == 194

    data = json.loads((tmp_path / L.CHECKPOINT_FILE).read_text())
    assert [(r["level"], r["marginal_C"]) for r in data["records"]] == [
        (1, 10), (3, 184)
    ]

    L._record_level(rep, 3, 190)
    assert [(r.level, r.marginal_C) for r in rep.records] == [(1, 10), (3, 190)]
    assert rep.total_marginal_C == 200


def test_checkpoint_normalizes_stale_total_from_unique_records(tmp_path):
    checkpoint = {
        "game": "staletotal",
        "reached": 2,
        "records": [
            {"level": 1, "marginal_C": 40, "reached": True},
            {"level": 2, "marginal_C": 7, "reached": True},
        ],
        "total_marginal_C": 12,
        "final_path": [1, 2],
        "validated": True,
    }
    (tmp_path / L.CHECKPOINT_FILE).write_text(json.dumps(checkpoint))

    rep = L._load_checkpoint(str(tmp_path))
    assert rep.total_marginal_C == 47

    L._save_checkpoint(str(tmp_path), rep)
    saved = json.loads((tmp_path / L.CHECKPOINT_FILE).read_text())
    assert saved["total_marginal_C"] == 47


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_checkpoint_host_write_rejects_alias_without_touching_canary(
        tmp_path, alias_kind):
    checkpoint = tmp_path / L.CHECKPOINT_FILE
    checkpoint.write_text(json.dumps({
        "game": "wa30",
        "reached": 1,
        "records": [{"level": 1, "marginal_C": 7, "reached": True}],
        "total_marginal_C": 7,
        "final_path": [1],
        "validated": True,
    }))
    rep = L._load_checkpoint(str(tmp_path))
    canary = tmp_path.parent / f"{tmp_path.name}-{alias_kind}-canary"
    canary.write_text("DO NOT CHANGE")
    checkpoint.unlink()
    if alias_kind == "symlink":
        checkpoint.symlink_to(canary)
    else:
        os.link(canary, checkpoint)
    with pytest.raises(L.WorkspaceTainted, match="host-write target"):
        L._save_checkpoint(str(tmp_path), rep)
    assert canary.read_text() == "DO NOT CHANGE"


def test_path_only_proposer_checkpoint_is_not_trusted_report(tmp_path):
    checkpoint = {
        "game": "pathonly",
        "final_path": [1, 2, 3],
        "validated": True,
    }
    (tmp_path / L.CHECKPOINT_FILE).write_text(json.dumps(checkpoint))

    assert L._load_checkpoint(str(tmp_path)) is None
    assert L._candidate_paths_from_checkpoint(str(tmp_path)) == [[1, 2, 3]]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("reached"),
        lambda value: value.update(reached=True),
        lambda value: value.update(total_marginal_C="7"),
        lambda value: value.update(records=[{"level": 1}]),
        lambda value: value.update(final_path=[True]),
        lambda value: value.update(validated="yes"),
        lambda value: value.update(untrusted_extra=True),
    ],
)
def test_malformed_checkpoint_never_becomes_trusted_report(tmp_path, mutation):
    checkpoint = {
        "game": "wa30",
        "reached": 1,
        "records": [
            {"level": 1, "marginal_C": 7, "reached": True},
        ],
        "total_marginal_C": 7,
        "final_path": [1],
        "validated": True,
    }
    mutation(checkpoint)
    (tmp_path / L.CHECKPOINT_FILE).write_text(json.dumps(checkpoint))

    # The real harness must classify malformed proposer bytes as untrusted
    # candidate material instead of raising later while reading bookkeeping.
    assert L._load_checkpoint(str(tmp_path)) is None


def test_fresh_replay_candidate_schema_is_discovered_and_parsed(tmp_path):
    candidate = tmp_path / "fresh_replay_level8_candidate.json"
    candidate.write_text(
        json.dumps({"game": "sk48", "levels": 8, "final_path": [1, 2, 3]})
    )
    assert candidate.as_posix() in L._candidate_path_files(
        str(tmp_path), 8
    )
    assert L._load_action_path(json.loads(candidate.read_text())) == [1, 2, 3]


def test_isolated_workspace_generation_never_reuses_old_tagged_scratch(
        tmp_path, monkeypatch):
    monkeypatch.setattr(L, "SCRATCH", str(tmp_path))
    first = Path(L.setup_workspace(
        "wa30", "same_tag", isolated_generation=True
    ))
    (first / "old_probe.py").write_text("old hypothesis\n")
    second = Path(L.setup_workspace(
        "wa30", "same_tag", isolated_generation=True
    ))
    assert first != second
    assert not (second / "old_probe.py").exists()


def test_candidate_discovery_cannot_harvest_sibling_or_symlinked_path(tmp_path):
    ws = tmp_path / "attempt"
    ws.mkdir()
    sibling = tmp_path / "win8_sibling.json"
    sibling.write_text(json.dumps({"final_path": [1, 2, 3]}))
    link = ws / "win8_link.json"
    link.symlink_to(sibling)
    assert L._candidate_path_files(str(ws), 8) == []


def test_candidate_discovery_prioritizes_frontier_and_is_bounded(tmp_path):
    for index in range(L.MAX_RECOVERY_PATH_CANDIDATES + 20):
        (tmp_path / f"level7_old_{index:03d}_candidate.json").write_text(
            json.dumps({"final_path": [1, 2, index % 7 + 1]})
        )
    frontier = tmp_path / "level9_frontier_candidate.json"
    frontier.write_text(json.dumps({"final_path": [1, 2, 3]}))

    found = L._candidate_path_files(str(tmp_path), 9)
    assert len(found) == L.MAX_RECOVERY_PATH_CANDIDATES
    assert found[0] == str(frontier)


def test_recovery_pairwise_replay_is_hard_bounded(tmp_path, monkeypatch):
    paths = []
    for index in range(L.MAX_RECOVERY_PATH_CANDIDATES + 20):
        path = tmp_path / f"level9_piece_{index:03d}_candidate.json"
        unique_path = [
            (index // 49) % 7 + 1,
            (index // 7) % 7 + 1,
            index % 7 + 1,
        ]
        path.write_text(json.dumps({"final_path": unique_path}))
        paths.append(str(path))
    (tmp_path / "solve.py").write_text("def solve(env): pass\n")
    (tmp_path / "players.py").write_text("ORIGINAL\n")
    monkeypatch.setattr(L, "_candidate_path_files", lambda ws, level: paths)
    monkeypatch.setattr(L, "_candidate_paths_from_log", lambda ws: [])
    monkeypatch.setattr(L, "_candidate_paths_from_checkpoint", lambda ws: [])
    monkeypatch.setattr(L, "_validated_prefix_floor", lambda *args: True)
    monkeypatch.setattr(L, "_verify_candidate_suffix", lambda *args: None)
    monkeypatch.setattr(L, "_verify_candidate_path", lambda *args: None)
    replay_calls = []

    def losing_replay(game, path):
        replay_calls.append(tuple(path))
        return 8, list(path), None

    monkeypatch.setattr(L, "_run_candidate_replay", losing_replay)
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)

    assert L.recover_discovered_path_artifact(
        "lf52", str(tmp_path), 9, [1], verbose=False
    ) is None
    assert len(replay_calls) <= (
        L.MAX_RECOVERY_PATH_CANDIDATES
        + L.MAX_RECOVERY_GLUE_ATTEMPTS
    )


def test_recovery_adopts_fresh_prefix_only_when_path_and_source_replay(
        tmp_path, monkeypatch):
    candidate = tmp_path / "fresh_replay_level8_candidate.json"
    candidate.write_text(json.dumps({"final_path": [1, 2, 3]}))
    (tmp_path / "solve.py").write_text("def solve(env): pass\n")
    (tmp_path / "players.py").write_text("ORIGINAL\n")

    monkeypatch.setattr(
        L, "_run_candidate_replay",
        lambda game, path: (8, list(path), None),
    )
    calls = []

    def fresh_source(game, solve_path, **kwargs):
        calls.append(kwargs)
        return 8, [1, 2, 3], None

    monkeypatch.setattr(L, "run_solve_file", fresh_source)
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)

    recovered = L.recover_discovered_path_artifact(
        "sk48", str(tmp_path), 8, [4, 4], verbose=False
    )
    assert recovered == (8, [1, 2, 3], None)
    assert calls == [{"resume_checkpoint": False}]
    assert (tmp_path / "players.py").read_text() == "ORIGINAL\n"


def test_recovery_rejects_fresh_path_when_workspace_source_does_not_replay(
        tmp_path, monkeypatch):
    (tmp_path / "fresh_replay_level8_candidate.json").write_text(
        json.dumps({"final_path": [1, 2, 3]})
    )
    (tmp_path / "solve.py").write_text("def solve(env): pass\n")
    (tmp_path / "players.py").write_text("ORIGINAL\n")

    monkeypatch.setattr(
        L, "_run_candidate_replay",
        lambda game, path: (8, list(path), None),
    )
    monkeypatch.setattr(
        L, "run_solve_file",
        lambda game, solve_path, **kwargs: (7, [1, 2], None),
    )
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)
    monkeypatch.setattr(L, "_validated_prefix_floor", lambda *args: False)
    monkeypatch.setattr(L, "_verify_candidate_path", lambda *args: None)

    assert L.recover_discovered_path_artifact(
        "sk48", str(tmp_path), 8, [4, 4], verbose=False
    ) is None
    assert (tmp_path / "players.py").read_text() == "ORIGINAL\n"


def test_frontier_marginal_baseline_uses_promoted_parent_across_wip(
        tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifact = artifacts / "wa30_legs"
    artifact.mkdir(parents=True)
    (artifact / "legs.py").write_text("PARENT_LEG\n")
    (artifact / "players.py").write_text("PARENT_PLAYER\n")
    L._save_checkpoint(
        str(artifact),
        L.Report(
            game="wa30",
            reached=1,
            total_marginal_C=7,
            records=[
                L.LevelRecord(level=1, marginal_C=7, reached=True)
            ],
            final_path=[1],
            validated=True,
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "legs.py").write_text("PARENT_LEG\nWIP_NOVELTY\n")
    (workspace / "players.py").write_text("PARENT_PLAYER\nWIP_PLAYER\n")
    monkeypatch.setenv("GKM_ARTIFACTS_ROOT", str(artifacts))

    assert L._frontier_marginal_baseline(
        "wa30", str(workspace), reached=1
    ) == ("PARENT_LEG\n", "PARENT_PLAYER\n")
    assert L._frontier_marginal_baseline(
        "wa30", str(workspace), reached=0
    ) == (
        "PARENT_LEG\nWIP_NOVELTY\n",
        "PARENT_PLAYER\nWIP_PLAYER\n",
    )


def test_generated_tester_supports_fresh_replay_without_checkpoint_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(L, "SCRATCH", str(tmp_path))
    ws = Path(L.setup_workspace("wa30", "fresh_replay"))
    tester = (ws / "gkm_try.py").read_text()
    task = L._propose_task("wa30", 9, "", [])

    assert 'os.environ.get("GKM_FRESH_REPLAY") != "1"' in tester
    assert "GKM_FRESH_REPLAY=1 python gkm_try.py" in task
    assert "`checkpoint.json` is supervisor-owned" in task
    assert "never edit, replace, delete, chmod, or regenerate it" in task
    assert "0..63 inclusive" in task
    assert "invalidates and terminates the complete proposer turn" in task


def test_repository_promoted_artifacts_are_clean_and_consistent():
    artifacts = Path(__file__).with_name("agent_solutions")
    checked = 0
    for artifact in sorted(artifacts.glob("*_legs")):
        checkpoint_path = artifact / L.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            continue
        checkpoint = json.loads(checkpoint_path.read_text())
        records = checkpoint["records"]
        levels = [record["level"] for record in records]
        assert L.promoted_artifact_taint_reason(str(artifact)) is None
        assert checkpoint["validated"] is True
        assert checkpoint["final_path"]
        assert len(levels) == len(set(levels))
        assert checkpoint["total_marginal_C"] == sum(
            record["marginal_C"] for record in records
        )
        checked += 1
    assert checked >= 8


def test_workspace_lock_rejects_overlapping_orchestrator(tmp_path):
    first = L._acquire_workspace_lock(str(tmp_path))
    try:
        assert L._workspace_lock_path(str(tmp_path)).parent != tmp_path
        assert not (tmp_path / ".orchestrate.lock").exists()
        try:
            L._acquire_workspace_lock(str(tmp_path))
        except RuntimeError as ex:
            assert "another orchestrator" in str(ex)
        else:
            raise AssertionError("overlapping workspace lock was accepted")

        code = (
            "import gkm_legs as L\n"
            "try:\n"
            f"    L._acquire_workspace_lock({str(tmp_path)!r})\n"
            "except RuntimeError:\n"
            "    print('BLOCKED')\n"
            "else:\n"
            "    raise SystemExit('overlap accepted')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(filter(None, (
                    os.path.dirname(L.__file__), os.environ.get("PYTHONPATH", "")
                ))),
            },
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "BLOCKED"
    finally:
        L._release_workspace_lock(first)


def test_boundary_rejects_legacy_workspace_lock_as_context_surface(tmp_path):
    shell = L.APB.scan_shell_command(
        "cat .orchestrate.lock",
        logical_path="turn.jsonl",
        line=1,
    )
    assert any(item.code == "hidden_control_surface" for item in shell)
    python = L.APB.scan_python_source(
        "open('.orchestrate.lock').read()\n", logical_path="probe.py"
    )
    assert any(item.code == "dynamic_or_external_file_access" for item in python)

    workspace = tmp_path / "gkm_legs_ws_lock_surface"
    workspace.mkdir()
    transcript = tmp_path / "turn.jsonl"
    transcript.write_text(json.dumps({
        "type": "item.completed",
        "item": {
            "id": "change-lock",
            "type": "file_change",
            "changes": [{"path": ".orchestrate.lock"}],
        },
    }) + "\n", encoding="utf-8")
    findings = L.APB.scan_codex_transcript(
        transcript, workspace_root=workspace
    )
    assert any(item.code == "file_change_hidden_surface" for item in findings)


def test_workspace_lock_rejects_symlink_and_hardlink_aliases(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-lock"
    outside.write_text("do not truncate\n", encoding="utf-8")
    lock_path = tmp_path / ".orchestrate.lock"
    lock_path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="unaliased regular"):
        L._acquire_workspace_lock(str(tmp_path))
    assert outside.read_text(encoding="utf-8") == "do not truncate\n"

    lock_path.unlink()
    os.link(outside, lock_path)
    with pytest.raises(RuntimeError, match="unaliased regular"):
        L._acquire_workspace_lock(str(tmp_path))
    assert outside.read_text(encoding="utf-8") == "do not truncate\n"


def test_lineage_lock_rejects_different_tags_for_same_artifact_root(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GKM_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    first = L._acquire_lineage_lock("sk48")
    try:
        with pytest.raises(RuntimeError, match="artifact lineage"):
            L._acquire_lineage_lock("sk48")
        independent_root = tmp_path / "candidate"
        monkeypatch.setenv("GKM_ARTIFACTS_ROOT", str(independent_root))
        candidate = L._acquire_lineage_lock("sk48")
        L._release_workspace_lock(candidate)
    finally:
        L._release_workspace_lock(first)


def test_codex_command_is_explicitly_metered_and_sandboxed(tmp_path):
    cmd = L._codex_command(str(tmp_path), "do the bounded task", None, "medium")
    joined = " ".join(cmd)
    assert cmd[:2] == ["codex", "exec"]
    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--strict-config" in cmd
    assert "--model gpt-5.6-sol" in joined
    assert 'model_reasoning_effort="medium"' in cmd
    assert 'web_search="disabled"' in cmd
    assert "sandbox_workspace_write.network_access=false" in cmd
    assert 'approval_policy="never"' in cmd
    assert "--sandbox workspace-write" in joined
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--add-dir" not in cmd

    xhigh_cmd = L._codex_command(str(tmp_path), "task", None, "xhigh")
    assert 'model_reasoning_effort="xhigh"' in xhigh_cmd
    max_cmd = L._codex_command(str(tmp_path), "task", None, "max")
    assert 'model_reasoning_effort="max"' in max_cmd

    try:
        L._codex_command(str(tmp_path), "task", None, "low")
    except ValueError as ex:
        assert all(name in str(ex) for name in ("medium", "high", "xhigh", "max"))
    else:
        raise AssertionError("unsupported Codex effort was accepted")


def test_native_codex_compatibility_launch_is_disabled_until_closure_sealed(
    monkeypatch,
):
    monkeypatch.setattr(
        L,
        "authoritative_level_target",
        lambda _game: (_ for _ in ()).throw(
            AssertionError("launch gate ran too late")
        ),
    )

    with pytest.raises(RuntimeError, match="dependency closure"):
        L.orchestrate(
            "lf52",
            max_level=9,
            proposer="codex",
            propose_fn=None,
            verbose=False,
        )


def test_codex_environment_does_not_forward_api_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak-either")
    monkeypatch.setenv("PATH", "/test/bin")
    env = L._codex_environment()
    assert env["PATH"] == "/test/bin"
    assert env["GKM_SANITIZE_PROPOSER_INTERRUPTS"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_codex_json_usage_parser_uses_turn_completed_only(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\n" +
        "diagnostic that is not JSON\n" +
        json.dumps({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 70,
                "output_tokens": 30,
                "reasoning_output_tokens": 20,
            },
        }) + "\n"
    )
    usage = L._codex_usage_from_jsonl(str(log))
    assert usage == {
        "thread_id": "thread-1",
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "output_tokens": 30,
        "reasoning_output_tokens": 20,
        "usage_reported": True,
        "observed_tokens": 130,
    }


def test_codex_agent_records_offline_fake_turn(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    reset = 1_800_000_000
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 6,
                    "resetsAt": reset,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }

    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "print('Reading additional input from stdin...', file=sys.stderr); "
                "print(json.dumps({'type':'thread.started','thread_id':'fake'})); "
                "print(json.dumps({'type':'turn.completed','usage':"
                "{'input_tokens':80,'cached_input_tokens':50,'output_tokens':20,"
                "'reasoning_output_tokens':12}}))"
            ),
        ],
    )
    real_popen = L.subprocess.Popen
    popen_calls = []

    def recording_popen(*args, **kwargs):
        popen_calls.append(kwargs.copy())
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(L.subprocess, "Popen", recording_popen)
    binding = L.CCS.exact_frontier_binding(
        tmp_path / "missing_artifact",
        game="fake",
        target_level=1,
    )

    record = L._codex_agent(
        str(ws),
        "offline fake task",
        None,
        1,
        reasoning_effort="medium",
        weekly_reserve=90,
        max_campaign_tokens=1_000,
        max_campaign_runs=1,
        ledger_path=str(ledger),
        run_label="fake:L1:propose",
        game="fake",
        target_level=1,
        frontier_binding=binding,
    )
    assert record["returncode"] == 0
    assert record["thread_id"] == "fake"
    assert record["observed_tokens"] == 100
    assert record["weekly_remaining_before"] == 94
    assert record["weekly_remaining_after"] == 94
    assert record["reasoning_effort"] == "medium"
    assert record["frontier_sha256"] == binding["frontier_sha256"]
    assert record["surviving_process_group"] is False
    assert len(popen_calls) == 1
    assert popen_calls[0]["stdin"] is L.subprocess.DEVNULL
    assert popen_calls[0]["stderr"] is not L.subprocess.STDOUT
    assert json.loads(ledger.read_text())["run_label"] == "fake:L1:propose"
    immutable = ws / record["transcript"]
    assert immutable.is_file()
    assert all(json.loads(line) for line in immutable.read_text().splitlines())
    assert (ws / "proposer_last.log").read_bytes() == immutable.read_bytes()
    protected = (
        tmp_path / ".proposer_transcripts" / "ws" / record["transcript"]
    )
    assert protected.is_file()
    assert protected.read_bytes() == immutable.read_bytes()
    diagnostics = ws / record["diagnostics"]
    protected_diagnostics = (
        tmp_path / ".proposer_transcripts" / "ws" / record["diagnostics"]
    )
    assert diagnostics.read_text() == "Reading additional input from stdin...\n"
    assert protected_diagnostics.read_bytes() == diagnostics.read_bytes()
    assert (ws / "proposer_last.stderr.log").read_bytes() == diagnostics.read_bytes()
    assert record["protected_diagnostics_status"] == "sealed"
    assert record["protected_diagnostics_sha256"] == L._sha256_file(
        str(protected_diagnostics)
    )

    L._record_codex_level_outcome(
        record,
        ledger_path=str(ledger),
        game="fake",
        level=1,
        reached_before=0,
        reached_after=1,
        path=[1, 2],
        marginal_C=17,
    )
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[1]["event"] == "codex_level_outcome"
    assert rows[1]["thread_id"] == "fake"
    assert rows[1]["codex_exec_transcript"] == record["transcript"]
    assert rows[1]["solved_target"] is True
    assert rows[1]["winning_marginal_C"] == 17
    assert rows[1]["parent_checkpoint_sha256"] == L.CCS.ZERO_SHA256
    assert rows[1]["frontier_sha256"] == binding["frontier_sha256"]


def test_codex_agent_releases_full_turn_lock_for_unlimited_provider(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "business",
                "primary": None,
                "secondary": None,
                "credits": {
                    "hasCredits": True,
                    "unlimited": True,
                    "balance": None,
                },
                "spendControlReached": False,
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    lock_path = f"{ledger}.lock"
    child_code = (
        "import fcntl,json; "
        f"h=open({lock_path!r},'r+'); "
        "fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); "
        "print(json.dumps({'type':'thread.started','thread_id':'unlimited'})); "
        "print(json.dumps({'type':'turn.completed','usage':{}}))"
    )
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable, "-c", child_code,
        ],
    )

    record = L._codex_agent(
        str(ws),
        "offline unlimited concurrency task",
        None,
        1,
        weekly_reserve=100,
        max_campaign_tokens=0,
        max_campaign_runs=0,
        ledger_path=str(ledger),
        run_label="unlimited:L1:propose",
        game="unlimited",
        target_level=1,
    )

    assert record["returncode"] == 0
    assert record["thread_id"] == "unlimited"
    assert record["weekly_remaining_before"] == 100
    assert json.loads(ledger.read_text())["run_label"] == (
        "unlimited:L1:propose"
    )


def test_codex_agent_immediately_terminates_public_action_violation(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 0,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    marker = L.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable,
            "-u",
            "-c",
            (
                "import json,time; "
                f"print(json.dumps({{'type':'probe','text':'{marker}'}}),"
                " flush=True); "
                "time.sleep(60)"
            ),
        ],
    )

    started = time.monotonic()
    with pytest.raises(
        L.ProposerProtocolViolation,
        match="outside the public protocol",
    ):
        L._codex_agent(
            str(ws),
            "offline protocol poison",
            None,
            1,
            ledger_path=str(ledger),
            run_label="poison:L1:propose",
            game="poison",
            target_level=1,
        )
    assert time.monotonic() - started < 10

    row = json.loads(ledger.read_text())
    assert row["failure_class"] == "taint"
    assert (
        row["failure_detail_class"]
        == "public_action_protocol_violation"
    )
    assert row["public_action_protocol_violation"] is True
    assert row["protected_transcript_status"] == "sealed"
    assert L._workspace_or_protected_taint_reason(str(ws)) is not None


@pytest.mark.parametrize("phase", ["live", "terminal"])
def test_codex_agent_applies_arena_control_live_and_terminal(
    tmp_path, monkeypatch, phase
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 0,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    child = (
        "import json,time; "
        "print(json.dumps({'type':'thread.started','thread_id':'gate'}),"
        " flush=True); "
        + (
            "time.sleep(60)"
            if phase == "live"
            else "print(json.dumps({'type':'turn.completed','usage':{}}), flush=True)"
        )
    )
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable, "-u", "-c", child
        ],
    )
    calls = []

    def boundary_reasons(_monitor, _path, *, final=False):
        calls.append(final)
        if phase == "live" and not final:
            return ("arena_module_control_drift: live",)
        if phase == "terminal" and final:
            return ("arena_module_control_drift: terminal",)
        return ()

    monkeypatch.setattr(L, "_codex_boundary_reasons", boundary_reasons)

    with pytest.raises(L.ProposerBoundaryViolation, match=phase):
        L._codex_agent(
            str(ws),
            f"offline {phase} arena-control test",
            None,
            1,
            ledger_path=str(ledger),
            run_label=f"gate:{phase}",
            game="gate",
            target_level=1,
        )

    assert (False in calls) if phase == "live" else (True in calls)
    row = json.loads(ledger.read_text())
    assert row["failure_class"] == "taint"
    assert row["failure_detail_class"] == "filesystem_boundary_violation"
    assert row["filesystem_boundary_violation_reason"].endswith(phase)


@pytest.mark.parametrize("fault", ["arena", "workspace"])
def test_codex_agent_rejects_boundary_before_process_creation(
    tmp_path, monkeypatch, fault
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 0,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda *_args: [sys.executable, "-c", "raise SystemExit(0)"],
    )
    if fault == "arena":
        monkeypatch.setattr(
            L,
            "_compatibility_arena_control_reason",
            lambda: "arena_module_control_drift: prelaunch",
        )
    else:
        (ws / "probe.py").write_text(
            "open('/etc/passwd').read()\n", encoding="utf-8"
        )
    popen_called = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("Popen called before boundary admission")

    monkeypatch.setattr(L.subprocess, "Popen", forbidden_popen)

    with pytest.raises(L.ProposerBoundaryViolation):
        L._codex_agent(
            str(ws),
            f"offline prelaunch {fault} test",
            None,
            1,
            ledger_path=str(ledger),
            run_label=f"prelaunch:{fault}",
            game="prelaunch",
            target_level=1,
        )

    assert popen_called is False
    row = json.loads(ledger.read_text())
    assert row["returncode"] is None
    assert row["failure_class"] == "taint"
    assert row["failure_detail_class"] == "filesystem_boundary_violation"
    assert row["protected_transcript_status"] == "sealed"
    assert row["protected_diagnostics_status"] == "sealed"


def test_codex_agent_fails_closed_when_protected_transcript_disappears(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 0,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable,
            "-c",
            "print('{\"type\":\"turn.completed\",\"usage\":{}}')",
        ],
    )

    def missing_protected_transcript(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(
        L, "_read_single_link_regular", missing_protected_transcript
    )
    with pytest.raises(
        L.ProposerEvidenceUnavailable,
        match="discarding this complete proposer generation",
    ):
        L._codex_agent(
            str(ws),
            "offline fake task",
            None,
            1,
            ledger_path=str(ledger),
            run_label="missing:L1:propose",
            game="missing",
            target_level=1,
        )

    assert not (ws / "proposer_last.log").exists()
    assert not list(ws.glob("codex_turn_*.jsonl"))
    row = json.loads(ledger.read_text())
    assert row["failure_class"] == "evidence"
    assert row["failure_detail_class"] == "protected_transcript_unavailable"
    assert row["protected_transcript_status"] == "unavailable"
    assert row["protected_transcript_sha256"] is None
    assert row["observed_tokens"] == 0


def test_codex_agent_discards_turn_with_surviving_background_process(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 0,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable,
            "-c",
            (
                "import json, subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                "print(json.dumps({'type':'turn.completed','usage':{}}))"
            ),
        ],
    )

    with pytest.raises(
        L.ProposerEvidenceUnavailable,
        match="spawned process remained alive",
    ):
        L._codex_agent(
            str(ws),
            "background child test",
            None,
            1,
            ledger_path=str(ledger),
            run_label="background:L1:propose",
            game="background",
            target_level=1,
        )

    row = json.loads(ledger.read_text())
    assert row["failure_class"] == "evidence"
    assert row["failure_detail_class"] == "surviving_process_group"
    assert row["surviving_process_group"] is True
    assert row["protected_transcript_status"] == "sealed"


def test_orchestrate_does_not_reuse_or_promote_evidence_lost_turn(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(
        L, "setup_workspace", lambda game, tag="": str(workspace)
    )
    monkeypatch.setattr(L.A, "validate", lambda *_args, **_kwargs: False)

    calls = []

    def evidence_lost_proposer(ws, level):
        calls.append(level)
        Path(ws, "legs.py").write_text(
            "def unaudited_leg(env):\n    env.step(1)\n"
        )
        Path(ws, "players.py").write_text(
            "def play_level_1(env):\n    unaudited_leg(env)\n"
        )
        raise L.ProposerEvidenceUnavailable("protected transcript vanished")

    report = L.orchestrate(
        "evidencelost",
        max_level=1,
        propose_fn=evidence_lost_proposer,
        verify_fn=lambda game, solve_path: (0, [], None),
        debrief_fn=lambda ws, level: None,
        transient_retries=2,
        verbose=False,
    )

    assert calls == [1]
    assert report.reached == 0
    assert not (artifact_root / "evidencelost_legs" / "players.py").exists()
    assert not (
        artifact_root / "evidencelost_legs" / "wip_context"
    ).exists()


def test_orchestrate_protocol_poison_writes_no_wip_or_promotion(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(
        L, "setup_workspace", lambda game, tag="": str(workspace)
    )
    monkeypatch.setattr(L.A, "validate", lambda *_args, **_kwargs: False)

    def poisoned_proposer(ws, level):
        Path(ws, "probe.py").write_text("print('derived after poison')\n")
        raise L.ProposerProtocolViolation("out-of-frame coordinate")

    with pytest.raises(
        L.WorkspaceTainted,
        match="no WIP or promotion was written",
    ):
        L.orchestrate(
            "protocolpoison",
            max_level=1,
            propose_fn=poisoned_proposer,
            verify_fn=lambda game, solve_path: (0, [], None),
            debrief_fn=lambda ws, level: None,
            verbose=False,
        )

    artifact = artifact_root / "protocolpoison_legs"
    assert not (artifact / "wip_context").exists()
    assert not (artifact / "players.py").exists()


def test_evidence_lost_debrief_restores_exact_pre_debrief_win(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, body in {
        "legs.py": "",
        "players.py": "",
        "solve.py": "",
        "legs_log.md": "",
    }.items():
        (workspace / name).write_text(body)
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(
        L, "setup_workspace", lambda game, tag="": str(workspace)
    )
    monkeypatch.setattr(L.A, "validate", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        L, "exact_level_boundary", lambda game, path, level: list(path)
    )

    def propose(ws, level):
        Path(ws, "legs.py").write_text(
            "def exact_leg(env):\n    env.step(1)\n"
        )
        Path(ws, "players.py").write_text(
            "def play_level_1(env):\n    exact_leg(env)\n"
        )

    def verify(game, solve_path):
        source = Path(solve_path).with_name("players.py").read_text()
        return (
            (1, [1], None)
            if "play_level_1" in source
            else (0, [], None)
        )

    def evidence_lost_debrief(ws, level):
        Path(ws, "players.py").write_text("# unaudited debrief rewrite\n")
        Path(ws, "unrecorded_probe.txt").write_text("unrecorded\n")
        raise L.ProposerEvidenceUnavailable("debrief transcript vanished")

    report = L.orchestrate(
        "debriefevidence",
        max_level=1,
        propose_fn=propose,
        verify_fn=verify,
        debrief_fn=evidence_lost_debrief,
        verbose=False,
    )

    assert report.reached == 1
    promoted = artifact_root / "debriefevidence_legs" / "players.py"
    assert "play_level_1" in promoted.read_text()
    assert "unaudited debrief" not in promoted.read_text()
    assert not (workspace / "unrecorded_probe.txt").exists()
    phases = [
        json.loads(path.read_text())["phase"]
        for path in (
            artifact_root
            / "debriefevidence_legs"
            / "wip_context"
            / "level_01"
        ).glob("*/metadata.json")
    ]
    assert "reached_before_debrief" in phases
    assert not any("evidence" in phase for phase in phases)


@pytest.mark.parametrize(
    "failure",
    [L.CreditOut("test credit"), L.ProposerInfrastructureError("test infra")],
)
def test_failed_debrief_fully_rolls_back_and_replays_before_promotion(
    tmp_path, monkeypatch, failure
):
    artifact_root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.chmod(0o750)
    for name in ("legs.py", "players.py", "solve.py", "legs_log.md"):
        (workspace / name).write_text(
            "# clean\n" if name.endswith(".py") else "",
            encoding="utf-8",
        )
    (workspace / ".git").mkdir()
    (workspace / ".git" / "sealed-baseline").write_text(
        "before\n", encoding="utf-8"
    )
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "sealed.cache").write_bytes(b"before")
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(L, "setup_workspace", lambda *a, **k: str(workspace))
    monkeypatch.setattr(L.A, "validate", lambda *_a, **_k: True)
    monkeypatch.setattr(L, "exact_level_boundary", lambda _g, path, _l: list(path))

    def propose(ws, _level):
        assert _level == 1, "rollback-recovered generation advanced to L2"
        Path(ws, "players.py").write_text(
            "def play_level_1(env):\n    pass\n", encoding="utf-8"
        )

    verify_calls = 0

    def verify(_game, solve_path):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls >= 3:
            assert (workspace / ".git" / "sealed-baseline").read_text() == (
                "before\n"
            )
            assert not (workspace / ".git" / "debrief-leak").exists()
            assert (workspace / "__pycache__" / "sealed.cache").read_bytes() == (
                b"before"
            )
            assert not (workspace / "__pycache__" / "debrief.cache").exists()
        source = Path(solve_path).with_name("players.py").read_text()
        return (1, [1], None) if "play_level_1" in source else (0, [], None)

    def failed_debrief(ws, _level):
        Path(ws).chmod(0o700)
        Path(ws, "players.py").write_text(
            "# partial debrief edit\n", encoding="utf-8"
        )
        Path(ws, "unsealed_note.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
        Path(ws, ".git", "sealed-baseline").unlink()
        Path(ws, ".git", "debrief-leak").write_text(
            "learned\n", encoding="utf-8"
        )
        Path(ws, "__pycache__", "sealed.cache").unlink()
        Path(ws, "__pycache__", "debrief.cache").write_bytes(b"learned")
        Path(ws, ".orchestrate.lock").write_text(
            "learned through obsolete lock\n", encoding="utf-8"
        )
        raise failure

    report = L.orchestrate(
        "debriefrollback",
        max_level=2,
        propose_fn=propose,
        verify_fn=verify,
        debrief_fn=failed_debrief,
        verbose=False,
    )
    assert report.reached == 1
    assert verify_calls >= 3  # startup, winning proposal, sealed rollback replay
    assert not (workspace / "unsealed_note.py").exists()
    assert (workspace / ".git" / "sealed-baseline").read_text() == "before\n"
    assert not (workspace / ".git" / "debrief-leak").exists()
    assert (workspace / "__pycache__" / "sealed.cache").read_bytes() == b"before"
    assert not (workspace / "__pycache__" / "debrief.cache").exists()
    assert not (workspace / ".orchestrate.lock").exists()
    assert workspace.stat().st_mode & 0o777 == 0o750
    promoted = artifact_root / "debriefrollback_legs" / "players.py"
    assert "play_level_1" in promoted.read_text()
    assert "partial debrief" not in promoted.read_text()


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt(), SystemExit("stop"), RuntimeError("unexpected")],
    ids=["keyboard", "system-exit", "runtime"],
)
def test_unexpected_debrief_failure_rolls_back_and_releases_both_locks(
    tmp_path, monkeypatch, failure
):
    artifact_root = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("legs.py", "players.py", "solve.py", "legs_log.md"):
        (workspace / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(L, "setup_workspace", lambda *a, **k: str(workspace))
    monkeypatch.setattr(L.A, "validate", lambda *_a, **_k: True)
    monkeypatch.setattr(L, "exact_level_boundary", lambda _g, path, _l: list(path))

    def propose(ws, _level):
        Path(ws, "players.py").write_text(
            "def play_level_1(env):\n    pass\n", encoding="utf-8"
        )

    def verify(_game, solve_path):
        source = Path(solve_path).with_name("players.py").read_text()
        return (1, [1], None) if "play_level_1" in source else (0, [], None)

    def debrief(ws, _level):
        Path(ws, "players.py").write_text("# partial\n", encoding="utf-8")
        Path(ws, "unsealed.txt").write_text("leak\n", encoding="utf-8")
        raise failure

    with pytest.raises(type(failure)):
        L.orchestrate(
            "unexpecteddebrief",
            max_level=1,
            propose_fn=propose,
            verify_fn=verify,
            debrief_fn=debrief,
            verbose=False,
        )

    assert "play_level_1" in (workspace / "players.py").read_text()
    assert not (workspace / "unsealed.txt").exists()
    lineage = L._acquire_lineage_lock("unexpecteddebrief")
    run = L._acquire_workspace_lock(str(workspace))
    L._release_workspace_lock(run)
    L._release_workspace_lock(lineage)


def test_single_link_evidence_reader_rejects_hardlink_alias(tmp_path):
    source = tmp_path / "source.jsonl"
    alias = tmp_path / "alias.jsonl"
    source.write_text('{"type":"turn.completed"}\n')
    os.link(source, alias)
    with pytest.raises(L.WorkspaceTainted, match="aliased/non-regular"):
        L._read_single_link_regular(str(source))


def test_checked_codex_frontier_rejects_source_change_and_promotion_race(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    artifact = artifact_root / "bindinggame_legs"
    artifact.mkdir(parents=True)
    checkpoint = {
        "game": "bindinggame",
        "reached": 1,
        "final_path": [1, 2],
        "records": [],
        "total_marginal_C": 0,
        "validated": True,
    }
    (artifact / "checkpoint.json").write_text(
        json.dumps(checkpoint, sort_keys=True)
    )
    for name in ("legs.py", "players.py", "solve.py"):
        (artifact / name).write_text(f"# {name}\n")
    monkeypatch.setenv("GKM_ARTIFACTS_ROOT", str(artifact_root))
    expected = L.CCS.exact_frontier_binding(
        artifact,
        game="bindinggame",
        target_level=2,
    )
    assert L._checked_codex_frontier_binding(
        "bindinggame", 2, "", expected
    ) == expected

    (artifact / "legs.py").write_text("# different exact parent\n")
    with pytest.raises(ValueError, match="scheduler decision"):
        L._checked_codex_frontier_binding(
            "bindinggame", 2, "", expected
        )

    (artifact / "legs.py").write_text("# legs.py\n")
    checkpoint["reached"] = 2
    checkpoint["final_path"].append(3)
    (artifact / "checkpoint.json").write_text(
        json.dumps(checkpoint, sort_keys=True)
    )
    with pytest.raises(ValueError, match="scheduler decision"):
        L._checked_codex_frontier_binding(
            "bindinggame", 2, "", expected
        )


def test_codex_hard_expiry_is_containment_not_solver_no_progress(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 6,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
        ],
    )

    with pytest.raises(L.ProposerContainmentTimeout):
        L._codex_agent(
            str(ws),
            "offline timeout",
            None,
            0.0005,
            allocation_policy="hard",
            weekly_reserve=90,
            max_campaign_tokens=1_000,
            max_campaign_runs=1,
            ledger_path=str(ledger),
        )

    record = json.loads(ledger.read_text())
    assert record["allocation_policy"] == "hard"
    assert record["allocation_expired"] is True
    assert record["timed_out"] is True
    assert record["failure_class"] == "containment"
    assert record["failure_detail_class"] == "hard_wall_time"


def test_codex_default_soft_expiry_drains_without_signalling(
    tmp_path, monkeypatch
):
    assert L.DEFAULT_CODEX_ALLOCATION_POLICY == "drain"
    assert (
        inspect.signature(L.orchestrate)
        .parameters["codex_allocation_policy"]
        .default
        == L.DEFAULT_CODEX_ALLOCATION_POLICY
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 6,
                    "resetsAt": 1_800_000_000,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }
    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            (
                "import json,time; time.sleep(0.05); "
                "print(json.dumps({'type':'thread.started','thread_id':'drain'})); "
                "print(json.dumps({'type':'turn.completed','usage':"
                "{'input_tokens':8,'cached_input_tokens':5,'output_tokens':2,"
                "'reasoning_output_tokens':1}}))"
            ),
        ],
    )

    record = L._codex_agent(
        str(ws),
        "offline drain",
        None,
        0.0002,
        weekly_reserve=90,
        max_campaign_tokens=1_000,
        max_campaign_runs=1,
        ledger_path=str(ledger),
    )

    assert record["allocation_policy"] == "drain"
    assert record["allocation_expired"] is True
    assert record["timed_out"] is False
    assert record["thread_id"] == "drain"
    assert record["usage_reported"] is True


def test_orchestration_loop_with_mocks_shows_reuse_trend(tmp_path, monkeypatch):
    """Drive the loop with a mock proposer that INVENTS legs on L1-L2 (learning the
    rules) and REUSES them on L3-L4 (no new legs). Marginal novelty must drop."""
    def mock_propose(ws, K):
        with open(os.path.join(ws, "players.py"), "a") as f:
            f.write(f"\n\ndef play_level_{K}(env):\n    leg_1(env)\n    leg_2(env) if {K} > 1 else None\n")
        if K <= 2:  # early levels: invent a new leg
            with open(os.path.join(ws, "legs.py"), "a") as f:
                f.write(f"\n\ndef leg_{K}(env):\n    for _ in range(3):\n        pass\n")

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [], None)   # empty path => real A.validate is skipped

    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))
    monkeypatch.setattr(
        L, "exact_level_boundary",
        lambda game, path, level: [level],
    )
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)

    # isolated, clean workspace (fake game name; mocks ignore the real game)
    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_legstest"), ignore_errors=True)
    rep = L.orchestrate("legstest", max_level=4, propose_fn=mock_propose,
                        verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                        verbose=False)
    assert rep.reached == 4
    by = {r.level: r.marginal_C for r in rep.records}
    assert set(by) == {1, 2, 3, 4}
    # L1 invents a leg; L2-L4 reuse with auto-solve (player-stub-only marginal cost)
    assert by[3] < by[1]                        # reuse cheaper than invention
    assert by[4] <= by[2]                       # not strictly increasing
    assert by[3] == by[4]                       # pure reuse: identical marginal cost
    assert rep.total_marginal_C == sum(by.values())
    level3_wip = artifact_root / "legstest_legs" / "wip_context" / "level_03"
    assert any(p.name.startswith("after_auto_solve_debrief_") for p in level3_wip.iterdir())


def test_orchestrate_replay_promotes_clean_interrupted_workspace_before_proposer(
    tmp_path, monkeypatch
):
    art = tmp_path / "artifacts" / "recover_legs"
    art.mkdir(parents=True)
    canonical = {
        "legs.py": "def retained(env):\n    pass\n",
        "players.py": "def play_level_1(env):\n    retained(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# retained\n",
    }
    for name, body in canonical.items():
        (art / name).write_text(body)

    ws = tmp_path / "workspace"
    ws.mkdir()
    for name, body in canonical.items():
        (ws / name).write_text(body)
    (ws / "players.py").write_text(
        canonical["players.py"]
        + "\ndef play_level_2(env):\n    retained(env)\n"
    )
    rep = L.Report(
        game="recover", reached=1,
        records=[L.LevelRecord(level=1, marginal_C=3, reached=True)],
        total_marginal_C=3, final_path=[1], validated=True,
    )
    L._save_checkpoint(str(ws), rep)

    proposed = []
    promoted = []
    monkeypatch.setattr(L, "setup_workspace", lambda game, tag="": str(ws))
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(art))
    monkeypatch.setattr(
        L, "promote_verified_artifact",
        lambda game, workspace, report, tag="", verbose=True:
            promoted.append((report.reached, list(report.final_path))) or True,
    )
    monkeypatch.setattr(
        L, "exact_level_boundary",
        lambda game, path, level: list(path[:level]),
    )
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)

    result = L.orchestrate(
        "recover",
        max_level=2,
        seed_artifact=False,
        propose_fn=lambda workspace, level: proposed.append(level),
        verify_fn=lambda game, solve_path: (2, [1, 2], None),
        debrief_fn=lambda workspace, level: None,
        verbose=False,
    )
    assert proposed == []
    assert result.reached == 2
    assert result.validated is True
    assert [record.level for record in result.records] == [1, 2]
    assert promoted
    assert all(item == (2, [1, 2]) for item in promoted)


def test_zero_seed_refuses_stale_workspace_beside_newer_validated_artifact(
    tmp_path, monkeypatch
):
    import pytest

    art = tmp_path / "artifacts" / "target_legs"
    art.mkdir(parents=True)
    ws = tmp_path / "workspace"
    ws.mkdir()
    for root in (art, ws):
        for name, body in {
            "legs.py": "def retained(env):\n    pass\n",
            "players.py": "def play_level_1(env):\n    retained(env)\n",
            "solve.py": "def solve(env):\n    return None\n",
            "legs_log.md": "# retained\n",
        }.items():
            (root / name).write_text(body)
    promoted = L.Report(
        game="target",
        reached=5,
        records=[L.LevelRecord(level=5, marginal_C=3, reached=True)],
        total_marginal_C=3,
        final_path=[1, 2, 3, 4, 5],
        validated=True,
    )
    L._save_checkpoint(str(art), promoted)
    stale = L.Report(
        game="target",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=3, reached=True)],
        total_marginal_C=3,
        final_path=[1],
        validated=True,
    )
    L._save_checkpoint(str(ws), stale)

    monkeypatch.setattr(L, "setup_workspace", lambda game, tag="": str(ws))
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(art))
    proposed = []
    with pytest.raises(ValueError, match="refusing zero-seed run"):
        L.orchestrate(
            "target",
            max_level=6,
            seed_artifact=False,
            propose_fn=lambda workspace, level: proposed.append(level),
            verify_fn=lambda game, solve_path: (1, [1], None),
            debrief_fn=lambda workspace, level: None,
            verbose=False,
        )
    assert proposed == []


def test_setup_workspace_builds_valid_dispatch():
    ws = L.setup_workspace("wa30")
    for f in ("legs.py", "players.py", "solve.py", "gkm_try.py", "legs_log.md", "perception.py"):
        assert os.path.exists(os.path.join(ws, f))
    import ast
    ast.parse(open(os.path.join(ws, "solve.py")).read())   # solve.py is valid Python
    ast.parse(open(os.path.join(ws, "perception.py")).read())


def test_codex_workspace_has_local_git_boundary(tmp_path):
    ws = tmp_path / "clean-room"
    ws.mkdir()
    for name, body in {
        "gkm_try.py": "print('ok')\n",
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n",
        "solve.py": "def solve(env):\n    pass\n",
        "legs_log.md": "# local\n",
    }.items():
        (ws / name).write_text(body)

    L._initialize_codex_workspace_git(str(ws))
    top = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    assert Path(top).resolve() == ws.resolve()

    (ws / "legs.py").write_text("def leg(env):\n    return 1\n")
    diff = subprocess.run(
        ["git", "-C", str(ws), "diff", "--", "legs.py"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    assert "return 1" in diff
    assert L._workspace_taint_reason(str(ws)) is None
    hooks_path = subprocess.run(
        ["git", "-C", str(ws), "config", "--get", "core.hooksPath"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    assert hooks_path == "/dev/null"
    with pytest.raises(L.WorkspaceTainted, match="pre-existing Git metadata"):
        L._initialize_codex_workspace_git(str(ws))


def test_solver_source_index_is_compact_and_navigable(tmp_path):
    (tmp_path / "legs.py").write_text(
        "def reusable_leg(env, steps=3):\n"
        "    \"\"\"Move through a bounded number of observed states.\"\"\"\n"
        "    for _ in range(steps):\n"
        "        observe(env)\n"
        "\n"
        "def other(env):\n"
        "    return reusable_leg(env)\n"
    )
    (tmp_path / "players.py").write_text(
        "def play_level_1(env):\n"
        "    reusable_leg(env)\n"
    )
    index = L._solver_source_index(str(tmp_path))
    assert "## legs.py" in index
    assert "L1--4" in index
    assert "`def reusable_leg(env, steps=3):`" in index
    assert "Move through a bounded number" in index
    assert "calls: observe, range" in index
    assert "for _ in range(steps)" not in index

    path = L._write_solver_source_index(str(tmp_path))
    assert Path(path).read_text() == index


def test_replay_harness_refuses_tainted_workspace():
    ws = L.setup_workspace("wa30", tag="taintrepro")
    with open(os.path.join(ws, "proposer_last.log"), "w") as f:
        f.write("cat environment_files/wa30/wa30.py\n")

    proc = subprocess.run(
        [sys.executable, "gkm_try.py"],
        cwd=ws,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode != 0
    assert "TAINTED WORKSPACE" in proc.stderr


def test_perception_seed_extracts_components_and_deltas(tmp_path):
    ws = L.setup_workspace("perceptiontest")
    import importlib.util
    spec = importlib.util.spec_from_file_location("perception", os.path.join(ws, "perception.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    import numpy as np
    frame = np.zeros((6, 6), dtype=int)
    frame[1:3, 1:3] = 4
    frame[4, 4] = 9
    blobs = P.connected_components(frame, colors=[4, 9])
    assert [(b.color, b.bbox, b.area) for b in blobs] == [
        (4, (1, 1, 2, 2), 4),
        (9, (4, 4, 4, 4), 1),
    ]
    after = frame.copy()
    after[2, 2] = 7
    delta = P.frame_delta(frame, after)
    assert delta["count"] == 1
    assert delta["bbox"] == (2, 2, 2, 2)

    class TinyEnv:
        def __init__(self, state=0):
            self.state = state

        def clone(self):
            return TinyEnv(self.state)

        def step(self, action, *coords):
            self.state += int(action)

        def frame(self):
            return np.asarray([[self.state]], dtype=int)

        def terminal(self):
            return False

    path = P.bounded_replay_bfs(
        TinyEnv(),
        goal_fn=lambda env, path: env.state >= 3,
        action_fn=lambda env: [1],
        max_states=10,
        max_depth=5,
    )
    assert path == [1, 1, 1]

    class StrictPublicEnv:
        actions = (1, 2, 3, 4, 6, 7)

        def __init__(self, state=0, calls=None):
            self.state = state
            self.calls = [] if calls is None else calls

        def clone(self):
            return StrictPublicEnv(self.state, self.calls)

        def step(self, action, *coords):
            self.calls.append((action, *coords))
            self.state += int(action)

        def frame(self):
            return np.asarray([[self.state]], dtype=int)

        def terminal(self):
            return False

    strict = StrictPublicEnv()
    with pytest.raises(ValueError, match="bare ACTION6"):
        P.action_deltas(strict, strict.actions)
    assert strict.calls == []

    strict = StrictPublicEnv()
    deltas = P.action_deltas(strict)
    assert set(deltas) == {1, 2, 3, 4, 7}
    assert all(call[0] != 6 for call in strict.calls)

    strict = StrictPublicEnv()
    assert P.bounded_bfs(
        strict,
        goal_fn=lambda env, path: False,
        max_states=100,
        max_depth=1,
    ) is None
    assert {call[0] for call in strict.calls} == {1, 2, 3, 4, 7}

    strict = StrictPublicEnv()
    assert P.safe_step(strict, (6, 12, 34)) == (6, 12, 34)
    assert strict.calls == [(6, 12, 34)]
    with pytest.raises(ValueError, match="0..63"):
        P.safe_step(strict, (6, 64, 0))
    assert strict.calls == [(6, 12, 34)]


def test_promote_and_seed_verified_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
    }.items():
        (ws / name).write_text(body)
    protected_dir = Path(L._protected_codex_transcript_dir(str(ws)))
    protected_dir.mkdir(parents=True)
    codex_turn = protected_dir / "codex_turn_20260722T000000000000Z_test.jsonl"
    codex_turn.write_text(json.dumps({"type": "thread.started", "thread_id": "clean"}) + "\n")

    rep = L.Report(
        game="artifacttest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=3, reached=True)],
        total_marginal_C=3,
        final_path=[1, 2, 3],
        validated=True,
    )
    assert L.promote_verified_artifact(
        "artifacttest",
        str(ws),
        rep,
        verbose=False,
        authorized_turn={"transcript": codex_turn.name, "diagnostics": None},
    )
    assert (artifact_root / "artifacttest_legs" / "README.md").exists()
    manifest = json.loads(
        (artifact_root / "artifacttest_legs" / "promotion_evidence" /
         "level_01" / "manifest.json").read_text()
    )
    assert len(manifest["codex_transcripts"]) == 1
    evidence_turn = (
        artifact_root / "artifacttest_legs" / "promotion_evidence" /
        "level_01" / manifest["codex_transcripts"][0]["path"]
    )
    assert evidence_turn.read_bytes() == codex_turn.read_bytes()

    (ws / "players.py").write_text("contaminated unfinished edit\n")
    seeded = L.seed_workspace_from_artifact("artifacttest", str(ws), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert "play_level_1" in (ws / "players.py").read_text()


def test_promotion_uses_host_protected_codex_transcript(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
    }.items():
        (ws / name).write_text(body)

    name = "codex_turn_20260728T000000000000Z_test.jsonl"
    workspace_bytes = (
        json.dumps({"type": "thread.started", "thread_id": "mutable"}) + "\n"
    ).encode()
    protected_bytes = (
        json.dumps({"type": "thread.started", "thread_id": "protected"}) + "\n"
    ).encode()
    (ws / name).write_bytes(workspace_bytes)
    (ws / "proposer_last.log").write_bytes(workspace_bytes)
    protected_dir = Path(L._protected_codex_transcript_dir(str(ws)))
    protected_dir.mkdir(parents=True)
    (protected_dir / name).write_bytes(protected_bytes)
    diagnostics_name = name.removesuffix(".jsonl") + ".stderr.log"
    diagnostics_bytes = b"deterministic CLI diagnostic\n"
    (protected_dir / diagnostics_name).write_bytes(diagnostics_bytes)
    failed_debrief_name = "codex_turn_20260728T000100000000Z_failed.jsonl"
    (protected_dir / failed_debrief_name).write_text(
        json.dumps({"type": "error", "message": "failed debrief"}) + "\n",
        encoding="utf-8",
    )

    rep = L.Report(
        game="protectedtranscript",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    assert L.promote_verified_artifact(
        "protectedtranscript",
        str(ws),
        rep,
        verbose=False,
        authorized_turn={
            "transcript": name,
            "diagnostics": diagnostics_name,
        },
    )
    evidence = (
        artifact_root / "protectedtranscript_legs" / "promotion_evidence" /
        "level_01"
    )
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert (evidence / "proposer_last.log").read_bytes() == protected_bytes
    assert (
        evidence / manifest["codex_transcripts"][0]["path"]
    ).read_bytes() == protected_bytes
    assert len(manifest["codex_diagnostics"]) == 1
    assert (
        evidence / manifest["codex_diagnostics"][0]["path"]
    ).read_bytes() == diagnostics_bytes
    assert manifest["authorized_turn_transcript"] == name
    assert failed_debrief_name not in {
        Path(item["path"]).name for item in manifest["codex_transcripts"]
    }


def test_tainted_workspace_cannot_promote_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
        "proposer_last.log": "sed -n '1,80p' environment_files/wa30/x/wa30.py\n",
    }.items():
        (ws / name).write_text(body)

    rep = L.Report(
        game="tainttest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    try:
        L.promote_verified_artifact("tainttest", str(ws), rep, verbose=False)
    except L.WorkspaceTainted as ex:
        assert "forbidden source/history access" in str(ex)
    else:
        raise AssertionError("tainted workspace promoted")
    assert not (artifact_root / "tainttest_legs" / "checkpoint.json").exists()


def test_private_runtime_introspection_taints_workspace(tmp_path):
    (tmp_path / "probe.py").write_text("print(env._game)\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "private game/runtime introspection" in reason


def test_oversized_taint_evidence_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MAX_TAINT_SCAN_BYTES", 10)
    path = tmp_path / "proposer_last.log"
    path.write_text("x" * 11)
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "oversized unscanned evidence" in reason


def test_runtime_enumeration_and_frame_data_taint_workspace(tmp_path):
    for index, body in enumerate(("print(vars(env))\n", "print(env._fd)\n")):
        path = tmp_path / f"probe_{index}.py"
        path.write_text(body)
        reason = L._workspace_taint_reason(str(tmp_path))
        assert reason is not None
        assert (
            "private game/runtime introspection" in reason
            or "runtime_introspection" in reason
        )
        path.unlink()


def test_other_catalog_game_source_taints_workspace(tmp_path):
    (tmp_path / "proposer_last.log").write_text("find .. -name 'g50t.py'\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "g50t.py" in reason


def test_game_named_probe_does_not_taint_workspace(tmp_path):
    (tmp_path / "probe_cd82.py").write_text("print('public API probe')\n")
    (tmp_path / "proposer_last.log").write_text(
        "python probe_cd82.py\n"
        "nl -ba probe_cd82.py\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_external_web_or_network_attempt_taints_workspace(tmp_path):
    attempts = (
        "curl https://example.com/public-scorecard.json\n",
        "python3 -c \"import requests; requests.get('https://example.com')\"\n",
        "use web_search to find the game\n",
        "import socket\nsocket.create_connection(('1.1.1.1', 443))\n",
    )
    for index, body in enumerate(attempts):
        path = tmp_path / f"network_attempt_{index}.txt"
        path.write_text(body)
        reason = L._workspace_taint_reason(str(tmp_path))
        assert reason is not None
        assert "external web/network access" in reason
        path.unlink()


def test_binary_probe_metadata_url_is_not_network_taint(tmp_path):
    # PNG-like binary metadata may legitimately name a specification or creator
    # URL. The immutable proposer transcript—not strings inside a binary
    # container—is the execution record for attempted network access.
    (tmp_path / "probe.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00XMP https://matplotlib.org/"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_loopback_reference_is_not_network_taint(tmp_path):
    (tmp_path / "client_note.txt").write_text(
        "The local service is http://127.0.0.1:8879/game/current\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_blocked_attempt_ledger_is_audit_evidence_not_execution_taint(tmp_path):
    (tmp_path / L.BLOCKED_ATTEMPTS_LOG).write_text(
        "bash: 'python3 -c \\\"print(env._game)\\\"'\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_adaptive_debrief_skips_literal_reuse_and_small_acquisitions():
    assert not L.should_run_debrief(
        "adaptive", auto_solved=True, pre_debrief_marginal_C=999
    )
    assert not L.should_run_debrief(
        "adaptive", auto_solved=False, pre_debrief_marginal_C=149
    )
    assert L.should_run_debrief(
        "adaptive", auto_solved=False, pre_debrief_marginal_C=150
    )
    assert L.should_run_debrief(
        "always", auto_solved=True, pre_debrief_marginal_C=0
    )
    assert not L.should_run_debrief(
        "never", auto_solved=False, pre_debrief_marginal_C=999
    )


def test_debrief_inline_code_mention_is_not_execution_taint(tmp_path):
    (tmp_path / "proposer_last.log").write_text(
        "The blocked `dir(legs)` command was recorded in the ledger.\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None

    (tmp_path / "probe.py").write_text("print(vars(env))\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert (
        "private game/runtime introspection" in reason
        or "runtime_introspection" in reason
    )


def test_public_clone_traceback_private_field_is_not_agent_taint(tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": (
                    "Traceback: clone() -> self._game = "
                    "copy.deepcopy(_clone._game)\n"
                ),
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "clone failed"},
        },
    ]
    (tmp_path / "proposer_last.log").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    (tmp_path / "probe.py").write_text("clone = env.clone()\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_codex_jsonl_diagnostics_do_not_expose_command_output_to_taint_scan(tmp_path):
    events = [
        "Reading additional input from stdin...",
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": "Traceback: clone() -> self._game\n",
            },
        }),
        "ERROR codex_models_manager: model refresh timed out",
    ]
    (tmp_path / "proposer_last.log").write_text("\n".join(events) + "\n")
    (tmp_path / "probe.py").write_text("clone = env.clone()\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_many_codex_diagnostics_do_not_reclassify_tool_output_as_taint(tmp_path):
    events = [
        f"ERROR codex_core::tools::router: benign diagnostic {index}"
        for index in range(7)
    ]
    events.insert(
        3,
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": "Traceback: clone() -> self._game and self._fd\n",
            },
        }),
    )
    (tmp_path / "proposer_last.log").write_text("\n".join(events) + "\n")
    (tmp_path / "probe.py").write_text("clone = env.clone()\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_codex_transport_https_diagnostics_are_not_proposer_network_taint(
        tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "error",
                "message": (
                    "Falling back from WebSockets to HTTPS transport. "
                    "request timed out"
                ),
            },
        },
        {
            "type": "error",
            "message": (
                "Reconnecting: error sending request for url "
                "(https://chatgpt.com/backend-api/codex/responses)"
            ),
        },
        {
            "type": "turn.failed",
            "error": {
                "message": (
                    "stream disconnected before completion: error sending "
                    "request for url "
                    "(https://chatgpt.com/backend-api/codex/responses)"
                ),
            },
        },
    ]
    (tmp_path / "proposer_last.log").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_codex_network_command_still_taints_with_transport_diagnostics(
        tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "error",
                "message": "Falling back to HTTPS transport",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "curl https://example.com/game-help",
                "aggregated_output": "network denied",
            },
        },
    ]
    (tmp_path / "proposer_last.log").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "external web/network access" in reason


def test_codex_jsonl_private_command_still_taints_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "python -c 'print(env._game)'",
            "aggregated_output": "anything\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "private game/runtime introspection" in reason


def test_codex_jsonl_host_process_command_taints_workspace_even_if_denied(
        tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "/bin/zsh -lc 'ps -axo pid,command'",
            "aggregated_output": "zsh: operation not permitted\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "host process introspection" in reason


def test_host_process_names_in_tool_output_do_not_taint_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "python probe.py",
            "aggregated_output": "debug: ps and lsof were unavailable\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    (tmp_path / "probe.py").write_text("print('public observation')\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_python_heredoc_variable_named_ps_does_not_taint_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"python - <<'PY'\n"
                "while queue:\n"
                " (bridges,ps),path=queue.popleft()\n"
                " occupied=bridges|ps\n"
                "PY\""
            ),
            "aggregated_output": "public symbolic search\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_process_command_after_python_heredoc_still_taints_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"python - <<'PY'\n"
                "ps = {'public': 'symbolic state'}\n"
                "PY\n"
                "ps -axo pid,command\""
            ),
            "aggregated_output": "anything\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "host process introspection" in reason


def test_unterminated_heredoc_fails_closed_for_process_scan(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "/bin/zsh -lc \"python - <<'PY'\nprint('incomplete')",
            "aggregated_output": "anything\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "host process introspection" in reason


def test_filtered_own_probe_process_monitoring_is_not_gameplay_taint(
        tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"ps -axo pid=,command= | "
                "rg 'probe_level7_worker.py --worker' || true\""
            ),
            "aggregated_output": "123 python probe_level7_worker.py --worker",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_exact_pgrep_of_own_named_probe_is_not_gameplay_taint(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"pgrep -af "
                "'probe_l7.py focused_search' || true\""
            ),
            "aggregated_output": "123 python probe_l7.py focused_search",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_macos_pgrep_of_own_named_probe_is_not_gameplay_taint(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"pgrep -fl "
                "'probe_l7_fresh_graph.py' || true\""
            ),
            "aggregated_output": "123 python probe_l7_fresh_graph.py",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_broad_pgrep_still_taints_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "/bin/zsh -lc \"pgrep -af python || true\"",
            "aggregated_output": "123 python probe_l7.py",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "host process introspection" in reason


def test_narrow_pgrep_does_not_mask_forbidden_process_control(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": (
                "/bin/zsh -lc \"pgrep -fl "
                "'probe_level7_search.py' || true; "
                "pkill -INT -f 'python probe_level7.py lower_search'\""
            ),
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "host process introspection" in reason


def test_raw_arena_capability_does_not_authorize_source_introspection(tmp_path):
    (tmp_path / "probe.py").write_text(
        "import inspect, gkm_arena as A\n"
        "print(dir(A), dir(A.Arena), dir(env))\n"
        "print(inspect.getsource(A.Arena.clone))\n"
    )
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "dynamic_or_process_import" in reason


def test_public_action_protocol_marker_quarantines_complete_turn(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "python gkm_try.py",
            "aggregated_output": (
                f"RESULT levels=0 err="
                f"{L.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER}: "
                "coordinate outside 0..63"
            ),
        },
    }
    (tmp_path / "codex_turn.jsonl").write_text(
        json.dumps(event) + "\n"
    )
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "public action protocol violation" in reason


def test_explicit_agent_protocol_self_invalidation_quarantines_turn(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": (
                "Turn invalidated: a probe attempted action 6 with x=65, "
                "outside the allowed 0..63 frame. I stopped immediately."
            ),
        },
    }
    (tmp_path / "codex_turn.jsonl").write_text(
        json.dumps(event) + "\n"
    )
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "self-reported public action protocol violation" in reason


def test_self_invalidated_protected_turn_cannot_be_snapshotted_as_wip(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "scratch" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "probe.py").write_text("print('ordinary probe')\n")
    protected = (
        workspace.parent
        / ".proposer_transcripts"
        / workspace.name
    )
    protected.mkdir(parents=True)
    event = {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": (
                "Turn invalidated: a probe attempted action 6 with x=65, "
                "outside the allowed 0..63 frame. I stopped immediately."
            ),
        },
    }
    (protected / "codex_turn_invalid.jsonl").write_text(
        json.dumps(event) + "\n"
    )
    with pytest.raises(
        L.WorkspaceTainted,
        match="self-reported public action protocol violation",
    ):
        L.snapshot_wip_context(
            "selfinvalidated",
            str(workspace),
            9,
            "after_propose",
            reached=8,
            verbose=False,
        )
    assert not (
        artifact_root / "selfinvalidated_legs" / "wip_context"
    ).exists()


def test_hypothetical_protocol_discussion_is_not_self_invalidation(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": (
                "A future turn would be invalidated if a probe attempted an "
                "out-of-frame coordinate outside 0..63."
            ),
        },
    }
    (tmp_path / "codex_turn.jsonl").write_text(
        json.dumps(event) + "\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_promoted_artifact_scan_excludes_forensic_wip(tmp_path):
    for name in L.PROMOTED_FILES:
        body = "# clean promoted evidence\n" if name.endswith(".py") else "clean promoted evidence\n"
        (tmp_path / name).write_text(body)
    dirty = tmp_path / "wip_context" / "level_01" / "attempt" / "files"
    dirty.mkdir(parents=True)
    (dirty / "probe.py").write_text("print(env._game)\n")

    assert L._workspace_taint_reason(str(tmp_path)) is not None
    assert L.promoted_artifact_taint_reason(str(tmp_path)) is None

    (tmp_path / "legs.py").write_text("print(env._game)\n")
    assert "legs.py" in L.promoted_artifact_taint_reason(str(tmp_path))


def test_action_path_accepts_coordinate_clicks_without_changing_key_paths():
    assert L._load_action_path([1, 5, 2]) == [1, 5, 2]
    assert L._load_action_path([[6, 12, 34], [6, 0, 63]]) == [
        [6, 12, 34], [6, 0, 63]
    ]
    assert L._load_action_path([[5, 12, 34]]) is None
    assert L._load_action_path([True]) is None
    assert L._load_action_path([7]) == [7]
    assert L._load_action_path([6]) is None
    assert L._load_action_path([8]) is None
    assert L._load_action_path([[6, True, 34]]) is None
    assert L._load_action_path([[6, -1, 34]]) is None
    assert L._load_action_path([[6, 64, 34]]) is None
    assert L._load_action_path([[6, 34, 64]]) is None
    assert L._action_path_key([1, [6, 12, 34], 5]) == (1, (6, 12, 34), 5)


def test_wip_context_snapshot_is_artifact_visible(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def old_leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    old_leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "old leg context\n",
        "proposer_last.log": "fresh probe observation\n",
    }.items():
        (ws / name).write_text(body)
    rep = L.Report(
        game="snaptest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    L._save_checkpoint(str(ws), rep)
    assert L.promote_verified_artifact("snaptest", str(ws), rep, verbose=False)
    snap = L.snapshot_wip_context("snaptest", str(ws), 2, "not_reached", 1, "probe failed", verbose=False)
    assert (artifact_root / "snaptest_legs" / "wip_context" / "level_02" / "latest.json").exists()
    metadata = json.loads(Path(snap, "metadata.json").read_text())
    assert metadata["taint_verdict"] == "clean"
    assert metadata["frontier_binding"]["game"] == "snaptest"
    assert metadata["frontier_binding"]["reached"] == 1
    assert metadata["frontier_binding"]["target_level"] == 2
    assert "fresh probe observation" in (os.path.join(snap, "files", "proposer_last.log") and
                                         open(os.path.join(snap, "files", "proposer_last.log")).read())

    strict_ws = tmp_path / "strict_ws"
    strict_ws.mkdir()
    strict_seed = L.seed_workspace_from_artifact(
        "snaptest",
        str(strict_ws),
        verbose=False,
        expected_wip_attempt=Path(snap).name,
    )
    assert strict_seed is not None and strict_seed.reached == 1
    assert (strict_ws / "proposer_last.log").read_text() == (
        "fresh probe observation\n"
    )

    (ws / "players.py").write_text("contaminated unfinished edit\n")
    seeded = L.seed_workspace_from_artifact("snaptest", str(ws), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert "play_level_1" in (ws / "players.py").read_text()
    # WIP snapshots are forensic only: seeding must NOT inject probe context back
    # into the workspace (that stitching caused analysis paralysis; see FINDINGS).
    assert not (ws / "wip_context.md").exists()


def test_expected_wip_restore_rejects_pointer_change(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "probe.py").write_text("print('partial')\n")
    snap = Path(L.snapshot_wip_context(
        "strict", str(ws), 1, "infrastructure_failure", 0,
        "transport", verbose=False,
    ))
    latest = snap.parent / "latest.json"
    pointer = json.loads(latest.read_text())
    pointer["attempt"] = "different_attempt"
    latest.write_text(json.dumps(pointer))
    retry = tmp_path / "retry"
    retry.mkdir()
    try:
        L._restore_wip_probes(
            "strict", str(retry), 1, verbose=False,
            expected_attempt=snap.name,
        )
    except ValueError as exc:
        assert "no longer eligible" in str(exc)
    else:
        raise AssertionError("changed WIP pointer was restored")


def test_wip_snapshot_and_restore_preserve_nested_agent_context(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs")
    )
    ws = tmp_path / "ws"
    (ws / "search" / "beam").mkdir(parents=True)
    (ws / "search" / "beam" / "frontier.json").write_text('{"best": 7}\n')
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "discard.pyc").write_bytes(b"cache")

    snap = L.snapshot_wip_context(
        "nested", str(ws), 1, "interrupted", 0, "retry", verbose=False
    )
    assert Path(snap, "files", "search", "beam", "frontier.json").exists()
    assert not Path(snap, "files", "__pycache__").exists()

    retry = tmp_path / "retry"
    retry.mkdir()
    L._restore_wip_probes("nested", str(retry), 1, verbose=False)
    assert (retry / "search" / "beam" / "frontier.json").read_text() == '{"best": 7}\n'


def test_wip_restore_rechecks_old_snapshots_and_falls_back_to_clean(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "probe_old.py").write_text("print('clean earlier context')\n")
    L.snapshot_wip_context(
        "policyupgrade", str(ws), 1, "first_clean", 0, verbose=False
    )

    (ws / "probe_new.py").write_text("print('newer context')\n")
    newest = Path(
        L.snapshot_wip_context(
            "policyupgrade", str(ws), 1, "second_clean", 0, verbose=False
        )
    )
    # Model a snapshot that passed an older scanner before a policy upgrade.
    # The current restorer must reassess it rather than trusting latest.json.
    (newest / "files" / "probe_private.py").write_text("print(env._game)\n")

    retry = tmp_path / "retry"
    retry.mkdir()
    L._restore_wip_probes(
        "policyupgrade", str(retry), 1, verbose=False
    )
    assert (retry / "probe_old.py").read_text() == "print('clean earlier context')\n"
    assert not (retry / "probe_new.py").exists()
    assert not (retry / "probe_private.py").exists()


def test_exact_level_boundary_returns_first_winning_prefix(monkeypatch):
    class FakeArena:
        def __init__(self, game):
            self.levels_completed = 0
            self.steps = 0

        def terminal(self):
            return False

        def step(self, action):
            self.steps += 1
            if self.steps == 3:
                self.levels_completed = 2

    monkeypatch.setattr(L.A, "Arena", FakeArena)
    assert L.exact_level_boundary("fake", [1, 2, 3, 4, 5], 2) == [1, 2, 3]
    assert L.exact_level_boundary("fake", [1, 2], 2) is None


def test_propose_task_is_minimal(tmp_path):
    """The proposer prompt is the known-good 7-sentence task; no artifact/probe
    context is stitched in (prompt bloat degraded the proposer; see FINDINGS)."""
    task = L._propose_task("ls20", 5, "raw substrate context", ["bfs_to_level_up"])

    assert "GOAL: make solve.py reach LEVEL 5" in task
    assert "bfs_to_level_up" in task
    assert "REUSE existing legs" in task
    assert "play_level_5" in task
    assert "python gkm_try.py" in task
    assert "forced reuse is not compression" in task
    assert "VERIFIED ARTIFACT CONTEXT" not in task
    assert "wip" not in task.lower()


def test_tagged_workspace_uses_canonical_artifact_dir():
    assert L.artifact_dir("ls20") == L.artifact_dir("ls20", tag="continue")


def test_artifact_root_override_isolates_candidate_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("GKM_ARTIFACTS_ROOT", str(tmp_path / "candidates"))
    assert L.artifact_dir("wa30") == str(tmp_path / "candidates" / "wa30_legs")


def test_transient_proposer_failure_is_retried(tmp_path, monkeypatch):
    """A dropped-connection proposal (short log with an API error banner) must be
    retried instead of read as a capability failure; a genuine full-transcript
    failure must NOT be retried."""
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))
    monkeypatch.setattr(
        L, "exact_level_boundary",
        lambda game, path, level: [level],
    )
    monkeypatch.setattr(L.A, "validate", lambda game, path, levels: True)

    calls = []

    def flaky_propose(ws, K):
        calls.append(K)
        if len(calls) == 1:  # first attempt: infrastructure failure, no work done
            with open(os.path.join(ws, "proposer_last.log"), "w") as f:
                f.write("API Error: Connection closed mid-response.\n")
            return
        with open(os.path.join(ws, "proposer_last.log"), "w") as f:
            f.write("wrote play_level_1 composing a new leg\n")
        with open(os.path.join(ws, "legs.py"), "a") as f:
            f.write("\n\ndef leg_1(env):\n    pass\n")
        with open(os.path.join(ws, "players.py"), "a") as f:
            f.write("\n\ndef play_level_1(env):\n    leg_1(env)\n")

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [], None)

    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_retrytest"), ignore_errors=True)
    rep = L.orchestrate("retrytest", max_level=1, propose_fn=flaky_propose,
                        verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                        verbose=False)
    assert calls == [1, 1]          # retried once after the transient failure
    assert rep.reached == 1


def test_transient_retry_can_be_disabled_for_cost_bounded_campaign(
        tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    calls = []

    def dropped_propose(ws, K):
        calls.append(K)
        with open(os.path.join(ws, "proposer_last.log"), "w") as f:
            f.write("API Error: Connection closed mid-response.\n")

    def no_clear(game, solve_path):
        return 0, [], None

    shutil.rmtree(
        os.path.join(L.SCRATCH, "gkm_legs_ws_retryoff"), ignore_errors=True
    )
    rep = L.orchestrate(
        "retryoff", max_level=1, propose_fn=dropped_propose,
        verify_fn=no_clear, debrief_fn=lambda w, k: None,
        transient_retries=0, verbose=False,
    )
    assert calls == [1]
    assert rep.reached == 0


def test_transient_detector_requires_short_log(tmp_path):
    ws = tmp_path
    (ws / "proposer_last.log").write_text("API Error: Connection closed mid-response.\n")
    assert L._transient_proposer_failure(str(ws))
    (ws / "proposer_last.log").write_text("probing level...\n" * 200 + "api error once, recovered\n")
    assert not L._transient_proposer_failure(str(ws))


def test_codex_terminal_error_classification_ignores_solver_prose(tmp_path):
    log = tmp_path / "turn.jsonl"
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "aggregated_output": (
                    "If the move budget is insufficient, optimize the prefix. "
                    "A probe also printed: Selected model is at capacity."
                ),
            },
        },
        {
            "type": "error",
            "message": "Selected model is at capacity. Please try a different model.",
        },
        {
            "type": "turn.failed",
            "error": {
                "message": "Selected model is at capacity. Please try a different model."
            },
        },
    ]
    log.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert L._codex_terminal_error_messages(str(log)) == [
        "Selected model is at capacity. Please try a different model."
    ]
    assert (
        L._classify_provider_error_message(
            "Selected model is at capacity. Please try a different model."
        )
        == "infrastructure"
    )
    assert (
        L._classify_provider_error_message(
            "If the move budget is insufficient, optimize the prefix."
        )
        == "other"
    )
    assert (
        L._classify_provider_error_message(
            "Your usage limit has been reached."
        )
        == "credit_out"
    )


def test_codex_usage_guard_only_labels_explicit_cost_blocks_as_credit_out():
    cost_messages = (
        "weekly Codex allowance is 5% remaining, at or below reserve",
        "weekly Codex allowance has only 1% headroom above reserve",
        "local campaign run cap reached (12/12)",
        "local campaign token cap reached (20/20)",
    )
    for message in cost_messages:
        assert L._usage_guard_error_is_cost_block(RuntimeError(message))
    infrastructure_messages = (
        "another Codex campaign turn holds ledger.lock",
        "timed out waiting for codex app-server request 2",
        "rate-limit response contained no usable limits",
    )
    for message in infrastructure_messages:
        assert not L._usage_guard_error_is_cost_block(RuntimeError(message))


def test_explicit_infrastructure_failure_retries_without_no_progress(
        tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L.A, "validate", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    calls = []

    def capacity_then_solve(ws, level):
        calls.append(level)
        if len(calls) == 1:
            raise L.ProposerInfrastructureError(
                "Selected model is at capacity. Please try a different model."
            )
        with open(os.path.join(ws, "legs.py"), "a") as stream:
            stream.write("\n\ndef leg_1(env):\n    pass\n")
        with open(os.path.join(ws, "players.py"), "a") as stream:
            stream.write("\n\ndef play_level_1(env):\n    leg_1(env)\n")

    def verify(_game, solve_path):
        players = open(
            os.path.join(os.path.dirname(solve_path), "players.py")
        ).read()
        reached = int("def play_level_1" in players)
        return reached, [1] if reached else [], None

    shutil.rmtree(
        os.path.join(L.SCRATCH, "gkm_legs_ws_infraretry"),
        ignore_errors=True,
    )
    report = L.orchestrate(
        "infraretry",
        max_level=1,
        propose_fn=capacity_then_solve,
        verify_fn=verify,
        debrief_fn=lambda _workspace, _level: None,
        verbose=False,
    )
    assert calls == [1, 1]
    assert report.reached == 1


def test_noop_proposal_is_retried(tmp_path):
    """A proposer that signs off without touching any code (e.g. backgrounded its
    probe and exited) is a no-attempt: retry. A short log WITH code changes is a
    real (cheap) attempt: no retry."""
    (tmp_path / "proposer_last.log").write_text(
        "I'll stop here and wait for the background search to notify me.\n")
    assert L._transient_proposer_failure(str(tmp_path), code_changed=False)
    assert not L._transient_proposer_failure(str(tmp_path), code_changed=True)


def test_auto_solve_failure_recorded_and_skipped(tmp_path):
    ws = str(tmp_path)
    legs = "def solve_all(env):\n    pass\n"
    assert not L._auto_solve_failed_before(ws, 5, legs)
    L._record_auto_solve_failure(ws, 5, legs)
    assert L._auto_solve_failed_before(ws, 5, legs)
    # a changed library invalidates the negative record; other levels unaffected
    assert not L._auto_solve_failed_before(ws, 5, legs + "def new_leg(env):\n    pass\n")
    assert not L._auto_solve_failed_before(ws, 6, legs)


def test_seed_restores_wip_probes_without_clobbering(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
    }.items():
        (ws / name).write_text(body)
    rep = L.Report(game="probetest", reached=1,
                   records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
                   total_marginal_C=1, final_path=[1], validated=True)
    L._save_checkpoint(str(ws), rep)
    assert L.promote_verified_artifact("probetest", str(ws), rep, verbose=False)

    # an interrupted L2 attempt leaves probes + a candidate players.py in scratch
    (ws / "probe_l2.py").write_text("print('probe knowledge')\n")
    (ws / "probe_l1_stale.py").write_text("print('wrong frontier')\n")
    (ws / "codex_turn_20260727T000000Z_probetest_L1_propose.jsonl").write_text(
        '{"type":"turn.started"}\n'
    )
    (ws / "codex_turn_20260727T000001Z_probetest_L2_propose.jsonl").write_text(
        '{"type":"turn.started"}\n'
    )
    (ws / "players.py").write_text("# UNVERIFIED candidate\n")
    L.snapshot_wip_context("probetest", str(ws), 2, "interrupted", 1, "killed", verbose=False)

    # scratch dies; a fresh seed must restore the probe but keep players.py verified
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    seeded = L.seed_workspace_from_artifact("probetest", str(ws2), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert (ws2 / "probe_l2.py").read_text() == "print('probe knowledge')\n"
    assert not (ws2 / "probe_l1_stale.py").exists()
    assert not (
        ws2 / "codex_turn_20260727T000000Z_probetest_L1_propose.jsonl"
    ).exists()
    assert (
        ws2 / "codex_turn_20260727T000001Z_probetest_L2_propose.jsonl"
    ).exists()
    assert "play_level_1" in (ws2 / "players.py").read_text()  # verified, not candidate
    # a probe already present in scratch is NOT overwritten by the older snapshot
    (ws2 / "probe_l2.py").write_text("newer scratch state\n")
    L._restore_wip_probes("probetest", str(ws2), 2, verbose=False)
    assert (ws2 / "probe_l2.py").read_text() == "newer scratch state\n"


def test_seed_restores_level_one_wip_without_promoted_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "attempt"
    ws.mkdir()
    (ws / "probe_l1.py").write_text("print('mapped mechanic')\n")
    (ws / "players.py").write_text("# UNVERIFIED candidate\n")
    L.snapshot_wip_context("unpromoted", str(ws), 1, "not_reached", 0,
                           "retry later", verbose=False)
    level_dir = artifact_root / "unpromoted_legs" / "wip_context" / "level_01"
    latest = json.loads((level_dir / "latest.json").read_text())["attempt"]
    cache = level_dir / latest / "files" / "__pycache__"
    cache.mkdir()
    (cache / "probe.pyc").write_bytes(b"cache")
    (level_dir / "frontier_scaffold.json").write_text(
        '{"version":"v2","created_at":"2026-07-24T00:00:00Z"}\n'
    )

    ws2 = tmp_path / "retry"
    ws2.mkdir()
    seeded = L.seed_workspace_from_artifact("unpromoted", str(ws2), verbose=False)
    assert seeded is None
    assert (ws2 / "probe_l1.py").read_text() == "print('mapped mechanic')\n"
    assert '"version":"v2"' in (ws2 / "frontier_scaffold.json").read_text()
    assert not (ws2 / "players.py").exists()


def test_seed_refuses_tainted_reviewed_frontier_scaffold(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    level_dir = (
        artifact_root
        / "unpromoted_legs"
        / "wip_context"
        / "level_01"
    )
    level_dir.mkdir(parents=True)
    (level_dir / "latest.json").write_text('{"attempt":"missing"}\n')
    (level_dir / "frontier_scaffold.json").write_text(
        '{"instruction":"inspect environment_files/ before solving"}\n'
    )

    ws = tmp_path / "retry"
    ws.mkdir()
    L._restore_wip_probes("unpromoted", str(ws), 1, verbose=False)

    assert not (ws / "frontier_scaffold.json").exists()


def test_parent_seed_preserves_clean_unpromoted_source_overlay(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    art = artifact_root / "overlay_legs"
    ws = tmp_path / "workspace"
    art.mkdir(parents=True)
    ws.mkdir()
    parent = L.Report(
        game="overlay",
        reached=1,
        total_marginal_C=7,
        records=[L.LevelRecord(level=1, marginal_C=7, reached=True)],
        final_path=[1, 2],
        validated=True,
    )
    L._save_checkpoint(str(art), parent)
    L._save_checkpoint(str(ws), parent)
    for root, marker in ((art, "PROMOTED"), (ws, "UNPROMOTED WIN")):
        (root / "legs.py").write_text(f"# {marker}\n")
        (root / "players.py").write_text(f"# {marker}\n")
        (root / "solve.py").write_text(f"# {marker}\n")
        (root / "legs_log.md").write_text(f"{marker}\n")

    overlay = L._clean_unpromoted_source_overlay("overlay", str(ws))
    assert "UNPROMOTED WIN" in overlay["players.py"]
    L.seed_workspace_from_artifact("overlay", str(ws), verbose=False)
    assert "PROMOTED" in (ws / "players.py").read_text()
    L._restore_source_overlay(str(ws), overlay)
    assert "UNPROMOTED WIN" in (ws / "players.py").read_text()
    assert json.loads((ws / "checkpoint.json").read_text())["reached"] == 1


def test_new_scratch_restores_clean_same_parent_wip_source_before_proposer(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    art = artifact_root / "crosswip_legs"
    old_ws = tmp_path / "old-workspace"
    new_ws = tmp_path / "new-workspace"
    for path in (art, old_ws, new_ws):
        path.mkdir(parents=True)
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(
        L, "setup_workspace", lambda game, tag="": str(new_ws)
    )
    monkeypatch.setattr(L.A, "validate", lambda game, path, level: True)
    monkeypatch.setattr(
        L, "exact_level_boundary",
        lambda game, path, level: list(path[:level]),
    )

    parent = L.Report(
        game="crosswip",
        reached=1,
        total_marginal_C=5,
        records=[L.LevelRecord(level=1, marginal_C=5, reached=True)],
        final_path=[1],
        validated=True,
    )
    for root in (art, old_ws):
        L._save_checkpoint(str(root), parent)
        (root / "legs.py").write_text(
            "def leg_1(env):\n    env.step(1)\n"
        )
        (root / "players.py").write_text(
            "def play_level_1(env):\n    leg_1(env)\n"
        )
        (root / "solve.py").write_text("def solve(env):\n    return None\n")
        (root / "legs_log.md").write_text("L1\n")

    (old_ws / "legs.py").write_text(
        "def leg_1(env):\n    env.step(1)\n\n"
        "def leg_2(env):\n    env.step(2)\n"
    )
    (old_ws / "players.py").write_text(
        "def play_level_1(env):\n    leg_1(env)\n\n"
        "def play_level_2(env):\n    leg_2(env)\n"
    )
    L.snapshot_wip_context(
        "crosswip", str(old_ws), 2, "not_reached", 1, "wall timeout",
        verbose=False,
    )

    def verify(_game, solve_path):
        players = Path(solve_path).with_name("players.py").read_text()
        levels = players.count("def play_level_")
        return levels, list(range(1, levels + 1)), None

    def should_not_propose(_workspace, _level):
        raise AssertionError("clean cross-workspace WIP paid for another proposer")

    report = L.orchestrate(
        "crosswip",
        max_level=2,
        propose_fn=should_not_propose,
        verify_fn=verify,
        debrief_fn=lambda workspace, level: None,
        verbose=False,
    )

    assert report.reached == 2
    assert "def leg_2" in (art / "legs.py").read_text()
    assert "def leg_2" in (new_ws / "legs.py").read_text()
    latest = json.loads(
        (art / "wip_context" / "level_02" / "latest.json").read_text()
    )
    assert latest["metadata"]["phase"] == "recovered_existing_workspace_solver"


def test_parent_seed_does_not_preserve_new_stale_or_tainted_source(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    art = artifact_root / "overlay_legs"
    ws = tmp_path / "workspace"
    art.mkdir(parents=True)
    ws.mkdir()
    parent = L.Report(
        game="overlay", reached=1, final_path=[1], validated=True
    )
    L._save_checkpoint(str(art), parent)
    (art / "players.py").write_text("# PROMOTED\n")
    (ws / "players.py").write_text("# NEW TEMPLATE\n")
    assert L._clean_unpromoted_source_overlay("overlay", str(ws)) == {}

    L._save_checkpoint(str(ws), parent)
    (ws / "players.py").write_text("# candidate\nprint(env._game)\n")
    assert L._clean_unpromoted_source_overlay("overlay", str(ws)) == {}


def test_orchestrate_recovers_clean_win_before_second_proposer_after_seed(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    art = artifact_root / "orphanseed_legs"
    ws = tmp_path / "workspace"
    art.mkdir(parents=True)
    ws.mkdir()
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    monkeypatch.setattr(L, "setup_workspace", lambda game, tag="": str(ws))
    monkeypatch.setattr(L.A, "validate", lambda game, path, level: True)
    monkeypatch.setattr(
        L, "exact_level_boundary",
        lambda game, path, level: list(path[:level]),
    )

    parent = L.Report(
        game="orphanseed",
        reached=1,
        total_marginal_C=5,
        records=[L.LevelRecord(level=1, marginal_C=5, reached=True)],
        final_path=[1],
        validated=True,
    )
    for root in (art, ws):
        L._save_checkpoint(str(root), parent)
        (root / "legs.py").write_text("def leg_1(env):\n    env.step(1)\n")
        (root / "players.py").write_text(
            "def play_level_1(env):\n    leg_1(env)\n"
        )
        (root / "solve.py").write_text("def solve(env):\n    return None\n")
        (root / "legs_log.md").write_text("L1\n")

    # This is the clean exact source written by the killed child. The parent
    # checkpoint is still L1 because normal promotion never ran.
    (ws / "legs.py").write_text(
        "def leg_1(env):\n    env.step(1)\n\n"
        "def leg_2(env):\n    env.step(2)\n"
    )
    (ws / "players.py").write_text(
        "def play_level_1(env):\n    leg_1(env)\n\n"
        "def play_level_2(env):\n    leg_2(env)\n"
    )
    (ws / "proposer_last.log").write_text(
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python gkm_try.py",
                "aggregated_output": "RESULT levels=2 replay_ok=True",
            },
        }) + "\n"
    )

    def verify(_game, solve_path):
        players = Path(solve_path).with_name("players.py").read_text()
        levels = players.count("def play_level_")
        return levels, list(range(1, levels + 1)), None

    def should_not_propose(_workspace, _level):
        raise AssertionError("orphan recovery paid for a second proposer")

    report = L.orchestrate(
        "orphanseed",
        max_level=2,
        propose_fn=should_not_propose,
        verify_fn=verify,
        debrief_fn=lambda workspace, level: None,
        verbose=False,
    )

    assert report.reached == 2
    assert "def leg_2" in (art / "legs.py").read_text()
    latest = json.loads(
        (art / "wip_context" / "level_02" / "latest.json").read_text()
    )
    assert latest["metadata"]["phase"] == "recovered_existing_workspace_solver"


def test_frontier_brief_distills_agent_messages_and_probe_index(tmp_path):
    ws = tmp_path / "brief"
    ws.mkdir()
    events = [
        {"type": "item.completed", "item": {
            "type": "agent_message",
            "text": "Observed   a compact board transition.",
        }},
        {"type": "item.completed", "item": {
            "type": "command_execution",
            "aggregated_output": "RAW PIXELS MUST NOT ENTER THE BRIEF",
        }},
    ]
    (ws / "proposer_last.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    (ws / "focused_probe.py").write_text("print('probe')\n")
    (ws / "codex_turn_huge.jsonl").write_text("ignored\n")
    L._save_checkpoint(
        str(ws),
        L.Report(
            game="briefgame",
            reached=1,
            final_path=[1] * 123,
            validated=True,
        ),
    )

    path = L._write_frontier_brief(str(ws), "briefgame", 2)
    assert path is not None
    text = (ws / "frontier_brief.md").read_text()
    assert "briefgame level 2" in text
    assert "Observed a compact board transition." in text
    assert "focused_probe.py" in text
    assert "RAW PIXELS" not in text
    assert "codex_turn_huge" not in text
    assert "unverified" in text.lower()
    assert "Exact parent boundary: level 1 at 123 actions." in text
    assert "477 of 600" in text


def test_frontier_brief_is_removed_when_no_prior_context(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    (ws / "frontier_brief.md").write_text("stale")
    assert L._write_frontier_brief(str(ws), "empty", 1) is None
    assert not (ws / "frontier_brief.md").exists()


def test_generated_frontier_brief_is_covered_by_workspace_taint_gate(tmp_path):
    ws = tmp_path / "tainted_brief"
    ws.mkdir()
    event = {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": "A traceback suggested reading env._game next.",
        },
    }
    (ws / "proposer_last.log").write_text(json.dumps(event) + "\n")
    L._write_frontier_brief(str(ws), "x", 1)
    assert "private game/runtime introspection" in L._workspace_taint_reason(str(ws))


def test_interrupt_snapshots_and_promotes(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    def propose(ws, K):
        if K == 1:
            with open(os.path.join(ws, "players.py"), "a") as f:
                f.write(f"\n\ndef play_level_1(env):\n    pass\n")
        else:
            with open(os.path.join(ws, "probe_l2.py"), "w") as f:
                    f.write("# half-done probe\n")
            raise KeyboardInterrupt  # user hits Ctrl-C mid-L2

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [1] * n, None)

    monkeypatch.setattr(L.A, "validate", lambda g, p, l: True)
    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_inttest"), ignore_errors=True)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        L.orchestrate("inttest", max_level=3, propose_fn=propose,
                      verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                      verbose=False)
    # L1 was promoted despite the interrupt; the L2 probe context was snapshotted
    art = artifact_root / "inttest_legs"
    assert (art / "players.py").exists()
    level2 = art / "wip_context" / "level_02"
    assert level2.exists()
    import json as _json
    latest = _json.loads((level2 / "latest.json").read_text())
    assert latest["metadata"]["phase"] == "interrupted"


def test_interrupt_does_not_create_resumable_wip_from_tainted_turn(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )

    def propose(ws, level):
        (Path(ws) / "probe_private.py").write_text("print(env._game)\n")
        raise KeyboardInterrupt

    shutil.rmtree(
        os.path.join(L.SCRATCH, "gkm_legs_ws_taintedinterrupt"),
        ignore_errors=True,
    )
    with pytest.raises(KeyboardInterrupt):
        L.orchestrate(
            "taintedinterrupt", max_level=1, propose_fn=propose,
            verify_fn=lambda game, path: (0, [], None),
            debrief_fn=lambda workspace, level: None, verbose=False,
        )
    assert not (
        artifact_root / "taintedinterrupt_legs" / "wip_context" / "level_01"
    ).exists()


def test_snapshot_wip_context_fails_closed_on_tainted_workspace(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "probe_private.py").write_text("print(env._game)\n")

    with pytest.raises(L.WorkspaceTainted):
        L.snapshot_wip_context(
            "taintedsnapshot", str(workspace), 2, "after_propose",
            reached=1, verbose=False,
        )

    assert not (
        artifact_root / "taintedsnapshot_legs" / "wip_context"
    ).exists()


def test_snapshot_wip_reopens_host_owned_protocol_transcript(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "scratch" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "probe.py").write_text("print('ordinary probe')\n")
    protected = (
        workspace.parent
        / ".proposer_transcripts"
        / workspace.name
    )
    protected.mkdir(parents=True)
    (protected / "codex_turn_poison.jsonl").write_text(
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "aggregated_output": (
                    f"{L.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER}: "
                    "coordinate action requires integer x,y in 0..63"
                ),
            },
        })
        + "\n"
    )

    assert L._workspace_taint_reason(str(workspace)) is None
    assert L._protected_transcript_taint_reason(str(workspace)) is not None
    with pytest.raises(
        L.WorkspaceTainted,
        match="protected_transcripts|proposer_transcripts|protocol violation",
    ):
        L.snapshot_wip_context(
            "protectedpoison",
            str(workspace),
            1,
            "interrupted",
            reached=0,
            verbose=False,
        )
    assert not (
        artifact_root / "protectedpoison_legs" / "wip_context"
    ).exists()


def test_snapshot_wip_reopens_host_owned_diagnostics_sideband(
    tmp_path, monkeypatch
):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L,
        "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    workspace = tmp_path / "scratch" / "workspace"
    workspace.mkdir(parents=True)
    protected = (
        workspace.parent / ".proposer_transcripts" / workspace.name
    )
    protected.mkdir(parents=True)
    (protected / "codex_turn_poison.stderr.log").write_text(
        f"{L.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER}: invalid action\n"
    )

    assert L._protected_transcript_taint_reason(str(workspace)) is not None
    with pytest.raises(L.WorkspaceTainted, match="protocol violation"):
        L.snapshot_wip_context(
            "diagnosticpoison",
            str(workspace),
            1,
            "interrupted",
            reached=0,
            verbose=False,
        )
    assert not (
        artifact_root / "diagnosticpoison_legs" / "wip_context"
    ).exists()
