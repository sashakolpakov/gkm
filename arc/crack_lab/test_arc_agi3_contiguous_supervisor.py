from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_contiguous_conformance as C
import arc_agi3_contiguous_pilot as P
import arc_agi3_contiguous_supervisor as B
import arc_agi3_exact_bundle as E

PYTHON_EXECUTABLE = Path(sys.executable).resolve()
PYTHON_EXECUTABLE_SHA256 = hashlib.sha256(
    PYTHON_EXECUTABLE.read_bytes()
).hexdigest()


class _PostIncidentMetaDriver:
    def __init__(self, *, valid: bool = True):
        self.valid = valid
        self.calls = 0

    def run_attached_stream(
        self,
        argv,
        *,
        timeout_seconds,
        stdout_path,
        stderr_path,
        stdout_limit_bytes,
        stderr_limit_bytes,
    ):
        del timeout_seconds, stdout_limit_bytes, stderr_limit_bytes
        self.calls += 1
        command = tuple(argv)
        assert command[1::2] == (
            "--configuration",
            "--request",
            "--response",
        )
        request = json.loads(
            Path(command[4]).read_text(encoding="ascii")
        )
        response = {
            "schema": B.POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_response",
            "protocol_sha256":
                B.POST_INCIDENT_META_PROTOCOL_SHA256,
            "request_sha256": hashlib.sha256(
                B._operator_lease_canonical_json(request) + b"\n"
            ).hexdigest(),
            "status": "DIAGNOSED" if self.valid else "UNBOUNDED_RETRY",
            "diagnosis_code": "controller_runtime_drift",
            "diagnosis_summary":
                "The sealed substrate observation should be inspected.",
            "socratic_challenge":
                "Could the same code instead indicate host runtime drift?",
            "recommended_operator_action":
                "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE",
            "scheduler_authority": False,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "retry_authority": False,
            "dispatch_authority": False,
            "promotion_authority": False,
        }
        Path(command[6]).write_bytes(
            B._operator_lease_canonical_json(response) + b"\n"
        )
        Path(command[6]).chmod(0o600)
        Path(stdout_path).write_bytes(b"")
        Path(stdout_path).chmod(0o600)
        Path(stderr_path).write_bytes(b"")
        Path(stderr_path).chmod(0o600)
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            output_overflow=False,
        )


def _post_incident_meta_fixture(
    tmp_path: Path, *, valid: bool = True
):
    campaign = (tmp_path / "campaign").resolve()
    campaign.mkdir(mode=0o700)
    campaign.chmod(0o700)
    executable = (tmp_path / "sealed-meta-driver").resolve()
    executable.write_bytes(b"#!/bin/sh\nexit 99\n")
    executable.chmod(0o700)
    configuration = (tmp_path / "sealed-meta-config.json").resolve()
    configuration.write_bytes(b'{"schema":1}\n')
    configuration.chmod(0o400)
    driver = _PostIncidentMetaDriver(valid=valid)
    diagnostic = B.PostIncidentMetaDiagnostic(
        campaign,
        operator_configuration_sha256="a" * 64,
        driver_executable=executable,
        driver_executable_sha256=hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        driver_configuration=configuration,
        driver_configuration_sha256=hashlib.sha256(
            configuration.read_bytes()
        ).hexdigest(),
        driver_attestation_sha256="b" * 64,
        operation_timeout_seconds=60,
        command_runner=driver,
    )
    projection = {
        "schema": B.POST_INCIDENT_META_SCHEMA,
        "kind":
            "arc_agi3_contiguous_substrate_incident_projection",
        "operator_incident": {
            "attempt_id": "attempt-1",
            "operation": "substrate_health_reprobe",
            "fault_domain": "controller_substrate",
            "operation_consecutive": 2,
            "domain_consecutive": 2,
            "threshold": 2,
            "reason_code":
                "deterministic_substrate_configuration_repeated",
        },
        "substrate_incident": {
            "attempt_id": "attempt-1",
            "substrate_identity_sha256": "c" * 64,
            "failure_receipt_sha256": "d" * 64,
            "failure_class": "DETERMINISTIC_CONFIGURATION",
            "failure_code": "runtime_manifest_drift",
            "health_probe_count": 1,
            "attempted_remediation_epochs_sha256": "e" * 64,
            "last_health_probe_sha256": "f" * 64,
        },
        "incident_event_sequence": 12,
        "incident_event_digest": "1" * 64,
    }
    return diagnostic, driver, projection


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_pass_conformance(path: Path) -> dict:
    nodeids = [invariant.nodeid for invariant in C.INVARIANTS]
    component_nodeids = [
        f"{relative}::test_synthetic_full_component_pass"
        for relative in C.COMPONENT_TEST_FILES
    ]
    value = C.build_result(
        pytest_exit_code=0,
        collected_nodeids=nodeids,
        outcomes={nodeid: "PASS" for nodeid in nodeids},
        pytest_output="synthetic exact PASS for preflight unit test",
        component_collect_exit_code=0,
        component_collected_nodeids=component_nodeids,
        component_pytest_exit_code=0,
        component_run_collected_nodeids=list(component_nodeids),
        component_outcomes={
            nodeid: "PASS" for nodeid in component_nodeids
        },
        component_pytest_output=(
            "synthetic full component PASS for preflight unit test"
        ),
    )
    C._write_new_result(path, value)
    return value


def test_darwin_scoped_process_abi_and_pid_count_contract(monkeypatch):
    assert B.ctypes.sizeof(B._DarwinProcBsdInfo) == 136
    assert {
        name: getattr(B._DarwinProcBsdInfo, name).offset
        for name, _kind in B._DarwinProcBsdInfo._fields_
    } == {
        "pbi_flags": 0,
        "pbi_status": 4,
        "pbi_xstatus": 8,
        "pbi_pid": 12,
        "pbi_ppid": 16,
        "pbi_uid": 20,
        "pbi_gid": 24,
        "pbi_ruid": 28,
        "pbi_rgid": 32,
        "pbi_svuid": 36,
        "pbi_svgid": 40,
        "rfu_1": 44,
        "pbi_comm": 48,
        "pbi_name": 64,
        "pbi_nfiles": 96,
        "pbi_pgid": 100,
        "pbi_pjobc": 104,
        "e_tdev": 108,
        "e_tpgid": 112,
        "pbi_nice": 116,
        "pbi_start_tvsec": 120,
        "pbi_start_tvusec": 128,
    }
    calls = []

    def list_children(identifier, buffer, buffer_bytes):
        assert identifier == 77
        capacity = buffer_bytes // B.ctypes.sizeof(B.ctypes.c_int)
        calls.append(capacity)
        if capacity == 32:
            for index in range(capacity):
                buffer[index] = 1000 + index
            # libproc returns a PID count, not a byte count.  Equality means
            # the result may have filled the buffer and must be retried.
            return capacity
        buffer[0] = 2001
        buffer[1] = 2002
        return 2

    monkeypatch.setattr(
        B,
        "_darwin_libproc",
        lambda: SimpleNamespace(proc_listchildpids=list_children),
    )
    assert B._darwin_scoped_pids("proc_listchildpids", 77) == {
        2001,
        2002,
    }
    assert calls == [32, 64]


def test_linux_process_stat_indices_and_proc_record_bounds(tmp_path):
    suffix = ["R", "11", "12", "13", *("0" for _ in range(15)), "987"]
    identity = B._parse_linux_process_identity(
        4321,
        f"4321 (name with ) delimiter) {' '.join(suffix)}\n",
    )
    assert identity == (11, 12, 13, "R", "linux:987")

    oversized = tmp_path / "oversized-proc-record"
    oversized.write_bytes(b"x" * 9)
    with pytest.raises(
        B.SupervisorContractError, match="hard byte bound"
    ):
        B._read_bounded_linux_proc_record(
            oversized, maximum_bytes=8, label="synthetic proc record"
        )


def test_linux_child_pid_count_and_tokens_fail_closed(monkeypatch):
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "MAX_SCOPED_PROCESS_IDENTITIES", 2)
    monkeypatch.setattr(B, "_linux_task_ids", lambda _pid: (10,))
    monkeypatch.setattr(
        B,
        "_read_bounded_linux_proc_record",
        lambda *_args, **_kwargs: "20 21 22\n",
    )
    with pytest.raises(
        B.SupervisorContractError, match="PID bound"
    ):
        B._scoped_child_pids(10)

    monkeypatch.setattr(
        B,
        "_read_bounded_linux_proc_record",
        lambda *_args, **_kwargs: "20 20\n",
    )
    with pytest.raises(B.SupervisorContractError, match="malformed"):
        B._scoped_child_pids(10)


def test_linux_child_inventory_reads_every_thread(monkeypatch):
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "_linux_task_ids", lambda _pid: (100, 104))
    observed = []

    def read_children(path, **_kwargs):
        observed.append(Path(path))
        if str(path).endswith("/task/100/children"):
            return "201\n"
        if str(path).endswith("/task/104/children"):
            return "202 203\n"
        raise AssertionError(f"unexpected scoped proc record: {path}")

    monkeypatch.setattr(
        B, "_read_bounded_linux_proc_record", read_children
    )
    assert B._scoped_child_pids(100) == {201, 202, 203}
    assert len(observed) == 2


def test_linux_descendant_signal_uses_only_bound_pidfd(monkeypatch):
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "_stable_handle_exited", lambda _fd: False)
    monkeypatch.setattr(
        B.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("raw numeric PID signal")
        ),
    )
    calls = []
    monkeypatch.setattr(
        B.signal,
        "pidfd_send_signal",
        lambda fd, signum, siginfo, flags: calls.append(
            (fd, signum, siginfo, flags)
        ),
        raising=False,
    )

    B._signal_exact_processes(
        {201: "linux:old"},
        B.signal.SIGKILL,
        {201: 77},
        owned_pgid=100,
        final=True,
    )
    assert calls == [(77, B.signal.SIGKILL, None, 0)]


def test_linux_pid_reuse_during_pidfd_binding_is_rejected(monkeypatch):
    root_pid = 100
    root = (os.getpid(), root_pid, root_pid, "R", "linux:root")
    child_calls = 0
    closed = []

    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "_scoped_group_pids", lambda _pgid: set())
    monkeypatch.setattr(
        B,
        "_scoped_child_pids",
        lambda pid: {201} if pid == root_pid else set(),
    )

    def identity(pid):
        nonlocal child_calls
        if pid == root_pid:
            return root
        assert pid == 201
        child_calls += 1
        started = "linux:old" if child_calls == 1 else "linux:reused"
        return (root_pid, root_pid, root_pid, "R", started)

    monkeypatch.setattr(B, "_process_identity", identity)
    monkeypatch.setattr(
        B, "_linux_open_stable_process_handle", lambda _pid: 77
    )
    monkeypatch.setattr(B.os, "close", lambda fd: closed.append(fd))
    identities = {}
    stable_handles = {}

    with pytest.raises(
        B.SupervisorContractError,
        match="changed while binding its stable handle",
    ):
        B._accumulate_related_identities(
            root_pid,
            root[4],
            identities,
            stable_handles,
        )
    assert identities == {}
    assert stable_handles == {}
    assert closed == [77]


def test_linux_exited_pidfd_ignores_later_numeric_pid_reuse(monkeypatch):
    root_pid = 100
    root = (os.getpid(), root_pid, root_pid, "R", "linux:root")
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "_scoped_group_pids", lambda _pgid: set())
    monkeypatch.setattr(B, "_scoped_child_pids", lambda _pid: set())
    monkeypatch.setattr(B, "_stable_handle_exited", lambda fd: fd == 77)

    def identity(pid):
        if pid == root_pid:
            return root
        raise AssertionError("re-observed an exited pidfd by numeric PID")

    monkeypatch.setattr(B, "_process_identity", identity)
    identities = {201: "linux:old"}
    stable_handles = {201: 77}
    B._accumulate_related_identities(
        root_pid,
        root[4],
        identities,
        stable_handles,
    )
    assert identities == {201: "linux:old"}
    assert stable_handles == {201: 77}


def test_linux_subreaper_inventory_adopts_fast_reparent(monkeypatch):
    root_pid = 100
    supervisor_pid = os.getpid()
    root = (
        supervisor_pid,
        root_pid,
        root_pid,
        "R",
        "linux:root",
    )
    adopted = (
        supervisor_pid,
        301,
        301,
        "R",
        "linux:adopted",
    )
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B, "_scoped_group_pids", lambda _pgid: set())

    def children(pid):
        if pid == supervisor_pid:
            return {root_pid, 301}
        return set()

    monkeypatch.setattr(B, "_scoped_child_pids", children)
    monkeypatch.setattr(
        B,
        "_process_identity",
        lambda pid: root if pid == root_pid else adopted,
    )
    monkeypatch.setattr(
        B, "_linux_open_stable_process_handle", lambda pid: pid + 1000
    )
    monkeypatch.setattr(B, "_stable_handle_exited", lambda _fd: False)
    identities = {}
    stable_handles = {}
    B._accumulate_related_identities(
        root_pid,
        root[4],
        identities,
        stable_handles,
        supervisor_pid,
    )
    assert identities == {301: "linux:adopted"}
    assert stable_handles == {301: 1301}


def test_linux_subreaper_rejects_multithreaded_scheduler(monkeypatch):
    scheduler_pid = os.getpid()
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(B.os, "pidfd_open", lambda *_args: 77, raising=False)
    monkeypatch.setattr(
        B.signal, "pidfd_send_signal", lambda *_args: None, raising=False
    )
    monkeypatch.setattr(B, "_linux_child_subreaper_enabled", lambda: False)
    monkeypatch.setattr(
        B,
        "_linux_task_ids",
        lambda _pid: (scheduler_pid, scheduler_pid + 1),
    )

    with pytest.raises(
        B.SupervisorContractError, match="single-threaded scheduler"
    ):
        B._begin_linux_subreaper_custody()


def test_linux_subreaper_seal_rechecks_children_after_reap(monkeypatch):
    custody = B._StartedProcessCustody(
        SimpleNamespace(pid=100, returncode=None),
        {},
        root_started="linux:root",
        linux_subreaper_active=True,
    )
    accumulations = 0
    reap_calls = 0
    post_reap_inventories = iter(({100, 301}, {100}))

    def accumulate(_custody):
        nonlocal accumulations
        accumulations += 1

    def reap(_custody):
        nonlocal reap_calls
        reap_calls += 1

    monkeypatch.setattr(B, "_accumulate_custody", accumulate)
    monkeypatch.setattr(B, "_signal_custody_descendants", lambda *_a, **_k: None)
    monkeypatch.setattr(B, "_custody_descendants_absent", lambda _c: True)
    monkeypatch.setattr(B, "_direct_child_exit_observed", lambda _pid: True)
    monkeypatch.setattr(B, "_reap_adopted_linux_descendants", reap)
    monkeypatch.setattr(
        B, "_scoped_child_pids", lambda _pid: set(next(post_reap_inventories))
    )

    B._seal_descendants_before_root_reap(custody, timeout_seconds=1)
    assert reap_calls == 2
    assert accumulations == 5


def test_descendant_seal_holds_root_until_exit_is_observed(monkeypatch):
    custody = B._StartedProcessCustody(
        SimpleNamespace(pid=100, returncode=None),
        {},
        root_started="root:one",
    )
    root_observations = iter((False, True))
    accumulation_calls = 0

    def accumulate(_custody):
        nonlocal accumulation_calls
        accumulation_calls += 1

    monkeypatch.setattr(B, "_accumulate_custody", accumulate)
    monkeypatch.setattr(B, "_signal_custody_descendants", lambda *_a, **_k: None)
    monkeypatch.setattr(B, "_custody_descendants_absent", lambda _c: True)
    monkeypatch.setattr(
        B,
        "_direct_child_exit_observed",
        lambda _pid: next(root_observations),
    )
    monkeypatch.setattr(B, "_reap_adopted_linux_descendants", lambda _c: None)
    monkeypatch.setattr(B.time, "sleep", lambda _seconds: None)

    B._seal_descendants_before_root_reap(custody, timeout_seconds=1)
    # One traversal waits for the root; the second plus its confirmation
    # traversal closes the descendant fixed point after reparenting.
    assert accumulation_calls == 3


def test_scoped_lineage_rejects_root_and_descendant_pid_reuse(monkeypatch):
    root_pid = 100
    root = (9, 100, 100, "R", "root:old")
    monkeypatch.setattr(B, "_scoped_group_pids", lambda _pgid: set())
    monkeypatch.setattr(B, "_scoped_child_pids", lambda _pid: set())
    monkeypatch.setattr(
        B, "_process_identity", lambda _pid: (*root[:4], "root:new")
    )
    with pytest.raises(
        B.SupervisorContractError, match="root changed its birth identity"
    ):
        B._accumulate_related_identities(root_pid, root[4], {})

    identities = {201: "child:old"}

    def reused_child(pid):
        if pid == root_pid:
            return root
        assert pid == 201
        return (root_pid, root_pid, root_pid, "R", "child:new")

    monkeypatch.setattr(B, "_process_identity", reused_child)
    with pytest.raises(
        B.SupervisorContractError, match="changed its birth identity"
    ):
        B._accumulate_related_identities(
            root_pid, root[4], identities
        )
    assert identities == {201: "child:old"}


def test_scoped_lineage_rechecks_parent_after_child_sample(monkeypatch):
    root_pid = 100
    root_old = (9, 100, 100, "R", "root:old")
    root_new = (9, 100, 100, "R", "root:new")
    root_calls = 0

    def identity(pid):
        nonlocal root_calls
        if pid == root_pid:
            root_calls += 1
            return root_new if root_calls >= 3 else root_old
        assert pid == 201
        return (root_pid, 100, 100, "R", "child:one")

    monkeypatch.setattr(B, "_process_identity", identity)
    monkeypatch.setattr(B, "_scoped_group_pids", lambda _pgid: set())
    monkeypatch.setattr(
        B,
        "_scoped_child_pids",
        lambda pid: {201} if pid == root_pid else set(),
    )
    with pytest.raises(
        B.SupervisorContractError, match="parent changed"
    ):
        B._accumulate_related_identities(root_pid, root_old[4], {})


def test_waitid_wnowait_retries_interrupt_without_reaping(monkeypatch):
    calls = []

    def waitid(kind, pid, flags):
        calls.append((kind, pid, flags))
        if len(calls) == 1:
            raise InterruptedError
        return SimpleNamespace(si_pid=pid)

    monkeypatch.setattr(B.os, "waitid", waitid)
    assert B._direct_child_exit_observed(321) is True
    assert len(calls) == 2
    assert calls[-1][2] & B.os.WNOWAIT


def test_darwin_group_signal_eperm_needs_scoped_absence(monkeypatch):
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Darwin")
    )
    monkeypatch.setattr(
        B.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError()),
    )
    monkeypatch.setattr(
        B, "_process_group_has_live_members", lambda _pgid: False
    )
    B._signal_owned_process_group(99, 15)
    monkeypatch.setattr(
        B, "_process_group_has_live_members", lambda _pgid: True
    )
    with pytest.raises(
        B.SupervisorContractError, match="cannot signal"
    ):
        B._signal_owned_process_group(99, 15)
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )
    monkeypatch.setattr(
        B, "_process_group_has_live_members", lambda _pgid: False
    )
    with pytest.raises(
        B.SupervisorContractError, match="cannot signal"
    ):
        B._signal_owned_process_group(99, 15)


def test_darwin_detached_descendant_is_never_raw_signaled(monkeypatch):
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Darwin")
    )
    monkeypatch.setattr(
        B,
        "_process_identity",
        lambda _pid: (1, 777, 777, "R", "darwin:birth"),
    )
    monkeypatch.setattr(
        B.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Darwin raw numeric PID signal")
        ),
    )
    identities = {201: "darwin:birth"}

    # TERM grace delegates cleanup to the trusted direct runner without a
    # TOCTOU-prone numeric signal.  Final containment must then fail closed.
    B._signal_exact_processes(
        identities,
        B.signal.SIGTERM,
        None,
        owned_pgid=100,
        final=False,
    )
    with pytest.raises(
        B.ScopedProcessContainmentError,
        match="cannot safely signal a detached numeric PID",
    ):
        B._signal_exact_processes(
            identities,
            B.signal.SIGKILL,
            None,
            owned_pgid=100,
            final=True,
        )


def test_postreap_emergency_is_observation_only(monkeypatch):
    custody = B._StartedProcessCustody(
        SimpleNamespace(pid=99, returncode=0),
        {101: "child:one"},
    )
    monkeypatch.setattr(
        B,
        "_signal_owned_process_group",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("post-reap numeric group signal")
        ),
    )
    monkeypatch.setattr(
        B,
        "_signal_exact_processes",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("post-reap exact PID signal")
        ),
    )

    def prove(selected):
        selected.sealed = True

    monkeypatch.setattr(B, "_prove_postreap_process_absence", prove)
    B._emergency_contain_started_process(custody)
    assert custody.sealed is True


def test_emergency_seals_before_single_wait_without_popen_kill(monkeypatch):
    events = []

    class FakeProcess:
        pid = 500
        returncode = None

        def kill(self):
            raise AssertionError("Popen.kill reaped or polled the root")

        def wait(self, *, timeout):
            assert timeout == 10
            events.append("wait")
            self.returncode = -9
            return self.returncode

    custody = B._StartedProcessCustody(
        FakeProcess(), {}, root_started="root:one"
    )
    monkeypatch.setattr(
        B,
        "_signal_owned_process_group",
        lambda _pgid, signum: events.append(("group", signum)),
    )
    monkeypatch.setattr(
        B,
        "_seal_descendants_before_root_reap",
        lambda _custody: events.append("seal-descendants"),
    )
    monkeypatch.setattr(
        B,
        "_prove_postreap_process_absence",
        lambda _custody: events.append("prove-absence"),
    )
    monkeypatch.setattr(
        B,
        "_release_custody_kernel_state",
        lambda _custody: events.append("release-kernel-state"),
    )

    B._emergency_contain_started_process(custody)
    assert custody.sealed is True
    assert events == [
        ("group", B.signal.SIGKILL),
        "seal-descendants",
        "wait",
        "prove-absence",
        "release-kernel-state",
    ]


def test_clean_exit_seals_group_before_one_deliberate_reap(monkeypatch):
    events = []

    class FakeProcess:
        pid = 500
        returncode = None

        def wait(self, *, timeout):
            assert timeout == 10
            assert self.returncode is None
            events.append("wait")
            self.returncode = 0
            return 0

    process = FakeProcess()
    custody = B._StartedProcessCustody(process, {})

    def identity(pid):
        assert pid == process.pid
        if process.returncode is not None:
            return None
        return (os.getpid(), pid, pid, "R", "root:one")

    monkeypatch.setattr(B, "_process_identity", identity)
    monkeypatch.setattr(
        B, "_accumulate_related_identities", lambda *_args: None
    )
    monkeypatch.setattr(
        B, "_direct_child_exit_observed", lambda _pid: True
    )
    monkeypatch.setattr(
        B,
        "_signal_owned_process_group",
        lambda _pgid, signum: events.append(("group", signum)),
    )
    monkeypatch.setattr(
        B, "_signal_exact_processes", lambda *_args: None
    )
    monkeypatch.setattr(
        B.os, "uname", lambda: SimpleNamespace(sysname="Linux")
    )

    def prove(selected):
        events.append("prove")
        selected.sealed = True

    monkeypatch.setattr(B, "_prove_postreap_process_absence", prove)
    monkeypatch.setattr(
        B, "_read_bounded_process_stream", lambda *_args, **_kwargs: ""
    )
    result = B._supervise_started_process(
        custody,
        argv=("/bin/true",),
        timeout_seconds=1,
        started_at_ns=1,
        stdout_file=SimpleNamespace(),
        stderr_file=SimpleNamespace(),
    )
    assert result.returncode == 0
    assert events == [
        ("group", 15),
        ("group", 9),
        "wait",
        "prove",
    ]


def test_bounded_process_stream_rejects_hard_overflow(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(B, "MAX_CONTROL_SUITE_STREAM_BYTES", 8)
    path = tmp_path / "captured-stream"
    path.write_bytes(b"123456789")
    with path.open("r+b") as stream:
        with pytest.raises(
            B.SupervisorContractError, match="hard byte bound"
        ):
            B._read_bounded_process_stream(stream, label="stdout")


def test_postlaunch_observer_failure_kills_and_reaps_owned_group(
    tmp_path, monkeypatch
):
    launched = []
    real_popen = B.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    def fail_observation(*_args, **_kwargs):
        raise B.SupervisorContractError("injected scoped observer failure")

    monkeypatch.setattr(B.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(B, "_accumulate_related_identities", fail_observation)
    with pytest.raises(
        B.ScopedProcessContainmentError,
        match="launch could not prove containment",
    ):
        B._run_bounded_process_group(
            ("/bin/sleep", "30"),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout_seconds=10,
        )
    assert len(launched) == 1
    assert launched[0].returncode is not None
    observed = B._process_identity(launched[0].pid)
    assert observed is None or observed[3].startswith("Z")


def test_postlaunch_keyboard_interrupt_kills_and_reaps_owned_group(
    tmp_path, monkeypatch
):
    launched = []
    real_popen = B.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(B.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        B,
        "_accumulate_related_identities",
        lambda *_args: (_ for _ in ()).throw(
            KeyboardInterrupt("injected interrupt")
        ),
    )
    with pytest.raises(
        B.ScopedProcessContainmentError,
        match="launch could not prove containment",
    ):
        B._run_bounded_process_group(
            ("/bin/sleep", "30"),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout_seconds=10,
        )
    assert len(launched) == 1
    assert launched[0].returncode is not None


def test_pending_sigint_between_fork_and_factory_return_is_contained(
    tmp_path, monkeypatch
):
    launched = []
    real_popen = B.subprocess.Popen

    class InterruptBeforeReturnPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            launched.append(self)
            # _start_scoped_process has SIGINT blocked here.  Delivery occurs
            # only after the child has a bound custody object.
            B.signal.raise_signal(B.signal.SIGINT)

    monkeypatch.setattr(B.subprocess, "Popen", InterruptBeforeReturnPopen)
    with pytest.raises(KeyboardInterrupt):
        B.ScopedProcessTree.launch(
            ("/bin/sleep", "30"),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
        )

    assert len(launched) == 1
    assert launched[0].returncode is not None
    observed = B._process_identity(launched[0].pid)
    assert observed is None or observed[3].startswith("Z")


def test_popen_init_baseexception_after_fork_is_contained(
    tmp_path, monkeypatch
):
    launched = []
    real_popen = B.subprocess.Popen

    class ConstructorFailurePopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            launched.append(self)
            raise RuntimeError("injected constructor failure after fork")

    monkeypatch.setattr(B.subprocess, "Popen", ConstructorFailurePopen)
    with pytest.raises(RuntimeError, match="constructor failure after fork"):
        B.ScopedProcessTree.launch(
            ("/bin/sleep", "30"),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
        )

    assert len(launched) == 1
    assert launched[0].returncode is not None
    observed = B._process_identity(launched[0].pid)
    assert observed is None or observed[3].startswith("Z")


def test_bounded_runner_contains_start_return_handoff_failure(
    tmp_path, monkeypatch
):
    real_start = B._start_scoped_process
    launched = []

    def fail_after_custody_publish(*args, **kwargs):
        custody = real_start(*args, **kwargs)
        launched.append(custody.process)
        raise KeyboardInterrupt("injected start-return handoff failure")

    monkeypatch.setattr(
        B, "_start_scoped_process", fail_after_custody_publish
    )
    with pytest.raises(KeyboardInterrupt, match="handoff failure"):
        B._run_bounded_process_group(
            ("/bin/sleep", "30"),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout_seconds=10,
        )

    assert len(launched) == 1
    assert launched[0].returncode is not None
    observed = B._process_identity(launched[0].pid)
    assert observed is None or observed[3].startswith("Z")


def test_postlaunch_cleanup_failure_surfaces_unproven(
    tmp_path, monkeypatch
):
    process = SimpleNamespace(pid=500, returncode=None)
    monkeypatch.setattr(
        B,
        "_process_identity",
        lambda _pid: (1, 1, 1, "R", "preflight"),
    )
    monkeypatch.setattr(B, "_scoped_child_pids", lambda _pid: set())
    monkeypatch.setattr(B.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        B,
        "_supervise_started_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            B.SupervisorContractError("injected observation failure")
        ),
    )
    monkeypatch.setattr(
        B,
        "_emergency_contain_started_process",
        lambda *_args: (_ for _ in ()).throw(
            B.SupervisorContractError("injected containment failure")
        ),
    )
    with pytest.raises(
        B.ScopedProcessContainmentError,
        match="launch could not prove containment",
    ):
        B._run_bounded_process_group(
            ("/bin/true",),
            cwd=tmp_path,
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout_seconds=1,
        )


def test_bounded_control_suite_accepts_path_resolved_argv0(tmp_path):
    result = B._run_bounded_process_group(
        ("true",),
        cwd=tmp_path,
        environment={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        timeout_seconds=2,
    )
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.captured_descendants_absent is True


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_normal_exit_seals_same_group_child_missed_by_lineage_poll(
    tmp_path, monkeypatch
):
    script = tmp_path / "same_group_orphan.py"
    pid_path = tmp_path / "same-group.pid"
    script.write_text(
        "\n".join((
            "import os",
            "import signal",
            "import sys",
            "import time",
            "child = os.fork()",
            "if child == 0:",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    with open(sys.argv[1], 'w', encoding='ascii') as out:",
            "        out.write(str(os.getpid()))",
            "        out.flush()",
            "        os.fsync(out.fileno())",
            "    while True:",
            "        time.sleep(1)",
            "while not os.path.exists(sys.argv[1]):",
            "    time.sleep(0.01)",
            "time.sleep(0.1)",
        ))
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        B, "_accumulate_related_identities", lambda *_args: None
    )
    result = B._run_bounded_process_group(
        (
            str(PYTHON_EXECUTABLE),
            "-I",
            "-E",
            "-s",
            "-B",
            str(script),
            str(pid_path),
        ),
        cwd=tmp_path,
        environment={
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": str(tmp_path),
        },
        timeout_seconds=3,
    )
    assert result.returncode == 0
    assert result.captured_descendant_count == 0
    child_pid = int(pid_path.read_text(encoding="ascii"))
    observed = B._process_identity(child_pid)
    assert observed is None or observed[3].startswith("Z")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_bounded_control_suite_kills_setsid_grandchild_and_reaps_parent(
    tmp_path,
):
    script = tmp_path / "escaped_grandchild.py"
    pid_path = tmp_path / "grandchild.pid"
    release_path = tmp_path / "release-grandchild"
    exited_path = tmp_path / "grandchild-exiting"
    script.write_text(
        "\n".join(
            (
                "import os",
                "import signal",
                "import sys",
                "import time",
                "child = os.fork()",
                "if child == 0:",
                "    os.setsid()",
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "    with open(sys.argv[1], 'w', encoding='ascii') as out:",
                "        out.write(str(os.getpid()))",
                "        out.flush()",
                "        os.fsync(out.fileno())",
                "    while not os.path.exists(sys.argv[2]):",
                "        time.sleep(1)",
                "    with open(sys.argv[3], 'wb') as out:",
                "        out.write(b'exiting\\n')",
                "        out.flush()",
                "        os.fsync(out.fileno())",
                "    raise SystemExit(0)",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "while True:",
                "    time.sleep(1)",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    command = (
        str(PYTHON_EXECUTABLE),
        "-I",
        "-E",
        "-s",
        "-B",
        str(script),
        str(pid_path),
        str(release_path),
        str(exited_path),
    )
    environment = {
        "HOME": str(tmp_path),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": str(tmp_path),
    }
    if os.uname().sysname == "Darwin":
        try:
            with pytest.raises(
                B.SupervisorContractError,
                match="could not prove containment",
            ):
                B._run_bounded_process_group(
                    command,
                    cwd=tmp_path,
                    environment=environment,
                    timeout_seconds=0.75,
                )
        finally:
            # Never strand the intentionally uncontained Darwin synthetic if
            # the assertion itself fails before the normal cleanup line.
            release_path.write_bytes(b"release\n")
            cleanup_deadline = time.monotonic() + 10
            while pid_path.exists() and not exited_path.exists():
                if time.monotonic() >= cleanup_deadline:
                    break
                time.sleep(0.01)
        deadline = time.monotonic() + 10
        while not exited_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # Darwin may retain an orphan zombie briefly.  The child-authored,
        # fsynced terminal marker proves the synthetic escapee executed its
        # cooperative cleanup path without relying on PID-reuse-prone polling.
        assert exited_path.read_bytes() == b"exiting\n"
        return
    result = B._run_bounded_process_group(
        command,
        cwd=tmp_path,
        environment=environment,
        timeout_seconds=0.75,
    )
    assert result.timed_out is True
    assert result.captured_descendant_count == 1
    assert result.captured_descendants_absent is True
    grandchild_pid = int(pid_path.read_text(encoding="ascii"))
    observed = B._process_identity(grandchild_pid)
    assert observed is None or observed[3].startswith("Z")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.skipif(
    os.uname().sysname != "Linux",
    reason="detached descendant sealing requires Linux pidfd/subreaper",
)
def test_scoped_process_tree_seals_sigterm_resistant_child_and_descendant(
    tmp_path,
):
    script = tmp_path / "stubborn_scoped_tree.py"
    child_path = tmp_path / "stubborn-child.pid"
    script.write_text(
        "\n".join((
            "import os",
            "import signal",
            "import sys",
            "import time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "child = os.fork()",
            "if child == 0:",
            "    os.setsid()",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    with open(sys.argv[1], 'w', encoding='ascii') as out:",
            "        out.write(str(os.getpid()))",
            "        out.flush()",
            "        os.fsync(out.fileno())",
            "    while True:",
            "        time.sleep(1)",
            "while True:",
            "    time.sleep(1)",
        )) + "\n",
        encoding="utf-8",
    )
    tree = B.ScopedProcessTree.launch(
        (
            str(PYTHON_EXECUTABLE),
            "-I",
            "-E",
            "-s",
            "-B",
            str(script),
            str(child_path),
        ),
        cwd=tmp_path,
        environment={
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": str(tmp_path),
        },
    )
    deadline = time.monotonic() + 5
    while not child_path.exists():
        assert time.monotonic() < deadline
        assert tree.observe_exit() is False
        time.sleep(0.01)
    # One final sample binds the setsid descendant before quarantine starts.
    assert tree.observe_exit() is False
    result = tree.seal(stop_requested=True, grace_seconds=0.2)

    assert result.returncode == -9
    assert result.forced_kill is True
    assert result.captured_descendant_count >= 1
    assert result.captured_descendants_absent is True
    child_pid = int(child_path.read_text(encoding="ascii"))
    observed = B._process_identity(child_pid)
    assert observed is None or observed[3].startswith("Z")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.skipif(
    os.uname().sysname != "Linux",
    reason="detached descendant sealing requires Linux pidfd/subreaper",
)
def test_scoped_process_tree_seals_descendant_after_root_exit_race(tmp_path):
    script = tmp_path / "exited_root_scoped_tree.py"
    child_path = tmp_path / "exited-root-child.pid"
    script.write_text(
        "\n".join((
            "import os",
            "import signal",
            "import sys",
            "import time",
            "child = os.fork()",
            "if child == 0:",
            "    os.setsid()",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    with open(sys.argv[1], 'w', encoding='ascii') as out:",
            "        out.write(str(os.getpid()))",
            "        out.flush()",
            "        os.fsync(out.fileno())",
            "    while True:",
            "        time.sleep(1)",
            "while not os.path.exists(sys.argv[1]):",
            "    time.sleep(0.01)",
            "time.sleep(0.15)",
        )) + "\n",
        encoding="utf-8",
    )
    tree = B.ScopedProcessTree.launch(
        (
            str(PYTHON_EXECUTABLE),
            "-I",
            "-E",
            "-s",
            "-B",
            str(script),
            str(child_path),
        ),
        cwd=tmp_path,
        environment={
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": str(tmp_path),
        },
    )
    deadline = time.monotonic() + 5
    exited = False
    while not exited:
        assert time.monotonic() < deadline
        exited = tree.observe_exit()
        time.sleep(0.01)
    result = tree.seal(stop_requested=True, grace_seconds=0.1)

    assert result.returncode == 0
    assert result.forced_kill is True
    assert result.captured_descendant_count >= 1
    assert result.captured_descendants_absent is True
    child_pid = int(child_path.read_text(encoding="ascii"))
    observed = B._process_identity(child_pid)
    assert observed is None or observed[3].startswith("Z")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.skipif(
    os.uname().sysname != "Linux",
    reason="detached descendant sealing requires Linux pidfd/subreaper",
)
def test_bounded_control_suite_rejects_normal_exit_orphan(tmp_path):
    script = tmp_path / "normal_exit_orphan.py"
    pid_path = tmp_path / "orphan.pid"
    script.write_text(
        "\n".join((
            "import os",
            "import signal",
            "import sys",
            "import time",
            "child = os.fork()",
            "if child == 0:",
            "    os.setsid()",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    with open(sys.argv[1], 'w', encoding='ascii') as out:",
            "        out.write(str(os.getpid()))",
            "        out.flush()",
            "        os.fsync(out.fileno())",
            "    while True:",
            "        time.sleep(1)",
            "while not os.path.exists(sys.argv[1]):",
            "    time.sleep(0.01)",
            "time.sleep(0.2)",
        ))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        B.SupervisorContractError,
        match="normally exited control suite left a descendant",
    ):
        B._run_bounded_process_group(
            (
                str(PYTHON_EXECUTABLE),
                "-I",
                "-E",
                "-s",
                "-B",
                str(script),
                str(pid_path),
            ),
            cwd=tmp_path,
            environment={
                "HOME": str(tmp_path),
                "LANG": "C",
                "LC_ALL": "C",
                "TMPDIR": str(tmp_path),
            },
            timeout_seconds=3,
        )
    orphan_pid = int(pid_path.read_text(encoding="ascii"))
    observed = B._process_identity(orphan_pid)
    assert observed is None or observed[3].startswith("Z")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.skipif(
    os.uname().sysname != "Linux",
    reason="detached descendant sealing requires Linux pidfd/subreaper",
)
def test_bounded_control_suite_contains_reparented_double_fork(
    tmp_path,
):
    script = tmp_path / "double_fork.py"
    pid_path = tmp_path / "double-fork.pid"
    script.write_text(
        "\n".join((
            "import os",
            "import signal",
            "import sys",
            "import time",
            "child = os.fork()",
            "if child == 0:",
            "    grandchild = os.fork()",
            "    if grandchild == 0:",
            "        with open(sys.argv[1], 'w', encoding='ascii') as out:",
            "            out.write(str(os.getpid()))",
            "            out.flush()",
            "            os.fsync(out.fileno())",
            "        time.sleep(0.2)",
            "        os.setsid()",
            "        signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "        while True:",
            "            time.sleep(1)",
            "    time.sleep(0.35)",
            "    raise SystemExit(0)",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "while True:",
            "    time.sleep(1)",
        ))
        + "\n",
        encoding="utf-8",
    )
    result = B._run_bounded_process_group(
        (
            str(PYTHON_EXECUTABLE),
            "-I",
            "-E",
            "-s",
            "-B",
            str(script),
            str(pid_path),
        ),
        cwd=tmp_path,
        environment={
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": str(tmp_path),
        },
        timeout_seconds=0.8,
    )
    assert result.timed_out is True
    assert result.captured_descendant_count >= 2
    grandchild_pid = int(pid_path.read_text(encoding="ascii"))
    observed = B._process_identity(grandchild_pid)
    assert observed is None or observed[3].startswith("Z")


def test_runtime_control_suite_scratch_leak_blocks_and_is_removed(
    tmp_path, monkeypatch,
):
    roots = []
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "must-survive.txt"
    outside_marker.write_text("survive", encoding="utf-8")
    real_private_scratch = B._private_system_scratch

    def tracked_private_scratch():
        root = real_private_scratch()
        roots.append(root)
        return root

    def leak_scratch(**kwargs):
        scratch = Path(kwargs["scratch_root"])
        locked = scratch / "leaked" / "read-only"
        locked.mkdir(parents=True)
        sealed = locked / "sealed.txt"
        sealed.write_text("junk", encoding="utf-8")
        sealed.chmod(0o400)
        locked.chmod(0o500)
        (scratch / "escape").symlink_to(
            outside,
            target_is_directory=True,
        )
        return {}

    monkeypatch.setattr(
        B, "_private_system_scratch", tracked_private_scratch
    )
    monkeypatch.setattr(
        B, "_run_control_suite_with_scratch", leak_scratch
    )
    with pytest.raises(
        B.SupervisorContractError,
        match="leaked scratch entries",
    ):
        B._run_control_suite(
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=Path("/unused"),
        )
    assert len(roots) == 1
    assert not roots[0].exists()
    assert outside_marker.read_text(encoding="utf-8") == "survive"


def test_private_system_scratch_sealing_failure_removes_residue(
    monkeypatch,
):
    created_roots = []
    private_tmp = Path("/private/tmp")
    parent = private_tmp if private_tmp.exists() else Path("/tmp")
    real_mkdir = B.os.mkdir

    def tracked_mkdir(path, mode=0o777, *, dir_fd=None):
        result = real_mkdir(path, mode, dir_fd=dir_fd)
        if (
            dir_fd is not None
            and isinstance(path, str)
            and path.startswith("a3c_")
        ):
            created_roots.append(parent / path)
        return result

    def fail_created_root_fchmod(_descriptor, _mode):
        if created_roots:
            raise PermissionError("injected scratch sealing failure")
        raise AssertionError("scratch fchmod preceded scratch creation")

    monkeypatch.setattr(B.os, "mkdir", tracked_mkdir)
    monkeypatch.setattr(B.os, "fchmod", fail_created_root_fchmod)
    with pytest.raises(
        B.SupervisorContractError,
        match="private scratch creation failed",
    ):
        B._private_system_scratch()
    assert len(created_roots) == 1
    assert not created_roots[0].exists()


def test_private_system_scratch_parent_rebind_cleans_by_descriptor(
    monkeypatch,
):
    created_roots = []
    private_tmp = Path("/private/tmp")
    parent = private_tmp if private_tmp.exists() else Path("/tmp")
    real_mkdir = B.os.mkdir
    real_lstat = B.os.lstat
    parent_lstat_count = 0

    def tracked_mkdir(path, mode=0o777, *, dir_fd=None):
        result = real_mkdir(path, mode, dir_fd=dir_fd)
        if (
            dir_fd is not None
            and isinstance(path, str)
            and path.startswith("a3c_")
        ):
            created_roots.append(parent / path)
        return result

    def rebound_parent_lstat(path):
        nonlocal parent_lstat_count
        observed = real_lstat(path)
        if Path(path) == parent:
            parent_lstat_count += 1
            if parent_lstat_count >= 2:
                fields = list(observed)
                fields[1] += 1
                return os.stat_result(fields)
        return observed

    monkeypatch.setattr(B.os, "mkdir", tracked_mkdir)
    monkeypatch.setattr(B.os, "lstat", rebound_parent_lstat)
    with pytest.raises(
        B.SupervisorContractError,
        match="not a shallow owned directory",
    ):
        B._private_system_scratch()
    assert parent_lstat_count >= 2
    assert len(created_roots) == 1
    assert not created_roots[0].exists()


def test_private_system_scratch_rejects_unsticky_group_writable_parent(
    monkeypatch,
):
    private_tmp = Path("/private/tmp")
    parent = private_tmp if private_tmp.exists() else Path("/tmp")
    real_lstat = B.os.lstat
    mkdir_called = False

    def unsafe_parent_lstat(path):
        observed = real_lstat(path)
        if Path(path) == parent:
            fields = list(observed)
            fields[0] = (
                (fields[0] | stat.S_IWGRP)
                & ~stat.S_ISVTX
            )
            return os.stat_result(fields)
        return observed

    def forbidden_mkdir(*_args, **_kwargs):
        nonlocal mkdir_called
        mkdir_called = True
        raise AssertionError("unsafe parent reached scratch creation")

    monkeypatch.setattr(B.os, "lstat", unsafe_parent_lstat)
    monkeypatch.setattr(B.os, "mkdir", forbidden_mkdir)
    with pytest.raises(
        B.SupervisorContractError,
        match="system scratch parent is unsafe",
    ):
        B._private_system_scratch()
    assert mkdir_called is False


def _checkpoint(game: str = "wa30", reached: int = 2) -> dict:
    records = [
        {"level": level, "marginal_C": level + 2, "reached": True}
        for level in range(1, reached + 1)
    ]
    return {
        "game": game,
        "reached": reached,
        "total_marginal_C": sum(row["marginal_C"] for row in records),
        "records": records,
        "final_path": [1, 2, 3],
        "validated": True,
    }


def _parent_checkpoint(game: str, reached: int) -> dict:
    value = _checkpoint(game=game, reached=reached)
    if reached == 0:
        value.update(
            total_marginal_C=0,
            records=[],
            final_path=[],
        )
    else:
        value["final_path"] = list(range(1, reached + 1))
    return value


def _promotion_bundle(
    root: Path,
    name: str,
    *,
    level: int,
    solver_text: str,
    parent_checkpoint_path: Path | None = None,
    parent_actions: list[int] | None = None,
) -> tuple[Path, dict]:
    if parent_checkpoint_path is None:
        parent_checkpoint_path = root / f"{name}-parent-checkpoint.json"
        parent = _parent_checkpoint("wa30", level - 1)
        if parent_actions is not None:
            parent["final_path"] = parent_actions
        _write_json(parent_checkpoint_path, parent)
    parent_checkpoint_sha256 = hashlib.sha256(
        parent_checkpoint_path.read_bytes()
    ).hexdigest()
    parent_checkpoint = json.loads(
        parent_checkpoint_path.read_text(encoding="utf-8")
    )
    parent_action_count = len(parent_checkpoint["final_path"])
    remaining_action_budget = B.MAX_REPLAY_ACTIONS - parent_action_count

    source = root / f"{name}-source"
    source.mkdir()
    parent_source = parent_checkpoint_path.parent
    parent_evidence = parent_source / "promotion_evidence"
    if level > 1 and parent_evidence.is_dir():
        shutil.copytree(parent_evidence, source / "promotion_evidence")
    exact_path = list(range(1, level + 1))
    checkpoint = _checkpoint(reached=level)
    checkpoint["final_path"] = exact_path
    _write_json(source / B.CHECKPOINT_NAME, checkpoint)
    source_payloads = {
        "legs.py": "LEGS = ()\n",
        "players.py": "PLAYERS = ()\n",
        "solve.py": solver_text,
        # Retain the legacy fixture name for the mutation/atomicity tests that
        # exercise arbitrary optional source files in addition to the shared
        # closed three-file core.
        "solver.py": solver_text,
    }
    for filename, payload in source_payloads.items():
        (source / filename).write_text(payload, encoding="utf-8")
    evidence = source / "promotion_evidence" / f"level_{level:02d}"
    evidence.mkdir(parents=True)
    winning = evidence / "winning_source.py"
    winning.write_text(solver_text, encoding="utf-8")
    evidence_files = evidence / "files"
    evidence_files.mkdir()
    shutil.copy2(
        source / B.CHECKPOINT_NAME,
        evidence_files / B.CHECKPOINT_NAME,
    )
    manifest = evidence / "manifest.json"
    parent_manifest = (
        source / "promotion_evidence"
        / f"level_{level - 1:02d}" / "manifest.json"
    )
    _write_json(
        manifest,
        {
            "schema": 1,
            "game": "wa30",
            "level": level,
            "parent_manifest": (
                parent_manifest.relative_to(source).as_posix()
                if level > 1 and parent_manifest.is_file()
                else None
            ),
            "parent_manifest_sha256": (
                hashlib.sha256(parent_manifest.read_bytes()).hexdigest()
                if level > 1 and parent_manifest.is_file()
                else None
            ),
            "validated": True,
            "taint_verdict": "clean",
            "promoted_files_sha256": {
                B.CHECKPOINT_NAME: hashlib.sha256(
                    (evidence_files / B.CHECKPOINT_NAME).read_bytes()
                ).hexdigest()
            },
        },
    )

    output = root / f"{name}-output"
    output.mkdir()
    exported_hashes = {}
    for filename, payload in source_payloads.items():
        exported = output / filename
        exported.write_text(payload, encoding="utf-8")
        exported_hashes[filename] = hashlib.sha256(
            exported.read_bytes()
        ).hexdigest()
    _write_json(
        output / B.CANDIDATE_NAME,
        {
            "schema": 1,
            "game": "wa30",
            "target_level": level,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "candidate_path": exact_path,
            "exported_files_sha256": exported_hashes,
        },
    )
    receipt = root / f"{name}-receipt.json"
    _write_json(
        receipt,
        {
            "schema": 1,
            "game": "wa30",
            "target_level": level,
            "authoritative_target": 9,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_action_count": parent_action_count,
            "remaining_action_budget": remaining_action_budget,
            "fresh_prefix_required": remaining_action_budget == 0,
            "candidate_manifest_sha256": hashlib.sha256(
                (output / B.CANDIDATE_NAME).read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(
                (source / B.CHECKPOINT_NAME).read_bytes()
            ).hexdigest(),
            "source_tree_sha256": B._tree_hash(source),
            "winning_source_path": winning.relative_to(source).as_posix(),
            "winning_source_sha256": hashlib.sha256(
                winning.read_bytes()
            ).hexdigest(),
            "promotion_manifest_path": manifest.relative_to(source).as_posix(),
            "promotion_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "exact_path": exact_path,
            "checks": {
                check: True for check in B.REQUIRED_PROMOTION_CHECKS
            },
        },
    )
    frontier = B.FrontierAdmission(
        game="wa30",
        reached=level - 1,
        next_level=level,
        authoritative_target=9,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        parent_action_count=parent_action_count,
        remaining_action_budget=remaining_action_budget,
        fresh_prefix_required=remaining_action_budget == 0,
    )
    return source, {
        "receipt_path": receipt,
        "frontier": frontier,
        "parent_checkpoint_path": parent_checkpoint_path,
        "candidate_output_root": output,
    }


def test_inventory_is_exact_not_merely_a_total():
    valid = B.authoritative_inventory()
    B.validate_inventory(valid)
    with pytest.raises(B.SupervisorContractError, match="25 games / 183"):
        B.validate_inventory({"one": 183})
    redistributed = dict(valid)
    redistributed["re86"] += 1
    redistributed["wa30"] -= 1
    with pytest.raises(
        B.SupervisorContractError, match="exactly match authoritative"
    ):
        B.validate_inventory(redistributed)
    malformed = dict(valid)
    malformed["re86"] = "8"
    with pytest.raises(B.SupervisorContractError, match="invalid entries"):
        B.validate_inventory(malformed)


def test_soft_allocation_drains_active_turn_without_signalling_it():
    running = B.decide_turn_drain(
        elapsed_seconds=89 * 60,
        soft_allocation_seconds=90 * 60,
        proposer_active=True,
    )
    assert running == B.TurnDrainDecision(
        phase="proposing",
        launch_new_turn=False,
        request_container_stop=False,
        force_container_teardown=False,
    )

    for elapsed in (90 * 60, 181 * 60, 24 * 60 * 60):
        draining = B.decide_turn_drain(
            elapsed_seconds=elapsed,
            soft_allocation_seconds=90 * 60,
            proposer_active=True,
        )
        assert draining == B.TurnDrainDecision(
            phase="draining",
            launch_new_turn=False,
            request_container_stop=False,
            force_container_teardown=False,
        )

    completed = B.decide_turn_drain(
        elapsed_seconds=181 * 60,
        soft_allocation_seconds=90 * 60,
        proposer_active=False,
    )
    assert completed == B.TurnDrainDecision(
        phase="allocation_complete",
        launch_new_turn=False,
        request_container_stop=False,
        force_container_teardown=False,
    )


def test_container_teardown_requires_independent_containment_fault_and_grace():
    stopping = B.decide_turn_drain(
        elapsed_seconds=10,
        soft_allocation_seconds=180 * 60,
        proposer_active=True,
        containment_fault=True,
    )
    assert stopping == B.TurnDrainDecision(
        phase="containment_stopping",
        launch_new_turn=False,
        request_container_stop=True,
        force_container_teardown=False,
    )

    teardown = B.decide_turn_drain(
        elapsed_seconds=10,
        soft_allocation_seconds=180 * 60,
        proposer_active=True,
        containment_fault=True,
        containment_grace_expired=True,
    )
    assert teardown == B.TurnDrainDecision(
        phase="containment_teardown",
        launch_new_turn=False,
        request_container_stop=True,
        force_container_teardown=True,
    )

    with pytest.raises(
        B.SupervisorContractError,
        match="without a containment fault",
    ):
        B.decide_turn_drain(
            elapsed_seconds=181 * 60,
            soft_allocation_seconds=180 * 60,
            proposer_active=True,
            containment_grace_expired=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("elapsed_seconds", -1),
        ("elapsed_seconds", float("nan")),
        ("soft_allocation_seconds", 0),
        ("soft_allocation_seconds", float("inf")),
        ("proposer_active", 1),
        ("containment_fault", 0),
    ),
)
def test_turn_drain_policy_rejects_malformed_inputs(field, value):
    arguments = {
        "elapsed_seconds": 1,
        "soft_allocation_seconds": 2,
        "proposer_active": True,
        "containment_fault": False,
    }
    arguments[field] = value
    with pytest.raises(B.SupervisorContractError, match=field):
        B.decide_turn_drain(**arguments)


def test_checkpoint_rejects_path_only_and_accounting_inconsistency(tmp_path):
    path = tmp_path / "checkpoint.json"
    _write_json(path, {"game": "wa30", "final_path": [1, 2], "validated": True})
    with pytest.raises(B.SupervisorContractError, match="schema mismatch"):
        B.load_trusted_checkpoint(
            path, expected_game="wa30", authoritative_target=9
        )
    value = _checkpoint()
    value["total_marginal_C"] += 1
    _write_json(path, value)
    with pytest.raises(B.SupervisorContractError, match="marginal total"):
        B.load_trusted_checkpoint(
            path, expected_game="wa30", authoritative_target=9
        )


def test_checkpoint_accepts_public_coordinate_actions_and_rejects_bad_tokens(
        tmp_path):
    path = tmp_path / "checkpoint.json"
    value = _checkpoint(reached=1)
    value["final_path"] = [4, [6, 45, 21], 2]
    _write_json(path, value)
    checkpoint = B.load_trusted_checkpoint(
        path, expected_game="wa30", authoritative_target=9
    )
    assert checkpoint.final_path == (4, [6, 45, 21], 2)

    for bad in (
        True,
        0,
        6,
        8,
        [6, 1],
        [5, 1, 2],
        [6, True, 2],
        [6, -1, 2],
        [6, 64, 2],
        [6, 2, 64],
    ):
        value["final_path"] = [bad]
        _write_json(path, value)
        with pytest.raises(B.SupervisorContractError, match="replay action"):
            B.load_trusted_checkpoint(
                path, expected_game="wa30", authoritative_target=9
            )


def test_authoritative_inventory_checkpoint_contract_is_game_agnostic(
    tmp_path,
):
    for game, target in B.authoritative_inventory().items():
        path = tmp_path / game / B.CHECKPOINT_NAME
        _write_json(
            path,
            _checkpoint(game=game, reached=min(1, target)),
        )
        checkpoint = B.load_trusted_checkpoint(
            path,
            expected_game=game,
            authoritative_target=target,
        )
        assert checkpoint.reached <= target


def test_checkpoint_requires_real_path_and_regular_host_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    value = _checkpoint(reached=1)
    value["final_path"] = []
    _write_json(path, value)
    with pytest.raises(B.SupervisorContractError, match="no replay path"):
        B.load_trusted_checkpoint(
            path, expected_game="wa30", authoritative_target=9
        )
    link = tmp_path / "checkpoint-link.json"
    link.symlink_to(path)
    with pytest.raises(B.SupervisorContractError, match="regular host-owned"):
        B.load_trusted_checkpoint(
            link, expected_game="wa30", authoritative_target=9
        )


def test_checkpoint_rejects_nonsequential_and_over_target(tmp_path):
    path = tmp_path / "checkpoint.json"
    value = _checkpoint(reached=2)
    value["records"][1]["level"] = 3
    _write_json(path, value)
    with pytest.raises(B.SupervisorContractError, match="exactly 1..2"):
        B.load_trusted_checkpoint(
            path, expected_game="wa30", authoritative_target=9
        )
    value = _checkpoint(reached=2)
    value["reached"] = 10
    _write_json(path, value)
    with pytest.raises(B.SupervisorContractError, match="outside"):
        B.load_trusted_checkpoint(
            path, expected_game="wa30", authoritative_target=9
        )


def test_frontier_admission_refuses_re86_l9_and_nonsequential_dispatch(tmp_path):
    path = tmp_path / "checkpoint.json"
    _write_json(path, _checkpoint(game="re86", reached=8))
    with pytest.raises(B.SupervisorContractError, match="already complete"):
        B.admit_next_frontier(
            path, expected_game="re86", requested_level=9
        )

    _write_json(path, _checkpoint(game="re86", reached=6))
    with pytest.raises(B.SupervisorContractError, match="nonsequential"):
        B.admit_next_frontier(
            path, expected_game="re86", requested_level=8
        )
    admitted = B.admit_next_frontier(
        path, expected_game="re86", requested_level=7
    )
    assert admitted.next_level == 7
    assert admitted.authoritative_target == 8
    assert admitted.parent_checkpoint_sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_exhausted_parent_admission_requires_fresh_prefix(tmp_path):
    path = tmp_path / "checkpoint.json"
    value = _checkpoint(game="wa30", reached=1)
    value["final_path"] = [1] * B.MAX_REPLAY_ACTIONS
    _write_json(path, value)

    admitted = B.admit_next_frontier(
        path, expected_game="wa30", requested_level=2
    )
    assert admitted.parent_action_count == 600
    assert admitted.remaining_action_budget == 0
    assert admitted.fresh_prefix_required is True

    value["final_path"].append(1)
    _write_json(path, value)
    with pytest.raises(B.SupervisorContractError, match="600-action cap"):
        B.admit_next_frontier(
            path, expected_game="wa30", requested_level=2
        )


def test_candidate_output_rejects_checkpoint_and_undeclared_file(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    solver = output / "legs.py"
    solver.write_text("def solve(): pass\n", encoding="utf-8")
    digest = hashlib.sha256(solver.read_bytes()).hexdigest()
    manifest = {
        "schema": 1,
        "game": "wa30",
        "target_level": 3,
        "parent_checkpoint_sha256": "a" * 64,
        "candidate_path": [1, 2, 3],
        "exported_files_sha256": {"legs.py": digest},
    }
    _write_json(output / B.CANDIDATE_NAME, manifest)
    B.validate_candidate_output(
        output,
        expected_game="wa30",
        expected_level=3,
        parent_checkpoint_sha256="a" * 64,
    )
    (output / "surprise.txt").write_text("smuggled", encoding="utf-8")
    with pytest.raises(B.SupervisorContractError, match="undeclared"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=3,
            parent_checkpoint_sha256="a" * 64,
        )
    (output / "surprise.txt").unlink()
    manifest["exported_files_sha256"] = {"checkpoint.json": "b" * 64}
    _write_json(output / B.CANDIDATE_NAME, manifest)
    with pytest.raises(B.SupervisorContractError, match="invalid declared"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=3,
            parent_checkpoint_sha256="a" * 64,
        )


def test_exact_bundle_manifests_are_atomic_and_fail_closed(tmp_path):
    archive = tmp_path / "archive"
    nested = archive / "nested"
    nested.mkdir(parents=True)
    evidence = nested / "evidence.json"
    evidence.write_text('{"exact":true}\n', encoding="utf-8")
    receipt = E.write_manifest_atomic(
        archive,
        bundle_id="test-exact-bundle",
    )
    assert receipt["status"] == "PASS"
    assert receipt["file_count"] == 1
    assert E.verify_manifest(
        archive,
        expected_bundle_id="test-exact-bundle",
    )["status"] == "PASS"

    changed = tmp_path / "changed"
    shutil.copytree(archive, changed)
    (changed / "nested" / "evidence.json").write_text(
        '{"exact":false}\n', encoding="utf-8"
    )
    with pytest.raises(E.ExactBundleError, match="hash differs"):
        E.verify_manifest(changed)

    missing = tmp_path / "missing"
    shutil.copytree(archive, missing)
    (missing / "nested" / "evidence.json").unlink()
    with pytest.raises(E.ExactBundleError, match="file set differs"):
        E.verify_manifest(missing)

    extra = tmp_path / "extra"
    shutil.copytree(archive, extra)
    (extra / "unlisted.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(E.ExactBundleError, match="file set differs"):
        E.verify_manifest(extra)

    stale = tmp_path / "stale"
    shutil.copytree(archive, stale)
    manifest_path = stale / E.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files_sha256"]["nested/evidence.json"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(E.ExactBundleError, match="hash differs"):
        E.verify_manifest(stale)

    directory_extra = tmp_path / "directory-extra"
    shutil.copytree(archive, directory_extra)
    (directory_extra / "empty").mkdir()
    with pytest.raises(E.ExactBundleError, match="directory set differs"):
        E.verify_manifest(directory_extra)

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "evidence.json").write_text(
        '{"exact":true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="injected"):
        E.write_manifest_atomic(
            interrupted,
            bundle_id="test-interrupted-bundle",
            fault_at="after_file_sync",
        )
    assert not (interrupted / E.MANIFEST_NAME).exists()
    assert not list(interrupted.glob(".*.pending-*"))

    archive_root = tmp_path / "closed-archive"
    first = archive_root / "first"
    second = archive_root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "evidence.json").write_text("{}\n", encoding="utf-8")
    (second / "evidence.json").write_text("{}\n", encoding="utf-8")
    E.write_manifest_atomic(first, bundle_id="first")
    E.write_manifest_atomic(second, bundle_id="second")
    archive_result = E.verify_archive(archive_root)
    assert archive_result["status"] == "PASS"
    assert archive_result["bundle_count"] == 2
    assert archive_result["retention_policy"] == {
        "max_files_per_bundle": 256,
        "max_directories_per_bundle": 64,
        "max_bytes_per_bundle": 16 * 1024 * 1024,
        "stale_operational_paths_forbidden": True,
    }

    stale_archive = tmp_path / "stale-archive"
    stale_bundle = stale_archive / "stale"
    stale_workspace = stale_bundle / "workspace"
    stale_workspace.mkdir(parents=True)
    (stale_workspace / "latest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    E.write_manifest_atomic(stale_bundle, bundle_id="stale")
    with pytest.raises(
        E.ExactBundleError, match="stale operational paths"
    ):
        E.verify_archive(stale_archive)

    unsealed = archive_root / "unsealed"
    unsealed.mkdir()
    with pytest.raises(E.ExactBundleError):
        E.verify_archive(archive_root)

    live_output = tmp_path / "live-output"
    live_output.mkdir()
    solver = live_output / "legs.py"
    solver.write_text("x = 1\n", encoding="utf-8")
    digest = hashlib.sha256(solver.read_bytes()).hexdigest()
    live_manifest = {
        "schema": 1,
        "game": "wa30",
        "target_level": 1,
        "parent_checkpoint_sha256": "a" * 64,
        "candidate_path": [1],
        "exported_files_sha256": {"legs.py": digest},
    }
    _write_json(live_output / B.CANDIDATE_NAME, live_manifest)
    B.validate_candidate_output(
        live_output,
        expected_game="wa30",
        expected_level=1,
        parent_checkpoint_sha256="a" * 64,
    )
    solver.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(B.SupervisorContractError, match="hash mismatch"):
        B.validate_candidate_output(
            live_output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )


def test_candidate_output_rejects_stale_parent_and_symlink(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    solver = output / "legs.py"
    solver.write_text("x = 1\n", encoding="utf-8")
    _write_json(
        output / B.CANDIDATE_NAME,
        {
            "schema": 1,
            "game": "wa30",
            "target_level": 3,
            "parent_checkpoint_sha256": "a" * 64,
            "candidate_path": [1],
            "exported_files_sha256": {
                "legs.py": hashlib.sha256(solver.read_bytes()).hexdigest()
            },
        },
    )
    with pytest.raises(B.SupervisorContractError, match="stale/wrong parent"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=3,
            parent_checkpoint_sha256="b" * 64,
        )
    (output / "escape").symlink_to(solver)
    with pytest.raises(B.SupervisorContractError, match="symlink"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=3,
            parent_checkpoint_sha256="a" * 64,
        )


def test_candidate_output_rejects_hardlinked_artifact_entry(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    solver = output / "legs.py"
    alias = output / "alias.py"
    solver.write_text("x = 1\n", encoding="utf-8")
    os.link(solver, alias)
    digest = hashlib.sha256(solver.read_bytes()).hexdigest()
    _write_json(
        output / B.CANDIDATE_NAME,
        {
            "schema": 1,
            "game": "wa30",
            "target_level": 1,
            "parent_checkpoint_sha256": "a" * 64,
            "candidate_path": [1],
            "exported_files_sha256": {
                "legs.py": digest,
                "alias.py": digest,
            },
        },
    )
    with pytest.raises(B.SupervisorContractError, match="hard-linked"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )


def test_candidate_output_enforces_host_work_quotas(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    solver = output / "legs.py"
    solver.write_text("x = 1\n", encoding="utf-8")
    _write_json(
        output / B.CANDIDATE_NAME,
        {
            "schema": 1,
            "game": "wa30",
            "target_level": 1,
            "parent_checkpoint_sha256": "a" * 64,
            "candidate_path": [1],
            "exported_files_sha256": {
                "legs.py": hashlib.sha256(solver.read_bytes()).hexdigest()
            },
        },
    )
    monkeypatch.setattr(B, "MAX_CANDIDATE_FILES", 1)
    with pytest.raises(B.SupervisorContractError, match="file-count quota"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )

    monkeypatch.setattr(B, "MAX_CANDIDATE_FILES", 256)
    monkeypatch.setattr(B, "MAX_CANDIDATE_TOTAL_BYTES", 1)
    with pytest.raises(B.SupervisorContractError, match="total-byte quota"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )

    monkeypatch.setattr(B, "MAX_CANDIDATE_TOTAL_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(B, "MAX_CANDIDATE_DEPTH", 1)
    nested = output / "nested"
    nested.mkdir()
    (nested / "probe.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(B.SupervisorContractError, match="path-depth quota"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )
    (nested / "probe.py").unlink()
    nested.rmdir()
    monkeypatch.setattr(B, "MAX_CANDIDATE_DEPTH", 8)
    sparse = output / "sparse.bin"
    with sparse.open("wb") as handle:
        handle.seek(1024 * 1024)
        handle.write(b"x")
    if sparse.stat().st_blocks * 512 < sparse.stat().st_size:
        with pytest.raises(B.SupervisorContractError, match="sparse file"):
            B.validate_candidate_output(
                output,
                expected_game="wa30",
                expected_level=1,
                parent_checkpoint_sha256="a" * 64,
            )


def test_candidate_schema_rejects_boolean_aliases_and_accepts_coordinate_actions(
        tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    solver = output / "legs.py"
    solver.write_text("x = 1\n", encoding="utf-8")
    manifest = {
        "schema": 1,
        "game": "wa30",
        "target_level": 1,
        "parent_checkpoint_sha256": "a" * 64,
        "candidate_path": [[6, 45, 21], 4],
        "exported_files_sha256": {
            "legs.py": hashlib.sha256(solver.read_bytes()).hexdigest()
        },
    }
    _write_json(output / B.CANDIDATE_NAME, manifest)
    assert B.validate_candidate_output(
        output,
        expected_game="wa30",
        expected_level=1,
        parent_checkpoint_sha256="a" * 64,
    ).candidate_path == ([6, 45, 21], 4)
    for bad in ([6, -1, 21], [6, 64, 21], [6, 45, 64]):
        manifest["candidate_path"] = [bad]
        _write_json(output / B.CANDIDATE_NAME, manifest)
        with pytest.raises(B.SupervisorContractError, match="replay action"):
            B.validate_candidate_output(
                output,
                expected_game="wa30",
                expected_level=1,
                parent_checkpoint_sha256="a" * 64,
            )
    manifest["candidate_path"] = [[6, 45, 21], 4]
    manifest["schema"] = True
    _write_json(output / B.CANDIDATE_NAME, manifest)
    with pytest.raises(B.SupervisorContractError, match="schema mismatch"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )
    manifest["schema"] = 1
    manifest["target_level"] = True
    _write_json(output / B.CANDIDATE_NAME, manifest)
    with pytest.raises(B.SupervisorContractError, match="wrong frontier"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )
    manifest["target_level"] = 1
    manifest["candidate_path"] = [1] * (B.MAX_REPLAY_ACTIONS + 1)
    _write_json(output / B.CANDIDATE_NAME, manifest)
    with pytest.raises(B.SupervisorContractError, match="at most 600"):
        B.validate_candidate_output(
            output,
            expected_game="wa30",
            expected_level=1,
            parent_checkpoint_sha256="a" * 64,
        )


def test_game_lock_rejects_live_owner_and_reclaims_stale_file(tmp_path):
    root = tmp_path / "supervisor"
    with B.GameLock(root, "wa30"):
        with pytest.raises(B.SupervisorContractError, match="live supervisor"):
            with B.GameLock(root, "wa30"):
                pass

    lock_path = root / "locks" / "wa30.lock"
    lock_path.write_text("pid=999999999\n", encoding="utf-8")
    with B.GameLock(root, "wa30"):
        assert lock_path.read_text(encoding="utf-8").startswith("pid=")


def test_operator_lease_rejects_live_second_owner_and_is_bounded(
    tmp_path,
):
    root = tmp_path / "campaign"
    first = B.OperatorLease(
        root,
        operator_configuration_sha256="a" * 64,
        acquire_timeout_seconds=0.05,
        heartbeat_interval_seconds=60,
    ).acquire()
    try:
        first.assert_healthy()
        with pytest.raises(
            B.SupervisorContractError,
            match="another live contiguous operator",
        ):
            B.OperatorLease(
                root,
                operator_configuration_sha256="a" * 64,
                acquire_timeout_seconds=0.05,
                heartbeat_interval_seconds=60,
            ).acquire()
        with first._state_lock:
            for _index in range(6):
                first._publish_heartbeat(status="ACTIVE")
        heartbeat_names = sorted(
            path.name
            for path in first.root.glob("heartbeat_*.json")
        )
        assert heartbeat_names == [
            "heartbeat_0.json",
            "heartbeat_1.json",
        ]
        assert first.current_path.is_file()
    finally:
        first.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_operator_lease_authenticates_stale_takeover_without_signalling_pid(
    tmp_path,
):
    root = tmp_path / "campaign"
    read_descriptor, write_descriptor = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_descriptor)
        try:
            lease = B.OperatorLease(
                root,
                operator_configuration_sha256="b" * 64,
                heartbeat_interval_seconds=60,
            ).acquire()
            os.write(
                write_descriptor,
                lease.owner_instance_id.encode("ascii"),
            )
        finally:
            os.close(write_descriptor)
        os._exit(0)
    os.close(write_descriptor)
    prior_owner = os.read(read_descriptor, 64).decode("ascii")
    os.close(read_descriptor)
    waited, status = os.waitpid(child, 0)
    assert waited == child
    assert os.WIFEXITED(status)
    assert prior_owner

    with B.OperatorLease(
        root,
        operator_configuration_sha256="b" * 64,
        heartbeat_interval_seconds=60,
    ) as replacement:
        assert replacement.owner_instance_id != prior_owner
        acquisition, _sha256 = replacement._read_authenticated(
            replacement._acquisition_path,
            label="replacement acquisition",
        )
        assert acquisition["takeover"] is True
        assert acquisition["prior_owner_instance_id"] == prior_owner
        assert acquisition["prior_heartbeat_status"] == "ACTIVE"
        assert acquisition["signals_prior_pid"] is False
        assert acquisition["takeover_authority"] == (
            "kernel_lock_absence_plus_authenticated_prior_receipt"
        )


@pytest.mark.parametrize("target", ["current", "authentication_key"])
def test_operator_lease_rejects_tampered_recovery_authority(
    tmp_path, target
):
    root = tmp_path / "campaign"
    with B.OperatorLease(
        root,
        operator_configuration_sha256="c" * 64,
        heartbeat_interval_seconds=60,
    ):
        pass
    lease_root = root / B.OPERATOR_LEASE_ROOT_NAME
    if target == "current":
        current = lease_root / "current.json"
        value = json.loads(current.read_text(encoding="ascii"))
        value["status"] = "ACTIVE"
        current.write_text(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
    else:
        key = lease_root / "host_authentication.key"
        key.write_bytes(b"x" * 32)
        key.chmod(0o600)
    with pytest.raises(
        B.SupervisorContractError,
        match="host authenticated",
    ):
        B.OperatorLease(
            root,
            operator_configuration_sha256="c" * 64,
            heartbeat_interval_seconds=60,
        ).acquire()


def test_operator_lease_accounts_for_crash_orphaned_acquisition(
    tmp_path,
):
    root = tmp_path / "campaign"
    with B.OperatorLease(
        root,
        operator_configuration_sha256="d" * 64,
        heartbeat_interval_seconds=60,
    ):
        pass
    observer = B.OperatorLease(
        root,
        operator_configuration_sha256="d" * 64,
        heartbeat_interval_seconds=60,
    )
    key = (observer.root / "host_authentication.key").read_bytes()
    current = B.OperatorLease.observe_current(
        root,
        operator_configuration_sha256="d" * 64,
    )
    prior = current["acquisition"]
    orphan_owner = "e" * 32
    orphan_body = {
        **prior,
        "owner_instance_id": orphan_owner,
        "owner_pid": os.getpid(),
        "owner_process_start_identity_sha256":
            B._operator_lease_process_start_identity(os.getpid()),
        "acquisition_sequence": 1,
        "acquired_at_ns": int(prior["acquired_at_ns"]) + 1,
        "takeover": True,
        "prior_owner_instance_id": prior["owner_instance_id"],
        "prior_acquisition_sha256":
            current["acquisition_sha256"],
        "prior_heartbeat_sha256": current["heartbeat_sha256"],
        "prior_heartbeat_status": "RELEASED",
        "recovered_orphan_acquisition_sha256s": [],
    }
    orphan = {
        **orphan_body,
        "host_authentication_sha256":
            B._operator_lease_hmac(key, orphan_body),
    }
    orphan_raw = B._operator_lease_canonical_json(orphan) + b"\n"
    orphan_path = (
        observer.acquisitions / f"00000001-{orphan_owner}.json"
    )
    B._write_new_regular_bytes(
        orphan_path,
        orphan_raw,
        label="test orphan acquisition",
    )
    B._fsync_directory(observer.acquisitions)
    orphan_sha256 = hashlib.sha256(orphan_raw).hexdigest()

    with B.OperatorLease(
        root,
        operator_configuration_sha256="d" * 64,
        heartbeat_interval_seconds=60,
    ) as recovered:
        acquisition, _digest = recovered._read_authenticated(
            recovered._acquisition_path,
            label="recovered acquisition",
        )
        assert acquisition[
            "recovered_orphan_acquisition_sha256s"
        ] == [orphan_sha256]
        assert orphan_path.is_file()

    with B.OperatorLease(
        root,
        operator_configuration_sha256="d" * 64,
        heartbeat_interval_seconds=60,
    ) as next_owner:
        acquisition, _digest = next_owner._read_authenticated(
            next_owner._acquisition_path,
            label="next acquisition",
        )
        assert acquisition[
            "recovered_orphan_acquisition_sha256s"
        ] == []


def test_locks_reject_unknown_games_and_symlinked_lock_files(tmp_path):
    root = tmp_path / "supervisor"
    for invalid_game in ("../not-a-game", "../wa30"):
        with pytest.raises(B.SupervisorContractError, match="authoritative"):
            with B.GameLock(root, invalid_game):
                pass
    assert not (root / "wa30.lock").exists()

    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True)
    outside = tmp_path / "outside-lock"
    outside.write_text("do not touch\n", encoding="utf-8")
    (lock_dir / "wa30.lock").symlink_to(outside)
    with pytest.raises(B.SupervisorContractError, match="regular host-owned"):
        with B.GameLock(root, "wa30"):
            pass
    assert outside.read_text(encoding="utf-8") == "do not touch\n"

    linked_root = tmp_path / "linked-supervisor"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(B.SupervisorContractError, match="game-lock root"):
        with B.GameLock(linked_root, "wa30"):
            pass

    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / ".promotion.lock").symlink_to(outside)
    with pytest.raises(B.SupervisorContractError, match="regular host-owned"):
        with B.StoreLock(store_root):
            pass
    assert outside.read_text(encoding="utf-8") == "do not touch\n"


def test_locks_reject_hardlinked_lock_files(tmp_path):
    game_root = tmp_path / "game-lock-root"
    lock_dir = game_root / "locks"
    lock_dir.mkdir(parents=True)
    outside_game = tmp_path / "outside-game-lock"
    outside_game.write_text("untrusted alias\n", encoding="utf-8")
    os.link(outside_game, lock_dir / "wa30.lock")
    with pytest.raises(B.SupervisorContractError, match="unaliased regular"):
        with B.GameLock(game_root, "wa30"):
            pass

    store_root = tmp_path / "artifact-store"
    store_root.mkdir()
    outside_store = tmp_path / "outside-store-lock"
    outside_store.write_text("untrusted alias\n", encoding="utf-8")
    os.link(outside_store, store_root / ".promotion.lock")
    with pytest.raises(B.SupervisorContractError, match="unaliased regular"):
        with B.StoreLock(store_root):
            pass


def test_promotion_receipt_blocks_failed_gate_and_post_admission_mutation(
        tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "blocked", level=1,
        solver_text="clean",
    )
    receipt = kwargs["receipt_path"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["checks"]["transcript_taint"] = False
    _write_json(receipt, data)
    store = B.VersionedArtifactStore(tmp_path / "store")
    with pytest.raises(B.SupervisorContractError, match="transcript_taint"):
        store.publish(source, **kwargs)

    data["checks"]["transcript_taint"] = True
    _write_json(receipt, data)
    (source / "solver.py").write_text("mutated", encoding="utf-8")
    with pytest.raises(B.SupervisorContractError, match="source-tree"):
        store.publish(source, **kwargs)
    assert store.current() is None


def test_promotion_requires_path_and_source_replay_from_zero(tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "full-replay", level=1, solver_text="clean",
    )
    store = B.VersionedArtifactStore(tmp_path / "store")
    receipt = kwargs["receipt_path"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["checks"]["path_replay_from_zero"] = False
    _write_json(receipt, data)
    with pytest.raises(
        B.SupervisorContractError, match="path_replay_from_zero"
    ):
        store.publish(source, **kwargs)
    assert store.current() is None

    data["checks"]["path_replay_from_zero"] = True
    data["checks"]["source_replay_from_zero"] = False
    _write_json(receipt, data)
    with pytest.raises(
        B.SupervisorContractError, match="source_replay_from_zero"
    ):
        store.publish(source, **kwargs)
    assert store.current() is None

    # A private probe or nested clone may suggest a path, but that suggestion
    # is still only proposer-controlled candidate data.  Even with all host
    # check names set to PASS, a path that differs from the independently
    # replayed exact host boundary cannot become publication authority.
    clone_source, clone_kwargs = _promotion_bundle(
        tmp_path, "clone-claim", level=1, solver_text="clone claim",
    )
    clone_output = clone_kwargs["candidate_output_root"]
    clone_manifest = clone_output / B.CANDIDATE_NAME
    clone_data = json.loads(clone_manifest.read_text(encoding="utf-8"))
    clone_data["candidate_path"] = [2]
    _write_json(clone_manifest, clone_data)
    clone_receipt = clone_kwargs["receipt_path"]
    clone_receipt_data = json.loads(
        clone_receipt.read_text(encoding="utf-8")
    )
    clone_receipt_data["candidate_manifest_sha256"] = hashlib.sha256(
        clone_manifest.read_bytes()
    ).hexdigest()
    _write_json(clone_receipt, clone_receipt_data)
    with pytest.raises(
        B.SupervisorContractError,
        match="exact replay boundary is not a prefix",
    ):
        store.publish(clone_source, **clone_kwargs)
    assert store.current() is None

    # Nor may the candidate synthesize its own all-PASS "host" receipt.  The
    # receipt ingress is disjoint from every proposer-controlled root, so only
    # the existing host replay path can supply promotion evidence.
    forged_source, forged_kwargs = _promotion_bundle(
        tmp_path, "candidate-authored-receipt", level=1,
        solver_text="forged authority",
    )
    forged_output = forged_kwargs["candidate_output_root"]
    candidate_receipt = forged_output / "claimed_host_receipt.json"
    candidate_receipt.write_bytes(
        forged_kwargs["receipt_path"].read_bytes()
    )
    forged_kwargs["receipt_path"] = candidate_receipt
    with pytest.raises(
        B.SupervisorContractError,
        match="host promotion receipt must be outside proposer-controlled",
    ):
        store.publish(forged_source, **forged_kwargs)
    assert store.current() is None


def test_promotion_winning_snapshot_must_match_declared_candidate_export(
        tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "snapshot-link", level=1, solver_text="clean",
    )
    receipt = kwargs["receipt_path"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    winning = source / data["winning_source_path"]
    winning.write_text("stale or unrelated source\n", encoding="utf-8")
    data["winning_source_sha256"] = hashlib.sha256(
        winning.read_bytes()
    ).hexdigest()
    data["source_tree_sha256"] = B._tree_hash(source)
    _write_json(receipt, data)
    with pytest.raises(B.SupervisorContractError, match="declared candidate"):
        B.VersionedArtifactStore(tmp_path / "store").publish(source, **kwargs)


def test_promotion_receipt_rejects_manifest_that_does_not_bind_clean_checkpoint(
        tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "bad-manifest", level=1,
        solver_text="clean",
    )
    receipt = kwargs["receipt_path"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = source / data["promotion_manifest_path"]
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["taint_verdict"] = "dirty"
    _write_json(manifest, manifest_data)
    data["promotion_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    data["source_tree_sha256"] = B._tree_hash(source)
    _write_json(receipt, data)
    with pytest.raises(
        B.SupervisorContractError, match="manifest chain"
    ):
        B.VersionedArtifactStore(tmp_path / "store").publish(
            source, **kwargs
        )


def test_promotion_recomputes_complete_manifest_chain(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "chain-one", level=1, solver_text="one"
    )
    first = store.publish(source1, **kwargs1)
    source2, kwargs2 = _promotion_bundle(
        tmp_path,
        "chain-two",
        level=2,
        parent_checkpoint_path=(
            store.versions / first["version"] / B.CHECKPOINT_NAME
        ),
        solver_text="two",
    )

    # A receipt saying ``manifest_chain: true`` is not enough. Make every
    # remaining hash self-consistent after deleting L1; recomputation must
    # still reject the incomplete history.
    shutil.rmtree(source2 / "promotion_evidence" / "level_01")
    manifest = source2 / "promotion_evidence" / "level_02" / "manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["parent_manifest"] = None
    manifest_data["parent_manifest_sha256"] = None
    _write_json(manifest, manifest_data)
    receipt = kwargs2["receipt_path"]
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["promotion_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    receipt_data["source_tree_sha256"] = B._tree_hash(source2)
    _write_json(receipt, receipt_data)

    with pytest.raises(B.SupervisorContractError, match="exactly complete"):
        store.publish(source2, **kwargs2)
    assert store.current() == first


def test_promotion_revalidates_every_historical_boundary_file(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "history-one", level=1, solver_text="one"
    )
    first = store.publish(source1, **kwargs1)
    source2, kwargs2 = _promotion_bundle(
        tmp_path,
        "history-two",
        level=2,
        parent_checkpoint_path=(
            store.versions / first["version"] / B.CHECKPOINT_NAME
        ),
        solver_text="two",
    )

    boundary = (
        source2 / "promotion_evidence" / "level_01"
        / "files" / B.CHECKPOINT_NAME
    )
    boundary.unlink()
    receipt = kwargs2["receipt_path"]
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["source_tree_sha256"] = B._tree_hash(source2)
    _write_json(receipt, receipt_data)
    with pytest.raises(
        B.SupervisorContractError, match="evidence mismatch"
    ):
        store.publish(source2, **kwargs2)


def test_publish_revalidates_frontier_from_parent_checkpoint(tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "forged-frontier", level=1, solver_text="clean",
    )
    frontier = kwargs["frontier"]
    kwargs["frontier"] = B.FrontierAdmission(
        game=frontier.game,
        reached=frontier.reached,
        next_level=frontier.next_level,
        authoritative_target=frontier.authoritative_target,
        parent_checkpoint_sha256="f" * 64,
        parent_action_count=frontier.parent_action_count,
        remaining_action_budget=frontier.remaining_action_budget,
        fresh_prefix_required=frontier.fresh_prefix_required,
    )
    with pytest.raises(
        B.SupervisorContractError, match="host-owned parent checkpoint"
    ):
        B.VersionedArtifactStore(tmp_path / "store").publish(
            source, **kwargs
        )


def test_publish_rejects_overlapping_artifact_roots(tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "overlap", level=1, solver_text="clean",
    )
    store = B.VersionedArtifactStore(source / "store")
    with pytest.raises(B.SupervisorContractError, match="pairwise disjoint"):
        store.publish(source, **kwargs)


def test_publish_rejects_proposer_controlled_parent_checkpoint(tmp_path):
    source, kwargs = _promotion_bundle(
        tmp_path, "controlled-parent", level=1, solver_text="clean",
    )
    kwargs["parent_checkpoint_path"] = source / B.CHECKPOINT_NAME
    with pytest.raises(
        B.SupervisorContractError, match="proposer-controlled roots"
    ):
        B.VersionedArtifactStore(tmp_path / "store").publish(
            source, **kwargs
        )


def test_store_rejects_symlinked_root_and_internal_directories(tmp_path):
    real = tmp_path / "real-store"
    real.mkdir()
    linked = tmp_path / "linked-store"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(B.SupervisorContractError, match="store root"):
        B.VersionedArtifactStore(linked).current()

    store = B.VersionedArtifactStore(tmp_path / "store")
    store.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    store.versions.symlink_to(outside, target_is_directory=True)
    with pytest.raises(B.SupervisorContractError, match="versions"):
        store.current()


def test_store_and_locks_reject_symlinked_ancestor(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(B.SupervisorContractError, match="symlinked path component"):
        B.VersionedArtifactStore(linked_parent / "store").current()
    with pytest.raises(B.SupervisorContractError, match="symlinked path component"):
        with B.StoreLock(linked_parent / "store"):
            pass
    with pytest.raises(B.SupervisorContractError, match="symlinked path component"):
        with B.GameLock(linked_parent / "supervisor", "wa30"):
            pass


def test_store_refuses_branch_that_does_not_extend_selected_checkpoint(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "one", level=1,
        solver_text="one",
    )
    first = store.publish(source1, **kwargs1)
    source2, kwargs2 = _promotion_bundle(
        tmp_path, "wrong-parent", level=2,
        solver_text="two", parent_actions=[2],
    )
    with pytest.raises(B.SupervisorContractError, match="currently selected"):
        store.publish(source2, **kwargs2)
    assert store.current() == first


def test_store_requires_exact_selected_parent_not_byte_identical_copy(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "one", level=1, solver_text="one",
    )
    first = store.publish(source1, **kwargs1)
    selected_parent = (
        store.versions / first["version"] / B.CHECKPOINT_NAME
    )
    copied_parent = tmp_path / "copied-selected-checkpoint.json"
    copied_parent.write_bytes(selected_parent.read_bytes())
    source2, kwargs2 = _promotion_bundle(
        tmp_path,
        "copied-parent",
        level=2,
        parent_checkpoint_path=copied_parent,
        solver_text="two",
    )
    with pytest.raises(
        B.SupervisorContractError, match="exact checkpoint.*currently selected"
    ):
        store.publish(source2, **kwargs2)
    assert store.current() == first


def test_store_rejects_dangling_current_pointer_symlink(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    store.root.mkdir()
    store.pointer.symlink_to(tmp_path / "missing-pointer-target")
    with pytest.raises(B.SupervisorContractError, match="current pointer"):
        store.current()


def test_store_lock_blocks_concurrent_publish_and_recovery(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "one", level=1,
        solver_text="one",
    )
    with B.StoreLock(store.root):
        with pytest.raises(B.SupervisorContractError, match="live publisher"):
            store.publish(source, **kwargs)
        with pytest.raises(B.SupervisorContractError, match="live publisher"):
            store.recover()
    assert store.current() is None


def test_versioned_promotion_rolls_back_before_pointer_and_recovers(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "one", level=1,
        solver_text="one",
    )
    first = store.publish(source1, **kwargs1)
    assert store.current() == first

    source2, kwargs2 = _promotion_bundle(
        tmp_path, "two", level=2,
        parent_checkpoint_path=store.versions / first["version"]
        / B.CHECKPOINT_NAME,
        solver_text="two",
    )
    with pytest.raises(RuntimeError, match="pre-pointer"):
        store.publish(source2, fault_at="after_version", **kwargs2)
    assert store.current() == first
    assert store.recover() == first


def test_versioned_promotion_post_pointer_is_complete(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1,
        solver_text="complete",
    )
    with pytest.raises(RuntimeError, match="post-pointer"):
        store.publish(source, fault_at="after_pointer", **kwargs)
    current = store.current()
    assert current is not None
    selected = store.versions / current["version"]
    assert (selected / "solver.py").read_text(encoding="utf-8") == "complete"


def test_selected_version_retains_and_revalidates_candidate_manifest(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "candidate-evidence", level=1, solver_text="complete",
    )
    pointer = store.publish(source, **kwargs)
    selected = store.versions / pointer["version"]
    candidate = (
        selected / "promotion_evidence" / "level_01"
        / B.CANDIDATE_EVIDENCE_NAME
    )
    assert candidate.is_file()
    candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
    assert candidate_data["candidate_path"] == [1]

    # Even a self-consistent receipt/pointer rewrite cannot sever the selected
    # exact boundary from the candidate path that produced it.
    candidate_data["candidate_path"] = [2]
    _write_json(candidate, candidate_data)
    receipt = selected / B.HOST_RECEIPT_NAME
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["candidate_manifest_sha256"] = hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    _write_json(receipt, receipt_data)
    pointer["promotion_receipt_sha256"] = hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    pointer["tree_sha256"] = B._tree_hash(selected)
    _write_json(store.pointer, pointer)
    with pytest.raises(
        B.SupervisorContractError, match="candidate manifest"
    ):
        store.current()


def test_versioned_store_rejects_symlink_and_corrupt_pointer(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1,
        solver_text="complete",
    )
    target = source / "solver.py"
    (source / "escape").symlink_to(target)
    with pytest.raises(B.SupervisorContractError, match="symlink"):
        store.publish(source, **kwargs)

    store.root.mkdir(parents=True, exist_ok=True)
    _write_json(
        store.pointer,
        {"schema": 1, "version": "../escape", "tree_sha256": "a" * 64},
    )
    with pytest.raises(B.SupervisorContractError, match="pointer schema"):
        store.current()


def test_current_reparses_selected_checkpoint_against_authority(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1,
        solver_text="complete",
    )
    pointer = store.publish(source, **kwargs)
    selected = store.versions / pointer["version"]

    checkpoint = selected / B.CHECKPOINT_NAME
    malformed = json.loads(checkpoint.read_text(encoding="utf-8"))
    malformed.pop("reached")
    _write_json(checkpoint, malformed)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    receipt = selected / B.HOST_RECEIPT_NAME
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["checkpoint_sha256"] = checkpoint_hash
    _write_json(receipt, receipt_data)

    pointer["checkpoint_sha256"] = checkpoint_hash
    pointer["promotion_receipt_sha256"] = hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    pointer["tree_sha256"] = B._tree_hash(selected)
    _write_json(store.pointer, pointer)

    with pytest.raises(B.SupervisorContractError, match="schema mismatch"):
        store.current()


def test_current_revalidates_embedded_receipt_not_only_pointer_hash(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1, solver_text="complete",
    )
    pointer = store.publish(source, **kwargs)
    selected = store.versions / pointer["version"]
    receipt = selected / B.HOST_RECEIPT_NAME
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["checks"]["transcript_taint"] = False
    _write_json(receipt, receipt_data)

    # Simulate a self-consistent pointer rewrite. Recovery must still inspect
    # the receipt's actual gate schema/results rather than accept its new hash.
    pointer["promotion_receipt_sha256"] = hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    pointer["tree_sha256"] = B._tree_hash(selected)
    _write_json(store.pointer, pointer)
    with pytest.raises(B.SupervisorContractError, match="failed host checks"):
        store.current()


def test_partial_copy_never_changes_current_pointer(tmp_path):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source1, kwargs1 = _promotion_bundle(
        tmp_path, "one", level=1,
        solver_text="one",
    )
    first = store.publish(source1, **kwargs1)
    source2, kwargs2 = _promotion_bundle(
        tmp_path, "two", level=2,
        parent_checkpoint_path=store.versions / first["version"]
        / B.CHECKPOINT_NAME,
        solver_text="two",
    )
    with pytest.raises(RuntimeError, match="partial-copy"):
        store.publish(source2, fault_at="partial_copy", **kwargs2)
    assert store.current() == first


def test_raced_in_symlink_is_not_followed_during_promotion(
        tmp_path, monkeypatch):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1, solver_text="clean",
    )
    outside = tmp_path / "outside-secret"
    outside.write_text("must not be copied\n", encoding="utf-8")
    real_copytree = B.shutil.copytree

    def copytree_then_inject_link(source_path, stage_path, *args, **options):
        result = real_copytree(source_path, stage_path, *args, **options)
        if Path(stage_path).parent.name == "staging":
            staged_solver = Path(stage_path) / "solver.py"
            staged_solver.unlink()
            staged_solver.symlink_to(outside)
        return result

    monkeypatch.setattr(B.shutil, "copytree", copytree_then_inject_link)
    with pytest.raises(B.SupervisorContractError, match="staged promotion"):
        store.publish(source, **kwargs)
    assert store.current() is None


def test_raced_receipt_path_is_not_followed_or_replaced(tmp_path, monkeypatch):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1, solver_text="clean",
    )
    outside = tmp_path / "outside-receipt-target"
    outside.write_text("must remain unchanged\n", encoding="utf-8")
    real_tree_hash = B._tree_hash

    def hash_then_inject_receipt(root, **options):
        digest = real_tree_hash(root, **options)
        root = Path(root)
        receipt = root / B.HOST_RECEIPT_NAME
        if root.parent.name == "staging" and not receipt.exists():
            receipt.symlink_to(outside)
        return digest

    monkeypatch.setattr(B, "_tree_hash", hash_then_inject_receipt)
    with pytest.raises(B.SupervisorContractError, match="path appeared"):
        store.publish(source, **kwargs)
    assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"
    assert store.current() is None


def test_publish_flushes_version_and_pointer_before_acknowledgement(
        tmp_path, monkeypatch):
    store = B.VersionedArtifactStore(tmp_path / "store")
    source, kwargs = _promotion_bundle(
        tmp_path, "source", level=1,
        solver_text="durable",
    )
    events = []
    real_tree = B._fsync_tree
    real_directory = B._fsync_directory

    def record_tree(path):
        events.append(("tree", Path(path).name))
        real_tree(path)

    def record_directory(path):
        events.append(("directory", Path(path).name))
        real_directory(path)

    monkeypatch.setattr(B, "_fsync_tree", record_tree)
    monkeypatch.setattr(B, "_fsync_directory", record_directory)
    pointer = store.publish(source, **kwargs)
    assert store.current() == pointer
    assert events[0][0] == "tree"
    assert ("directory", "versions") in events
    assert ("directory", "store") in events
    assert events.index(("directory", "versions")) < events.index(
        ("directory", "store")
    )


def _terminal_stub(
    base: dict, tmp_path: Path, image_digest: str
) -> dict:
    return {
        **base,
        "launch_authority": True,
        "container_image_digest": image_digest,
        "frozen_release_receipt_path": str(
            tmp_path / "release.json"
        ),
        "frozen_release_receipt_sha256": "e" * 64,
        "frozen_release_levels": 183,
        "production_scenario_driver_receipt_path": str(
            tmp_path / "scenario_driver_receipt.json"
        ),
        "production_scenario_driver_receipt_sha256": "f" * 64,
        "production_scenario_receipts_sha256": "d" * 64,
        "production_scenario_verification_environment_sha256":
            "b" * 64,
        "terminal_evidence_sha256": "c" * 64,
    }


def _launch_receipt_inputs(
    terminal: dict, tmp_path: Path
) -> dict:
    runtime_manifest = (tmp_path / "python-runtime.json").resolve()
    terminal.update({
        "suite_runtime_manifest_path": str(runtime_manifest),
        "suite_runtime_manifest_sha256": "7" * 64,
    })
    return {
        "python_runtime_manifest": runtime_manifest,
        "python_runtime_manifest_sha256": "7" * 64,
        "pilot_gate_receipt": (tmp_path / "pilot-gate.json").resolve(),
        "pilot_authentication_key": (tmp_path / "pilot.key").resolve(),
        "pilot_production_stack_attestation_sha256": "2" * 64,
    }


def _receipt_authority_fixture(tmp_path: Path) -> dict:
    runtime_manifest = (tmp_path / "runtime-manifest.json").resolve()
    release = (tmp_path / "release.json").resolve()
    scenario = (tmp_path / "scenario.json").resolve()
    control_root = (tmp_path / "controls").resolve()
    supplied = {
        "status": "PASS",
        "launch_authority": True,
        "registry_sha256": "1" * 64,
        "launch_requirements_sha256": "2" * 64,
        "control_contract_sha256": "3" * 64,
        "inventory_sha256": B.authoritative_inventory_sha256(),
        "container_image_digest": "sha256:" + "4" * 64,
        "frozen_release_receipt_path": str(release),
        "frozen_release_receipt_sha256": "5" * 64,
        "frozen_release_levels": 183,
        "production_scenario_driver_receipt_path": str(scenario),
        "production_scenario_driver_receipt_sha256": "6" * 64,
        "production_scenario_receipts_sha256": "7" * 64,
        "production_scenario_verification_environment_sha256": "8" * 64,
        "suite_execution_policy_sha256": "9" * 64,
        "scenario_receipts_sha256": {"S01": "a" * 64},
        "component_suite_inventory_sha256": "b" * 64,
        "component_suite_outcomes_sha256": "c" * 64,
        "suite_loaded_control_modules_sha256": "d" * 64,
        "suite_source_loaded_sha256": "e" * 64,
        "suite_interpreter_path": str(PYTHON_EXECUTABLE),
        "suite_interpreter_sha256": PYTHON_EXECUTABLE_SHA256,
        "suite_runtime_manifest_path": str(runtime_manifest),
        "suite_runtime_manifest_sha256": "f" * 64,
        "execution_control_root": str(control_root),
        "execution_control_snapshot_sha256": "3" * 64,
        "execution_control_snapshot_immutable": True,
        "workspace_root_inventory_start_sha256": "0" * 64,
        "workspace_root_inventory_end_sha256": "0" * 64,
        "games": 25,
        "levels": 183,
        "terminal_evidence_sha256": "a" * 64,
    }
    runtime = dict(supplied)
    runtime["terminal_evidence_sha256"] = "b" * 64
    gate_path = (tmp_path / "pilot-gate.json").resolve()
    pilot = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_pilot_gate",
        "status": "PASS",
        "full_campaign_launch_gate": "UNLOCKED",
        "pilot_games": ["ft09", "lp85"],
        "pilot_targets": [6, 8],
        "pilot_lineage_canonical": False,
        "image_digest": supplied["container_image_digest"],
        "control_contract_sha256": supplied[
            "control_contract_sha256"
        ],
        "production_stack_attestation_path": str(
            (tmp_path / "production_stack_attestation.json").resolve()
        ),
        "production_stack_attestation_sha256": "c" * 64,
        "pilot_manifest_sha256": P.PILOT_MANIFEST_SHA256,
        "receipt_sha256": "d" * 64,
        "file_sha256": "e" * 64,
        "path": str(gate_path),
        "meta_handoff_count": 1,
    }
    return {
        "targets": B.authoritative_inventory(),
        "supplied_terminal": supplied,
        "runtime_terminal": runtime,
        "pilot_gate": pilot,
        "pilot_gate_receipt": gate_path,
        "requested_image_digest": supplied["container_image_digest"],
        "python_runtime_manifest": runtime_manifest,
        "python_runtime_manifest_sha256": "f" * 64,
        "production_stack_attestation_sha256": "c" * 64,
    }


def _launch_validation_kwargs(tmp_path: Path) -> dict:
    return {
        "canonical_root": tmp_path,
        "environments_root": tmp_path,
        "repository": Path(__file__).resolve().parents[2],
    }


def test_launch_attestation_is_fail_closed(tmp_path):
    prelaunch = tmp_path / "prelaunch.json"
    _write_pass_conformance(prelaunch)
    with pytest.raises(
        B.SupervisorContractError,
        match="not genuine terminal conformance",
    ):
        B.validate_launch_attestation(
            prelaunch, **_launch_validation_kwargs(tmp_path)
        )

    asserted = tmp_path / "caller-booleans.json"
    _write_json(asserted, {
        "schema": 1,
        "image_digest": "sha256:" + "a" * 64,
        "suite_exit_code": 0,
        "isolation_checks": {"everything": True},
        "fault_checks": {"everything": True},
    })
    with pytest.raises(
        B.SupervisorContractError,
        match="not genuine terminal conformance",
    ):
        B.validate_launch_attestation(
            asserted, **_launch_validation_kwargs(tmp_path)
        )


def test_launch_attestation_requires_pid_isolation_and_safe_teardown(
    tmp_path,
):
    for asserted_value in (True, False):
        path = tmp_path / f"asserted-{asserted_value}.json"
        _write_json(path, {
            "schema": 1,
            "suite_exit_code": 0,
            "isolation_checks": {
                "private_pid_namespace": asserted_value,
                "no_host_pid_visibility": asserted_value,
                "supervisor_owned_timeout_teardown": asserted_value,
            },
            "fault_checks": {
                "detached_child_cannot_survive_container_exit":
                    asserted_value,
                "timeout_teardown_cannot_signal_sibling_lane":
                    asserted_value,
            },
        })
        with pytest.raises(
            B.SupervisorContractError,
            match="not genuine terminal conformance",
        ):
            B.validate_launch_attestation(
                path, **_launch_validation_kwargs(tmp_path)
            )


def test_launch_attestation_rejects_extra_checks_and_symlink(tmp_path):
    path = tmp_path / "prelaunch.json"
    _write_pass_conformance(path)
    link = tmp_path / "attestation-link.json"
    link.symlink_to(path)
    with pytest.raises(
        B.SupervisorContractError,
        match="not genuine terminal conformance",
    ):
        B.validate_launch_attestation(
            link, **_launch_validation_kwargs(tmp_path)
        )


def test_launch_preflight_binds_tested_image_and_current_control_tree(
        tmp_path, monkeypatch):
    path = tmp_path / "terminal-conformance.json"
    conformance = _write_pass_conformance(path)
    digest = "sha256:" + "a" * 64
    observed = []

    def pass_runtime_suite(**_kwargs):
        observed.append("PASS")
        return conformance

    terminal = _terminal_stub(conformance, tmp_path, digest)
    receipt_inputs = _launch_receipt_inputs(terminal, tmp_path)
    monkeypatch.setattr(
        B,
        "validate_launch_attestation",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        B.Conformance,
        "bind_terminal_launch_authority",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(B, "_run_control_suite", pass_runtime_suite)
    with pytest.raises(
        B.SupervisorContractError, match="exact live ft09 then lp85"
    ):
        B.launch_preflight(
            path,
            requested_image_digest=digest,
            conformance_result=path,
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
            **receipt_inputs,
        )
    assert observed == ["PASS"]
    with pytest.raises(B.SupervisorContractError, match="tested image"):
        B.launch_preflight(
            path,
            requested_image_digest="sha256:" + "b" * 64,
            conformance_result=path,
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
        )
    with pytest.raises(
        B.SupervisorContractError, match="one terminal conformance receipt"
    ):
        B.launch_preflight(
            path,
            requested_image_digest=digest,
            conformance_result=tmp_path / "another.json",
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
        )


def test_launch_preflight_requires_observed_runtime_suite(tmp_path, monkeypatch):
    path = tmp_path / "terminal-conformance.json"
    conformance = _write_pass_conformance(path)
    digest = "sha256:" + "a" * 64

    def fail_runtime_suite(**_kwargs):
        raise B.SupervisorContractError("observed suite failure")

    terminal = _terminal_stub(conformance, tmp_path, digest)
    receipt_inputs = _launch_receipt_inputs(terminal, tmp_path)
    monkeypatch.setattr(
        B,
        "validate_launch_attestation",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(B, "_run_control_suite", fail_runtime_suite)
    with pytest.raises(B.SupervisorContractError, match="observed suite"):
        B.launch_preflight(
            path,
            requested_image_digest=digest,
            conformance_result=path,
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
            **receipt_inputs,
        )


def test_launch_preflight_rejects_runtime_manifest_substitution(
    tmp_path, monkeypatch
):
    path = tmp_path / "terminal-conformance.json"
    digest = "sha256:" + "a" * 64
    prior = _terminal_stub(
        _write_pass_conformance(path), tmp_path, digest
    )
    prior.update({
        "suite_runtime_manifest_path":
            str(tmp_path / "approved-runtime.json"),
        "suite_runtime_manifest_sha256": "b" * 64,
    })
    monkeypatch.setattr(
        B,
        "validate_launch_attestation",
        lambda *_args, **_kwargs: prior,
    )
    monkeypatch.setattr(
        B,
        "_run_control_suite",
        lambda **_kwargs: pytest.fail(
            "runtime substitution must fail before execution"
        ),
    )
    receipt_inputs = _launch_receipt_inputs(prior, tmp_path)
    receipt_inputs.update({
        "python_runtime_manifest": (
            tmp_path / "substituted-runtime.json"
        ),
        "python_runtime_manifest_sha256": "c" * 64,
    })
    with pytest.raises(
        B.SupervisorContractError,
        match="Python runtime manifest",
    ):
        B.launch_preflight(
            path,
            requested_image_digest=digest,
            conformance_result=path,
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
            **receipt_inputs,
        )


def test_launch_preflight_rejects_prelaunch_only_conformance(
    tmp_path, monkeypatch
):
    path = tmp_path / "prelaunch.json"
    _write_pass_conformance(path)
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        B,
        "_run_control_suite",
        lambda **_kwargs: pytest.fail(
            "prelaunch-only receipt must fail before runtime execution"
        ),
    )
    with pytest.raises(
        B.SupervisorContractError,
        match="not genuine terminal conformance",
    ):
        B.launch_preflight(
            path,
            requested_image_digest=digest,
            conformance_result=path,
            canonical_root=tmp_path,
            environments_root=tmp_path,
            python_executable=PYTHON_EXECUTABLE,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            runtime_control_snapshot_root=tmp_path / "snapshot",
        )


def test_receipt_derived_launch_authority_has_no_static_latch(tmp_path):
    inputs = _receipt_authority_fixture(tmp_path)
    first = B._derive_receipt_launch_authority(**inputs)
    second = B._derive_receipt_launch_authority(**inputs)
    assert first == second
    assert first["status"] == "PASS"
    assert first["authority_source"] == "verified_receipts_only"
    assert first["games"] == 25
    assert first["levels"] == 183
    assert first["authority_sha256"] == hashlib.sha256(
        B._operator_lease_canonical_json({
            key: value
            for key, value in first.items()
            if key != "authority_sha256"
        })
    ).hexdigest()
    assert not hasattr(B, "CONTIGUOUS_LAUNCH_READY")


@pytest.mark.parametrize(
    ("scope", "field", "bad_value"),
    (
        ("both", "frozen_release_levels", 182),
        ("both", "frozen_release_receipt_sha256", None),
        (
            "runtime",
            "production_scenario_driver_receipt_sha256",
            "0" * 64,
        ),
        ("both", "production_scenario_receipts_sha256", "invalid"),
        (
            "both",
            "production_scenario_verification_environment_sha256",
            "invalid",
        ),
        ("runtime", "control_contract_sha256", "0" * 64),
        ("runtime", "registry_sha256", "0" * 64),
        ("both", "container_image_digest", "sha256:" + "0" * 64),
        ("both", "suite_runtime_manifest_sha256", "0" * 64),
        ("supplied", "terminal_evidence_sha256", "invalid"),
    ),
)
def test_receipt_derived_launch_authority_rejects_terminal_inverse(
    tmp_path, scope, field, bad_value
):
    inputs = _receipt_authority_fixture(tmp_path)
    selected = copy.deepcopy(inputs)
    if scope in {"both", "supplied"}:
        selected["supplied_terminal"][field] = bad_value
    if scope in {"both", "runtime"}:
        selected["runtime_terminal"][field] = bad_value
    with pytest.raises(B.SupervisorContractError):
        B._derive_receipt_launch_authority(**selected)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("status", "BLOCKED"),
        ("full_campaign_launch_gate", "LOCKED"),
        ("pilot_games", ["lp85", "ft09"]),
        ("pilot_targets", [6, 7]),
        ("image_digest", "sha256:" + "0" * 64),
        ("control_contract_sha256", "0" * 64),
        ("production_stack_attestation_sha256", "0" * 64),
        ("file_sha256", None),
        ("receipt_sha256", "invalid"),
        ("meta_handoff_count", 0),
    ),
)
def test_receipt_derived_launch_authority_rejects_pilot_inverse(
    tmp_path, field, bad_value
):
    inputs = _receipt_authority_fixture(tmp_path)
    inputs["pilot_gate"][field] = bad_value
    with pytest.raises(
        B.SupervisorContractError, match="pilot and production-stack"
    ):
        B._derive_receipt_launch_authority(**inputs)


def test_receipt_derived_launch_authority_rejects_inventory_and_runtime_args(
    tmp_path,
):
    inputs = _receipt_authority_fixture(tmp_path)
    inputs["targets"].pop("lf52")
    with pytest.raises(
        B.SupervisorContractError, match="another inventory"
    ):
        B._derive_receipt_launch_authority(**inputs)

    inputs = _receipt_authority_fixture(tmp_path)
    inputs["python_runtime_manifest_sha256"] = "0" * 64
    with pytest.raises(
        B.SupervisorContractError, match="exact runtime manifest"
    ):
        B._derive_receipt_launch_authority(**inputs)


def test_full_launch_reopens_exact_ordered_pilot_gate(
    tmp_path, monkeypatch
):
    attestation = tmp_path / "terminal-conformance.json"
    attestation.write_bytes(b"{}\n")
    attestation.chmod(0o400)
    digest = "sha256:" + "a" * 64
    prior = _terminal_stub(
        _write_pass_conformance(tmp_path / "base.json"),
        tmp_path,
        digest,
    )
    receipt_inputs = _launch_receipt_inputs(prior, tmp_path)
    observed = {
        "calls": 0,
        "terminal_reopens": 0,
        "runtime_reopens": 0,
    }

    def validate_terminal(*_args, **_kwargs):
        observed["terminal_reopens"] += 1
        return prior

    monkeypatch.setattr(
        B, "validate_launch_attestation", validate_terminal
    )
    monkeypatch.setattr(
        B,
        "_run_control_suite",
        lambda **_kwargs: {"execution_control_root": str(tmp_path)},
    )
    monkeypatch.setattr(
        B.Conformance,
        "bind_terminal_launch_authority",
        lambda *_args, **_kwargs: dict(prior),
    )
    def verify(
        receipt_path,
        *,
        authentication_key_path,
        expected_image_digest,
        expected_control_contract_sha256,
        expected_production_stack_attestation_sha256,
    ):
        observed.update({
            "calls": observed["calls"] + 1,
            "receipt": receipt_path,
            "key": authentication_key_path,
            "image": expected_image_digest,
            "control": expected_control_contract_sha256,
            "stack": expected_production_stack_attestation_sha256,
        })
        return {
            "schema": 1,
            "kind": "arc_agi3_contiguous_pilot_gate",
            "status": "PASS",
            "full_campaign_launch_gate": "UNLOCKED",
            "pilot_games": ["ft09", "lp85"],
            "pilot_targets": [6, 8],
            "pilot_lineage_canonical": False,
            "image_digest": digest,
            "control_contract_sha256": prior[
                "control_contract_sha256"
            ],
            "production_stack_attestation_path": str(
                (tmp_path / "production_stack_attestation.json").resolve()
            ),
            "production_stack_attestation_sha256": "2" * 64,
            "file_sha256": "1" * 64,
            "receipt_sha256": "3" * 64,
            "path": str(Path(receipt_path).resolve()),
            "pilot_manifest_sha256": P.PILOT_MANIFEST_SHA256,
            "meta_handoff_count": 1,
        }

    monkeypatch.setattr(P, "verify_pilot_gate_receipt", verify)

    def reopen_runtime(*_args, **_kwargs):
        observed["runtime_reopens"] += 1
        return {}

    monkeypatch.setattr(
        B.RuntimeManifest,
        "load_runtime_manifest",
        reopen_runtime,
    )
    result = B.launch_preflight(
        attestation,
        requested_image_digest=digest,
        conformance_result=attestation,
        canonical_root=tmp_path,
        environments_root=tmp_path,
        python_executable=PYTHON_EXECUTABLE,
        python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
        runtime_control_snapshot_root=tmp_path,
        **receipt_inputs,
    )
    assert observed == {
        "calls": 2,
        "terminal_reopens": 2,
        "runtime_reopens": 1,
        "receipt": receipt_inputs["pilot_gate_receipt"],
        "key": receipt_inputs["pilot_authentication_key"],
        "image": digest,
        "control": prior["control_contract_sha256"],
        "stack": "2" * 64,
    }
    assert result["pilot_gate_receipt_sha256"] == "1" * 64
    assert result["pilot_manifest_sha256"] == (
        P.PILOT_MANIFEST_SHA256
    )
    assert result["pilot_meta_handoff_count"] == 1
    assert result["launch_authority"] == "RECEIPT_DERIVED"
    assert result["launch_authority_evidence"]["games"] == 25
    assert result["launch_authority_evidence"]["levels"] == 183
    assert result["launch_authority_sha256"] == (
        result["launch_authority_evidence"]["authority_sha256"]
    )


def test_post_incident_meta_diagnostic_is_once_and_quarantine_only(
    tmp_path,
):
    diagnostic, driver, projection = _post_incident_meta_fixture(
        tmp_path
    )
    first = diagnostic.run_once(projection)
    second = diagnostic.run_once(projection)
    assert first == second
    assert driver.calls == 1
    assert first["status"] == "DIAGNOSED"
    assert first["diagnostic_available"] is True
    assert first["human_intervention_required"] is True
    assert first["runner_remained_paused"] is True
    assert first["meta_proposer_invocation_count"] == 1
    for field in (
        "scheduler_authority",
        "solver_authority",
        "wip_authority",
        "cost_authority",
        "retry_authority",
        "dispatch_authority",
        "promotion_authority",
    ):
        assert first[field] is False
    request = json.loads(
        diagnostic.request_path.read_text(encoding="ascii")
    )
    assert request["input_contains_game_source"] is False
    assert request["input_contains_wip"] is False
    assert request["input_contains_candidate"] is False
    assert request["result_authority"] == "quarantine_only"


def test_post_incident_meta_invalid_output_fails_to_paused_handoff(
    tmp_path,
):
    diagnostic, driver, projection = _post_incident_meta_fixture(
        tmp_path, valid=False
    )
    result = diagnostic.run_once(projection)
    assert driver.calls == 1
    assert result["status"] == "FAILED"
    assert result["failure_code"] == "invalid_response"
    assert result["diagnostic_available"] is False
    assert result["human_intervention_required"] is True
    assert result["runner_remained_paused"] is True
    assert result["retry_authority"] is False
    assert result["promotion_authority"] is False


def test_post_incident_meta_ambiguous_crash_never_reinvokes(
    tmp_path,
):
    diagnostic, driver, projection = _post_incident_meta_fixture(
        tmp_path
    )
    request, _raw, request_sha256 = diagnostic._request(projection)
    diagnostic._ensure_layout()
    B._post_incident_meta_write(
        diagnostic.request_path,
        request,
        label="test post-incident meta request",
    )
    B._post_incident_meta_write(
        diagnostic.intent_path,
        {
            "schema": B.POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_invocation_intent",
            "protocol_sha256":
                B.POST_INCIDENT_META_PROTOCOL_SHA256,
            "request_sha256": request_sha256,
            "incident_identity_sha256":
                diagnostic.incident_identity_sha256,
            "invocation_sequence": 1,
            "maximum_invocations": 1,
            "timeout_seconds": 60,
            "result_authority": "quarantine_only",
        },
        label="test post-incident meta intent",
    )
    result = diagnostic.run_once(projection)
    assert driver.calls == 0
    assert result["status"] == "AMBIGUOUS_INTERRUPTION"
    assert result["failure_code"] == (
        "operator_interrupted_during_driver"
    )
    assert diagnostic.run_once(projection) == result
    assert driver.calls == 0


def test_post_incident_meta_recovery_reopens_response_and_detects_tamper(
    tmp_path,
):
    diagnostic, _driver, projection = _post_incident_meta_fixture(
        tmp_path
    )
    diagnostic.run_once(projection)
    diagnostic.response_path.chmod(0o600)
    diagnostic.response_path.write_bytes(b'{"schema":1}\n')
    diagnostic.response_path.chmod(0o400)
    with pytest.raises(
        B.SupervisorContractError,
        match="response binding changed",
    ):
        diagnostic.run_once(projection)


def test_post_incident_meta_restart_reuses_episode_after_probe_progress(
    tmp_path,
):
    diagnostic, driver, projection = _post_incident_meta_fixture(
        tmp_path
    )
    first = diagnostic.run_once(projection)
    progressed = {
        **projection,
        "substrate_incident": {
            **projection["substrate_incident"],
            "health_probe_count": 2,
            "attempted_remediation_epochs_sha256": "2" * 64,
            "last_health_probe_sha256": "3" * 64,
        },
    }
    assert diagnostic.run_once(progressed) == first
    assert driver.calls == 1


def test_post_incident_meta_allows_one_episode_per_distinct_incident(
    tmp_path,
):
    first, driver, projection = _post_incident_meta_fixture(
        tmp_path
    )
    first.run_once(projection)
    second = B.PostIncidentMetaDiagnostic(
        first.campaign_root,
        operator_configuration_sha256=(
            first.operator_configuration_sha256
        ),
        driver_executable=first.driver_executable,
        driver_executable_sha256=(
            first.driver_executable_sha256
        ),
        driver_configuration=first.driver_configuration,
        driver_configuration_sha256=(
            first.driver_configuration_sha256
        ),
        driver_attestation_sha256=(
            first.driver_attestation_sha256
        ),
        operation_timeout_seconds=(
            first.operation_timeout_seconds
        ),
        command_runner=driver,
    )
    next_projection = {
        **projection,
        "operator_incident": {
            **projection["operator_incident"],
            "attempt_id": "attempt-2",
        },
        "substrate_incident": {
            **projection["substrate_incident"],
            "attempt_id": "attempt-2",
            "failure_receipt_sha256": "4" * 64,
        },
        "incident_event_sequence": 29,
        "incident_event_digest": "5" * 64,
    }
    result = second.run_once(next_projection)
    assert result["status"] == "DIAGNOSED"
    assert driver.calls == 2
    assert first.root != second.root
    assert len(tuple(first.collection_root.iterdir())) == 2
