from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_contiguous_conformance as C
import arc_agi3_contiguous_scenario_driver as D
import arc_agi3_contiguous_supervisor as S
import arc_agi3_release_gate as G


def _component_pass_kwargs():
    nodeids = [
        f"{path}::test_synthetic_full_component_pass"
        for path in C.COMPONENT_TEST_FILES
    ]
    return {
        "component_collect_exit_code": 0,
        "component_collected_nodeids": nodeids,
        "component_pytest_exit_code": 0,
        "component_run_collected_nodeids": list(nodeids),
        "component_outcomes": {
            nodeid: "PASS" for nodeid in nodeids
        },
        "component_pytest_output":
            "synthetic full component PASS for unit receipt",
    }


def _pass_result(repository: Path | None = None, **kwargs):
    nodeids = [item.nodeid for item in C.INVARIANTS]
    outcomes = {nodeid: "PASS" for nodeid in nodeids}
    return C.build_result(
        pytest_exit_code=0,
        collected_nodeids=nodeids,
        outcomes=outcomes,
        pytest_output="all canonical cases passed",
        **_component_pass_kwargs(),
        repository=repository,
        **kwargs,
    )


def _complete_loaded_module_receipt(
    repository: Path,
) -> dict:
    snapshot = C.control_contract_snapshot(repository=repository)
    required = sorted({
        C.SUITE_CONTROL_PATH,
        *C.COMPONENT_TEST_FILES,
    })
    records = [
        {
            "module": f"sealed_control_{index:02d}",
            "path": relative,
            "sha256": snapshot["files_sha256"][relative],
        }
        for index, relative in enumerate(required)
    ]
    records.sort(key=lambda record: (record["module"], record["path"]))
    return {
        "complete": True,
        "records": records,
        "sha256": C._sha256(C._canonical_json({
            "records": records,
            "required_paths": required,
        })),
        "summary": {
            "required": len(required),
            "represented": len(required),
            "records": len(records),
            "missing_required": 0,
            "conflicting_origins": 0,
            "unsealed_local_modules": 0,
        },
    }


def _immutable_pass_result(tmp_path: Path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    digest = C.control_contract_sha256(repository)
    snapshot = tmp_path / digest
    C.materialize_immutable_control_snapshot(repository, snapshot)
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_bytes(b"{}\n")
    monkeypatch.setattr(
        C,
        "loaded_control_modules_snapshot",
        lambda *_args, **_kwargs:
            _complete_loaded_module_receipt(snapshot),
    )
    base = _pass_result(
        snapshot,
        suite_runtime_manifest_path=str(runtime_manifest),
        suite_runtime_manifest_sha256=C._sha256(
            runtime_manifest.read_bytes()
        ),
    )
    return snapshot, runtime_manifest, base


def test_registry_represents_every_required_invariant_exactly_once():
    values = C.validate_registry()
    launch = C.launch_requirements_snapshot()["body"]
    assert {value.invariant_id for value in values} == (
        {
            requirement["invariant_id"]
            for requirement in launch["requirements"]
        }
    )
    assert len({value.invariant_id for value in values}) == len(values)
    assert len({value.nodeid for value in values}) == len(values)
    assert {
        value.component for value in values
    } == {
        "conformance",
        "supervisor",
        "runner",
        "scheduler",
        "orchestrator",
        "watchdog",
        "pilot",
        "container_backend",
        "app_server_transport",
        "taint",
        "source_schema",
        "proposer_worker",
        "boundary_certifier",
        "arena_rpc",
        "container_worker",
        "release_gate",
    }

    duplicate = (*values, replace(values[0], invariant_id="another"))
    with pytest.raises(C.ConformanceError, match="registry"):
        C.validate_registry(duplicate)
    missing = values[1:]
    with pytest.raises(C.ConformanceError, match="registry"):
        C.validate_registry(missing)


def test_loaded_control_modules_classify_pseudo_origins_by_control_name(
    monkeypatch,
):
    baseline = C.loaded_control_modules_snapshot()
    monkeypatch.setitem(
        sys.modules,
        "attempt_solver_runtime",
        SimpleNamespace(__file__="<attempt-solver>"),
    )
    monkeypatch.setitem(
        sys.modules,
        "synthetic_frozen_dependency",
        SimpleNamespace(__file__="<frozen importlib._bootstrap>"),
    )
    observed = C.loaded_control_modules_snapshot()
    assert observed["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"]
    )
    assert observed["summary"]["unsealed_local_modules"] == (
        baseline["summary"]["unsealed_local_modules"]
    )
    monkeypatch.setitem(
        sys.modules,
        "solve",
        SimpleNamespace(__file__="<attempt-solver>"),
    )
    sealed_name = C.loaded_control_modules_snapshot()
    assert sealed_name["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"] + 1
    )


def test_loaded_control_module_origins_have_no_ambient_conflict():
    observed = C.loaded_control_modules_snapshot()
    repository = Path(__file__).resolve().parents[2]
    local_unsealed = sorted({
        path.relative_to(repository).as_posix()
        for module in sys.modules.values()
        if isinstance(getattr(module, "__file__", None), str)
        and not str(module.__file__).startswith("<")
        for path in [Path(os.path.abspath(module.__file__)).resolve()]
        if path.is_relative_to(repository)
        and path.relative_to(repository).as_posix()
        not in C.CONTROL_CONTRACT_FILES
    })
    assert observed["summary"]["conflicting_origins"] == 0, observed
    # Ordinary repository pytest imports its root conftest before this
    # component suite.  The hermetic immutable runner has no such file.
    assert local_unsealed in ([], ["conftest.py"])
    assert observed["summary"]["unsealed_local_modules"] == len(
        local_unsealed
    )


def test_loaded_control_modules_reject_forged_control_pseudo_origin(
    monkeypatch,
):
    baseline = C.loaded_control_modules_snapshot()
    monkeypatch.setitem(
        sys.modules,
        "arc_agi3_contiguous_runner",
        SimpleNamespace(__file__="<forged>"),
    )
    observed = C.loaded_control_modules_snapshot()
    assert observed["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"] + 1
    )


def test_loaded_control_modules_reject_missing_control_named_origin(
    tmp_path, monkeypatch,
):
    baseline = C.loaded_control_modules_snapshot()
    missing = tmp_path / "arc_agi3_contiguous_runner.py"
    monkeypatch.setitem(
        sys.modules,
        "forged_missing_control",
        SimpleNamespace(__file__=str(missing)),
    )
    observed = C.loaded_control_modules_snapshot()
    assert observed["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"] + 1
    )


def test_loaded_control_modules_reject_control_alias_origin(
    tmp_path, monkeypatch,
):
    repository = Path(__file__).resolve().parents[2]
    baseline = C.loaded_control_modules_snapshot()
    alias = tmp_path / "arc_agi3_contiguous_runner.py"
    alias.symlink_to(
        repository
        / "arc"
        / "crack_lab"
        / "arc_agi3_contiguous_runner.py"
    )
    monkeypatch.setitem(
        sys.modules,
        "forged_control_alias",
        SimpleNamespace(__file__=str(alias)),
    )
    observed = C.loaded_control_modules_snapshot()
    assert observed["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"] + 1
    )


def test_loaded_control_modules_reject_missing_local_origin(
    monkeypatch,
):
    repository = Path(__file__).resolve().parents[2]
    baseline = C.loaded_control_modules_snapshot()
    monkeypatch.setitem(
        sys.modules,
        "deleted_local_control_candidate",
        SimpleNamespace(
            __file__=str(
                repository
                / "arc"
                / "crack_lab"
                / "deleted_control_candidate.py"
            )
        ),
    )
    missing = C.loaded_control_modules_snapshot()
    assert missing["summary"]["conflicting_origins"] == (
        baseline["summary"]["conflicting_origins"] + 1
    )


def test_loaded_control_modules_classify_existing_extra_local_as_unsealed(
    tmp_path, monkeypatch,
):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    extra = tmp_path / "arc" / "crack_lab" / "extra_local.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    baseline = C.loaded_control_modules_snapshot(repository=tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "existing_unsealed_local_candidate",
        SimpleNamespace(__file__=str(extra)),
    )
    unsealed = C.loaded_control_modules_snapshot(repository=tmp_path)
    assert unsealed["summary"]["unsealed_local_modules"] == (
        baseline["summary"]["unsealed_local_modules"] + 1
    )


def test_full_component_allowlist_is_exact_control_bound_inventory():
    snapshot = C.component_test_files_snapshot()
    assert tuple(
        record["path"] for record in snapshot["files"]
    ) == C.COMPONENT_TEST_FILES
    assert len(C.COMPONENT_TEST_FILES) == 27
    assert set(C.COMPONENT_TEST_FILES) <= set(C.CONTROL_CONTRACT_FILES)


def test_authoritative_inventory_metadata_is_exact_public_25_183_input():
    snapshot = C.authoritative_inventory_metadata_snapshot()
    assert snapshot["games"] == 25
    assert snapshot["levels"] == 183
    assert tuple(
        record["path"] for record in snapshot["files"]
    ) == C.AUTHORITATIVE_INVENTORY_METADATA_FILES
    assert set(C.AUTHORITATIVE_INVENTORY_METADATA_FILES) <= set(
        C.CONTROL_CONTRACT_FILES
    )
    assert tuple(
        relative
        for relative in C.CONTROL_CONTRACT_FILES
        if relative.startswith("environment_files/")
    ) == C.AUTHORITATIVE_INVENTORY_METADATA_FILES
    assert all(
        Path(relative).parts[0] == "environment_files"
        and Path(relative).name == "metadata.json"
        for relative in C.AUTHORITATIVE_INVENTORY_METADATA_FILES
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "omission",
        "cross_game_substitution",
        "twenty_sixth_entry",
        "ancestor_symlink",
        "hardlink_alias",
    ],
)
def test_authoritative_inventory_metadata_drift_fails_closed(
    tmp_path, mutation
):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    first, second = C.AUTHORITATIVE_INVENTORY_METADATA_FILES[:2]
    if mutation == "omission":
        (tmp_path / first).unlink()
    elif mutation == "cross_game_substitution":
        shutil.copyfile(tmp_path / second, tmp_path / first)
    elif mutation == "twenty_sixth_entry":
        extra = (
            tmp_path
            / "environment_files"
            / "zz99"
            / "deadbeef"
            / "metadata.json"
        )
        extra.parent.mkdir(parents=True)
        shutil.copyfile(tmp_path / first, extra)
    elif mutation == "ancestor_symlink":
        ancestor = (tmp_path / first).parent.parent
        outside = tmp_path / "outside_inventory_ancestor"
        ancestor.rename(outside)
        os.symlink(
            outside,
            ancestor,
            target_is_directory=True,
        )
    else:
        alias = tmp_path / "metadata_hardlink_alias.json"
        os.link(tmp_path / first, alias)
    with pytest.raises(
        C.ConformanceError,
        match="authoritative inventory metadata|control path|control file",
    ):
        C.control_contract_snapshot(repository=tmp_path)


def test_control_contract_rejects_symlinked_code_ancestor(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    crack_lab = tmp_path / "arc" / "crack_lab"
    escaped = tmp_path / "escaped_crack_lab"
    crack_lab.rename(escaped)
    crack_lab.symlink_to(escaped, target_is_directory=True)
    with pytest.raises(
        C.ConformanceError,
        match="control path is unavailable or symlinked",
    ):
        C.control_contract_snapshot(repository=tmp_path)


def test_component_pytest_basetemp_is_explicit_and_removed(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(scratch))
    session, basetemp, identity = C._new_component_pytest_basetemp()
    assert session == scratch / "p"
    assert basetemp == scratch / "p" / "t"
    assert stat.S_IMODE(session.stat().st_mode) == 0o700
    basetemp.mkdir()
    (basetemp / "owned.txt").write_text("owned", encoding="utf-8")
    C._remove_component_pytest_basetemp(
        session,
        expected_identity=identity,
    )
    assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    "unsafe_kind",
    ("relative", "symlink", "file", "wrong_mode", "inside_checkout"),
)
def test_component_pytest_basetemp_rejects_unsafe_private_root(
    tmp_path, monkeypatch, unsafe_kind,
):
    if unsafe_kind == "relative":
        monkeypatch.chdir(tmp_path)
        root = Path("relative-scratch")
        root.mkdir(mode=0o700)
        raw = str(root)
    elif unsafe_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        root = tmp_path / "scratch-link"
        root.symlink_to(target, target_is_directory=True)
        raw = str(root)
    elif unsafe_kind == "file":
        root = tmp_path / "scratch-file"
        root.write_text("not a directory", encoding="utf-8")
        raw = str(root)
    elif unsafe_kind == "wrong_mode":
        root = tmp_path / "scratch-mode"
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        raw = str(root)
    else:
        repository = tmp_path / "checkout"
        repository.mkdir()
        root = repository / "scratch"
        root.mkdir(mode=0o700)
        monkeypatch.setattr(C, "_repository_root", lambda: repository)
        raw = str(root)
    monkeypatch.setenv("TMPDIR", raw)
    with pytest.raises(C.ConformanceError, match="TMPDIR"):
        C._new_component_pytest_basetemp()


def test_component_pytest_basetemp_rejects_fixed_name_collision(
    tmp_path, monkeypatch,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    collision = scratch / "p"
    collision.mkdir(mode=0o700)
    marker = collision / "must-survive.txt"
    marker.write_text("survive", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(scratch))
    with pytest.raises(C.ConformanceError, match="failed closed"):
        C._new_component_pytest_basetemp()
    assert marker.read_text(encoding="utf-8") == "survive"


def test_component_pytest_basetemp_is_short_outside_long_cwd(
    tmp_path, monkeypatch,
):
    long_cwd = tmp_path
    while len(os.fsencode(long_cwd)) <= 120:
        long_cwd = long_cwd / ("long-working-directory-" + "x" * 20)
    long_cwd.mkdir(parents=True)
    monkeypatch.chdir(long_cwd)
    scratch = S._private_system_scratch()
    metadata = scratch.stat(follow_symlinks=False)
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
    )
    monkeypatch.setenv("TMPDIR", str(scratch))
    try:
        session, basetemp, session_identity = (
            C._new_component_pytest_basetemp()
        )
        assert session == scratch / "p"
        assert basetemp == scratch / "p" / "t"
        assert len(os.fsencode(basetemp / "arena.sock")) < 104
        C._remove_component_pytest_basetemp(
            session,
            expected_identity=session_identity,
        )
    finally:
        C._remove_owned_private_tree(
            scratch,
            expected_identity=identity,
            label="short scratch inverse",
        )


def test_suite_scratch_gate_rejects_unrelated_entry_without_removing_it(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(scratch))
    C._require_empty_suite_scratch(phase="inverse start")
    unrelated = scratch / "unrelated.txt"
    unrelated.write_text("retain", encoding="utf-8")
    with pytest.raises(C.ConformanceError, match="not empty"):
        C._require_empty_suite_scratch(phase="inverse end")
    assert unrelated.read_text(encoding="utf-8") == "retain"


def test_component_pytest_failure_cleanup_is_confined_to_exact_session(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    unrelated = scratch / "unrelated-preexisting.txt"
    unrelated.write_text("retain", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "must-survive.txt"
    outside_marker.write_text("survive", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(scratch))

    def deliberate_failure(arguments, *, plugins):
        del plugins
        option = arguments.index("--basetemp")
        basetemp = Path(arguments[option + 1])
        locked = basetemp / "read-only"
        locked.mkdir(parents=True)
        locked_file = locked / "sealed.txt"
        locked_file.write_text("sealed", encoding="utf-8")
        locked_file.chmod(0o400)
        locked.chmod(0o500)
        (basetemp / "escape").symlink_to(
            outside,
            target_is_directory=True,
        )
        raise RuntimeError("deliberate component failure")

    monkeypatch.setattr(pytest, "main", deliberate_failure)
    with pytest.raises(RuntimeError, match="deliberate component failure"):
        C._run_component_pytest()
    assert unrelated.read_text(encoding="utf-8") == "retain"
    assert outside_marker.read_text(encoding="utf-8") == "survive"
    assert sorted(path.name for path in scratch.iterdir()) == [
        unrelated.name
    ]


def test_every_full_component_case_collects_from_explicit_files():
    exit_code, collected, output = C._collect_component_pytest()
    assert exit_code == 0, output
    facts = C._component_collection_facts(collected)
    assert not facts["duplicate_nodeids"]
    assert not facts["unknown_nodeids"]
    assert not facts["missing_files"]
    assert len(collected) > len(C.INVARIANTS)


def test_full_component_inventory_rejects_unknown_or_missing_file(
    tmp_path,
):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    unknown = (
        tmp_path / "arc" / "crack_lab"
        / "test_arc_agi3_unreviewed.py"
    )
    unknown.write_text("def test_unreviewed(): pass\n", encoding="utf-8")
    with pytest.raises(C.ConformanceError, match="unknown file"):
        C.component_test_files_snapshot(tmp_path)
    unknown.unlink()
    (
        tmp_path / C.COMPONENT_TEST_FILES[-1]
    ).unlink()
    with pytest.raises(C.ConformanceError, match="missing or unknown"):
        C.component_test_files_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("failure", "FAIL"),
        ("skip", "SKIP"),
        ("xfail", "XFAIL"),
        ("xpass", "XPASS"),
        ("collection_drift", "PASS"),
    ],
)
def test_owner_pass_cannot_mask_nonpass_full_component_suite(
    mutation, expected_status
):
    owner_nodeids = [item.nodeid for item in C.INVARIANTS]
    component = _component_pass_kwargs()
    if mutation == "collection_drift":
        component["component_run_collected_nodeids"] = list(
            reversed(component["component_run_collected_nodeids"])
        )
    else:
        target = component["component_run_collected_nodeids"][0]
        component["component_outcomes"][target] = expected_status
    result = C.build_result(
        pytest_exit_code=0,
        collected_nodeids=owner_nodeids,
        outcomes={nodeid: "PASS" for nodeid in owner_nodeids},
        pytest_output="owner-only green aggregate",
        **component,
    )
    assert result["status"] == "FAIL"
    assert result["component_suite_status"] == "FAIL"
    with pytest.raises(C.ConformanceError, match="not PASS"):
        C.validate_result(result)


def test_full_component_receipt_rejects_forged_outcome_digest():
    result = _pass_result()
    forged = copy.deepcopy(result)
    forged["component_suite_outcomes"][0]["status"] = "SKIP"
    with pytest.raises(
        C.ConformanceError, match="failed, skipped, or missing"
    ):
        C.validate_result(forged)


def test_pytest_recorder_distinguishes_xfail_and_xpass():
    recorder = C._PytestRecorder()
    recorder.pytest_runtest_logreport(SimpleNamespace(
        nodeid="x.py::test_xfail",
        wasxfail="known issue",
        skipped=True,
        failed=False,
        passed=False,
        when="call",
    ))
    recorder.pytest_runtest_logreport(SimpleNamespace(
        nodeid="x.py::test_xpass",
        wasxfail="known issue",
        skipped=False,
        failed=False,
        passed=True,
        when="call",
    ))
    assert recorder.outcomes == {
        "x.py::test_xfail": "XFAIL",
        "x.py::test_xpass": "XPASS",
    }


def test_recursive_canonical_suite_execution_is_forbidden(monkeypatch):
    monkeypatch.setenv(C._ACTIVE_RUN_ENVIRONMENT_KEY, "outer-suite")
    with pytest.raises(C.ConformanceError, match="recursive"):
        C.run()


@pytest.mark.parametrize(
    "mutation",
    [
        "deleted_requirement",
        "extra_requirement",
        "renamed_requirement",
        "scenario_owner_mismatch",
        "test_owner_mismatch",
    ],
)
def test_independent_launch_requirements_fail_closed_on_registry_drift(
    mutation,
):
    launch = copy.deepcopy(C.launch_requirements_snapshot()["body"])
    if mutation == "deleted_requirement":
        launch["requirements"].pop()
    elif mutation == "extra_requirement":
        launch["requirements"].append({
            "invariant_id": "unexpected_extra_requirement",
            "component": "runner",
            "owner_nodeid":
                "arc/crack_lab/test_unexpected.py::test_extra",
            "scenario_id": "S07",
        })
    elif mutation == "renamed_requirement":
        launch["requirements"][0]["invariant_id"] = (
            "renamed_required_invariant"
        )
    elif mutation == "scenario_owner_mismatch":
        launch["scenario_owners"][0]["owner"] = (
            "arc_agi3_contiguous_s02_v1"
        )
    elif mutation == "test_owner_mismatch":
        launch["requirements"][0]["owner_nodeid"] += "_renamed"
    with pytest.raises(C.ConformanceError):
        C.validate_registry(launch_requirements=launch)


def test_result_verification_reopens_selected_launch_requirements(
    tmp_path,
):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    result = _pass_result(tmp_path)
    manifest_path = (
        tmp_path / C.LAUNCH_REQUIREMENTS_CONTROL_PATH
    )
    manifest = json.loads(manifest_path.read_bytes())
    manifest["requirements"][0]["owner_nodeid"] += "_substituted"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C.ConformanceError):
        C.validate_registry(repository=tmp_path)
    with pytest.raises(C.ConformanceError):
        C.validate_result(result, repository=tmp_path)


def test_every_registered_nodeid_collects_before_launch():
    values = C.validate_registry()
    exit_code, collected, output = C._collect_registered_pytest(values)
    assert exit_code == 0, output
    assert collected == [value.nodeid for value in values]


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "skip", "unexpected", "exit_nonzero"],
)
def test_green_aggregate_cannot_mask_skips_duplicates_or_omissions(
    mutation,
):
    nodeids = [item.nodeid for item in C.INVARIANTS]
    collected = list(nodeids)
    outcomes = {nodeid: "PASS" for nodeid in nodeids}
    exit_code = 0
    if mutation == "missing":
        collected.pop()
    elif mutation == "duplicate":
        collected.append(collected[0])
    elif mutation == "skip":
        outcomes[collected[0]] = "SKIP"
    elif mutation == "unexpected":
        collected.append("arc/crack_lab/test_unknown.py::test_unknown")
    elif mutation == "exit_nonzero":
        exit_code = 1
    result = C.build_result(
        pytest_exit_code=exit_code,
        collected_nodeids=collected,
        outcomes=outcomes,
        pytest_output="synthetic green-looking aggregate",
        **_component_pass_kwargs(),
    )
    assert result["status"] == "FAIL"
    with pytest.raises(C.ConformanceError, match="not PASS"):
        C.validate_result(result)


def test_pass_artifact_is_exact_current_contract_and_canonical(tmp_path):
    result = _pass_result()
    assert C.validate_result(result) == result
    path = tmp_path / "conformance.json"
    C._write_new_result(path, result)
    assert C.load_result(path) == result
    assert path.read_bytes() == C._canonical_json(result) + b"\n"
    with pytest.raises(FileExistsError):
        C._write_new_result(path, result)


def test_conformance_receipt_binds_runtime_manifest_bytes(tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_bytes(b"{}\n")
    nodeids = [item.nodeid for item in C.INVARIANTS]
    result = C.build_result(
        pytest_exit_code=0,
        collected_nodeids=nodeids,
        outcomes={nodeid: "PASS" for nodeid in nodeids},
        pytest_output="runtime-bound PASS",
        **_component_pass_kwargs(),
        suite_runtime_manifest_path=str(manifest),
        suite_runtime_manifest_sha256=C._sha256(
            manifest.read_bytes()
        ),
    )
    assert C.validate_result(result) == result
    manifest.write_bytes(b'{"substituted":true}\n')
    with pytest.raises(
        C.ConformanceError, match="runtime manifest bytes changed"
    ):
        C.validate_result(result)


def test_terminal_launch_authority_binds_release_and_image(
    tmp_path, monkeypatch
):
    snapshot, runtime_manifest, base = _immutable_pass_result(
        tmp_path, monkeypatch
    )
    image = "sha256:" + "a" * 64
    release_path = tmp_path / "release.json"
    release_path.write_text("{}\n", encoding="utf-8")
    scenario = D.run(
        repository=snapshot,
        runtime_manifest_path=runtime_manifest,
        runtime_manifest_sha256=C._sha256(
            runtime_manifest.read_bytes()
        ),
        output_root=tmp_path / "scenario",
    )
    assert scenario["status"] == "BLOCKED"
    with pytest.raises(
        C.ConformanceError,
        match="production S01--S12 observations are not exact PASS",
    ):
        C.bind_terminal_launch_authority(
            base,
            container_image_digest=image,
            release_receipt_path=release_path,
            scenario_driver_receipt_path=Path(
                scenario["receipt_path"]
            ),
            canonical_root=tmp_path,
            environments_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("container_image_digest", "sha256:" + "9" * 64),
        ("frozen_release_receipt_path", "/tmp/substituted-release.json"),
        ("frozen_release_receipt_sha256", "8" * 64),
        ("frozen_release_levels", 182),
        (
            "production_scenario_driver_receipt_path",
            "/tmp/substituted-scenario.json",
        ),
        ("production_scenario_driver_receipt_sha256", "7" * 64),
        ("production_scenario_receipts_sha256", "6" * 64),
        (
            "production_scenario_verification_environment_sha256",
            "5" * 64,
        ),
    ],
)
def test_terminal_evidence_field_mutation_blocks(
    tmp_path, monkeypatch, field, replacement
):
    snapshot, _runtime_manifest, base = _immutable_pass_result(
        tmp_path, monkeypatch
    )
    terminal = {
        **base,
        "launch_authority": True,
        "container_image_digest": "sha256:" + "a" * 64,
        "frozen_release_receipt_path": str(
            tmp_path / "release.json"
        ),
        "frozen_release_receipt_sha256": "b" * 64,
        "frozen_release_levels": 183,
        "production_scenario_driver_receipt_path": str(
            tmp_path / "scenario_driver_receipt.json"
        ),
        "production_scenario_driver_receipt_sha256": "c" * 64,
        "production_scenario_receipts_sha256": "d" * 64,
        "production_scenario_verification_environment_sha256":
            "e" * 64,
    }
    terminal["terminal_evidence_sha256"] = C._sha256(
        C._canonical_json(C._terminal_evidence_body(terminal))
    )
    assert C.validate_result(terminal, repository=snapshot) == terminal
    mutated = copy.deepcopy(terminal)
    mutated[field] = replacement
    with pytest.raises(C.ConformanceError):
        C.validate_result(mutated, repository=snapshot)


def test_scenario_verifier_scratch_leak_blocks_and_is_removed(
    tmp_path, monkeypatch
):
    snapshot, runtime_manifest, base = _immutable_pass_result(
        tmp_path, monkeypatch
    )
    scenario = D.run(
        repository=snapshot,
        runtime_manifest_path=runtime_manifest,
        runtime_manifest_sha256=C._sha256(
            runtime_manifest.read_bytes()
        ),
        output_root=tmp_path / "scenario",
    )
    leaked_roots = []

    def leak_scratch(
        _argv,
        *,
        cwd,
        environment,
        timeout_seconds,
        scratch_root,
    ):
        assert cwd == snapshot / ".neutral"
        assert environment["TMPDIR"] == str(scratch_root)
        assert timeout_seconds == 120
        leaked_roots.append(Path(scratch_root))
        (Path(scratch_root) / "leaked").write_text(
            "junk", encoding="utf-8"
        )
        return SimpleNamespace(
            returncode=0,
            stdout="{}\n",
            stderr="",
            timed_out=False,
            captured_descendants_absent=True,
        )

    monkeypatch.setattr(
        S, "_run_bounded_process_group", leak_scratch
    )
    with pytest.raises(
        C.ConformanceError,
        match="did not complete",
    ):
        C._verify_production_scenario_authority(
            base,
            Path(scenario["receipt_path"]),
        )
    assert len(leaked_roots) == 1
    assert not leaked_roots[0].exists()


def test_immutable_control_snapshot_crash_never_publishes_partial_target(
    tmp_path, monkeypatch
):
    repository = Path(__file__).resolve().parents[2]
    digest = C.control_contract_sha256(repository)
    target = tmp_path / digest
    real_rename = C.os.rename

    def crash_before_publish(source, destination):
        assert Path(source).name.startswith(f".{digest}.staging.")
        assert Path(destination) == target
        raise KeyboardInterrupt("simulated publisher loss")

    monkeypatch.setattr(C.os, "rename", crash_before_publish)
    with pytest.raises(KeyboardInterrupt, match="publisher loss"):
        C.materialize_immutable_control_snapshot(repository, target)
    assert not target.exists()
    assert any(
        path.name.startswith(f".{digest}.staging.")
        for path in tmp_path.iterdir()
    )

    monkeypatch.setattr(C.os, "rename", real_rename)
    snapshot = C.materialize_immutable_control_snapshot(
        repository, target
    )
    assert snapshot["sha256"] == digest
    assert C.validate_immutable_control_snapshot(target) == snapshot
    assert not any(
        path.name.startswith(f".{digest}.staging.")
        for path in tmp_path.iterdir()
    )


def test_forged_pass_cannot_edit_case_or_summary():
    result = _pass_result()
    forged_case = copy.deepcopy(result)
    forged_case["cases"][0]["nodeid"] = forged_case["cases"][1]["nodeid"]
    with pytest.raises(C.ConformanceError, match="case registry"):
        C.validate_result(forged_case)

    forged_summary = json.loads(json.dumps(result))
    forged_summary["summary"]["skipped"] = 1
    with pytest.raises(C.ConformanceError, match="green aggregate"):
        C.validate_result(forged_summary)


def test_control_contract_has_one_cross_module_digest_authority():
    canonical = C.control_contract_snapshot()
    assert tuple(S.CONTROL_CONTRACT_FILES) == C.CONTROL_CONTRACT_FILES
    assert S.control_contract_sha256() == canonical["sha256"]
    assert G._control_contract_snapshot(
        G._default_control_files()
    ) == canonical


def test_control_contract_mutation_invalidates_canonical_digest(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    files: dict[str, Path] = {}
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
        files[relative] = destination
    before = C.control_contract_snapshot(
        files, expected_order=C.CONTROL_CONTRACT_FILES
    )
    mutated = files[C.CONTROL_CONTRACT_FILES[0]]
    mutated.write_bytes(mutated.read_bytes() + b"\n# mutation\n")
    after = C.control_contract_snapshot(
        files, expected_order=C.CONTROL_CONTRACT_FILES
    )
    assert after["sha256"] != before["sha256"]
    assert (
        after["files_sha256"][C.CONTROL_CONTRACT_FILES[0]]
        != before["files_sha256"][C.CONTROL_CONTRACT_FILES[0]]
    )
    assert G._control_contract_snapshot(files) == after


def test_workspace_root_leak_forces_fail():
    repository = Path(__file__).resolve().parents[2]
    start = C.workspace_root_inventory(repository)
    end = copy.deepcopy(start)
    end["entries"].append({
        "name": ".a3cb_leaked_fixture",
        "kind": "directory",
        "device": 1,
        "inode": 1,
        "mode": 0o700,
        "nlink": 1,
        "uid": 1,
    })
    end["entries"].sort(key=lambda record: record["name"])
    end["forbidden_entries"].append(".a3cb_leaked_fixture")
    end["forbidden_entries"].sort()
    end["sha256"] = C._sha256(C._canonical_json({
        "root": end["root"],
        "entries": end["entries"],
        "forbidden_entries": end["forbidden_entries"],
    }))
    nodeids = [item.nodeid for item in C.INVARIANTS]
    result = C.build_result(
        pytest_exit_code=0,
        collected_nodeids=nodeids,
        outcomes={nodeid: "PASS" for nodeid in nodeids},
        pytest_output="synthetic workspace leak inverse",
        **_component_pass_kwargs(),
        repository=repository,
        workspace_inventory_start=start,
        workspace_inventory_end=end,
    )
    assert result["status"] == "FAIL"
    assert result["workspace_root_inventory_stable"] is False
    with pytest.raises(C.ConformanceError, match="not PASS"):
        C.validate_result(result)


def test_workspace_same_name_inode_substitution_forces_fail():
    repository = Path(__file__).resolve().parents[2]
    start = C.workspace_root_inventory(repository)
    end = copy.deepcopy(start)
    assert end["entries"]
    end["entries"][0]["inode"] += 1
    end["sha256"] = C._sha256(C._canonical_json({
        "root": end["root"],
        "entries": end["entries"],
        "forbidden_entries": end["forbidden_entries"],
    }))
    nodeids = [item.nodeid for item in C.INVARIANTS]
    result = C.build_result(
        pytest_exit_code=0,
        collected_nodeids=nodeids,
        outcomes={nodeid: "PASS" for nodeid in nodeids},
        pytest_output="synthetic same-name substitution inverse",
        **_component_pass_kwargs(),
        repository=repository,
        workspace_inventory_start=start,
        workspace_inventory_end=end,
    )
    assert result["status"] == "FAIL"
    assert result["workspace_root_inventory_stable"] is False
    with pytest.raises(C.ConformanceError, match="not PASS"):
        C.validate_result(result)


def test_mid_suite_control_mutation_forces_fail(
    tmp_path, monkeypatch
):
    repository = Path(__file__).resolve().parents[2]
    for relative in C.CONTROL_CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    values = C.validate_registry()
    expected = [value.nodeid for value in values]
    component = _component_pass_kwargs()
    component_nodeids = list(dict.fromkeys([
        *component["component_collected_nodeids"],
        *expected,
    ]))
    calls = 0

    def mutate_during_suite():
        nonlocal calls
        calls += 1
        target = tmp_path / C.CONTROL_CONTRACT_FILES[0]
        target.write_bytes(target.read_bytes() + b"\n# mid-suite mutation\n")
        return (
            0,
            component_nodeids,
            {nodeid: "PASS" for nodeid in component_nodeids},
            "single synthetic full component execution passed\n",
        )

    monkeypatch.setattr(
        C,
        "_run_component_pytest",
        mutate_during_suite,
    )
    monkeypatch.setattr(
        C,
        "_run_registered_pytest",
        lambda _values: pytest.fail(
            "owner nodeids must not execute a second time"
        ),
    )
    # Exercise the implementation directly so the canonical outer suite's
    # recursion guard remains closed while this unit simulates its inner run.
    result = C._run_once(repository=tmp_path)
    assert calls == 1
    assert result["status"] == "FAIL"
    assert result["control_contract_stable"] is False
    assert (
        result["control_contract_start_sha256"]
        != result["control_contract_end_sha256"]
    )
    with pytest.raises(C.ConformanceError, match="not PASS"):
        C.validate_result(result, repository=tmp_path)
