from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import arc_agi3_python_runtime_manifest as R


@pytest.fixture(scope="module")
def observed_runtime() -> tuple[Path, dict]:
    interpreter = Path(sys.executable)
    if not (interpreter.parent.parent / "pyvenv.cfg").is_file():
        pytest.skip("focused runtime-manifest evidence requires a venv")
    try:
        value = R.build_runtime_manifest(interpreter)
    except R.RuntimeManifestError as exc:
        pytest.skip(f"selected test venv cannot provide pytest: {exc}")
    return interpreter, value


def test_runtime_manifest_binds_symlink_venv_stdlib_native_and_pytest(
    tmp_path,
    observed_runtime,
):
    interpreter, value = observed_runtime
    assert value["interpreter"]["requested_path"] == str(interpreter)
    assert (
        value["interpreter"]["resolution"]["resolved_target"]["kind"]
        == "file"
    )
    assert value["pyvenv_cfg"]["path"] == str(
        interpreter.parent.parent / "pyvenv.cfg"
    )
    assert value["standard_library_manifest"]["entry_count"] > 0
    assert value["native_extension_manifest"]["entry_count"] > 0
    names = {
        item["name"]
        for item in value["package_runtime_probe"]["distributions"]
    }
    assert "pytest" in names
    assert all(
        dependency in names
        for item in value["package_runtime_probe"]["distributions"]
        for dependency in item["requires"]
    )
    assert "-S" in R.base_probe_command(interpreter)
    assert "-S" in R.suite_command(
        interpreter,
        site_root=Path(value["base_runtime_probe"]["purelib"]),
        suite_path=tmp_path / "suite.py",
        runtime_manifest_path=tmp_path / "runtime.json",
        runtime_manifest_sha256="a" * 64,
    )

    path = tmp_path / "runtime.json"
    digest = R.write_new_runtime_manifest(path, value)
    assert R.load_runtime_manifest(
        path,
        expected_sha256=digest,
        python_executable=interpreter,
        python_executable_sha256=(
            value["interpreter"]["resolved_sha256"]
        ),
    ) == value


@pytest.mark.parametrize(
    "mutation",
    (
        "interpreter_resolution",
        "pyvenv",
        "stdlib",
        "native",
        "pytest_dependency",
        "dependency_closure",
    ),
)
def test_runtime_manifest_rejects_each_runtime_authority_drift(
    observed_runtime,
    mutation,
):
    interpreter, original = observed_runtime
    value = copy.deepcopy(original)
    if mutation == "interpreter_resolution":
        links = value["interpreter"]["resolution"]["symlinks"]
        if not links:
            pytest.skip("selected venv interpreter has no symlink chain")
        links[0]["target"] += ".substituted"
    elif mutation == "pyvenv":
        value["pyvenv_cfg"]["sha256"] = "0" * 64
    elif mutation == "stdlib":
        value["standard_library_manifest"]["entries_sha256"] = "0" * 64
    elif mutation == "native":
        value["native_extension_manifest"]["entries_sha256"] = "0" * 64
    elif mutation == "pytest_dependency":
        value["pytest_dependency_manifests"][-1][
            "files_sha256"
        ] = "0" * 64
    else:
        pytest_item = next(
            item
            for item in value["package_runtime_probe"]["distributions"]
            if item["name"] == "pytest"
        )
        dependency = pytest_item["requires"][0]
        value["package_runtime_probe"]["distributions"] = [
            item
            for item in value["package_runtime_probe"]["distributions"]
            if item["name"] != dependency
        ]
        value["pytest_dependency_manifests"] = [
            item
            for item in value["pytest_dependency_manifests"]
            if item["name"] != dependency
        ]
    with pytest.raises(R.RuntimeManifestError):
        R.revalidate_runtime_files(value)


def test_runtime_manifest_rejects_manifest_file_substitution(
    tmp_path,
    observed_runtime,
):
    interpreter, value = observed_runtime
    path = tmp_path / "runtime.json"
    digest = R.write_new_runtime_manifest(path, value)
    with pytest.raises(
        R.RuntimeManifestError, match="pinned bytes"
    ):
        R.load_runtime_manifest(
            path,
            expected_sha256="0" * 64,
            python_executable=interpreter,
            python_executable_sha256=(
                value["interpreter"]["resolved_sha256"]
            ),
        )
    assert digest != "0" * 64


def test_hermetic_suite_bootstrap_disables_site_and_loads_bound_pytest(
    tmp_path,
    observed_runtime,
):
    interpreter, value = observed_runtime
    suite = tmp_path / "probe_suite.py"
    suite.write_text(
        "\n".join((
            "import json",
            "import pytest",
            "import sys",
            "print(json.dumps({",
            "    'pytest_version': pytest.__version__,",
            "    'sys_path': sys.path,",
            "}, sort_keys=True, separators=(',', ':')))",
        ))
        + "\n",
        encoding="utf-8",
    )
    site_root = Path(value["base_runtime_probe"]["purelib"])
    command = R.suite_command(
        interpreter,
        site_root=site_root,
        suite_path=suite,
        runtime_manifest_path=tmp_path / "runtime.json",
        runtime_manifest_sha256="a" * 64,
    )
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env={"LANG": "C", "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        shell=False,
        close_fds=True,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    observed = json.loads(completed.stdout)
    assert observed["pytest_version"] == (
        value["package_runtime_probe"]["pytest_version"]
    )
    assert observed["sys_path"] == [
        *value["base_runtime_probe"]["isolated_sys_path"],
        str(tmp_path),
        str(site_root),
    ]
    assert not any(
        path
        for path in observed["sys_path"]
        if path != str(site_root)
        and "site-packages" in path
    )


def test_hermetic_suite_bootstrap_fails_without_bound_pytest(
    tmp_path,
    observed_runtime,
):
    interpreter, _ = observed_runtime
    suite = tmp_path / "requires_pytest.py"
    suite.write_text("import pytest\n", encoding="utf-8")
    empty_site = tmp_path / "empty-site"
    empty_site.mkdir()
    completed = subprocess.run(
        R.suite_command(
            interpreter,
            site_root=empty_site,
            suite_path=suite,
            runtime_manifest_path=tmp_path / "runtime.json",
            runtime_manifest_sha256="a" * 64,
        ),
        cwd=tmp_path,
        env={"LANG": "C", "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        shell=False,
        close_fds=True,
    )
    assert completed.returncode != 0
    assert b"No module named 'pytest'" in completed.stderr
