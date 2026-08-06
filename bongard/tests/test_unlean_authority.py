from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

from bongard.semantic_calibration_command import (
    freeze_stage_a_source_dependencies,
)


_ISOLATED_ACCEPTANCE = r"""
from __future__ import annotations

import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import re
import shutil
import sys


class _RejectSemanticChecker(importlib.abc.MetaPathFinder):
    attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "bongard.semantic_checker":
            self.attempts += 1
            raise AssertionError(
                "authoritative pipeline imported bongard.semantic_checker"
            )
        return None


checker_guard = _RejectSemanticChecker()
sys.meta_path.insert(0, checker_guard)

forbidden_executables = frozenset({"lean", "lake", "elan"})
forbidden_token = re.compile(r"(?<![A-Za-z0-9_.-])(?:lean|lake|elan)(?![A-Za-z0-9_.-])")


def _guard_executable_probe(value):
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str):
        return
    if Path(value).name.lower() in forbidden_executables or forbidden_token.search(value.lower()):
        raise AssertionError(f"authoritative pipeline probed optional checker executable: {value!r}")


real_which = shutil.which


def guarded_which(command, *args, **kwargs):
    _guard_executable_probe(command)
    return real_which(command, *args, **kwargs)


shutil.which = guarded_which


def audit_executable_launch(event, args):
    if event not in {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"}:
        return
    pending = list(args)
    while pending:
        value = pending.pop()
        if isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, (str, bytes)):
            _guard_executable_probe(value)


sys.addaudithook(audit_executable_launch)

work = Path(sys.argv[1]).resolve()
package_root = Path(sys.argv[2]).resolve()
mode = sys.argv[3]
if mode not in {"build", "replay"}:
    raise AssertionError("unknown isolated acceptance mode")

import bongard

if not Path(bongard.__file__).resolve().is_relative_to(package_root):
    raise AssertionError("acceptance child imported bongard outside isolated package")

from bongard.artifacts import canonical_digest
from bongard.semantic_calibration_command import (
    StageAPersistenceConfig,
    load_stage_a_command_receipt,
    persist_stage_a_outcome,
)
from bongard.semantic_gated_dev_validation import (
    GatedDevAcceptancePolicy,
    GatedDevTaskRun,
    GatedDevValidationArtifact,
    _capture_gated_dev_task_replay_bytes,
    capture_gated_dev_replay_bytes,
    plan_gated_dev_validation,
    run_gated_dev_validation,
)
from bongard.semantic_calibration_campaign import (
    verify_semantic_campaign_against_corpus,
)
from bongard.semantic_calibration_command import StageACommandReceipt
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.tests.test_semantic_calibration_command import _fake_success_outcome
from bongard.tests.test_semantic_gated_dev_validation import (
    _LAUNCHER_VERSION,
    _corpus,
    _stage_a,
    _stage_a_command_receipt,
    _stage_b_transport,
)
from bongard.transport import CloudPolicyCacheSnapshot
from bongard import semantic_gated_dev_validation as stage_b_module
from bongard import exposure as exposure_module


command_work = work / "stage-a-command"
outcome, _, _, command_config, _ = _fake_success_outcome(command_work)
command_persistence = StageAPersistenceConfig(
    artifact_directory=command_work / "artifacts",
    exposure_directory=command_work / "exposure",
    cache_snapshot_directory=command_work / "cache",
)
command_result = persist_stage_a_outcome(outcome, command_persistence)
command_receipt = load_stage_a_command_receipt(
    command_result.command_receipt_path,
    command_result.command_receipt_digest,
)
command_receipt_bytes = command_result.command_receipt_path.read_bytes()

# Stage-A exposure events intentionally use the wall clock in production.
# Freeze that external input so this test varies checker bytes and nothing else.
exposure_module._utc_now = lambda: "2026-08-06T12:00:00Z"
stage_b_work = work / "stage-b"
corpus, manifest = _corpus(stage_b_work)
stage_b_campaign_path = stage_b_work / "stage-a-campaign.json"
stage_b_receipt_path = stage_b_work / "stage-a-command-receipt.json"
if mode == "build":
    campaign = _stage_a(corpus, manifest)
    stage_b_receipt = _stage_a_command_receipt(stage_b_work, campaign)
    stage_b_receipt_path.write_bytes(stage_b_receipt.receipt_payload)
else:
    campaign, _ = verify_semantic_campaign_against_corpus(
        json.loads(stage_b_campaign_path.read_bytes()),
        corpus=corpus,
        corpus_manifest=manifest,
    )
    stage_b_receipt_payload = stage_b_receipt_path.read_bytes()
    stage_b_receipt_data = json.loads(stage_b_receipt_payload)
    stage_b_receipt = StageACommandReceipt.from_bytes(
        stage_b_receipt_payload,
        expected_receipt_digest=stage_b_receipt_data[
            "command_receipt_digest"
        ],
    )
if stage_b_receipt.source_dependencies != outcome.source_dependencies:
    raise AssertionError("Stage-A command and Stage-B handoff froze different sources")
policy = build_visual_semantic_policy(
    campaign.calibration.family,
    prospective_protocol=campaign.calibration.protocol,
)
predecessor = (
    campaign.score_batch.commitment_batch.proposal_archive.exposure_successor
)
acceptance = GatedDevAcceptancePolicy(
    confidence_level=0.001,
    minimum_selected_clusters=1,
    minimum_gate_passed_clusters=1,
    minimum_gate_coverage_lower=0.0,
    minimum_both_query_correct_lower=0.0,
    minimum_fully_determinate_lower=0.0,
    maximum_any_abstention_upper=1.0,
    maximum_any_error_upper=1.0,
)
plan = plan_gated_dev_validation(
    corpus,
    source_corpus_manifest=manifest,
    expected_source_corpus_manifest_digest=manifest.digest,
    expected_split_source_digest=manifest.split.source_digest,
    stage_a_campaign=campaign,
    stage_a_command_receipt=stage_b_receipt,
    visual_semantic_policy=policy,
    exposure_predecessor=predecessor,
    expected_exposure_predecessor_digest=predecessor.digest,
    public_seed=hashlib.sha256(
        b"externally committed stage-b seed"
    ).hexdigest(),
    selection_seed_provenance=(
        "fixture external beacon fixed before DEV task identities were inspected"
    ),
    requested_task_count=1,
    exposure_observed_at="2026-08-06T12:00:00Z",
    cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
    acceptance_policy=acceptance,
    families=("bd",),
    task_max_workers=1,
)
transport, call_count = _stage_b_transport()
stage_b_module.codex_cli_authenticated_fingerprint = (
    lambda executable, *, expected_launcher_digest: {
        "version": _LAUNCHER_VERSION,
        "launcher_digest": expected_launcher_digest,
    }
)


def scorer_must_not_run(*args, **kwargs):
    del args, kwargs
    raise AssertionError("direct-only Stage-B acceptance fixture called scorer")


artifact_directory = stage_b_work / "artifacts"
if mode == "build":
    artifact = run_gated_dev_validation(
        corpus,
        plan,
        source_corpus_manifest=manifest,
        stage_a_campaign=campaign,
        stage_a_command_receipt=stage_b_receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        exposure_output_directory=stage_b_work / "exposure",
        artifact_output_directory=artifact_directory,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        label_nonce_root="isolated-unlean-stage-b-label-root",
        proposer_transport=transport,
        scorer_transport=scorer_must_not_run,
    )
    if call_count() != 1:
        raise AssertionError(
            "Stage-B acceptance fixture did not execute exact task count"
        )
    replay = capture_gated_dev_replay_bytes(
        corpus,
        artifact,
        source_corpus_manifest=manifest,
    )
    artifact_data = artifact.to_data()
else:
    artifact_paths = tuple(
        artifact_directory.glob("*.gated-dev-validation.json")
    )
    if len(artifact_paths) != 1:
        raise AssertionError("Stage-B replay requires one frozen artifact")
    artifact_data = json.loads(artifact_paths[0].read_bytes())
    task_runs = tuple(
        GatedDevTaskRun.from_data(value)
        for value in artifact_data["task_runs"]
    )
    replay = {
        run.selection.task_id: _capture_gated_dev_task_replay_bytes(
            corpus,
            manifest,
            plan,
            run,
        )
        for run in task_runs
    }
    artifact = GatedDevValidationArtifact.from_data(
        artifact_data,
        stage_a_campaign=campaign,
        stage_a_command_receipt=stage_b_receipt,
        corpus=corpus,
        source_corpus_manifest=manifest,
        replay_bytes_by_task=replay,
    )
decoded = GatedDevValidationArtifact.from_data(
    artifact_data,
    stage_a_campaign=campaign,
    stage_a_command_receipt=stage_b_receipt,
    corpus=corpus,
    source_corpus_manifest=manifest,
    replay_bytes_by_task=replay,
)
if decoded.digest != artifact.digest:
    raise AssertionError("Stage-B cold replay changed artifact identity")

replay_manifest = {
    task_id: {
        blob_id: {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for blob_id, payload in sorted(blobs.items())
    }
    for task_id, blobs in sorted(replay.items())
}
artifact_paths = tuple(
    path
    for path in artifact_directory.glob("*.gated-dev-validation.json")
    if path.name.startswith(artifact.digest)
)
if len(artifact_paths) != 1:
    raise AssertionError("Stage-B did not persist one content-addressed artifact")
artifact_bytes = artifact_paths[0].read_bytes()

if checker_guard.attempts or "bongard.semantic_checker" in sys.modules:
    raise AssertionError("optional semantic checker entered authoritative imports")

print(
    json.dumps(
        {
            "stage_a": {
                "source_identity": outcome.source_dependencies.to_data(),
                "source_identity_digest": outcome.source_dependencies.digest,
                "command_config": command_config.to_data(),
                "command_config_digest": command_config.digest,
                "command_receipt": command_receipt.to_data(),
                "command_receipt_digest": command_receipt.receipt_digest,
                "command_receipt_file_sha256": hashlib.sha256(
                    command_receipt_bytes
                ).hexdigest(),
            },
            "stage_b": {
                "plan_digest": plan.digest,
                "stage_a_command_receipt_digest": (
                    stage_b_receipt.receipt_digest
                ),
                "replay_manifest_digest": canonical_digest(replay_manifest),
                "task_replay_receipt_digests": [
                    item.digest for item in artifact.task_replay_receipts
                ],
                "artifact_digest": artifact.digest,
                "artifact_file_sha256": hashlib.sha256(
                    artifact_bytes
                ).hexdigest(),
            },
            "guard": {
                "semantic_checker_import_attempts": checker_guard.attempts,
                "semantic_checker_imported": (
                    "bongard.semantic_checker" in sys.modules
                ),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


def _copy_isolated_bongard_package(destination: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    package = destination / "bongard"
    shutil.copytree(
        source,
        package,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "crack_lab",
            "manuscript",
            "runs",
        ),
    )
    return package


def _run_isolated_acceptance(
    root: Path,
    work: Path,
    *,
    mode: str,
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_ISOLATED_ACCEPTANCE),
            str(work),
            str(root / "bongard"),
            mode,
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "isolated authoritative acceptance child failed:\n"
            + completed.stderr
        )
    return json.loads(completed.stdout)


def test_checker_absence_and_arbitrary_bytes_are_authoritatively_identical(
    tmp_path: Path,
) -> None:
    isolated_root = tmp_path / "isolated-package"
    package = _copy_isolated_bongard_package(isolated_root)
    checker = package / "semantic_checker.py"
    checker.unlink()
    work = tmp_path / "fixed-authoritative-work"

    absent = _run_isolated_acceptance(isolated_root, work, mode="build")

    # Deliberately not valid Python: any accidental import must fail before an
    # authoritative identity could be produced.
    checker.write_bytes(b"\x00arbitrary optional checker bytes\xff\n")
    arbitrary = _run_isolated_acceptance(
        isolated_root,
        work,
        mode="replay",
    )

    assert arbitrary == absent
    assert absent["guard"] == {
        "semantic_checker_import_attempts": 0,
        "semantic_checker_imported": False,
    }
    assert len(absent["stage_a"]["source_identity_digest"]) == 64
    assert absent["stage_a"]["command_config"]["reference_execution"] == (
        "python-only/v1"
    )
    assert absent["stage_a"]["command_receipt"][
        "python_predicate_authoritative"
    ] is True
    assert absent["stage_a"]["command_receipt"][
        "optional_checker_may_affect_result"
    ] is False
    assert absent["stage_b"]["artifact_digest"]
    assert len(absent["stage_b"]["task_replay_receipt_digests"]) == 1


def test_real_authoritative_source_boundary_has_one_exact_sidecar_exclusion() -> None:
    package = Path(__file__).resolve().parents[1]
    frozen = freeze_stage_a_source_dependencies(package)
    frozen_paths = {relative_path for relative_path, _, _ in frozen.entries}
    expected_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.relative_to(package).parts[0]
        not in {"tests", "crack_lab", "manuscript"}
        and "__pycache__" not in path.relative_to(package).parts
        and path.relative_to(package).as_posix() != "semantic_checker.py"
    }

    assert frozen_paths == expected_paths
    assert "semantic_checker.py" not in frozen_paths


def test_authoritative_modules_never_import_the_optional_checker() -> None:
    package = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if (
            relative.parts[0] in {"tests", "crack_lab", "manuscript"}
            or relative.as_posix() == "semantic_checker.py"
            or "__pycache__" in relative.parts
        ):
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = {module}
                imported.update(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )
            else:
                continue
            if any(
                name == "semantic_checker"
                or name == "bongard.semantic_checker"
                or name.endswith(".semantic_checker")
                for name in imported
            ):
                violations.append(
                    f"{relative.as_posix()}:{getattr(node, 'lineno', 0)}"
                )

    assert violations == []
