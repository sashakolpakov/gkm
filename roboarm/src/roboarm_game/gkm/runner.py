"""Headless-Codex Godel-Kolmogorov loop with a proposal-only safety boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..interface import KNOWN_INTERFACE, KNOWN_WORLD_INTERACTION
from ..observation import OBSERVATION_SCHEMA_VERSION
from .accounting import (
    canonical_json_sha256,
    free_energy,
    source_accounting,
)
from .arena import RoboArmConnector
from .replay import (
    ProposalRun,
    exact_path_replay,
    run_proposal_source,
    write_json,
)
from .safety_fsa import (
    SafetyAutomaton,
    SafetyPolicy,
    first_success,
    public_attempt_projection,
)
from .scenario import (
    ProposalBundle,
    ScenarioContractError,
    canonical_sha256,
    validate_proposal_bundle,
)
from .taint import (
    PROTECTED_WORKSPACE_FILES,
    inspect_generation,
    protected_manifest,
    sha256_file,
)
from .workspace import PROMOTED_SOURCE_FILES, materialize_workspace

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "gkm"
CODEX_REASONING_EFFORTS = {"medium", "high", "xhigh", "max"}
CODEX_PERMISSION_PROFILE = "roboarm_proposer"
PROPOSER_BOUNDARY_MARKER = "PROPOSER_ACTUATION_BOUNDARY_VIOLATION"


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    campaign_id: str | None = None
    seed: int = 0
    model: str = "gpt-5.6-sol"
    provider: str = "codex"
    reasoning_effort: str = "high"
    proposer_timeout_seconds: int = 1_200
    max_generations: int = 8
    max_scenarios_per_generation: int = 8
    max_actions_per_scenario: int = 160
    max_committed_actions: int = 2_000
    max_clone_actions: int = 12_000
    max_contact_load: float = 0.95
    require_failed_attempt: bool = True


@dataclass(frozen=True, slots=True)
class CampaignResult:
    campaign_id: str
    root: str
    proposer_exit_code: int | None
    clean_generation: bool
    protocol_clean: bool
    genuine_failed_attempt: bool
    source_changed: bool
    source_verified: bool
    path_replayed: bool
    promoted: bool
    exact_actions: int
    committed_actions: int
    clone_actions: int
    marginal_description: int
    literal_action_cost: int
    free_energy: float | None
    failure_reason: str | None
    proposer_generations: int = 0
    proposed_scenarios: int = 0
    fsa_rejections: int = 0
    revised_after_failure: bool = False


Proposer = Callable[[Path, str, Path, Path, CampaignConfig], int]

PROPOSER_FILE_CATEGORIES = {
    "README.md": "public_apparatus_documentation",
    "ROUND.md": "public_round_documentation",
    "solver_index.md": "public_proposal_source_documentation",
    "interface.py": "public_action_contract",
    "protocol.py": "public_environment_shape_contract",
    "scenario_contract.py": "untrusted_scenario_authoring_contract",
    "perception.py": "generic_public_evidence_helpers",
    "gkm_propose.py": "offline_proposal_source_harness",
    "evidence.json": "host_sealed_public_observations",
    "legs.py": "zero_seed_or_retained_proposal_source",
    "players.py": "zero_seed_or_retained_proposal_source",
    "solve.py": "zero_seed_or_retained_proposal_source",
}


def _safe_text(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def _copy_source(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in PROMOTED_SOURCE_FILES:
        source_path = source / name
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"candidate source is missing: {name}")
        destination_path = destination / name
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError(destination_path)
        shutil.copyfile(source_path, destination_path)


def _source_changed(seed_source: Path, candidate: Path) -> bool:
    return any(
        (seed_source / name).read_bytes()
        != (candidate / name).read_bytes()
        for name in PROMOTED_SOURCE_FILES
    )


def _source_digest(workspace: Path) -> str:
    return str(source_accounting(None, workspace)["source_tree_sha256"])


def _source_manifest(workspace: Path) -> dict[str, str]:
    return {
        name: sha256_file(workspace / name)
        for name in PROMOTED_SOURCE_FILES
    }


def _initialize_proposer_git(workspace: Path) -> None:
    """Create the clean local Git boundary used by the reference runner."""

    git_directory = workspace / ".git"
    if git_directory.exists() or git_directory.is_symlink():
        raise ValueError("fresh proposer workspace already contains Git state")
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "init",
            "--quiet",
            str(workspace),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for key, value in (
        ("user.name", "RoboArm Godel-Kolmogorov clean-room"),
        ("user.email", "roboarm-gkm@invalid"),
        ("core.hooksPath", "/dev/null"),
    ):
        subprocess.run(
            ["git", "-C", str(workspace), "config", key, value],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(workspace),
            "add",
            "--",
            *sorted(PROPOSER_FILE_CATEGORIES),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(workspace),
            "commit",
            "--quiet",
            "-m",
            "sealed proposer generation baseline",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _public_evidence(
    *,
    initial_observation: dict[str, object],
    attempts: list[dict[str, object]],
    generation: int,
    feedback: list[dict[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 2,
        "kind": "roboarm_host_sealed_public_evidence",
        "game_id": "rb01-v1",
        "round_id": "rb01-round-1",
        "seed": initial_observation.get("seed"),
        "generation": generation,
        "initial_observation": copy.deepcopy(initial_observation),
        "attempts": copy.deepcopy(attempts),
        "host_feedback": copy.deepcopy(feedback),
        "authority_boundary": {
            "proposer_role": "scenario_and_program_proposal_only",
            "connector_visible_to_proposer": False,
            "unix_socket_visible_to_proposer": False,
            "model_may_write_observations": False,
            "model_may_write_safety_verdict": False,
            "experiment_execution": "isolated_authoritative_digital_twin",
            "candidate_execution":
                "preflight_then_single_use_fsa_permit",
        },
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _proposer_payload_manifest(
    workspace: Path,
    prompt: str,
) -> dict[str, object]:
    expected = set(PROTECTED_WORKSPACE_FILES) | set(PROMOTED_SOURCE_FILES)
    actual = {
        path.name
        for path in workspace.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise ValueError(
            "unexpected proposer payload files: "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    files: list[dict[str, object]] = []
    for name in sorted(expected):
        path = workspace / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"proposer payload file is missing or linked: {name}"
            )
        files.append(
            {
                "path": name,
                "category": PROPOSER_FILE_CATEGORIES[name],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    prompt_bytes = prompt.encode("utf-8")
    return {
        "schema_version": 2,
        "scope": "potentially visible to the external coding proposer",
        "files": files,
        "prompt": {
            "bytes": len(prompt_bytes),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        },
        "authority": {
            "can_propose_scenarios": True,
            "can_write_retained_source": True,
            "can_actuate_connector": False,
            "can_write_observed_facts": False,
            "can_write_safety_verdicts": False,
        },
        "explicitly_excluded": [
            "connector object, socket, token, and live environment handle",
            "private simulator mechanics and world state",
            "canonical mechanics trace and oracle",
            "browser implementation and replay exports",
            "parent arc/ research content",
            "repository history",
            "credentials and host environment values",
        ],
        "execution_note": (
            "Codex emits declarative scenarios offline. After the model turn "
            "ends, the host validates the closed schema, performs isolated "
            "preflight, and admits only an FSA-authorized candidate."
        ),
    }


def proposer_prompt(generation: int = 1) -> str:
    return f"""\
You are the live headless-Codex coding proposer for generation {generation} of
one clean Godel-Kolmogorov machine RoboArm acquisition round. This follows the
program-growth discipline used by the read-only ARC research—retained legs,
falsifiable probes, exact frontier evidence, marginal source growth, and
replay-gated promotion—but the payload and authority boundary are
RoboArm-specific and ARC-API-independent.

Read `README.md`, `ROUND.md`, `solver_index.md`, `interface.py`,
`scenario_contract.py`, `perception.py`, and `evidence.json` in full. The known
apparatus contract is:

{KNOWN_INTERFACE}

The generic world-interaction contract is:

{KNOWN_WORLD_INTERACTION}

You do not have an Arena, connector client, socket, token, clone handle, or
`step()` capability. Do not try to create one. Your job is to formulate
falsifiable `experiment` scenarios and, only when prior host-sealed evidence
supports one, a `candidate` scenario. The trusted host runs every sequence after
your turn through a deterministic finite-state safety gate:

proposal -> closed-schema validation -> isolated simulator preflight ->
deterministic safety verdict -> optional single-use commit permit ->
observed facts -> independent verification.

The model is never the oracle. Do not write `passed`, `observedStatus`, reward,
terminal state, frames, authorization, safety verdicts, or claimed outcomes
into a scenario. `scenario_contract.py` defines the only allowed proposal
fields. Run `python3 gkm_propose.py` locally to validate your source and inspect
the emitted declarative JSON; that command performs no simulation.

Preserve real hypothesis changes in executable source. Put reusable
perception-to-hypothesis and scenario-construction skills in `legs.py`; keep
`players.propose_level_1(evidence)` a thin composition of imported legs, and
keep `solve.py` as the stable dispatcher. Base every revision on the exact
host-sealed RGB camera frames, paired controller feedback, and
dispositions in `evidence.json`. A rejected motion or empty grasp is useful if
it genuinely falsifies a hypothesis. Do not manufacture a theatrical failure
or copy a supplied path. A clone/preflight success is evidence but cannot
promote; a later generation must resubmit a safe supported candidate after
observing an earlier genuine failure.

CLEAN-ROOM BOUNDARY: work only inside the current workspace. Do not inspect
parent directories, absolute host paths, installed `roboarm_game` modules,
repository history, environment/dynamics/geometry/oracle source, private or
underscore-prefixed runtime state, host processes, environment variables,
credentials, or any network service. Do not modify `evidence.json`, public
contract/harness files, or host evidence. Use foreground bounded offline tools
only. The canonical mechanics-test action path is absent and must not be
sought. The workspace has its own sealed local Git baseline so `git diff`
cannot traverse into the parent repository.

Finish this turn after `python3 gkm_propose.py` emits one or more valid,
bounded, genuinely motivated scenarios. The host—not you—will execute them and
return sealed observations to a later generation.
"""


def _proposer_site_packages() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages")
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one pinned proposer site-packages directory"
        )
    return candidates[0].resolve(strict=True)


def _codex_environment(campaign_tmp: Path) -> dict[str, str]:
    campaign_tmp.mkdir(parents=True, exist_ok=True)
    zdot = campaign_tmp / "zdot"
    zdot.mkdir(exist_ok=True)
    keep = {
        "HOME",
        "CODEX_HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "SSL_CERT_FILE",
        "TERM",
        "USER",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in keep
    }
    environment.update(
        {
            "PATH": os.pathsep.join(
                (
                    "/opt/homebrew/bin",
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin",
                )
            ),
            "PYTHONPATH": str(_proposer_site_packages()),
            "TMPDIR": str(campaign_tmp),
            "XDG_CACHE_HOME": str(campaign_tmp / "xdg"),
            "ZDOTDIR": str(zdot),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _toml_inline_map(values: dict[str, str]) -> str:
    return "{" + ",".join(
        f"{json.dumps(key)}={json.dumps(value)}"
        for key, value in values.items()
    ) + "}"


def _codex_permission_configs(
    workspace: Path,
    campaign_tmp: Path,
) -> list[str]:
    workspace_root = workspace.resolve(strict=True)
    temporary_root = campaign_tmp.resolve(strict=True)
    if not temporary_root.is_relative_to(workspace_root):
        raise ValueError(
            "proposer temporary storage must stay inside its workspace"
        )
    filesystem = {
        ":root": "deny",
        ":minimal": "read",
        ":slash_tmp": "deny",
        "/private/tmp": "deny",
        "/opt/homebrew": "read",
        str(workspace_root): "write",
        str(_proposer_site_packages()): "read",
        str(temporary_root): "write",
    }
    return [
        "project_doc_max_bytes=0",
        "project_doc_fallback_filenames=[]",
        "features.network_proxy.enabled=true",
        f'permissions.{CODEX_PERMISSION_PROFILE}.extends=":workspace"',
        (
            f"permissions.{CODEX_PERMISSION_PROFILE}.filesystem="
            f"{_toml_inline_map(filesystem)}"
        ),
        f"permissions.{CODEX_PERMISSION_PROFILE}.network.enabled=false",
        (
            f"permissions.{CODEX_PERMISSION_PROFILE}.network.unix_sockets="
            "{}"
        ),
        f'default_permissions="{CODEX_PERMISSION_PROFILE}"',
    ]


def _sandbox_python() -> str:
    for candidate in (
        Path("/opt/homebrew/bin/python3"),
        Path("/usr/bin/python3"),
    ):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(
        "no minimal Python executable is available for proposal replay"
    )


def _sandboxed_proposal_source(
    workspace: Path,
    campaign_tmp: Path,
    *,
    timeout_seconds: int,
) -> ProposalRun:
    """Run untrusted retained source in the no-network Codex OS sandbox."""

    executable_value = shutil.which("codex")
    if executable_value is None:
        raise FileNotFoundError("codex executable is unavailable")
    environment = _codex_environment(campaign_tmp)
    configs = _codex_permission_configs(workspace, campaign_tmp)
    command = [
        str(Path(executable_value).resolve(strict=True)),
        "sandbox",
        *[
            item
            for setting in configs
            for item in ("--config", setting)
        ],
        "--",
        _sandbox_python(),
        "gkm_propose.py",
    ]
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process_group(process)
        stdout, stderr = process.communicate()
    residual_group = _process_group_exists(process.pid)
    group_quiesced = True
    if residual_group:
        group_quiesced = _stop_process_group(process)
    returncode = process.returncode
    if timed_out:
        returncode = 124
        stderr += "\nPROPOSAL_SOURCE_TIMEOUT\n"
    elif residual_group or not group_quiesced:
        returncode = 70
        stderr += "\nPROPOSAL_SOURCE_PROCESS_GROUP_NOT_QUIESCED\n"
    result: dict[str, object] | None = None
    for line in stdout.splitlines():
        if not line.startswith("SCENARIO_PROPOSALS "):
            continue
        try:
            value = json.loads(line[len("SCENARIO_PROPOSALS ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result = value
    return ProposalRun(
        returncode=int(returncode or 0),
        stdout=stdout,
        stderr=stderr,
        result=result,
    )


def run_codex_proposer(
    workspace: Path,
    prompt: str,
    transcript: Path,
    stderr_path: Path,
    config: CampaignConfig,
) -> int:
    """Run one contained noninteractive proposal-only Codex generation."""

    last_message = transcript.with_suffix(".last.md")
    executable_value = shutil.which("codex")
    if executable_value is None:
        raise FileNotFoundError("codex executable is unavailable")
    codex_executable = str(Path(executable_value).resolve(strict=True))
    campaign_tmp = workspace / ".tmp" / "codex"
    codex_environment = _codex_environment(campaign_tmp)
    permission_configs = _codex_permission_configs(
        workspace,
        campaign_tmp,
    )
    if config.reasoning_effort not in CODEX_REASONING_EFFORTS:
        raise ValueError(
            "unsupported Codex reasoning effort: "
            f"{config.reasoning_effort!r}"
        )
    if config.provider != "codex":
        raise ValueError(
            f"unsupported proposer provider: {config.provider!r}"
        )
    command = [
        codex_executable,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--strict-config",
        "--model",
        config.model,
        "--config",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "--config",
        'web_search="disabled"',
        *[
            item
            for setting in permission_configs
            for item in ("--config", setting)
        ],
        "--config",
        'approval_policy="never"',
        "--cd",
        str(workspace),
        "--skip-git-repo-check",
        "--ignore-rules",
        "--color",
        "never",
        prompt,
    ]
    transcript.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    transcript_fd = os.open(transcript, flags, 0o644)
    stderr_fd = os.open(stderr_path, flags, 0o644)
    with (
        os.fdopen(transcript_fd, "w", encoding="utf-8") as stdout_stream,
        os.fdopen(stderr_fd, "w", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=codex_environment,
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        deadline = time.monotonic() + config.proposer_timeout_seconds
        marker_offset = 0
        marker_carry = ""
        timed_out = False
        boundary_marker = False
        while process.poll() is None:
            stdout_stream.flush()
            try:
                with transcript.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as stream:
                    stream.seek(marker_offset)
                    appended = stream.read()
                    marker_offset = stream.tell()
            except OSError:
                appended = ""
            combined = marker_carry + appended
            if PROPOSER_BOUNDARY_MARKER in combined:
                boundary_marker = True
                _stop_process_group(process)
                stderr_stream.write(
                    "\nPROPOSER_ACTUATION_BOUNDARY_VIOLATION\n"
                )
                break
            marker_carry = combined[
                -max(0, len(PROPOSER_BOUNDARY_MARKER) - 1) :
            ]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _stop_process_group(process)
                break
            try:
                process.wait(timeout=min(0.5, remaining))
            except subprocess.TimeoutExpired:
                continue

        returncode = process.poll()
        residual_group = _process_group_exists(process.pid)
        group_quiesced = True
        if residual_group:
            group_quiesced = _stop_process_group(process)
        stdout_stream.flush()
        stderr_stream.flush()
        write_json(
            transcript.with_suffix(".containment.json"),
            {
                "schema_version": 2,
                "pid": process.pid,
                "returncode": returncode,
                "timed_out": timed_out,
                "boundary_marker": boundary_marker,
                "residual_process_group": residual_group,
                "process_group_quiesced": group_quiesced,
                "permission_profile": CODEX_PERMISSION_PROFILE,
                "network_proxy_enabled": True,
                "sandbox_network_enabled": False,
                "allowlisted_unix_sockets": [],
                "actuation_channel_present": False,
                "web_search_disabled": True,
            },
        )
        _write_last_codex_message(transcript, last_message)

        if not group_quiesced:
            stderr_stream.write(
                "\nPROPOSER_PROCESS_GROUP_NOT_QUIESCED\n"
            )
            stderr_stream.flush()
            return 70
        if boundary_marker:
            return 65
        if timed_out:
            stderr_stream.write("\nPROPOSER_TIMEOUT\n")
            stderr_stream.flush()
            return 124
    return int(returncode or 0)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = 8.0,
) -> bool:
    process_group = process.pid
    if not _process_group_exists(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + grace_seconds
    while (
        _process_group_exists(process_group)
        and time.monotonic() < deadline
    ):
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return False
    return not _process_group_exists(process_group)


def _write_last_codex_message(
    transcript: Path,
    destination: Path,
) -> None:
    latest: str | None = None
    try:
        lines = transcript.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if (
            not isinstance(item, dict)
            or item.get("type") != "agent_message"
        ):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            latest = text
    if latest is not None:
        _safe_text(destination, latest.rstrip() + "\n")


def _failed_observation(
    attempt: Mapping[str, object],
) -> bool:
    return bool(
        isinstance(attempt.get("observed_failure_evidence"), list)
        and attempt.get("observed_failure_evidence")
    )


def _fsa_rejected(attempt: Mapping[str, object]) -> bool:
    return str(attempt.get("disposition", "")).endswith("rejected") or (
        attempt.get("disposition") == "candidate_rejected_by_fsa"
    )


def _failed_browser_artifact(
    attempt: Mapping[str, object],
) -> dict[str, object]:
    trace = attempt.get("preflight")
    proposal = attempt.get("proposal")
    if not isinstance(trace, Mapping):
        raise ValueError("failed attempt has no authoritative preflight")
    raw_steps = trace.get("steps")
    raw_actions = trace.get("actions")
    if not isinstance(raw_steps, list) or not isinstance(raw_actions, list):
        raise ValueError("failed attempt trace is malformed")
    cutoff = len(raw_steps)
    for index, step in enumerate(raw_steps, 1):
        if not isinstance(step, Mapping):
            continue
        snapshot = step.get("visual_state")
        if not isinstance(snapshot, Mapping):
            continue
        robot = snapshot.get("robot")
        events = snapshot.get("events")
        event_kinds = (
            {
                str(event.get("kind"))
                for event in events
                if isinstance(event, Mapping)
                and isinstance(event.get("kind"), str)
            }
            if isinstance(events, list)
            else set()
        )
        if (
            isinstance(robot, Mapping)
            and robot.get("rejected") is True
        ) or "gripper_closed_empty" in event_kinds:
            cutoff = index
            break
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_kind": "gkm",
        "disposition": "failed",
        "trace_role": "fsa_preflight",
        "game_id": "rb01-v1",
        "scenario": trace.get("scenario"),
        "seed": trace.get("seed"),
        "attempt_id": attempt.get("attempt_id"),
        "generation": attempt.get("generation"),
        "scenario_id": (
            proposal.get("scenario_id")
            if isinstance(proposal, Mapping)
            else None
        ),
        "hypothesis": (
            proposal.get("hypothesis")
            if isinstance(proposal, Mapping)
            else None
        ),
        "expected_observation": (
            proposal.get("expected_observation")
            if isinstance(proposal, Mapping)
            else None
        ),
        "observed_failure_evidence": copy.deepcopy(
            attempt.get("observed_failure_evidence")
        ),
        "replay_stage": "failed_preflight",
        "fsa_receipt_sha256": attempt.get("receipt_sha256"),
        "sensor_contract_id": trace.get("sensor_contract_id"),
        "frame_encoding": trace.get("frame_encoding"),
        "frame_shape": copy.deepcopy(trace.get("frame_shape")),
        "camera_model": copy.deepcopy(trace.get("camera_model")),
        "initial_frame_sha256": trace.get("initial_frame_sha256"),
        "initial_frame_b64": trace.get("initial_frame_b64"),
        "initial_telemetry_sha256": trace.get(
            "initial_telemetry_sha256"
        ),
        "initial_telemetry": copy.deepcopy(
            trace.get("initial_telemetry")
        ),
        "initial_visual_state": trace.get("initial_visual_state"),
        "actions": copy.deepcopy(raw_actions[:cutoff]),
        "steps": copy.deepcopy(raw_steps[:cutoff]),
    }


def _success_browser_artifact(
    replay: dict[str, object],
    *,
    campaign_id: str,
    source_digest: str,
    promotion_receipt: str,
    fsa_receipt: str,
) -> dict[str, object]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_kind": "gkm",
        "disposition": "promoted",
        "trace_role": "verification",
        "replay_stage": "independent_exact_replay",
        "campaign_id": campaign_id,
        "game_id": replay.get("game_id"),
        "scenario": replay.get("scenario"),
        "seed": replay.get("seed"),
        "source_tree_sha256": source_digest,
        "promotion_receipt_sha256": promotion_receipt,
        "fsa_receipt_sha256": fsa_receipt,
        "sensor_contract_id": replay.get("sensor_contract_id"),
        "frame_encoding": replay.get("frame_encoding"),
        "frame_shape": copy.deepcopy(replay.get("frame_shape")),
        "camera_model": copy.deepcopy(replay.get("camera_model")),
        "initial_frame_sha256": replay.get("initial_frame_sha256"),
        "initial_frame_b64": replay.get("initial_frame_b64"),
        "initial_telemetry_sha256": replay.get(
            "initial_telemetry_sha256"
        ),
        "initial_telemetry": copy.deepcopy(
            replay.get("initial_telemetry")
        ),
        "initial_visual_state": replay.get("initial_visual_state"),
        "actions": replay.get("exact_actions"),
        "steps": replay.get("steps"),
    }


def _committed_success_browser_artifact(
    attempt: Mapping[str, object],
    *,
    campaign_id: str,
    source_digest: str,
    promotion_receipt: str,
) -> dict[str, object]:
    trace = attempt.get("commit")
    proposal = attempt.get("proposal")
    if not isinstance(trace, Mapping):
        raise ValueError("successful attempt has no committed trace")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_kind": "gkm",
        "disposition": "promoted",
        "trace_role": "fsa_commit",
        "replay_stage": "discovery_commit",
        "campaign_id": campaign_id,
        "game_id": trace.get("game_id"),
        "scenario": trace.get("scenario"),
        "seed": trace.get("seed"),
        "attempt_id": attempt.get("attempt_id"),
        "generation": attempt.get("generation"),
        "scenario_id": (
            proposal.get("scenario_id")
            if isinstance(proposal, Mapping)
            else None
        ),
        "hypothesis": (
            proposal.get("hypothesis")
            if isinstance(proposal, Mapping)
            else None
        ),
        "expected_observation": (
            proposal.get("expected_observation")
            if isinstance(proposal, Mapping)
            else None
        ),
        "observed_failure_evidence": [],
        "source_tree_sha256": source_digest,
        "promotion_receipt_sha256": promotion_receipt,
        "fsa_receipt_sha256": attempt.get("receipt_sha256"),
        "sensor_contract_id": trace.get("sensor_contract_id"),
        "frame_encoding": trace.get("frame_encoding"),
        "frame_shape": copy.deepcopy(trace.get("frame_shape")),
        "camera_model": copy.deepcopy(trace.get("camera_model")),
        "initial_frame_sha256": trace.get("initial_frame_sha256"),
        "initial_frame_b64": trace.get("initial_frame_b64"),
        "initial_telemetry_sha256": trace.get(
            "initial_telemetry_sha256"
        ),
        "initial_telemetry": copy.deepcopy(
            trace.get("initial_telemetry")
        ),
        "initial_visual_state": trace.get("initial_visual_state"),
        "actions": copy.deepcopy(trace.get("actions")),
        "steps": copy.deepcopy(trace.get("steps")),
    }


def _representative_failed_attempts(
    attempts: Sequence[Mapping[str, object]],
    *,
    limit: int = 3,
) -> list[Mapping[str, object]]:
    """Select chronological failures while preferring distinct mechanisms."""

    if limit <= 0:
        return []
    eligible = [attempt for attempt in attempts if _failed_observation(attempt)]
    selected: list[Mapping[str, object]] = []
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for attempt in eligible:
        raw = attempt.get("observed_failure_evidence")
        kinds = tuple(
            sorted(
                str(value)
                for value in raw
                if isinstance(value, str)
            )
        ) if isinstance(raw, list) else ()
        trace = attempt.get("preflight")
        raw_steps = (
            trace.get("steps")
            if isinstance(trace, Mapping)
            else None
        )
        rejection_reasons = tuple(
            sorted(
                {
                    str(reason)
                    for step in raw_steps
                    if isinstance(step, Mapping)
                    and isinstance(step.get("telemetry"), Mapping)
                    and isinstance(
                        step["telemetry"].get("motion"),
                        Mapping,
                    )
                    and (
                        reason := step["telemetry"]["motion"].get(
                            "reason"
                        )
                    )
                    and isinstance(reason, str)
                }
            )
        ) if isinstance(raw_steps, list) else ()
        signature = (kinds, rejection_reasons)
        if signature not in seen_signatures:
            selected.append(attempt)
            seen_signatures.add(signature)
        if len(selected) >= limit:
            return selected
    for attempt in eligible:
        if not any(attempt is chosen for chosen in selected):
            selected.append(attempt)
        if len(selected) >= limit:
            break
    return selected


def _result(
    *,
    campaign_id: str,
    root: Path,
    proposer_exit_code: int | None = None,
    clean_generation: bool = False,
    protocol_clean: bool = False,
    genuine_failed_attempt: bool = False,
    source_changed: bool = False,
    source_verified: bool = False,
    path_replayed: bool = False,
    promoted: bool = False,
    exact_actions: int = 0,
    committed_actions: int = 0,
    clone_actions: int = 0,
    marginal_description: int = 0,
    literal_action_cost: int = 0,
    free_energy_value: float | None = None,
    failure_reason: str | None = None,
    proposer_generations: int = 0,
    proposed_scenarios: int = 0,
    fsa_rejections: int = 0,
    revised_after_failure: bool = False,
) -> CampaignResult:
    return CampaignResult(
        campaign_id=campaign_id,
        root=str(root),
        proposer_exit_code=proposer_exit_code,
        clean_generation=clean_generation,
        protocol_clean=protocol_clean,
        genuine_failed_attempt=genuine_failed_attempt,
        source_changed=source_changed,
        source_verified=source_verified,
        path_replayed=path_replayed,
        promoted=promoted,
        exact_actions=exact_actions,
        committed_actions=committed_actions,
        clone_actions=clone_actions,
        marginal_description=marginal_description,
        literal_action_cost=literal_action_cost,
        free_energy=free_energy_value,
        failure_reason=failure_reason,
        proposer_generations=proposer_generations,
        proposed_scenarios=proposed_scenarios,
        fsa_rejections=fsa_rejections,
        revised_after_failure=revised_after_failure,
    )


def run_campaign(
    config: CampaignConfig = CampaignConfig(),
    *,
    proposer: Proposer | None = None,
) -> CampaignResult:
    """Run a zero-seed Godel-Kolmogorov acquisition and all host gates."""

    if (
        config.max_generations < 2
        or not 1 <= config.max_scenarios_per_generation <= 8
        or not 1 <= config.max_actions_per_scenario <= 160
    ):
        raise ValueError(
            "campaign requires at least two generations, 1..8 scenarios "
            "per generation, and 1..160 actions per scenario"
        )
    artifact_root = config.artifact_root.resolve(strict=False)
    project_root = PROJECT_ROOT.resolve(strict=True)
    if not artifact_root.is_relative_to(project_root):
        raise ValueError("campaign artifacts must stay below roboarm/")
    artifact_root.mkdir(parents=True, exist_ok=True)
    campaign_id = config.campaign_id or (
        f"rb01-{uuid.uuid4().hex[:12]}"
    )
    if not campaign_id or any(
        character
        not in
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in campaign_id
    ):
        raise ValueError("campaign_id contains unsupported characters")
    campaign_root = artifact_root / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=False)
    evidence_root = campaign_root / "evidence"
    evidence_root.mkdir()
    campaign_started = time.monotonic()
    timing: dict[str, float] = {}

    def finish(result: CampaignResult) -> CampaignResult:
        timing["campaign_total_seconds"] = round(
            time.monotonic() - campaign_started,
            6,
        )
        write_json(
            evidence_root / "campaign_timing.json",
            {
                "schema_version": 2,
                "clock": "time.monotonic",
                **timing,
            },
        )
        write_json(campaign_root / "campaign_result.json", asdict(result))
        return result

    write_json(
        evidence_root / "campaign_config.json",
        {
            "schema_version": 2,
            "campaign_id": campaign_id,
            **{
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in asdict(config).items()
                if key not in {"artifact_root", "campaign_id"}
            },
            "authority_model": "proposal_only_then_safety_fsa",
        },
    )

    connector = RoboArmConnector(
        seed=config.seed,
        scenario="round-1",
        max_committed_actions=config.max_committed_actions,
        max_preflight_actions=config.max_clone_actions,
    )
    fsa = SafetyAutomaton(
        connector,
        policy=SafetyPolicy(
            max_contact_load=config.max_contact_load,
        ),
    )
    initial_observation = connector.initial_observation()
    write_json(
        evidence_root / "initial_public_observation.json",
        initial_observation,
    )
    public_attempts: list[dict[str, object]] = []
    host_feedback: list[dict[str, object]] = []
    all_attempts: list[dict[str, object]] = []
    generation_workspaces: dict[int, Path] = {}
    generation_source_digests: dict[int, str] = {}
    generation_input_evidence: dict[int, dict[str, object]] = {}
    proposer_exit_codes: list[int] = []
    proposed_scenarios = 0
    fsa_rejections = 0
    first_failed_attempt: dict[str, object] | None = None
    first_failed_generation: int | None = None
    successful_attempt: Mapping[str, object] | None = None
    successful_generation: int | None = None
    successful_bundle: ProposalBundle | None = None
    final_workspace: Path | None = None
    parent_source: Path | None = None
    clean_so_far = True
    failure_reason: str | None = None
    seed_source = campaign_root / "parents" / "level_00"
    proposer_fn = proposer if proposer is not None else run_codex_proposer
    production_source_sandbox = proposer is None

    for generation in range(1, config.max_generations + 1):
        public_evidence = _public_evidence(
            initial_observation=initial_observation,
            attempts=public_attempts,
            generation=generation,
            feedback=host_feedback,
        )
        generation_input_evidence[generation] = copy.deepcopy(
            public_evidence
        )
        workspace = (
            campaign_root
            / "workspaces"
            / f"generation_{generation:03d}"
        )
        generation_workspaces[generation] = workspace
        layout = materialize_workspace(
            workspace,
            write_root=campaign_root,
            public_evidence=public_evidence,
            generation=generation,
            parent_source=parent_source,
        )
        _initialize_proposer_git(layout.root)
        baseline = protected_manifest(layout.root)
        if generation == 1:
            _copy_source(layout.root, seed_source)
        prompt = proposer_prompt(generation)
        transcript = (
            evidence_root
            / f"proposer_generation_{generation:03d}.jsonl"
        )
        stderr_path = (
            evidence_root
            / f"proposer_generation_{generation:03d}.stderr"
        )
        _safe_text(
            evidence_root
            / f"proposer_prompt_{generation:03d}.md",
            prompt,
        )
        write_json(
            evidence_root
            / f"proposer_payload_manifest_{generation:03d}.json",
            _proposer_payload_manifest(layout.root, prompt),
        )
        print(
            f"Godel-Kolmogorov machine proposal generation {generation}: "
            f"provider={config.provider} model={config.model} "
            f"effort={config.reasoning_effort}",
            flush=True,
        )
        started = time.monotonic()
        try:
            proposer_exit = proposer_fn(
                layout.root,
                prompt,
                transcript,
                stderr_path,
                config,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            failure_reason = (
                "proposer infrastructure failure: "
                f"{type(error).__name__}: {error}"
            )
            clean_so_far = False
            break
        timing[
            f"proposer_generation_{generation:03d}_seconds"
        ] = round(time.monotonic() - started, 6)
        proposer_exit_codes.append(proposer_exit)
        taint = inspect_generation(layout.root, transcript, baseline)
        write_json(
            evidence_root
            / f"generation_admission_{generation:03d}.json",
            {
                "schema_version": 2,
                "generation": generation,
                "clean": taint.clean,
                "reasons": list(taint.reasons),
                "proposer_exit_code": proposer_exit,
            },
        )
        if not taint.clean:
            clean_so_far = False
            failure_reason = "; ".join(taint.reasons)
            break

        source_before_execution = _source_manifest(layout.root)
        source_timeout = min(config.proposer_timeout_seconds, 300)
        source_run = (
            _sandboxed_proposal_source(
                layout.root,
                layout.root / ".tmp" / "proposal_source",
                timeout_seconds=source_timeout,
            )
            if production_source_sandbox
            else run_proposal_source(
                layout.root,
                timeout_seconds=source_timeout,
            )
        )
        write_json(
            evidence_root
            / f"proposal_source_run_{generation:03d}.json",
            {
                "schema_version": 1,
                "generation": generation,
                "returncode": source_run.returncode,
                "stdout": source_run.stdout,
                "stderr": source_run.stderr,
                "result": source_run.result,
                "os_sandboxed": production_source_sandbox,
                "network_policy": (
                    "deny"
                    if production_source_sandbox
                    else "injected_test_not_os_enforced"
                ),
                "unix_socket_policy": (
                    "deny_all"
                    if production_source_sandbox
                    else "injected_test_not_os_enforced"
                ),
            },
        )
        protected_after_execution = protected_manifest(layout.root)
        source_after_execution = _source_manifest(layout.root)
        if (
            protected_after_execution != baseline
            or source_after_execution != source_before_execution
        ):
            clean_so_far = False
            failure_reason = (
                "proposal source execution mutated protected input or "
                "retained source"
            )
            write_json(
                evidence_root
                / f"source_execution_admission_{generation:03d}.json",
                {
                    "schema_version": 1,
                    "generation": generation,
                    "accepted": False,
                    "protected_files_stable":
                        protected_after_execution == baseline,
                    "retained_source_stable":
                        source_after_execution
                        == source_before_execution,
                },
            )
            break
        generation_source_digests[generation] = _source_digest(
            layout.root
        )
        final_workspace = layout.root
        parent_source = layout.root

        try:
            if source_run.returncode != 0 or source_run.result is None:
                raise ScenarioContractError(
                    "proposal source did not emit a valid bundle"
                )
            bundle = validate_proposal_bundle(
                source_run.result,
                expected_generation=generation,
                max_scenarios=config.max_scenarios_per_generation,
                max_actions=config.max_actions_per_scenario,
            )
        except ScenarioContractError as error:
            host_feedback.append(
                {
                    "generation": generation,
                    "kind": "proposal_contract_rejected",
                    "detail": str(error),
                }
            )
            write_json(
                evidence_root
                / f"proposal_contract_{generation:03d}.json",
                {
                    "schema_version": 1,
                    "generation": generation,
                    "accepted": False,
                    "error": str(error),
                },
            )
            if proposer_exit not in {0, None}:
                failure_reason = (
                    "proposer exited without an admissible scenario bundle"
                )
                break
            continue

        proposed_scenarios += len(bundle.scenarios)
        write_json(
            evidence_root
            / f"proposed_scenarios_{generation:03d}.json",
            {
                **bundle.as_dict(),
                "proposal_bundle_sha256": bundle.sha256,
                "proposer_authored_outcomes": False,
            },
        )
        commit_enabled = first_failed_generation is not None
        fsa_result = fsa.run_bundle(
            bundle,
            commit_enabled=commit_enabled,
        )
        write_json(
            evidence_root
            / f"safety_fsa_generation_{generation:03d}.json",
            fsa_result,
        )
        attempts = fsa_result["attempts"]
        if not isinstance(attempts, list):
            raise RuntimeError("safety FSA returned malformed attempts")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise RuntimeError("safety FSA attempt is not an object")
            all_attempts.append(attempt)
            public_attempts.append(public_attempt_projection(attempt))
            if _fsa_rejected(attempt):
                fsa_rejections += 1
            if (
                first_failed_attempt is None
                and _failed_observation(attempt)
            ):
                first_failed_attempt = attempt
                first_failed_generation = generation
        candidate = first_success(attempts)
        if candidate is not None:
            successful_attempt = candidate
            successful_generation = generation
            successful_bundle = bundle
            break

    generations_run = len(proposer_exit_codes)
    connector_evidence = connector.evidence()
    write_json(
        evidence_root / "exploration_connector.json",
        connector_evidence,
    )
    write_json(
        evidence_root / "observed_attempt_ledger.json",
        {
            "schema_version": 1,
            "kind": "host_observed_attempt_ledger",
            "attempts": all_attempts,
            "receipt_sha256": canonical_sha256(all_attempts),
        },
    )
    write_json(
        evidence_root / "public_feedback_ledger.json",
        {
            "schema_version": 1,
            "kind": "proposer_visible_observed_facts",
            "attempts": public_attempts,
            "receipt_sha256": canonical_sha256(public_attempts),
        },
    )

    last_exit = proposer_exit_codes[-1] if proposer_exit_codes else None
    changed = bool(
        final_workspace is not None
        and _source_changed(seed_source, final_workspace)
    )
    if successful_attempt is None or successful_generation is None:
        result = _result(
            campaign_id=campaign_id,
            root=campaign_root,
            proposer_exit_code=last_exit,
            clean_generation=clean_so_far,
            protocol_clean=clean_so_far,
            genuine_failed_attempt=False,
            source_changed=changed,
            committed_actions=connector.committed_actions,
            clone_actions=connector.preflight_actions,
            failure_reason=(
                failure_reason
                or "generation budget ended without an FSA-verified commit"
            ),
            proposer_generations=generations_run,
            proposed_scenarios=proposed_scenarios,
            fsa_rejections=fsa_rejections,
        )
        return finish(result)
    assert final_workspace is not None
    assert successful_bundle is not None

    revised_after_failure = bool(
        first_failed_generation is not None
        and first_failed_generation < successful_generation
        and generation_source_digests.get(first_failed_generation)
        != generation_source_digests.get(successful_generation)
    )
    genuine_failed = bool(
        first_failed_attempt is not None
        and revised_after_failure
    )
    if first_failed_attempt is not None:
        write_json(
            campaign_root / "browser" / "failed_attempt.json",
            _failed_browser_artifact(first_failed_attempt),
        )

    accounting = source_accounting(seed_source, final_workspace)
    write_json(evidence_root / "source_accounting.json", accounting)

    # Fresh source replay: reconstruct the exact final-generation input,
    # execute retained proposal source offline, select the same candidate, then
    # pass it through a fresh connector/FSA. No proposer-written path is trusted.
    verification_workspace = (
        campaign_root / "verification" / "source_replay"
    )
    materialize_workspace(
        verification_workspace,
        write_root=campaign_root,
        public_evidence=generation_input_evidence[successful_generation],
        generation=successful_generation,
        parent_source=final_workspace,
    )
    verification_protected_before = protected_manifest(
        verification_workspace
    )
    verification_source_before = _source_manifest(
        verification_workspace
    )
    source_replay_started = time.monotonic()
    verification_source_timeout = min(
        config.proposer_timeout_seconds,
        300,
    )
    source_run = (
        _sandboxed_proposal_source(
            verification_workspace,
            verification_workspace / ".tmp" / "proposal_source",
            timeout_seconds=verification_source_timeout,
        )
        if production_source_sandbox
        else run_proposal_source(
            verification_workspace,
            timeout_seconds=verification_source_timeout,
        )
    )
    timing["source_replay_seconds"] = round(
        time.monotonic() - source_replay_started,
        6,
    )
    write_json(
        evidence_root / "source_replay_run.json",
        {
            "schema_version": 2,
            "returncode": source_run.returncode,
            "stdout": source_run.stdout,
            "stderr": source_run.stderr,
            "result": source_run.result,
            "os_sandboxed": production_source_sandbox,
            "network_policy": (
                "deny"
                if production_source_sandbox
                else "injected_test_not_os_enforced"
            ),
            "unix_socket_policy": (
                "deny_all"
                if production_source_sandbox
                else "injected_test_not_os_enforced"
            ),
        },
    )
    source_verified = False
    verification_success: Mapping[str, object] | None = None
    verification_connector_evidence: dict[str, object] = {}
    try:
        if (
            protected_manifest(verification_workspace)
            != verification_protected_before
            or _source_manifest(verification_workspace)
            != verification_source_before
        ):
            raise ScenarioContractError(
                "fresh source replay mutated protected input or source"
            )
        if source_run.returncode != 0 or source_run.result is None:
            raise ScenarioContractError(
                "fresh proposal source emitted no bundle"
            )
        verification_bundle = validate_proposal_bundle(
            source_run.result,
            expected_generation=successful_generation,
            max_scenarios=config.max_scenarios_per_generation,
            max_actions=config.max_actions_per_scenario,
        )
        successful_proposal_sha = successful_attempt.get(
            "proposal_sha256"
        )
        matching = tuple(
            proposal
            for proposal in verification_bundle.scenarios
            if proposal.sha256 == successful_proposal_sha
        )
        if len(matching) != 1:
            raise ScenarioContractError(
                "fresh source did not reproduce the admitted candidate"
            )
        selected_bundle = ProposalBundle(
            generation=successful_generation,
            scenarios=matching,
        )
        verification_connector = RoboArmConnector(
            seed=config.seed,
            scenario="round-1",
            max_committed_actions=320,
            max_preflight_actions=320,
        )
        verification_fsa = SafetyAutomaton(
            verification_connector,
            policy=SafetyPolicy(
                max_contact_load=config.max_contact_load,
            ),
        )
        verification_result = verification_fsa.run_bundle(
            selected_bundle,
            commit_enabled=True,
        )
        verification_connector_evidence = (
            verification_connector.evidence()
        )
        verification_success = first_success(
            verification_result["attempts"]
        )
        source_verified = verification_success is not None
    except (ScenarioContractError, ValueError) as error:
        verification_result = {
            "schema_version": 1,
            "accepted": False,
            "error": str(error),
        }
    write_json(
        evidence_root / "verification_safety_fsa.json",
        verification_result,
    )
    write_json(
        evidence_root / "verification_connector.json",
        verification_connector_evidence,
    )
    if not source_verified or verification_success is None:
        result = _result(
            campaign_id=campaign_id,
            root=campaign_root,
            proposer_exit_code=last_exit,
            clean_generation=True,
            protocol_clean=True,
            genuine_failed_attempt=genuine_failed,
            source_changed=changed,
            committed_actions=connector.committed_actions,
            clone_actions=connector.preflight_actions,
            marginal_description=int(
                accounting["marginal_description"]
            ),
            literal_action_cost=int(
                accounting["literal_action_cost"]
            ),
            failure_reason=(
                "candidate source did not reproduce an FSA-authorized "
                "fresh commit"
            ),
            proposer_generations=generations_run,
            proposed_scenarios=proposed_scenarios,
            fsa_rejections=fsa_rejections,
            revised_after_failure=revised_after_failure,
        )
        return finish(result)

    verification_commit = verification_success.get("commit")
    if not isinstance(verification_commit, Mapping):
        raise RuntimeError("verified attempt has no committed trace")
    actions_value = verification_commit.get("actions")
    if not isinstance(actions_value, list):
        raise RuntimeError("verified commit has no exact actions")
    actions = [int(action) for action in actions_value]
    write_json(
        evidence_root / "candidate_path.json",
        {
            "schema_version": 1,
            "target_level": 1,
            "actions": actions,
            "source": "host_verified_fsa_commit",
            "fsa_receipt_sha256":
                verification_success.get("receipt_sha256"),
        },
    )
    exact_replay_started = time.monotonic()
    replay = exact_path_replay(actions, seed=config.seed)
    timing["exact_path_replay_seconds"] = round(
        time.monotonic() - exact_replay_started,
        6,
    )
    write_json(evidence_root / "exact_path_replay.json", replay)
    exact = replay.get("exact_actions")
    path_replayed = bool(
        isinstance(exact, list)
        and exact
        and int(replay.get("levels_completed", 0)) >= 1
    )
    priced = int(accounting["priced_complexity"])
    energy = free_energy(1, priced)

    if not path_replayed:
        result = _result(
            campaign_id=campaign_id,
            root=campaign_root,
            proposer_exit_code=last_exit,
            clean_generation=True,
            protocol_clean=True,
            genuine_failed_attempt=genuine_failed,
            source_changed=changed,
            source_verified=True,
            committed_actions=connector.committed_actions,
            clone_actions=connector.preflight_actions,
            marginal_description=int(
                accounting["marginal_description"]
            ),
            literal_action_cost=int(
                accounting["literal_action_cost"]
            ),
            free_energy_value=energy,
            failure_reason="candidate path failed independent replay",
            proposer_generations=generations_run,
            proposed_scenarios=proposed_scenarios,
            fsa_rejections=fsa_rejections,
            revised_after_failure=revised_after_failure,
        )
        return finish(result)

    if config.require_failed_attempt and not genuine_failed:
        result = _result(
            campaign_id=campaign_id,
            root=campaign_root,
            proposer_exit_code=last_exit,
            clean_generation=True,
            protocol_clean=True,
            genuine_failed_attempt=False,
            source_changed=changed,
            source_verified=True,
            path_replayed=True,
            exact_actions=len(exact),
            committed_actions=connector.committed_actions,
            clone_actions=connector.preflight_actions,
            marginal_description=int(
                accounting["marginal_description"]
            ),
            literal_action_cost=int(
                accounting["literal_action_cost"]
            ),
            free_energy_value=energy,
            failure_reason=(
                "candidate replayed, but no earlier genuine failed "
                "observation followed by a retained source revision exists"
            ),
            proposer_generations=generations_run,
            proposed_scenarios=proposed_scenarios,
            fsa_rejections=fsa_rejections,
            revised_after_failure=revised_after_failure,
        )
        return finish(result)

    exact_actions = [int(value) for value in exact]
    promotion_dir = campaign_root / "promotions" / "level_01"
    _copy_source(final_workspace, promotion_dir)
    promotion: dict[str, object] = {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "game_id": "rb01-v1",
        "through_level": 1,
        "replay_validated": True,
        "source_tree_sha256": accounting["source_tree_sha256"],
        "source_files": {
            name: sha256_file(promotion_dir / name)
            for name in PROMOTED_SOURCE_FILES
        },
        "successful_generation": successful_generation,
        "proposal_bundle_sha256": successful_bundle.sha256,
        "discovery_fsa_receipt_sha256":
            successful_attempt.get("receipt_sha256"),
        "verification_fsa_receipt_sha256":
            verification_success.get("receipt_sha256"),
        "exact_actions": exact_actions,
        "exact_action_count": len(exact_actions),
        "path_replay_receipt_sha256": replay["receipt_sha256"],
        "accounting": accounting,
        "genuine_failed_attempt_observed": genuine_failed,
        "qualifying_failure_attempts": [
            {
                "attempt_id": attempt.get("attempt_id"),
                "generation": attempt.get("generation"),
                "observed_failure_evidence": copy.deepcopy(
                    attempt.get("observed_failure_evidence")
                ),
                "fsa_receipt_sha256": attempt.get("receipt_sha256"),
            }
            for attempt in all_attempts
            if _failed_observation(attempt)
        ],
        "revised_after_failure": revised_after_failure,
        "proposer_had_actuation_authority": False,
    }
    promotion["promotion_receipt_sha256"] = canonical_json_sha256(
        promotion
    )
    write_json(promotion_dir / "promotion.json", promotion)
    browser_attempt_names: list[str] = []
    for index, failed_attempt in enumerate(
        _representative_failed_attempts(all_attempts),
        1,
    ):
        filename = (
            "failed_attempt.json"
            if index == 1
            else f"failed_attempt_{index:03d}.json"
        )
        write_json(
            campaign_root / "browser" / filename,
            _failed_browser_artifact(failed_attempt),
        )
        browser_attempt_names.append(filename)
    committed_success_artifact = _committed_success_browser_artifact(
        successful_attempt,
        campaign_id=campaign_id,
        source_digest=str(accounting["source_tree_sha256"]),
        promotion_receipt=str(
            promotion["promotion_receipt_sha256"]
        ),
    )
    write_json(
        campaign_root / "browser" / "successful_commit.json",
        committed_success_artifact,
    )
    browser_attempt_names.append("successful_commit.json")
    success_artifact = _success_browser_artifact(
        replay,
        campaign_id=campaign_id,
        source_digest=str(accounting["source_tree_sha256"]),
        promotion_receipt=str(
            promotion["promotion_receipt_sha256"]
        ),
        fsa_receipt=str(
            verification_success.get("receipt_sha256")
        ),
    )
    write_json(
        campaign_root / "browser" / "successful_attempt.json",
        success_artifact,
    )
    browser_attempt_names.append("successful_attempt.json")
    write_json(
        campaign_root / "browser" / "manifest.json",
        {
            "schema_version": 3,
            "campaign_id": campaign_id,
            "attempts": browser_attempt_names,
            "failure_replays": len(browser_attempt_names) - 2,
            "success_replays": 2,
        },
    )

    result = _result(
        campaign_id=campaign_id,
        root=campaign_root,
        proposer_exit_code=last_exit,
        clean_generation=True,
        protocol_clean=True,
        genuine_failed_attempt=genuine_failed,
        source_changed=changed,
        source_verified=True,
        path_replayed=True,
        promoted=True,
        exact_actions=len(exact_actions),
        committed_actions=connector.committed_actions,
        clone_actions=connector.preflight_actions,
        marginal_description=int(accounting["marginal_description"]),
        literal_action_cost=int(accounting["literal_action_cost"]),
        free_energy_value=energy,
        failure_reason=None,
        proposer_generations=generations_run,
        proposed_scenarios=proposed_scenarios,
        fsa_rejections=fsa_rejections,
        revised_after_failure=revised_after_failure,
    )
    return finish(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one proposal-only, safety-FSA-gated RoboArm "
            "Godel-Kolmogorov machine campaign"
        )
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--game", choices=("rb01-v1",), default="rb01-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", choices=("codex",), default="codex")
    parser.add_argument(
        "--reasoning-effort",
        "--effort",
        dest="reasoning_effort",
        choices=tuple(sorted(CODEX_REASONING_EFFORTS)),
        default="high",
    )
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument(
        "--generations",
        "--rounds",
        dest="generations",
        type=int,
        default=8,
    )
    parser.add_argument("--scenarios-per-generation", type=int, default=8)
    parser.add_argument("--actions-per-scenario", type=int, default=160)
    parser.add_argument("--committed-budget", type=int, default=2_000)
    parser.add_argument("--clone-budget", type=int, default=12_000)
    parser.add_argument("--max-contact-load", type=float, default=0.95)
    parser.add_argument(
        "--allow-no-failed-attempt",
        action="store_true",
        help=(
            "test-only: do not require failure -> feedback -> source revision"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = CampaignConfig(
        campaign_id=arguments.campaign_id,
        seed=arguments.seed,
        model=arguments.model,
        provider=arguments.provider,
        reasoning_effort=arguments.reasoning_effort,
        proposer_timeout_seconds=max(1, arguments.minutes) * 60,
        max_generations=arguments.generations,
        max_scenarios_per_generation=
            arguments.scenarios_per_generation,
        max_actions_per_scenario=arguments.actions_per_scenario,
        max_committed_actions=arguments.committed_budget,
        max_clone_actions=arguments.clone_budget,
        max_contact_load=arguments.max_contact_load,
        require_failed_attempt=not arguments.allow_no_failed_attempt,
    )
    result = run_campaign(config)
    print("CAMPAIGN_RESULT", json.dumps(asdict(result), sort_keys=True))
    return 0 if result.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CODEX_PERMISSION_PROFILE",
    "CampaignConfig",
    "CampaignResult",
    "PROJECT_ROOT",
    "_codex_environment",
    "_codex_permission_configs",
    "main",
    "proposer_prompt",
    "run_campaign",
    "run_codex_proposer",
]
