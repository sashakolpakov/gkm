from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_containment_canary_operator as O


def _spec(tmp_path: Path):
    campaign = (tmp_path / "campaign").resolve()
    generation_id = str(uuid.uuid4())
    generation = campaign / "generations" / generation_id
    generation.mkdir(parents=True)
    fields = {}
    for name, relative in (
        ("input_dir", "input"),
        ("scratch_dir", "scratch"),
        ("workspace_dir", "workspace"),
        ("output_dir", "output"),
        ("app_server_state_dir", "state/codex_home"),
    ):
        path = generation / relative
        path.mkdir(parents=True)
        fields[name] = str(path)
    return SimpleNamespace(
        campaign_id=str(uuid.uuid4()),
        generation_id=generation_id,
        attempt_id=str(uuid.uuid4()),
        generation_dir=str(generation),
        **fields,
    )


def _operator(tmp_path: Path, environment: dict[str, str]):
    roots = {}
    for name in ("repository", "home", "control", "sibling"):
        path = (tmp_path / name).resolve()
        path.mkdir()
        roots[name] = path
    credentials = (tmp_path / "auth" / "auth.json").resolve()
    credentials.parent.mkdir()
    credentials.write_text('{"access_token":"untouched"}\n')
    os.chmod(credentials, 0o600)
    return (
        O.HostContainmentCanaryOperator(
            repository_root=roots["repository"],
            home_root=roots["home"],
            credential_source_path=credentials,
            controller_control_root=roots["control"],
            sibling_lane_root=roots["sibling"],
            environment=environment,
        ),
        credentials,
    )


def _write_terminal_reveal(
    spec,
    planting: O.CanaryPlanting,
) -> tuple[str, str]:
    root = (
        Path(spec.generation_dir).parent.parent
        / "containment_canary_reveals"
    )
    root.mkdir(mode=0o700)
    path = root / f"{spec.generation_id}.json"
    value = {
        "schema": 1,
        "kind": "contiguous_controller_canary_reveal",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "canary_commitments": [
            item.commitment()
            for item in sorted(
                planting.canaries, key=lambda item: item.category
            )
        ],
        "canary_placement_descriptors_sha256":
            planting.placement_descriptors_sha256,
        "reveal": O.Taint.build_live_canary_reveal(
            planting.canaries
        ),
        "teardown_observation_sha256": "a" * 64,
    }
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return str(path), hashlib.sha256(raw).hexdigest()


def test_operator_plants_six_exact_markers_without_mutating_credentials(
    tmp_path: Path,
):
    environment: dict[str, str] = {}
    operator, credentials = _operator(tmp_path, environment)
    spec = _spec(tmp_path)
    before = credentials.read_bytes()

    planting = operator.provision(spec)
    verified = operator.verify(spec, planting)

    assert len(verified) == 6
    assert len({item.value for item in verified}) == 6
    assert credentials.read_bytes() == before
    receipt_raw = Path(planting.receipt_path).read_bytes()
    assert all(
        item.value.encode("ascii") not in receipt_raw
        for item in verified
    )
    receipt = json.loads(receipt_raw)
    assert receipt["all_six_present_before_launch"] is True
    assert receipt["values_retained"] is False
    assert len(receipt["placement_descriptors"]) == 6
    for item in verified:
        if item.category == "environment":
            assert environment[item.location_name] == item.value
        else:
            path = Path(item.location_name)
            assert path.read_text(encoding="ascii") == item.value
            assert stat.S_IMODE(
                path.stat(follow_symlinks=False).st_mode
            ) == 0o400


@pytest.mark.parametrize(
    "mutation",
    ("omission", "substitution", "early_environment_cleanup"),
)
def test_operator_rejects_missing_or_substituted_planting(
    tmp_path: Path,
    mutation: str,
):
    environment: dict[str, str] = {}
    operator, _credentials = _operator(tmp_path, environment)
    spec = _spec(tmp_path)
    planting = operator.provision(spec)
    selected = next(
        item
        for item in planting.canaries
        if item.category == (
            "environment"
            if mutation == "early_environment_cleanup"
            else "repository"
        )
    )
    if mutation == "omission":
        Path(selected.location_name).unlink()
    elif mutation == "substitution":
        path = Path(selected.location_name)
        path.unlink()
        path.write_text(selected.value, encoding="ascii")
        os.chmod(path, 0o400)
    else:
        del environment[selected.location_name]

    with pytest.raises(
        O.CanaryOperatorError,
        match="unavailable|missing or substituted",
    ):
        operator.verify(spec, planting)


def test_cleanup_requires_terminal_reveal_and_is_idempotent(
    tmp_path: Path,
):
    environment: dict[str, str] = {}
    operator, credentials = _operator(tmp_path, environment)
    spec = _spec(tmp_path)
    credential_before = credentials.read_bytes()
    planting = operator.provision(spec)

    with pytest.raises(
        O.CanaryOperatorError,
        match="terminal canary reveal",
    ):
        operator.cleanup(
            spec,
            planting,
            reveal_path=str(
                Path(spec.generation_dir).parent.parent
                / "containment_canary_reveals"
                / f"{spec.generation_id}.json"
            ),
            reveal_sha256="0" * 64,
        )

    reveal_path, reveal_sha256 = _write_terminal_reveal(
        spec, planting
    )
    cleanup = operator.cleanup(
        spec,
        planting,
        reveal_path=reveal_path,
        reveal_sha256=reveal_sha256,
    )
    repeated = operator.cleanup(
        spec,
        planting,
        reveal_path=reveal_path,
        reveal_sha256=reveal_sha256,
    )

    assert repeated == cleanup
    assert credentials.read_bytes() == credential_before
    assert Path(reveal_path).exists()
    assert stat.S_IMODE(
        Path(cleanup.intent_path).stat().st_mode
    ) == 0o400
    assert stat.S_IMODE(
        Path(cleanup.receipt_path).stat().st_mode
    ) == 0o400
    for item in planting.canaries:
        if item.category == "environment":
            assert item.location_name not in environment
        else:
            assert not Path(item.location_name).exists()
            assert not Path(item.location_name).parent.exists()


def test_cleanup_resumes_after_crash_following_durable_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    environment: dict[str, str] = {}
    operator, _credentials = _operator(tmp_path, environment)
    spec = _spec(tmp_path)
    planting = operator.provision(spec)
    reveal_path, reveal_sha256 = _write_terminal_reveal(
        spec, planting
    )
    original = operator._unlink_exact_marker
    calls = 0

    def crash_after_first(descriptor, item):
        nonlocal calls
        original(descriptor, item)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated supervisor crash")

    monkeypatch.setattr(
        operator, "_unlink_exact_marker", crash_after_first
    )
    with pytest.raises(RuntimeError, match="simulated supervisor crash"):
        operator.cleanup(
            spec,
            planting,
            reveal_path=reveal_path,
            reveal_sha256=reveal_sha256,
        )
    intent_path, receipt_path = operator._cleanup_paths(spec)
    assert intent_path.exists()
    assert not receipt_path.exists()

    monkeypatch.setattr(
        operator, "_unlink_exact_marker", original
    )
    cleanup = operator.cleanup(
        spec,
        planting,
        reveal_path=reveal_path,
        reveal_sha256=reveal_sha256,
    )
    assert Path(cleanup.receipt_path).exists()


def test_cleanup_rejects_substituted_path_after_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    environment: dict[str, str] = {}
    operator, _credentials = _operator(tmp_path, environment)
    spec = _spec(tmp_path)
    planting = operator.provision(spec)
    reveal_path, reveal_sha256 = _write_terminal_reveal(
        spec, planting
    )

    def crash_before_removal(_descriptor, _item):
        raise RuntimeError("simulated supervisor crash")

    monkeypatch.setattr(
        operator, "_unlink_exact_marker", crash_before_removal
    )
    with pytest.raises(RuntimeError, match="simulated supervisor crash"):
        operator.cleanup(
            spec,
            planting,
            reveal_path=reveal_path,
            reveal_sha256=reveal_sha256,
        )
    selected = next(
        item
        for item in planting.canaries
        if item.category == "auth_source"
    )
    path = Path(selected.location_name)
    path.unlink()
    path.write_text(selected.value, encoding="ascii")
    os.chmod(path, 0o400)
    monkeypatch.setattr(
        operator,
        "_unlink_exact_marker",
        O.HostContainmentCanaryOperator._unlink_exact_marker,
    )

    with pytest.raises(
        O.CanaryOperatorError,
        match="cleanup target was substituted",
    ):
        operator.cleanup(
            spec,
            planting,
            reveal_path=reveal_path,
            reveal_sha256=reveal_sha256,
        )
    assert path.exists()
