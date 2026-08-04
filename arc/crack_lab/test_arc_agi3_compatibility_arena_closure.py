from __future__ import annotations

import copy
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

import arc_agi3_arena_rpc as HostRpc
import arc_agi3_arena_rpc_client as ClientRpc
import arc_agi3_compatibility_arena_closure as Closure
import arc_agi3_container_backend as Container


class _Arena:
    def __init__(self, game: str):
        self.game = game
        self.value = 0
        self.path: list[object] = []
        self._levels = 0

    @property
    def actions(self):
        return (1, 2, 6, 7)

    @property
    def levels_completed(self):
        return self._levels

    def terminal(self):
        return len(self.path) >= 6

    def frame(self):
        return [[self.value, 0], [0, self._levels]]

    def reset(self):
        self.value = 0
        self.path = []
        self._levels = 0
        return self.frame()

    def step(self, action, x=None, y=None):
        if action == 6:
            self.value = int(x) + int(y)
            self.path.append([6, int(x), int(y)])
        else:
            self.value += int(action)
            self.path.append(int(action))
        if self.value >= 5:
            self._levels = 1
        return self.frame()

    def clone(self):
        return copy.deepcopy(self)


class _SimulatedProcessCrash(BaseException):
    pass


def _binding() -> HostRpc.ArenaSessionBinding:
    return HostRpc.ArenaSessionBinding(
        campaign_id="campaign-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
        game="zz99",
        parent_level=0,
        target_level=1,
        parent_checkpoint_sha256="a" * 64,
        frontier_sha256="b" * 64,
        exploration_mode="continue_parent",
    )


def _prepared(tmp_path: Path) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "closure"
    return root, Closure.prepare_closure(root)


def test_client_closure_is_exact_and_has_no_host_read_capability():
    snapshot = Closure.canonical_closure_snapshot()
    assert snapshot["client"]["import_roots"] == list(
        Closure.EXACT_CLIENT_IMPORT_ROOTS
    )
    assert snapshot["client"]["local_import_closure"] == [
        Closure.CLIENT_NAME
    ]
    assert snapshot["client"]["repository_imports"] == []
    assert snapshot["client"]["engine_imports"] == []
    assert snapshot["client"]["filesystem_calls"] == []
    assert snapshot["client"]["private_game_state_accesses"] == []


def test_client_import_closure_is_stdlib_plus_pinned_numpy_only():
    imports = set(Closure.EXACT_CLIENT_IMPORT_ROOTS)
    assert imports - sys.stdlib_module_names == {"numpy"}
    client_source = Closure._LOADED_CLIENT_RAW.decode("utf-8")
    assert "import numpy as np" in client_source


def test_pinned_solver_image_hash_binds_and_import_probes_numpy():
    root = Path(Closure.__file__).resolve().parent
    lock_path = root / "container" / "arc_agi3_solver_requirements.lock"
    lock_raw = lock_path.read_bytes()
    lock_digest = hashlib.sha256(lock_raw).hexdigest()
    lock = lock_raw.decode("ascii")
    recipe = (
        root / "container" / "Containerfile.arc-agi3-contiguous"
    ).read_text(encoding="utf-8")
    assert "numpy==2.4.4" in lock
    assert lock.count("--hash=sha256:") == 2
    assert (
        "81f4a14bee47aec54f883e0cad2d73986640c1590eb9bfaaba7ad17394481e6e"
        in lock
    )
    assert (
        "f9e75681b59ddaa5e659898085ae0eaea229d054f2ac0c7e563a62205a700121"
        in lock
    )
    assert lock_digest in recipe
    assert "--require-hashes" in recipe
    assert "--only-binary=:all:" in recipe
    assert (
        "import numpy; assert numpy.__version__ == \"2.4.4\""
        in recipe
    )
    assert Container.trusted_worker_hashes()[
        Container.LABEL_SOLVER_REQUIREMENTS_SHA256
    ] == lock_digest


def test_remote_frame_preserves_numpy_uint8_interface():
    import numpy as np

    observed = ClientRpc.RemoteArena._frame_array([[1, 2], [3, 4]])
    assert isinstance(observed, np.ndarray)
    assert observed.dtype == np.uint8
    assert observed.shape == (2, 2)
    assert observed.tobytes() == bytes((1, 2, 3, 4))


@pytest.mark.parametrize(
    "injection",
    [
        "\nSECRET = '.env'\n",
        "\nSECRET = 'environment_files'\n",
        "\nimport importlib.util\n",
        "\nimport gkm_legs\n",
        "\nimport gkm_arena\n",
        "\nfrom gkm_arena import Arena\nRAW_ARENA = Arena\n",
        "\nimport os\nCWD = os.getcwd()\n",
        "\ndef leak(value):\n    return value._game\n",
        "\nSECRET = '../private/source.py'\n",
        "\nimport pathlib\n",
    ],
)
def test_client_source_inverse_rejects_env_source_parent_and_private_access(
    injection: str,
):
    raw = Closure._LOADED_CLIENT_RAW + injection.encode("utf-8")
    with pytest.raises(Closure.CompatibilityClosureError):
        Closure.analyze_client_source(raw)


def test_prepared_closure_reopens_with_exact_receipt(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    assert sorted(path.name for path in root.iterdir()) == list(
        Closure.EXACT_INVENTORY
    )
    assert not (root / "gkm_try.py").exists()
    observed = Closure.validate_closure(
        root, prepared["receipt_sha256"]
    )
    assert observed["status"] == "PASS"
    assert observed["client_sha256"] == prepared["client_sha256"]
    assert (
        observed["content_manifest_sha256"]
        == prepared["content_manifest_sha256"]
    )
    assert observed["launch_authorized"] is False
    assert "per-turn RPC" in observed["remaining_gate"]


def test_shadow_import_or_extra_file_invalidates_closure(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    (root / "arc_agi3_arena_rpc.py").write_text(
        "raise RuntimeError('shadow')\n", encoding="utf-8"
    )
    with pytest.raises(
        Closure.CompatibilityClosureError, match="shadow"
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])


def test_content_manifest_is_reproducible_and_receipt_is_instance_bound(
    tmp_path: Path,
):
    first_root, first = _prepared(tmp_path / "first")
    second_root, second = _prepared(tmp_path / "second")
    assert (
        first["content_manifest_sha256"]
        == second["content_manifest_sha256"]
    )
    assert (
        (first_root / Closure.CONTENT_MANIFEST_NAME).read_bytes()
        == (second_root / Closure.CONTENT_MANIFEST_NAME).read_bytes()
    )
    assert first["receipt_sha256"] != second["receipt_sha256"]


@pytest.mark.parametrize(
    "checkpoint",
    [
        "staging_created",
        "client_fsynced",
        "content_manifest_fsynced",
        "receipt_fsynced",
        "staging_directory_fsynced",
        "before_atomic_publication",
    ],
)
def test_prepublication_crash_preserves_staging_and_requires_fresh_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)

    def crash(selected: str) -> None:
        if selected == checkpoint:
            raise _SimulatedProcessCrash(checkpoint)

    original = Closure._publication_checkpoint
    monkeypatch.setattr(Closure, "_publication_checkpoint", crash)
    with pytest.raises(_SimulatedProcessCrash, match=checkpoint):
        Closure.prepare_closure(root)
    assert not root.exists()
    assert staging.is_dir()
    staging_metadata = staging.stat()
    retained = {
        path.name: path.read_bytes()
        for path in staging.iterdir()
    }

    monkeypatch.setattr(Closure, "_publication_checkpoint", original)
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    assert staging.is_dir()
    assert staging.stat().st_ino == staging_metadata.st_ino
    assert {
        path.name: path.read_bytes()
        for path in staging.iterdir()
    } == retained

    fresh_parent = root.parent / ("fresh-attempt-" + checkpoint)
    fresh_parent.mkdir(mode=0o700)
    fresh_root = fresh_parent / "closure"
    prepared = Closure.prepare_closure(fresh_root)
    observed = Closure.validate_closure(
        fresh_root, prepared["receipt_sha256"]
    )
    assert observed["status"] == "PASS"
    assert observed["launch_authorized"] is False


def test_ordinary_prepublication_fault_cleans_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)

    def fail(selected: str) -> None:
        if selected == "content_manifest_fsynced":
            raise RuntimeError("injected write-path fault")

    monkeypatch.setattr(Closure, "_publication_checkpoint", fail)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="failed before publication",
    ) as failure:
        Closure.prepare_closure(root)
    assert type(failure.value) is Closure.CompatibilityClosureError
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert not root.exists()
    assert not staging.exists()


def test_crash_during_partial_file_write_preserves_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    original_write = Closure.os.write
    calls = 0

    def partial_then_crash(descriptor: int, payload) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            partial = payload[: max(1, len(payload) // 2)]
            return original_write(descriptor, partial)
        raise _SimulatedProcessCrash("partial file write")

    monkeypatch.setattr(Closure.os, "write", partial_then_crash)
    with pytest.raises(_SimulatedProcessCrash, match="partial file write"):
        Closure.prepare_closure(root)
    assert not root.exists()
    assert staging.is_dir()
    assert (staging / Closure.CLIENT_NAME).stat().st_size < len(
        Closure._LOADED_CLIENT_RAW
    )
    retained = (staging / Closure.CLIENT_NAME).read_bytes()

    monkeypatch.setattr(Closure.os, "write", original_write)
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    assert (staging / Closure.CLIENT_NAME).read_bytes() == retained

    fresh_parent = root.parent / "fresh-attempt-partial-write"
    fresh_parent.mkdir(mode=0o700)
    fresh_root = fresh_parent / "closure"
    prepared = Closure.prepare_closure(fresh_root)
    assert Closure.validate_closure(
        fresh_root, prepared["receipt_sha256"]
    )["status"] == "PASS"


def test_file_fsync_fault_cleans_non_authoritative_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    original_fsync = Closure.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected file fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(Closure.os, "fsync", fail_first_fsync)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="failed before publication",
    ) as failure:
        Closure.prepare_closure(root)
    assert type(failure.value) is Closure.CompatibilityClosureError
    assert isinstance(failure.value.__cause__, OSError)
    assert not root.exists()
    assert not staging.exists()


def test_parent_fsync_fault_preserves_complete_published_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    original_fsync = Closure.os.fsync
    failed = False

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed and root.is_dir() and not staging.exists():
            failed = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(Closure.os, "fsync", fail_parent_fsync)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="failed after publication",
    ) as failure:
        Closure.prepare_closure(root)
    assert isinstance(failure.value.__cause__, OSError)
    assert failed is True
    assert root.is_dir()
    assert not staging.exists()

    monkeypatch.setattr(Closure.os, "fsync", original_fsync)
    receipt_sha256 = hashlib.sha256(
        (root / Closure.RECEIPT_NAME).read_bytes()
    ).hexdigest()
    assert Closure.validate_closure(root, receipt_sha256)["status"] == "PASS"


def test_atomic_publication_never_replaces_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)

    def race(selected: str) -> None:
        if selected == "before_atomic_publication":
            root.mkdir(mode=0o700)
            (root / "sentinel").write_bytes(b"do not replace\n")

    monkeypatch.setattr(Closure, "_publication_checkpoint", race)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="appeared during atomic publication",
    ):
        Closure.prepare_closure(root)
    assert (root / "sentinel").read_bytes() == b"do not replace\n"
    assert not staging.exists()


def test_crash_after_atomic_rename_leaves_one_complete_valid_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)

    def crash(selected: str) -> None:
        if selected == "published_before_parent_fsync":
            raise _SimulatedProcessCrash(selected)

    monkeypatch.setattr(Closure, "_publication_checkpoint", crash)
    with pytest.raises(
        _SimulatedProcessCrash, match="published_before_parent_fsync"
    ):
        Closure.prepare_closure(root)
    assert root.is_dir()
    assert not staging.exists()
    receipt_raw = (root / Closure.RECEIPT_NAME).read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    observed = Closure.validate_closure(root, receipt_sha256)
    assert observed["status"] == "PASS"
    assert observed["launch_authorized"] is False


def test_preexisting_staging_with_unexpected_entry_is_preserved(
    tmp_path: Path,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    unexpected = staging / "unrecognized"
    unexpected.write_bytes(b"preserve me\n")
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    assert unexpected.read_bytes() == b"preserve me\n"
    assert not root.exists()


def test_preexisting_empty_staging_is_preserved(tmp_path: Path):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    before = staging.stat()
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    after = staging.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert list(staging.iterdir()) == []
    assert not root.exists()


def test_preexisting_0400_public_prefix_staging_is_preserved(
    tmp_path: Path,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    ambiguous = staging / Closure.CLIENT_NAME
    retained = Closure._LOADED_CLIENT_RAW[:128]
    ambiguous.write_bytes(retained)
    ambiguous.chmod(0o400)
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    assert ambiguous.read_bytes() == retained
    assert ambiguous.stat().st_mode & 0o777 == 0o400
    assert not root.exists()


def test_preexisting_allowed_name_with_nonpublisher_bytes_is_preserved(
    tmp_path: Path,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    ambiguous = staging / Closure.CLIENT_NAME
    ambiguous.write_bytes(b"not this publisher's partial client\n")
    ambiguous.chmod(0o400)
    with pytest.raises(
        Closure.CompatibilityStagingAmbiguityError,
        match="staging directory must not already exist",
    ):
        Closure.prepare_closure(root)
    assert ambiguous.read_bytes() == b"not this publisher's partial client\n"
    assert not root.exists()


def test_same_uid_allowed_name_race_is_not_erased_by_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    retained = Closure._LOADED_CLIENT_RAW[:128]

    def race(selected: str) -> None:
        if selected == "staging_created":
            injected = staging / Closure.CLIENT_NAME
            injected.write_bytes(retained)
            injected.chmod(0o400)

    monkeypatch.setattr(Closure, "_publication_checkpoint", race)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="ambiguous state",
    ) as failure:
        Closure.prepare_closure(root)
    assert not isinstance(
        failure.value, Closure.CompatibilityStagingAmbiguityError
    )
    assert not root.exists()
    assert staging.is_dir()
    assert (staging / Closure.CLIENT_NAME).read_bytes() == retained


def test_quarantined_staging_observation_is_bounded_and_canonical(
    tmp_path: Path,
):
    root = (tmp_path / "closure").resolve()
    absent = Closure.observe_quarantined_staging(root)
    absent_body = dict(absent)
    absent_digest = absent_body.pop("observation_sha256")
    assert absent["status"] == "ABSENT"
    assert absent["present"] is False
    assert absent["entries"] == []
    assert absent["max_depth"] == 1
    assert absent_digest == hashlib.sha256(
        Closure._canonical_json(absent_body)
    ).hexdigest()

    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    partial = staging / Closure.CLIENT_NAME
    retained = Closure._LOADED_CLIENT_RAW[:128]
    partial.write_bytes(retained)
    partial.chmod(0o400)
    observed = Closure.observe_quarantined_staging(root)
    observed_body = dict(observed)
    observed_digest = observed_body.pop("observation_sha256")
    assert observed["status"] == "AMBIGUOUS"
    assert observed["present"] is True
    assert observed["ambiguity_reasons"] == [
        Closure.STAGING_PROVENANCE_AMBIGUITY
    ]
    assert observed["staging_root"] == str(staging)
    assert observed["total_bytes"] == len(retained)
    assert observed["entries"] == [{
        "name": Closure.CLIENT_NAME,
        "type": "regular",
        "identity": observed["entries"][0]["identity"],
        "size": len(retained),
        "content_observed": True,
        "observed_bytes": len(retained),
        "sha256": hashlib.sha256(retained).hexdigest(),
        "ambiguity_reason": None,
    }]
    assert observed_digest == hashlib.sha256(
        Closure._canonical_json(observed_body)
    ).hexdigest()
    assert partial.read_bytes() == retained


@pytest.mark.parametrize("kind", ["nested", "symlink", "oversize"])
def test_quarantined_staging_observes_ambiguity_without_deletion(
    tmp_path: Path,
    kind: str,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    entry = staging / "ambiguous"
    if kind == "nested":
        entry.mkdir()
        (entry / "retained").write_bytes(b"nested\n")
    elif kind == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"outside\n")
        entry.symlink_to(target)
    else:
        entry.touch(mode=0o600)
        with entry.open("r+b") as stream:
            stream.truncate(
                Closure.MAX_STAGING_OBSERVATION_ENTRY_BYTES + 1
            )
        entry.chmod(0o400)
    observed = Closure.observe_quarantined_staging(root)
    assert observed == Closure.observe_quarantined_staging(root)
    assert observed["status"] == "AMBIGUOUS"
    assert observed["present"] is True
    assert len(observed["entries"]) == 1
    record = observed["entries"][0]
    assert record["name"] == "ambiguous"
    assert record["type"] == {
        "nested": "directory",
        "symlink": "symlink",
        "oversize": "regular",
    }[kind]
    assert record["content_observed"] is False
    assert record["observed_bytes"] == 0
    assert record["sha256"] is None
    assert record["ambiguity_reason"] is not None
    assert set(observed["ambiguity_reasons"]) == {
        Closure.STAGING_PROVENANCE_AMBIGUITY,
        "ambiguous:" + record["ambiguity_reason"],
    }
    assert entry.exists() or entry.is_symlink()
    if kind == "nested":
        assert (entry / "retained").read_bytes() == b"nested\n"
    elif kind == "symlink":
        assert entry.is_symlink()
        assert entry.read_bytes() == b"outside\n"
    else:
        assert entry.stat().st_size == (
            Closure.MAX_STAGING_OBSERVATION_ENTRY_BYTES + 1
        )


def test_quarantined_staging_observes_over_entry_bound_without_deletion(
    tmp_path: Path,
):
    root = (tmp_path / "closure").resolve()
    staging = root.parent / Closure._staging_name(root)
    staging.mkdir(mode=0o700)
    retained: dict[str, bytes] = {}
    for index in range(Closure.MAX_STAGING_OBSERVATION_ENTRIES + 1):
        name = f"retained-{index:02d}.bin"
        raw = f"retained-{index}\n".encode("ascii")
        entry = staging / name
        entry.write_bytes(raw)
        entry.chmod(0o400)
        retained[name] = raw

    observed = Closure.observe_quarantined_staging(root)
    assert observed == Closure.observe_quarantined_staging(root)
    assert observed["status"] == "AMBIGUOUS"
    assert observed["present"] is True
    assert observed["inventory_observed"] is False
    assert observed["entry_count_lower_bound"] == (
        Closure.MAX_STAGING_OBSERVATION_ENTRIES + 1
    )
    assert observed["entries"] == []
    assert observed["total_bytes"] == 0
    assert set(observed["ambiguity_reasons"]) == {
        Closure.STAGING_PROVENANCE_AMBIGUITY,
        "staging_inventory_exceeds_entry_bound",
    }
    assert {
        entry.name: entry.read_bytes() for entry in staging.iterdir()
    } == retained


def test_crash_atomic_publication_fault_matrix_is_behavioral(tmp_path: Path):
    """Exercise every publication boundary owned by the release invariant."""

    checkpoints = (
        "staging_created",
        "client_fsynced",
        "content_manifest_fsynced",
        "receipt_fsynced",
        "staging_directory_fsynced",
        "before_atomic_publication",
    )

    def case(name: str) -> Path:
        selected = tmp_path / name
        selected.mkdir()
        return selected

    for checkpoint in checkpoints:
        with pytest.MonkeyPatch.context() as isolated:
            test_prepublication_crash_preserves_staging_and_requires_fresh_root(
                case("prepublication_" + checkpoint),
                isolated,
                checkpoint,
            )
    with pytest.MonkeyPatch.context() as isolated:
        test_ordinary_prepublication_fault_cleans_partial_staging(
            case("ordinary_prepublication_fault"), isolated
        )
    with pytest.MonkeyPatch.context() as isolated:
        test_crash_during_partial_file_write_preserves_staging(
            case("partial_write"), isolated
        )
    with pytest.MonkeyPatch.context() as isolated:
        test_file_fsync_fault_cleans_non_authoritative_staging(
            case("file_fsync"), isolated
        )
    with pytest.MonkeyPatch.context() as isolated:
        test_atomic_publication_never_replaces_a_racing_destination(
            case("exclusive_destination_race"), isolated
        )
    with pytest.MonkeyPatch.context() as isolated:
        test_crash_after_atomic_rename_leaves_one_complete_valid_publication(
            case("post_rename_crash"), isolated
        )
    with pytest.MonkeyPatch.context() as isolated:
        test_parent_fsync_fault_preserves_complete_published_closure(
            case("parent_fsync"), isolated
        )
    test_preexisting_staging_with_unexpected_entry_is_preserved(
        case("ambiguous_inventory")
    )
    test_preexisting_empty_staging_is_preserved(
        case("ambiguous_empty")
    )
    test_preexisting_0400_public_prefix_staging_is_preserved(
        case("ambiguous_public_prefix")
    )
    test_preexisting_allowed_name_with_nonpublisher_bytes_is_preserved(
        case("ambiguous_bytes")
    )
    with pytest.MonkeyPatch.context() as isolated:
        test_same_uid_allowed_name_race_is_not_erased_by_failure_cleanup(
            case("same_uid_allowed_name_race"), isolated
        )
    test_quarantined_staging_observation_is_bounded_and_canonical(
        case("bounded_staging_observation")
    )
    for kind in ("nested", "symlink", "oversize"):
        test_quarantined_staging_observes_ambiguity_without_deletion(
            case("unsafe_staging_observation_" + kind), kind
        )
    test_quarantined_staging_observes_over_entry_bound_without_deletion(
        case("over_entry_bound_staging_observation")
    )


def test_validated_cleanup_and_republication_are_composable(tmp_path: Path):
    root, first = _prepared(tmp_path)
    Closure.remove_closure(root, first["receipt_sha256"])
    assert not root.exists()
    second = Closure.prepare_closure(root)
    assert Closure.validate_closure(
        root, second["receipt_sha256"]
    )["status"] == "PASS"
    assert (
        first["content_manifest_sha256"]
        == second["content_manifest_sha256"]
    )
    assert first["receipt_sha256"] != second["receipt_sha256"]


def test_cleanup_rejects_root_substituted_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root, prepared = _prepared(tmp_path)
    displaced = root.with_name("validated-original")
    original_validate = Closure.validate_closure

    def validate_then_substitute(destination, receipt_sha256):
        observed = original_validate(destination, receipt_sha256)
        root.rename(displaced)
        root.mkdir(mode=0o700)
        for name in Closure.EXACT_INVENTORY:
            target = root / name
            target.write_bytes((displaced / name).read_bytes())
            target.chmod(0o400)
        return observed

    monkeypatch.setattr(
        Closure, "validate_closure", validate_then_substitute
    )
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="reopened a substituted root",
    ):
        Closure.remove_closure(root, prepared["receipt_sha256"])
    assert root.is_dir()
    assert displaced.is_dir()


def test_symlink_substitution_invalidates_closure(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    source = root / Closure.CLIENT_NAME
    target = tmp_path / "outside.py"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(
        Closure.CompatibilityClosureError, match="symlinked"
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])


def test_hardlink_substitution_invalidates_closure(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    source = root / Closure.CLIENT_NAME
    target = tmp_path / "outside.py"
    target.write_bytes(source.read_bytes())
    source.unlink()
    os.link(target, source)
    with pytest.raises(
        Closure.CompatibilityClosureError, match="unaliased"
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])


def test_closure_path_drift_invalidates_receipt(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    moved = tmp_path / "moved"
    root.rename(moved)
    with pytest.raises(
        Closure.CompatibilityClosureError, match="path"
    ):
        Closure.validate_closure(moved, prepared["receipt_sha256"])


def test_symlinked_ancestor_cannot_prepare_or_validate_closure(
    tmp_path: Path,
):
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="physical canonical|symlinked ancestor",
    ):
        Closure.prepare_closure(alias / "new-closure")
    assert not (physical / "new-closure").exists()

    root, prepared = _prepared(physical)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="physical canonical|symlinked ancestor",
    ):
        Closure.validate_closure(
            alias / root.name,
            prepared["receipt_sha256"],
        )


def test_root_swap_during_inventory_fails_final_descriptor_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root, prepared = _prepared(tmp_path)
    observed = Closure.validate_closure(
        root, prepared["receipt_sha256"]
    )
    assert observed["launch_authorized"] is False
    assert all(
        (root / name).stat().st_mode & 0o777 == 0o400
        for name in Closure.EXACT_INVENTORY
    )
    moved = tmp_path / "moved-closure"
    original_scandir = Closure.os.scandir
    swapped = False

    def swap_then_scan(directory):
        nonlocal swapped
        if not swapped and isinstance(directory, int):
            swapped = True
            root.rename(moved)
            root.symlink_to(moved, target_is_directory=True)
        return original_scandir(directory)

    monkeypatch.setattr(Closure.os, "scandir", swap_then_scan)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="aliased|ancestor|physical",
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])
    assert swapped is True


@pytest.mark.parametrize("name", Closure.EXACT_INVENTORY)
def test_byte_identical_file_substitution_fails_final_identity_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
):
    root, prepared = _prepared(tmp_path)
    original_snapshot = Closure.canonical_closure_snapshot
    substituted = False

    def snapshot_then_substitute():
        nonlocal substituted
        snapshot = original_snapshot()
        if not substituted:
            substituted = True
            path = root / name
            raw = path.read_bytes()
            path.unlink()
            path.write_bytes(raw)
            path.chmod(0o400)
        return snapshot

    monkeypatch.setattr(
        Closure, "canonical_closure_snapshot", snapshot_then_substitute
    )
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="changed after validation read",
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])
    assert substituted is True


def test_control_snapshot_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root, prepared = _prepared(tmp_path)
    drifted = copy.deepcopy(Closure._LOADED_COMPONENTS)
    first = next(iter(drifted.values()))
    first["sha256"] = "0" * 64
    monkeypatch.setattr(Closure, "_component_snapshot", lambda: drifted)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="changed after module import",
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])


def test_compatibility_closure_has_no_standalone_launch_authority(
    tmp_path: Path,
):
    assert Closure._authority_projection()["launch_authorized"] is False
    root, prepared = _prepared(tmp_path)
    assert prepared["launch_authorized"] is False
    assert Closure.validate_closure(
        root, prepared["receipt_sha256"]
    )["launch_authorized"] is False


@pytest.mark.parametrize("name", [Closure.CLIENT_NAME, Closure.RECEIPT_NAME])
def test_mutable_closure_file_mode_invalidates_custody(
    tmp_path: Path,
    name: str,
):
    root, prepared = _prepared(tmp_path)
    (root / name).chmod(0o600)
    with pytest.raises(
        Closure.CompatibilityClosureError,
        match="custody, owner, links, or mode",
    ):
        Closure.validate_closure(root, prepared["receipt_sha256"])


def test_client_or_receipt_hash_drift_invalidates_closure(tmp_path: Path):
    root, prepared = _prepared(tmp_path)
    client = root / Closure.CLIENT_NAME
    client.chmod(0o600)
    client.write_bytes(client.read_bytes() + b"\n# drift\n")
    with pytest.raises(Closure.CompatibilityClosureError):
        Closure.validate_closure(root, prepared["receipt_sha256"])

    other_root, other = _prepared(tmp_path / "other")
    with pytest.raises(
        Closure.CompatibilityClosureError, match="receipt hash"
    ):
        Closure.validate_closure(other_root, "0" * 64)
    assert other["receipt_sha256"] != "0" * 64


def test_canonical_client_interoperates_with_existing_host_rpc():
    scratch = Path(tempfile.mkdtemp(
        prefix="gkm-rpc-client-", dir="/tmp"
    )).resolve()
    socket_path = scratch / "s"
    assert len(os.fsencode(socket_path)) <= (
        Container.MAX_PORTABLE_UNIX_SOCKET_PATH_BYTES
    )
    transcript = scratch / "host.jsonl"
    host = HostRpc.ArenaHostSession(
        "zz99",
        binding=_binding(),
        parent_path=(),
        arena_factory=_Arena,
        token="t" * 64,
        real_step_cap=6,
        total_step_cap=40,
        reset_cap=8,
    )
    server = HostRpc.ArenaRpcServer(host, socket_path, transcript)
    thread = server.start_thread()
    try:
        with ClientRpc.ArenaRpcClient(socket_path, host.token) as client:
            assert client.root.levels_completed == 0
            client.root.step(2)
            client.root.step(2)
            client.root.step(1)
            assert client.root.levels_completed == 1
        thread.join(timeout=5)
        server.wait(1)
        assert host.host_result().path == (2, 2, 1)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        shutil.rmtree(scratch)


def test_client_and_host_share_exact_schema_and_mac():
    value = {"schema": HostRpc.RPC_SCHEMA, "kind": "session"}
    assert ClientRpc.RPC_SCHEMA == HostRpc.RPC_SCHEMA
    assert HostRpc.ArenaRpcClient is ClientRpc.ArenaRpcClient
    assert HostRpc.RemoteArena is ClientRpc.RemoteArena
    assert HostRpc.ArenaRpcError is ClientRpc.ArenaRpcError
    assert HostRpc.ArenaRpcContractError is ClientRpc.ArenaRpcContractError
    assert ClientRpc._wire_mac("t" * 64, value) == HostRpc._wire_mac(
        "t" * 64, value
    )


def test_malformed_socket_paths_close_the_allocated_transport(monkeypatch):
    created = []

    class TrackingSocket:
        def __init__(self):
            self.closed = False

        def settimeout(self, _seconds):
            return None

        def connect(self, _path):
            raise OSError("invalid socket path")

        def close(self):
            self.closed = True

    class InvalidPathLike:
        def __fspath__(self):
            raise TypeError("invalid path")

    def create_socket(*_args):
        observed = TrackingSocket()
        created.append(observed)
        return observed

    monkeypatch.setattr(ClientRpc.socket, "socket", create_socket)
    for invalid in ("", object(), InvalidPathLike()):
        with pytest.raises((TypeError, OSError)):
            ClientRpc.ArenaRpcClient(invalid, "t" * 64)
    assert len(created) == 3
    assert all(observed.closed for observed in created)


def test_container_and_workers_expose_only_purified_client():
    root = Path(Closure.__file__).resolve().parent
    recipe = (
        root / "container" / "Containerfile.arc-agi3-contiguous"
    ).read_text(encoding="utf-8")
    container_worker = (root / "arc_agi3_container_worker.py").read_text(
        encoding="utf-8"
    )
    proposer_worker = (root / "arc_agi3_proposer_worker.py").read_text(
        encoding="utf-8"
    )
    assert "COPY arc/crack_lab/arc_agi3_arena_rpc_client.py" in recipe
    assert "site-packages/arc_agi3_arena_rpc.py" not in recipe
    assert (
        "from arc_agi3_arena_rpc_client import ArenaRpcClient"
        in container_worker
    )
    assert (
        "from arc_agi3_arena_rpc_client import ArenaRpcClient"
        in proposer_worker
    )
