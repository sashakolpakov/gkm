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


def test_compatibility_closure_has_no_production_call_site_or_authority():
    root = Path(Closure.__file__).resolve().parent
    for path in root.glob("*.py"):
        if path.name in {
            "arc_agi3_compatibility_arena_closure.py",
            "test_arc_agi3_compatibility_arena_closure.py",
        }:
            continue
        source = path.read_text(encoding="utf-8")
        assert "prepare_closure(" not in source
        assert "validate_closure(" not in source
    assert Closure._authority_projection()["launch_authorized"] is False


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
