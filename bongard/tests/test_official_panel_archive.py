from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_json
from bongard.official_panel_archive import (
    ARCHIVE_SCHEMA,
    OfficialPanelArchive,
    OfficialPanelArchiveError,
    OfficialPanelReceipt,
    ReleasedOfficialPanel,
)
from bongard.release import OfficialReleaseDescriptor


PANEL_ID = "bd/bd_triangle-circle_0000/1/0.png"
MEMBER = "ShapeBongard_V2/bd/images/bd_triangle-circle_0000/1/0.png"
FF_PANEL_ID = "ff/ff_nact2_5_0042/0/6.png"
FF_MEMBER = "ShapeBongard_V2/ff/images/ff_nact2_5_0042/0/6.png"
HD_PANEL_ID = "hd/hd_has_curve-exist_quadrangle_0011/1/3.png"
HD_MEMBER = (
    "ShapeBongard_V2/hd/images/hd_has_curve-exist_quadrangle_0011/1/3.png"
)
PRECOMMIT = "sha256:" + "c" * 64
EXPOSURE = "sha256:" + "d" * 64


def _png() -> bytes:
    image = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon([(4, 27), (16, 4), (28, 27)], outline="black", width=2)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@pytest.fixture
def official_zip(tmp_path: Path):
    payload = _png()
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr(MEMBER, payload)
        bundle.writestr(FF_MEMBER, payload)
        bundle.writestr(HD_MEMBER, payload)
        bundle.writestr("ShapeBongard_V2/README.txt", b"synthetic fixture\n")
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-panel-archive-test",
        archive_filename=archive_path.name,
        archive_sha256=(
            "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
        ),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + "1" * 64,
        split_size_bytes=1,
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=(("bd", 1), ("ff", 1), ("hd", 1)),
        primary_split_counts=(("test", 0), ("train", 3), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256="sha256:" + "3" * 64,
        corpus_manifest_sha256="sha256:" + "4" * 64,
    )
    return descriptor, archive_path, payload


def _load(official_zip) -> OfficialPanelArchive:
    descriptor, archive_path, _payload = official_zip
    return OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )


def test_load_read_release_and_cold_replay_exact_official_bytes(
    official_zip,
) -> None:
    descriptor, _archive_path, expected_png = official_zip
    archive = _load(official_zip)
    identity = archive.identity_data()
    assert identity["schema"] == ARCHIVE_SCHEMA
    assert identity["release_descriptor_digest"] == descriptor.digest
    assert identity["archive_digest"] == descriptor.archive_sha256
    assert archive.record_digest.startswith("sha256:")
    assert identity["python_is_canonical_authority"] is True
    assert identity["lean_present"] is False
    assert identity["lean_required"] is False

    payload, receipt = archive.read_panel(PANEL_ID)
    assert payload == expected_png
    assert receipt.panel_id == PANEL_ID
    assert receipt.archive_member == MEMBER
    assert receipt.sha256 == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert receipt.release_descriptor_digest == descriptor.digest
    assert receipt.archive_digest == descriptor.archive_sha256
    assert OfficialPanelReceipt.from_data(receipt.to_data()) == receipt
    assert archive.verify_panel(payload, receipt) is receipt
    assert archive.verify_panel(payload, receipt.to_data()) == receipt

    released = ReleasedOfficialPanel.release(
        archive,
        PANEL_ID,
        execution_precommit_digest=PRECOMMIT,
        exposure_successor_digest=EXPOSURE,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )
    assert released.exact_png_bytes == expected_png
    assert released.execution_precommit_digest == PRECOMMIT
    assert released.exposure_successor_digest == EXPOSURE
    released.cold_verify(
        archive,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )

    encoded = canonical_json(released.to_data())
    decoded = json.loads(encoded)
    assert canonical_json(decoded) == encoded
    restored = ReleasedOfficialPanel.from_data(decoded)
    assert restored == released
    assert restored.to_data() == released.to_data()
    restored.cold_verify(
        archive,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )


@pytest.mark.parametrize(
    ("panel_id", "member"),
    ((PANEL_ID, MEMBER), (FF_PANEL_ID, FF_MEMBER), (HD_PANEL_ID, HD_MEMBER)),
)
def test_archive_reads_every_official_family_namespace(
    official_zip, panel_id: str, member: str
) -> None:
    _descriptor, _archive_path, expected_png = official_zip
    archive = _load(official_zip)
    payload, receipt = archive.read_panel(panel_id)
    assert payload == expected_png
    assert receipt.panel_id == panel_id
    assert receipt.archive_member == member
    assert archive.verify_panel(payload, receipt) == receipt


def test_archive_and_panel_tamper_fail_cold_replay(official_zip) -> None:
    _descriptor, archive_path, expected_png = official_zip
    archive = _load(official_zip)
    payload, receipt = archive.read_panel(PANEL_ID)

    changed_payload = bytearray(payload)
    changed_payload[-5] ^= 1
    with pytest.raises(OfficialPanelArchiveError, match="cold replay"):
        archive.verify_panel(bytes(changed_payload), receipt)

    receipt_data = deepcopy(receipt.to_data())
    receipt_data["central_directory_digest"] = "sha256:" + "0" * 64
    with pytest.raises(OfficialPanelArchiveError):
        OfficialPanelReceipt.from_data(receipt_data)

    released = ReleasedOfficialPanel.release(
        archive,
        PANEL_ID,
        execution_precommit_digest=PRECOMMIT,
        exposure_successor_digest=EXPOSURE,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )
    released_data = deepcopy(released.to_data())
    released_data["exact_png_base64"] = base64.b64encode(
        expected_png + b"foreign"
    ).decode("ascii")
    with pytest.raises(OfficialPanelArchiveError):
        ReleasedOfficialPanel.from_data(released_data)

    original_archive = archive_path.read_bytes()
    changed_archive = bytearray(original_archive)
    changed_archive[20] ^= 1
    archive_path.write_bytes(bytes(changed_archive))
    with pytest.raises(OfficialPanelArchiveError, match="changed after pinning"):
        archive.read_panel(PANEL_ID)


def test_wrong_release_precommit_and_exposure_addresses_fail_closed(
    official_zip,
) -> None:
    descriptor, archive_path, _payload = official_zip
    with pytest.raises(OfficialPanelArchiveError, match="release descriptor"):
        OfficialPanelArchive.load(
            descriptor,
            archive_path,
            expected_release_descriptor_digest="sha256:" + "0" * 64,
        )

    archive = _load(official_zip)
    with pytest.raises(OfficialPanelArchiveError, match="precommit"):
        ReleasedOfficialPanel.release(
            archive,
            PANEL_ID,
            execution_precommit_digest="not-an-address",
            exposure_successor_digest=EXPOSURE,
            expected_execution_precommit_digest=PRECOMMIT,
            expected_exposure_successor_digest=EXPOSURE,
        )
    with pytest.raises(OfficialPanelArchiveError, match="exposure"):
        ReleasedOfficialPanel.release(
            archive,
            PANEL_ID,
            execution_precommit_digest=PRECOMMIT,
            exposure_successor_digest="sha256:" + "f" * 63,
            expected_execution_precommit_digest=PRECOMMIT,
            expected_exposure_successor_digest=EXPOSURE,
        )
    with pytest.raises(OfficialPanelArchiveError, match="external commitment"):
        ReleasedOfficialPanel.release(
            archive,
            PANEL_ID,
            execution_precommit_digest=PRECOMMIT,
            exposure_successor_digest=EXPOSURE,
            expected_execution_precommit_digest="sha256:" + "e" * 64,
            expected_exposure_successor_digest=EXPOSURE,
        )
    with pytest.raises(OfficialPanelArchiveError, match="durable commitment"):
        ReleasedOfficialPanel.release(
            archive,
            PANEL_ID,
            execution_precommit_digest=PRECOMMIT,
            exposure_successor_digest=EXPOSURE,
            expected_execution_precommit_digest=PRECOMMIT,
            expected_exposure_successor_digest="sha256:" + "f" * 64,
        )

    released = ReleasedOfficialPanel.release(
        archive,
        PANEL_ID,
        execution_precommit_digest=PRECOMMIT,
        exposure_successor_digest=EXPOSURE,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )
    for field, replacement in (
        ("execution_precommit_digest", "sha256:" + "e" * 64),
        ("exposure_successor_digest", "sha256:" + "f" * 64),
    ):
        changed = deepcopy(released.to_data())
        changed[field] = replacement
        with pytest.raises(OfficialPanelArchiveError):
            ReleasedOfficialPanel.from_data(changed)
    with pytest.raises(OfficialPanelArchiveError, match="external commitment"):
        released.cold_verify(
            archive,
            expected_execution_precommit_digest="sha256:" + "e" * 64,
            expected_exposure_successor_digest=EXPOSURE,
        )
    with pytest.raises(OfficialPanelArchiveError, match="durable commitment"):
        released.cold_verify(
            archive,
            expected_execution_precommit_digest=PRECOMMIT,
            expected_exposure_successor_digest="sha256:" + "f" * 64,
        )


def test_namespace_missing_member_and_noncanonical_archives_are_rejected(
    official_zip,
) -> None:
    archive = _load(official_zip)
    for panel_id in (
        "../bd_triangle-circle_0000/1/0.png",
        "bd/bd_triangle-circle_0000/2/0.png",
        "bd/bd_triangle-circle_0000/1/7.png",
    ):
        with pytest.raises(OfficialPanelArchiveError, match="namespace"):
            archive.read_panel(panel_id)
    with pytest.raises(OfficialPanelArchiveError, match="absent"):
        archive.read_panel("bd/bd_missing_0000/1/0.png")

    payload, receipt = archive.read_panel(PANEL_ID)
    extra_receipt = deepcopy(receipt.to_data())
    extra_receipt["extra"] = True
    with pytest.raises(OfficialPanelArchiveError, match="fields differ"):
        OfficialPanelReceipt.from_data(extra_receipt)

    released = ReleasedOfficialPanel.release(
        archive,
        PANEL_ID,
        execution_precommit_digest=PRECOMMIT,
        exposure_successor_digest=EXPOSURE,
        expected_execution_precommit_digest=PRECOMMIT,
        expected_exposure_successor_digest=EXPOSURE,
    )
    noncanonical_base64 = deepcopy(released.to_data())
    noncanonical_base64["exact_png_base64"] += "\n"
    with pytest.raises(OfficialPanelArchiveError, match="malformed"):
        ReleasedOfficialPanel.from_data(noncanonical_base64)
    assert payload == released.exact_png_bytes


def test_archive_constructor_binding_and_predecompression_bound(
    official_zip,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _load(official_zip)
    with pytest.raises(TypeError):
        OfficialPanelArchive(  # type: ignore[call-arg]
            release_descriptor_digest=archive.release_descriptor_digest,
            archive_digest=archive.archive_digest,
            archive_size_bytes=archive.archive_size_bytes,
            central_directory_digest=archive.central_directory_digest,
            archive_path=archive.archive_path,
            archive_identity=archive.archive_identity,
            members=archive.members,
            record_digest=archive.record_digest,
        )
    for field in ("central_directory_digest", "record_digest"):
        tampered_archive = object.__new__(OfficialPanelArchive)
        for name in OfficialPanelArchive.__dataclass_fields__:
            object.__setattr__(tampered_archive, name, getattr(archive, name))
        object.__setattr__(tampered_archive, field, "sha256:" + "0" * 64)
        with pytest.raises(OfficialPanelArchiveError, match="binding"):
            tampered_archive.__post_init__()

    oversized_path = tmp_path / "ShapeBongard_V2.zip"
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (4_000_001 - 8)
    with zipfile.ZipFile(
        oversized_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr(MEMBER, oversized)
    archive_bytes = oversized_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-oversized-panel-test",
        archive_filename=oversized_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + "1" * 64,
        split_size_bytes=1,
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=(("bd", 1),),
        primary_split_counts=(("test", 0), ("train", 1), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256="sha256:" + "3" * 64,
        corpus_manifest_sha256="sha256:" + "4" * 64,
    )
    oversized_archive = OfficialPanelArchive.load(
        descriptor,
        oversized_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    reads = 0

    def forbidden_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("oversized member must not be decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden_read)
    with pytest.raises(OfficialPanelArchiveError, match="pre-decompression"):
        oversized_archive.read_panel(PANEL_ID)
    assert reads == 0


def test_module_imports_no_lean_or_historical_runner() -> None:
    source_path = Path(__file__).parents[1] / "official_panel_archive.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    lowered = tuple(name.lower() for name in imported)
    assert not any("lean" in name for name in lowered)
    assert not any("runner" in name for name in lowered)
    assert not any(
        name in {
            "bongard.multimodal_headless_runner",
            "bongard.grounded_headless_runner",
            "bongard.relational_headless_runner",
            "bongard.prototype_scene_headless_runner",
        }
        for name in lowered
    )
