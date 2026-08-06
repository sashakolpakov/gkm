from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bongard.corpus import CorpusManifest, SplitIndex
from bongard.release import (
    OfficialReleaseDescriptor,
    ReleaseIdentityError,
)


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _descriptor(
    *,
    archive: bytes = b"official archive",
    split: bytes = b"official split",
    task_ids: tuple[str, ...] = ("bd_one_0000", "ff_one_0000", "hd_one_0000"),
    manifest_digest: str = "sha256:" + "4" * 64,
) -> dict[str, object]:
    return {
        "schema": "gkm.shape-bongard-official-release.v1",
        "release_id": "ShapeBongard_V2",
        "archive": {
            "filename": "ShapeBongard_V2.zip",
            "sha256": _address(archive),
            "size_bytes": len(archive),
        },
        "split": {
            "filename": "ShapeBongard_V2_split.json",
            "sha256": _address(split),
            "size_bytes": len(split),
        },
        "upstream": {
            "repository": "https://github.com/NVlabs/Bongard-LOGO",
            "commit": "9df7c78ee9c6a2ff041b48d9ed407359aac259c3",
        },
        "family_counts": {"bd": 1, "ff": 1, "hd": 1},
        "primary_split_counts": {"test": 1, "train": 1, "val": 1},
        "regime_counts": {"BA": 1, "CM": 0, "FF": 0, "NV": 0},
        "task_ids_sha256": _address(
            "".join(f"{task_id}\n" for task_id in sorted(task_ids)).encode("utf-8")
        ),
        "corpus_manifest_sha256": manifest_digest,
    }


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    )


def test_descriptor_loads_only_canonical_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    raw = _descriptor()
    _write_canonical(path, raw)

    descriptor = OfficialReleaseDescriptor.load(path)

    assert descriptor.release_id == "ShapeBongard_V2"
    assert descriptor.family_counts == (("bd", 1), ("ff", 1), ("hd", 1))
    assert OfficialReleaseDescriptor.from_dict(descriptor.to_dict()) == descriptor
    assert descriptor.digest.startswith("sha256:")

    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(ReleaseIdentityError, match="canonical JSON"):
        OfficialReleaseDescriptor.load(path)

    raw["invented"] = True
    _write_canonical(path, raw)
    with pytest.raises(ReleaseIdentityError, match="fields differ"):
        OfficialReleaseDescriptor.load(path)


def test_archive_and_split_are_exact_byte_identities(tmp_path: Path) -> None:
    archive_payload = b"official archive"
    split_payload = b"official split"
    descriptor = OfficialReleaseDescriptor.from_dict(
        _descriptor(archive=archive_payload, split=split_payload)
    )
    archive = tmp_path / "ShapeBongard_V2.zip"
    split = tmp_path / "ShapeBongard_V2_split.json"
    archive.write_bytes(archive_payload)
    split.write_bytes(split_payload)

    descriptor.verify_archive(archive)
    descriptor.verify_split(split)

    archive.write_bytes(archive_payload + b"tampered")
    with pytest.raises(ReleaseIdentityError, match="archive identity"):
        descriptor.verify_archive(archive)
    with pytest.raises(ReleaseIdentityError, match="filename"):
        descriptor.verify_split(tmp_path / "ShapeBongard_V2.zip")


def test_extracted_corpus_check_binds_inventory_split_and_manifest(tmp_path: Path) -> None:
    split_payload = b"official split"
    split_path = tmp_path / "ShapeBongard_V2_split.json"
    split_path.write_bytes(split_payload)
    task_ids = ("bd_one_0000", "ff_one_0000", "hd_one_0000")
    manifest = CorpusManifest(
        layout="fixture",
        family_counts=(("bd", 1), ("ff", 1), ("hd", 1)),
        tasks=(),
        split=SplitIndex.empty(),
        digest="sha256:" + "4" * 64,
    )
    manifest_digest = manifest.digest
    descriptor = OfficialReleaseDescriptor.from_dict(
        _descriptor(
            split=split_payload,
            task_ids=task_ids,
            manifest_digest=manifest_digest,
        )
    )
    groups = {
        "train": (task_ids[0],),
        "val": (task_ids[1],),
        "test": (task_ids[2],),
        "FF": (),
        "BA": (task_ids[2],),
        "CM": (),
        "NV": (),
    }

    class FakeCorpus:
        def __init__(self) -> None:
            self.task_ids = task_ids
            self.family_counts = {"bd": 1, "ff": 1, "hd": 1}
            self.split = SimpleNamespace(
                source_path=split_path,
                canonical_groups=groups,
            )

        def validate_complete(self, *, require_split: bool) -> None:
            assert require_split

        def build_manifest(self) -> CorpusManifest:
            return manifest

    corpus = FakeCorpus()
    assert descriptor.verify_corpus(corpus, manifest=manifest) is manifest

    bad_manifest = CorpusManifest(
        layout="decoy",
        family_counts=manifest.family_counts,
        tasks=(),
        split=SplitIndex.empty(),
        digest="sha256:" + "5" * 64,
    )
    with pytest.raises(ReleaseIdentityError, match="supplied corpus manifest differs"):
        descriptor.verify_corpus(corpus, manifest=bad_manifest)


def test_supplied_manifest_cannot_replace_fresh_corpus_rebuild(tmp_path: Path) -> None:
    split_payload = b"official split"
    split_path = tmp_path / "ShapeBongard_V2_split.json"
    split_path.write_bytes(split_payload)
    task_ids = ("bd_one_0000", "ff_one_0000", "hd_one_0000")
    rebuilt = CorpusManifest(
        layout="fixture",
        family_counts=(("bd", 1), ("ff", 1), ("hd", 1)),
        tasks=(),
        split=SplitIndex.empty(),
        digest="sha256:" + "4" * 64,
    )
    descriptor = OfficialReleaseDescriptor.from_dict(
        _descriptor(
            split=split_payload,
            task_ids=task_ids,
            manifest_digest=rebuilt.digest,
        )
    )
    decoy = CorpusManifest(
        layout="decoy",
        family_counts=rebuilt.family_counts,
        tasks=(),
        split=SplitIndex.empty(),
        digest="sha256:" + "5" * 64,
    )

    class FakeCorpus:
        family_counts = {"bd": 1, "ff": 1, "hd": 1}
        split = SimpleNamespace(
            source_path=split_path,
            canonical_groups={
                "train": (task_ids[0],),
                "val": (task_ids[1],),
                "test": (task_ids[2],),
                "FF": (),
                "BA": (task_ids[2],),
                "CM": (),
                "NV": (),
            },
        )

        def __init__(self) -> None:
            self.task_ids = task_ids

        def validate_complete(self, *, require_split: bool) -> None:
            assert require_split

        def build_manifest(self) -> CorpusManifest:
            return rebuilt

    with pytest.raises(ReleaseIdentityError, match="supplied corpus manifest differs"):
        descriptor.verify_corpus(FakeCorpus(), manifest=decoy)


def test_descriptor_rejects_placeholder_or_malformed_identities() -> None:
    raw = _descriptor()
    raw["task_ids_sha256"] = "pending"
    with pytest.raises(ReleaseIdentityError, match="task_ids_sha256"):
        OfficialReleaseDescriptor.from_dict(raw)

    raw = _descriptor()
    raw["upstream"] = {
        "repository": "https://github.com/NVlabs/Bongard-LOGO",
        "commit": "main",
    }
    with pytest.raises(ReleaseIdentityError, match="40-hex"):
        OfficialReleaseDescriptor.from_dict(raw)
