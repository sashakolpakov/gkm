from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from bongard.corpus import ShapeBongardCorpus
from bongard.image_audit import (
    ImageAuditError,
    ImageExpectationError,
    ImageExpectations,
    audit_corpus_images,
)
import bongard.image_audit as image_audit


def _save_png(
    path: Path,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (11, 7),
    colour: object = (20, 40, 60),
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = None
    if metadata:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in sorted(metadata.items()):
            pnginfo.add_text(key, value)
    Image.new(mode, size, colour).save(path, format="PNG", pnginfo=pnginfo)


def _corpus(tmp_path: Path, *, metadata: dict[str, str] | None = None) -> ShapeBongardCorpus:
    root = tmp_path / "ShapeBongard_V2"
    task_id = "ff_audit_0000"
    for label in ("1", "0"):
        for index in range(7):
            _save_png(
                root / "ff" / "images" / task_id / label / f"{index}.png",
                colour=(index, 1 if label == "1" else 0, 100),
                metadata=metadata,
            )
    return ShapeBongardCorpus.from_root(root)


def test_report_is_compact_canonical_deterministic_and_manifest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path, metadata={"audit-note": "same on every panel"})
    manifest = corpus.build_manifest()

    # Supplying a manifest must not quietly rebuild (and rehash) it.
    monkeypatch.setattr(
        corpus,
        "build_manifest",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected second manifest build")),
    )
    first = audit_corpus_images(corpus, corpus_manifest=manifest)
    second = audit_corpus_images(corpus, corpus_manifest=manifest)

    assert first == second
    assert first.task_count == 1
    assert first.panel_count == 14
    assert first.family_task_counts == (("ff", 1),)
    assert first.family_panel_counts == (("ff", 14),)
    assert first.format_counts == (("PNG", 14),)
    assert first.mode_counts == (("RGB", 14),)
    assert first.size_counts == ((11, 7, 14),)
    assert first.info_key_set_counts == ((('audit-note',), 14),)
    assert first.frame_count_counts == ((1, 14),)
    assert first.corpus_manifest_digest == manifest.digest
    assert first.content_summary_digest.startswith("sha256:")
    assert first.property_summary_digest.startswith("sha256:")
    assert first.digest.startswith("sha256:")
    serialised = json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)
    assert str(corpus.root) not in serialised
    assert "decoded" not in serialised


def test_expected_properties_are_explicit_and_strict_mode_fails_with_report(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    observed = audit_corpus_images(corpus)
    assert observed.anomaly_count == 0

    expected = ImageExpectations("RGB", 11, 7, (), 1)
    confirmed = audit_corpus_images(
        corpus,
        expected_properties=expected,
        require_expected_properties=True,
    )
    assert confirmed.expectations == expected
    assert confirmed.anomaly_count == 0

    wrong = ImageExpectations("L", 9, 9, ("invented",), 2)
    diagnostic = audit_corpus_images(
        corpus, expected_properties=wrong, max_anomalies=3
    )
    assert diagnostic.anomaly_count == 14 * 4
    assert len(diagnostic.anomalies) == 3
    assert diagnostic.anomalies_truncated is True

    with pytest.raises(ImageExpectationError) as caught:
        audit_corpus_images(
            corpus,
            expected_properties=wrong,
            require_expected_properties=True,
            max_anomalies=2,
        )
    assert caught.value.report.anomaly_count == 14 * 4
    assert len(caught.value.report.anomalies) == 2

    with pytest.raises(ValueError, match="explicit expected_properties"):
        audit_corpus_images(corpus, require_expected_properties=True)


@pytest.mark.parametrize("damage", ["signature", "truncated", "trailing"])
def test_fake_corrupt_or_non_exact_png_fails_closed(tmp_path: Path, damage: str) -> None:
    corpus = _corpus(tmp_path)
    victim = corpus.task("ff_audit_0000").positive[0]
    payload = victim.read_bytes()
    if damage == "signature":
        victim.write_bytes(b"not-a-png")
    elif damage == "truncated":
        victim.write_bytes(payload[:-9])
    else:
        victim.write_bytes(payload + b"hidden trailing payload")

    with pytest.raises(ImageAuditError):
        audit_corpus_images(corpus)


def test_manifest_detects_valid_png_content_change(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    manifest = corpus.build_manifest()
    victim = corpus.task("ff_audit_0000").negative[0]
    _save_png(victim, colour=(255, 0, 0))

    with pytest.raises(ImageAuditError, match="differs from manifest"):
        audit_corpus_images(corpus, corpus_manifest=manifest)


def test_symlink_and_non_regular_png_are_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    victim = corpus.task("ff_audit_0000").positive[0]
    target = tmp_path / "outside.png"
    _save_png(target)
    victim.unlink()
    try:
        victim.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ImageAuditError, match="symlink"):
        audit_corpus_images(corpus)

    victim.unlink()
    victim.mkdir()
    with pytest.raises(ImageAuditError, match="not a regular file"):
        audit_corpus_images(corpus)


def test_task_directory_symlink_followed_by_discovery_is_still_rejected(
    tmp_path: Path,
) -> None:
    source = _corpus(tmp_path / "source")
    source_task = source.task("ff_audit_0000").root
    root = tmp_path / "linked" / "ShapeBongard_V2"
    task_link = root / "ff" / "images" / "ff_audit_0000"
    task_link.parent.mkdir(parents=True)
    try:
        task_link.symlink_to(source_task, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    linked = ShapeBongardCorpus.from_root(root)

    with pytest.raises(ImageAuditError, match="symlink"):
        audit_corpus_images(linked)


def test_mutation_during_decode_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    victim = corpus.task("ff_audit_0000").positive[0]
    real_decode = image_audit._decode_png_snapshot
    mutated = False

    def mutate_once(snapshot: object, *, max_pixels: int, max_frames: int):
        nonlocal mutated
        properties = real_decode(
            snapshot, max_pixels=max_pixels, max_frames=max_frames  # type: ignore[arg-type]
        )
        if not mutated:
            mutated = True
            _save_png(victim, colour=(199, 198, 197))
        return properties

    monkeypatch.setattr(image_audit, "_decode_png_snapshot", mutate_once)
    with pytest.raises(ImageAuditError, match="changed during decode"):
        audit_corpus_images(corpus)


def test_safety_limits_fail_before_unbounded_decode(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    panel_size = corpus.task("ff_audit_0000").positive[0].stat().st_size

    with pytest.raises(ImageAuditError, match="byte safety limit"):
        audit_corpus_images(corpus, max_panel_bytes=panel_size - 1)
    with pytest.raises(ImageAuditError, match="pixel safety limit"):
        audit_corpus_images(corpus, max_pixels=10)


def test_manifest_object_tampering_is_rejected_before_panel_reads(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    manifest = corpus.build_manifest()
    object.__setattr__(manifest, "digest", "sha256:" + "0" * 64)

    with pytest.raises(ImageAuditError, match="manifest content"):
        audit_corpus_images(corpus, corpus_manifest=manifest)


def test_invalid_expectation_and_limit_arguments_are_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(ValueError, match="in order"):
        ImageExpectations("RGB", 11, 7, ("z", "a"), 1)
    with pytest.raises(ValueError, match="max_frames"):
        audit_corpus_images(corpus, max_frames=0)
    with pytest.raises(ValueError, match="max_anomalies"):
        audit_corpus_images(corpus, max_anomalies=0)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable")
def test_fifo_with_png_suffix_is_rejected_without_opening(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    victim = corpus.task("ff_audit_0000").negative[0]
    victim.unlink()
    os.mkfifo(victim)

    with pytest.raises(ImageAuditError, match="not a regular file"):
        audit_corpus_images(corpus)
