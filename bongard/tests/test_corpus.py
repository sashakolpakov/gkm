from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from bongard.corpus import (
    CorpusLayoutError,
    CorpusValidationError,
    ShapeBongardCorpus,
)


PNG = b"\x89PNG\r\n\x1a\nsynthetic-panel-"


def _task(root: Path, family: str, task_id: str, *, component: str = "images") -> None:
    for label in ("1", "0"):
        label_dir = root / family / component / task_id / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            (label_dir / f"{index}.png").write_bytes(
                PNG + f"{task_id}/{label}/{index}".encode("utf-8")
            )


def _archive(root: Path) -> dict[str, str]:
    ids = {
        "train": "ff_train_0000",
        "val": "bd_val_0000",
        "FF": "ff_test_0000",
        "BA": "bd_test_0000",
        "CM": "hd_comb_0000",
        "NV": "hd_novel_0000",
    }
    for key, task_id in ids.items():
        _task(root, task_id.split("_", 1)[0], task_id)
    split = {
        "train": [ids["train"]],
        "val": [ids["val"]],
        "test_ff": [ids["FF"]],
        "test_bd": [ids["BA"]],
        "test_hd_comb": [ids["CM"]],
        "test_hd_novel": [ids["NV"]],
    }
    (root / "ShapeBongard_V2_split.json").write_text(json.dumps(split), encoding="utf-8")
    return ids


def test_discovers_wrapped_archive_and_normalises_official_split(tmp_path: Path) -> None:
    root = tmp_path / "materials" / "ShapeBongard_V2"
    ids = _archive(root)

    corpus = ShapeBongardCorpus.discover(tmp_path)

    assert corpus.root == root.resolve()
    assert corpus.layout == "archive"
    assert len(corpus) == 6
    assert corpus.family_counts == {"ff": 2, "bd": 2, "hd": 2}
    assert corpus.assignment(ids["train"]).split == "train"
    assert corpus.assignment(ids["val"]).split == "val"
    for regime in ("FF", "BA", "CM", "NV"):
        assignment = corpus.assignment(ids[regime])
        assert assignment.split == "test"
        assert assignment.regime == regime
        assert [task.task_id for task in corpus.tasks_in_split(regime)] == [ids[regime]]
    assert set(task.task_id for task in corpus.tasks_in_split("test")) == {
        ids["FF"], ids["BA"], ids["CM"], ids["NV"]
    }


def test_loads_exactly_seven_paths_per_polarity(tmp_path: Path) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_only_0000")

    corpus = ShapeBongardCorpus.from_root(root)
    task = corpus.task("ff_only_0000")

    assert len(task.positive_paths) == 7
    assert len(task.negative_paths) == 7
    assert all(path.is_absolute() and path.suffix == ".png" for path in task.panels)
    assert "/1/" in task.positive_paths[0].as_posix()
    assert "/0/" in task.negative_paths[0].as_posix()


def test_generator_png_layout_is_supported(tmp_path: Path) -> None:
    _task(tmp_path, "hd", "hd_generated_0000", component="png")

    corpus = ShapeBongardCorpus.discover(tmp_path)

    assert corpus.layout == "generator"
    assert corpus.task_ids == ("hd_generated_0000",)


def test_malformed_task_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_broken_0000")
    (root / "ff" / "images" / "ff_broken_0000" / "1" / "6.png").unlink()

    with pytest.raises(CorpusValidationError, match="expected 7 PNGs"):
        ShapeBongardCorpus.from_root(root)


def test_manifest_is_relocatable_and_detects_content_change(tmp_path: Path) -> None:
    first = tmp_path / "first" / "ShapeBongard_V2"
    second = tmp_path / "second" / "ShapeBongard_V2"
    _task(first, "ff", "ff_hash_0000")
    shutil.copytree(first, second)

    first_corpus = ShapeBongardCorpus.from_root(first)
    second_corpus = ShapeBongardCorpus.from_root(second)
    first_manifest = first_corpus.build_manifest()
    second_manifest = second_corpus.build_manifest()

    assert first_manifest.digest == second_manifest.digest
    panel = first_corpus.task("ff_hash_0000").positive[0]
    panel.write_bytes(PNG + b"changed")
    changed = first_corpus.build_manifest()
    assert changed.digest != second_manifest.digest
    task_manifest = changed.tasks[0]
    assert task_manifest.digest.startswith("sha256:")
    assert len(task_manifest.panels) == 14
    assert all(panel.sha256.startswith("sha256:") for panel in task_manifest.panels)
    assert all("path" not in panel.to_dict() for panel in task_manifest.panels)


def test_manifest_rejects_fake_png(tmp_path: Path) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_fake_0000")
    (root / "ff" / "images" / "ff_fake_0000" / "0" / "0.png").write_bytes(b"not png")
    corpus = ShapeBongardCorpus.from_root(root)

    with pytest.raises(CorpusValidationError, match="no PNG signature"):
        corpus.build_manifest()


def test_manifest_rejects_symlink_substitution_after_discovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_link_0000")
    corpus = ShapeBongardCorpus.from_root(root)
    panel = corpus.task("ff_link_0000").positive[0]
    target = tmp_path / "substitute.png"
    target.write_bytes(PNG + b"substitute")
    panel.unlink()
    panel.symlink_to(target)

    with pytest.raises(CorpusValidationError, match="regular no-follow"):
        corpus.build_manifest()


def test_manifest_rejects_path_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_race_0000")
    corpus = ShapeBongardCorpus.from_root(root)
    panel = corpus.task("ff_race_0000").positive[0]
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(PNG + b"different-inode")
    real_open = os.open
    replaced = False

    def raced_open(path, flags, *args):
        nonlocal replaced
        if Path(path) == panel and not replaced:
            panel.unlink()
            replacement.rename(panel)
            replaced = True
        return real_open(path, flags, *args)

    monkeypatch.setattr("bongard.corpus.os.open", raced_open)
    with pytest.raises(CorpusValidationError, match="changed while being opened"):
        corpus.build_manifest()
    assert replaced


def test_split_unknown_overlap_and_unclassified_tasks_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_one_0000")
    _task(root, "ff", "ff_two_0000")
    split = {"train": ["ff_one_0000"], "val": ["ff_one_0000"]}
    (root / "ShapeBongard_V2_split.json").write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(CorpusValidationError, match="overlap"):
        ShapeBongardCorpus.from_root(root)

    split = {"train": ["ff_missing_0000"]}
    (root / "ShapeBongard_V2_split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="absent tasks"):
        ShapeBongardCorpus.from_root(root)

    split = {"train": ["ff_one_0000"]}
    (root / "ShapeBongard_V2_split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="unclassified"):
        ShapeBongardCorpus.from_root(root)


def test_complete_validation_reports_official_count_contract(tmp_path: Path) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _task(root, "ff", "ff_only_0000")
    corpus = ShapeBongardCorpus.from_root(root)

    with pytest.raises(CorpusValidationError, match="3600"):
        corpus.validate_complete(require_split=False)


def test_discovery_rejects_ambiguous_roots(tmp_path: Path) -> None:
    _task(tmp_path / "left", "ff", "ff_left_0000")
    _task(tmp_path / "right", "ff", "ff_right_0000")

    with pytest.raises(CorpusLayoutError, match="ambiguous"):
        ShapeBongardCorpus.discover(tmp_path)
