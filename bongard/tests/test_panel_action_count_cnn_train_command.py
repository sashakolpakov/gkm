"""Synthetic, no-corpus tests for the fit-only supervised CNN command."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

import bongard.panel_action_count_cnn_train_command as command


def _png_bytes(offset: int = 0) -> bytes:
    image = Image.new("L", (48, 40), 255)
    draw = ImageDraw.Draw(image)
    draw.line((6 + offset, 5, 35, 30), fill=0, width=3)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _observation(*, panel: str, cohort: str, labels, raw: bytes) -> dict[str, object]:
    return {
        "fit_cohort": cohort,
        "label_triple": list(labels),
        "panel_id": panel,
        "png_sha256": command._address(raw),
        "png_size_bytes": len(raw),
        "metric_strata": {
            "crossing_task": False,
            "line_decoration": "normal_only",
            "thin_task": False,
        },
    }


def test_preprocessing_and_content_keys_ignore_path_identity() -> None:
    raw = _png_bytes()
    ink = command.preprocess_png_bytes(raw)
    assert ink.shape == (96, 96)
    assert ink.dtype == np.uint8
    assert int(ink.max()) > 0
    digest = command._address(raw)
    first = command.content_epoch_key(260810, 3, digest)
    renamed = command.content_epoch_key(260810, 3, digest)
    assert first == renamed
    assert len({command.d4_transform(ink, index).tobytes() for index in range(8)}) >= 4


def test_duplicate_digest_groups_are_path_independent_and_fail_on_leakage() -> None:
    raw = _png_bytes()
    rows = [
        _observation(
            panel=f"hd/renamed_task/1/{index}.png",
            cohort="train",
            labels=(4, 0, 1),
            raw=raw,
        )
        for index in range(2)
    ]
    groups, audit = command._audit_digest_groups(rows)
    assert len(groups) == 1
    assert groups[0]["multiplicity"] == 2
    assert groups[0]["label_triple"] == [4, 0, 1]
    assert audit["duplicate_group_count"] == 1

    different_label = [dict(row) for row in rows]
    different_label[1]["label_triple"] = [3, 0, 1]
    with pytest.raises(command.ActionCountCNNFitError, match="different label"):
        command._audit_digest_groups(different_label)

    cross_cohort = [dict(row) for row in rows]
    cross_cohort[1]["fit_cohort"] = "validation"
    with pytest.raises(command.ActionCountCNNFitError, match="leaks across"):
        command._audit_digest_groups(cross_cohort)


def test_artifacts_and_checkpoints_are_write_once_or_verify_identical(tmp_path: Path) -> None:
    first = command._seal_body({"schema": "synthetic", "value": 1})
    second = command._seal_body({"schema": "synthetic", "value": 2})
    artifact = tmp_path / "record.json"
    original = command._write_fsynced(artifact, first)
    assert command._write_fsynced(artifact, first) == original
    with pytest.raises(command.ActionCountCNNFitError, match="refusing to overwrite"):
        command._write_fsynced(artifact, second)
    assert artifact.read_bytes() == original

    pytest.importorskip("torch")
    model = command.build_model(seed=260810)
    checkpoint = {
        "architecture_id": command.ARCHITECTURE_ID,
        "catalog_class_values": list(command.CATALOG_VALUES),
        "config_digest": "sha256:" + "1" * 64,
        "selected_epoch": 0,
        "state_dict": model.state_dict(),
    }
    checkpoint_path = tmp_path / "model.pt"
    raw = command._save_checkpoint(checkpoint_path, checkpoint)
    assert command._save_checkpoint(checkpoint_path, checkpoint) == raw
    changed = {**checkpoint, "selected_epoch": 1}
    with pytest.raises(command.ActionCountCNNFitError, match="refusing to overwrite"):
        command._save_checkpoint(checkpoint_path, changed)
    assert checkpoint_path.read_bytes() == raw


def _synthetic_groups() -> list[command.MaterializedGroup]:
    groups = []
    for index in range(18):
        array = np.zeros((96, 96), dtype=np.uint8)
        array[8 + index : 12 + index, 10:80] = 255
        array[20:75, 8 + (index * 3) % 70 : 10 + (index * 3) % 70] = 180
        cohort = "train" if index < 12 else "validation"
        catalog = (1, 2)[index % 2]
        labels = (index % 3, (index // 3) % 2, catalog)
        groups.append(
            command.MaterializedGroup(
                cohort=cohort,
                digest=command._address(array.tobytes()),
                ink=array,
                labels=labels,
                multiplicity=1,
            )
        )
    return groups


def test_cpu_training_and_fresh_model_replay_are_bit_exact() -> None:
    pytest.importorskip("torch")
    first = command.train_core(_synthetic_groups(), epochs=2, seed=260810)
    second = command.train_core(_synthetic_groups(), epochs=2, seed=260810)
    assert command.state_dict_digest(first["state"]) == command.state_dict_digest(
        second["state"]
    )
    assert first["best_epoch"] == second["best_epoch"]
    assert first["best_metrics"] == second["best_metrics"]
    assert first["predictions"] == second["predictions"]


def test_cli_is_fit_only_and_correction_is_immutable() -> None:
    parser = command._parser()
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {"precommit-fit", "train-fit", "replay-fit"}
    with pytest.raises(TypeError):
        command.FIT_CORRECTION["future_calibration_and_evaluation_require_fresh_superseding_cohorts"] = False
