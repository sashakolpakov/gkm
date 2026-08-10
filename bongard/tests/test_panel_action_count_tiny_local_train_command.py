"""Focused tests for the bounded development-only tiny trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bongard import panel_action_count_tiny_local_dev_command as core
from bongard import panel_action_count_tiny_local_train_command as subject


SHA_A = "sha256:" + "a" * 64


def _interval(center: float) -> dict[str, float]:
    return {"center": center, "lower": center - 0.01, "upper": center + 0.01}


def _descriptor():
    return (
        {"kind": "line", "line_length": _interval(0.4)},
        {
            "arc_radius": _interval(0.3),
            "arc_sweep_magnitude": _interval(0.5),
            "kind": "arc",
        },
    )


def _group(
    *,
    cohort: str,
    digest_digit: int,
    multiplicity: int,
    catalog: int,
    descriptor=True,
) -> subject.TrainingGroup:
    array = np.zeros((64, 64), dtype=np.uint8)
    array[10 + digest_digit : 13 + digest_digit, 8:56] = 255
    return subject.TrainingGroup(
        cohort=cohort,
        png_sha256=f"sha256:{digest_digit:064x}",
        ink=array,
        straight=1,
        arc=1,
        catalog=catalog,
        multiplicity=multiplicity,
        descriptor_targets=_descriptor() if descriptor else None,
    )


def test_release_path_uses_images_component_and_ignores_decoy(tmp_path) -> None:
    panel_id = "hd/hd_example_0001/1/0.png"
    expected = tmp_path / "hd/images/hd_example_0001/1/0.png"
    decoy = tmp_path / "hd/hd_example_0001/1/0.png"
    expected.parent.mkdir(parents=True)
    decoy.parent.mkdir(parents=True)
    expected.write_bytes(b"official")
    decoy.write_bytes(b"decoy")
    found = subject._panel_path(tmp_path, panel_id)
    assert found == expected
    assert found.read_bytes() == b"official"
    with pytest.raises(subject.TinyLocalTrainingError, match="target family"):
        subject._panel_path(
            tmp_path, "hd/hd_convex-has_four_straight_lines_0000/1/0.png"
        )


def test_stable_panel_read_rejects_symlinked_task_parent(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "hd/images"
    real_task = images / "real-task"
    real_side = real_task / "1"
    real_side.mkdir(parents=True)
    (real_side / "0.png").write_bytes(b"png")
    (images / "linked-task").symlink_to(real_task, target_is_directory=True)
    linked = images / "linked-task/1/0.png"
    with pytest.raises(subject.TinyLocalTrainingError, match="parent contains"):
        subject._stable_panel_bytes(linked, dataset_root=dataset)


def test_group_normalized_descriptor_loss_ignores_occurrence_multiplicity() -> None:
    model = core.build_model().train()
    pixels = torch.rand((2, 2, 64, 64), generator=torch.Generator().manual_seed(7))
    output = model(pixels)
    first = [_group(cohort="train", digest_digit=1, multiplicity=1, catalog=1)]
    second = [_group(cohort="train", digest_digit=1, multiplicity=10_000, catalog=1)]
    loss_a = subject.group_normalized_loss(
        {key: value[:1] for key, value in output.items()}, first
    )
    loss_b = subject.group_normalized_loss(
        {key: value[:1] for key, value in output.items()}, second
    )
    assert torch.equal(
        loss_a["descriptor_classification_group_normalized"],
        loss_b["descriptor_classification_group_normalized"],
    )
    assert torch.equal(
        loss_a["descriptor_geometry_group_normalized"],
        loss_b["descriptor_geometry_group_normalized"],
    )


def test_descriptor_gap_keeps_count_and_catalog_losses() -> None:
    model = core.build_model().train()
    pixels = torch.rand((1, 2, 64, 64), generator=torch.Generator().manual_seed(8))
    loss = subject.group_normalized_loss(
        model(pixels),
        [_group(cohort="train", digest_digit=2, multiplicity=50, catalog=2, descriptor=False)],
    )
    assert loss["descriptor_eligible_group_count"] == 0
    assert float(loss["descriptor_classification_group_normalized"]) == 0.0
    assert float(loss["catalog_and_count_occurrence_weighted"].detach()) > 0.0
    loss["total"].backward()


def test_synthetic_train_core_runs_six_epochs_under_deadline() -> None:
    groups = (
        _group(cohort="train", digest_digit=1, multiplicity=5_600, catalog=1),
        _group(cohort="train", digest_digit=2, multiplicity=5_600, catalog=2),
        _group(cohort="validation", digest_digit=3, multiplicity=696, catalog=1),
        _group(cohort="validation", digest_digit=4, multiplicity=696, catalog=2),
    )
    trained = subject.train_core(groups, deadline=core.WallDeadline(seconds=30))
    assert trained["best_epoch"] in range(6)
    assert len(trained["history"]) == 6
    assert trained["metrics"]["panel_occurrences"] == 1_392
    assert core.state_dict_digest(trained["state"]).startswith("sha256:")


def test_checkpoint_is_write_once_and_source_bound(tmp_path) -> None:
    model = core.build_model()
    payload = {
        "architecture_id": core.ARCHITECTURE_ID,
        "config_digest": core.successor_config_digest(),
        "selected_epoch": 0,
        "source_sha256": core.source_sha256(),
        "state_dict": model.state_dict(),
        "state_dict_sha256": core.state_dict_digest(model.state_dict()),
        "training_precommit_record_digest": SHA_A,
    }
    path = tmp_path / "model.pt"
    first = subject._save_checkpoint(path, payload)
    assert subject._save_checkpoint(path, payload) == first
    changed = dict(payload)
    changed["selected_epoch"] = 1
    with pytest.raises(subject.TinyLocalTrainingError, match="overwrite"):
        subject._save_checkpoint(path, changed)


def test_cli_has_no_later_cohort_entrypoint() -> None:
    source = Path(subject.__file__).read_text()
    assert 'commands.add_parser("prepare")' in source
    assert 'commands.add_parser(name)' in source
    for forbidden in (
        "--calibration",
        "--evaluation",
        "--same-family",
        "--query",
        "--target",
    ):
        assert forbidden not in source


def test_train_cli_returns_nonzero_for_typed_failed_gate(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        subject,
        "run_training",
        lambda **_: {
            "record_digest": SHA_A,
            "validation_gate": {"passed": False},
        },
    )
    assert subject.main(
        [
            "train",
            "--repository-root",
            "/repo",
            "--dataset-root",
            "/dataset",
            "--output-root",
            "/output",
        ]
    ) == 2
    assert '"validation_gate_passed": false' in capsys.readouterr().out


def test_real_metadata_prepare_cli_reads_zero_pngs(tmp_path, monkeypatch) -> None:
    repository = Path(subject.__file__).resolve().parents[1]
    dataset = repository / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
    output = tmp_path / "tiny-prepare"

    def forbidden_panel_read(*_args, **_kwargs):
        raise AssertionError("prepare attempted to read a PNG")

    monkeypatch.setattr(subject, "_stable_panel_bytes", forbidden_panel_read)
    monkeypatch.setattr(
        core,
        "synthetic_runtime_probe",
        lambda **_: {
            "frozen_batch_size": 128,
            "median_seconds_per_frozen_batch": 0.1,
            "parameter_count": core.parameter_count(),
            "synthetic_only": True,
        },
    )
    assert subject.main(
        [
            "prepare",
            "--repository-root",
            str(repository),
            "--dataset-root",
            str(dataset),
            "--fit-precommit",
            str(
                repository
                / "downloads/ShapeBongard_V2_full/panel_action_count_cnn_fit_20260810_v3/fit_pixel_precommit.json"
            ),
            "--failed-baseline",
            str(
                repository
                / "downloads/ShapeBongard_V2_full/panel_action_count_cnn_fit_20260810_v3/fit_result.json"
            ),
            "--retired-spatial-outcome",
            str(
                repository
                / "bongard/data/panel_action_count_spatial_dev_outcome_20260810_v1.json"
            ),
            "--descriptor-conflict-audit",
            str(
                repository
                / "bongard/data/panel_action_local_duplicate_conflict_audit_20260810_v1.json"
            ),
            "--output-root",
            str(output),
        ]
    ) == 0
    authorization = subject._load(output / "authorization.json", label="authorization")
    precommit = subject._load(output / "training_precommit.json", label="precommit")
    core_precommit = subject._load(output / "core_precommit.json", label="core precommit")
    assert authorization["pixels_read_by_prepare"] == 0
    assert precommit["pixels_read_by_precommit"] == 0
    assert core_precommit["training_entrypoint_status"] == "live_runnable_development_only"
    assert precommit["descriptor_conflict_audit_record_digest"] == (
        subject.COMMITTED_CONFLICT_AUDIT_DIGEST
    )
