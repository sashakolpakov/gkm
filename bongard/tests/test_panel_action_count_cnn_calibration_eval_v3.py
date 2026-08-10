"""Synthetic-only tests for the closed V3 calibration/evaluation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bongard.canonical import canonical_digest
import bongard.panel_action_count_cnn_calibration_eval_v3 as runner


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _panel_ids(stage: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    task_ids = tuple(f"hd_synthetic_{stage}_{index:04}" for index in range(100))
    panels = tuple(
        f"hd/{task_id}/{folder}/{panel}.png"
        for task_id in task_ids
        for folder in (1, 0)
        for panel in range(7)
    )
    return task_ids, panels


def _authority(stage: str) -> runner.V3Authority:
    task_ids, panel_ids = _panel_ids(stage)
    plan = {
        "record_digest": SHA_A,
        "postprediction_target_authority": {
            "frozen_label_source_bindings": {
                "catalog_algorithm_digest": SHA_A,
                "catalog_audit_record_digest": SHA_B,
                "catalog_authority_source_sha256": SHA_C,
                "hd_action_program_raw_sha256": SHA_D,
            },
            "source_sha256": runner.V3_POSTPREDICTION_SOURCE_SHA256,
        },
    }
    manifest = {"record_digest": SHA_B}
    return runner.V3Authority(
        stage=stage,
        plan=plan,
        plan_raw=b"synthetic-plan",
        manifest=manifest,
        manifest_raw=f"synthetic-{stage}-manifest".encode(),
        task_ids=task_ids,
        panel_ids=panel_ids,
    )


def _fit() -> runner.FitAuthority:
    return runner.FitAuthority(
        authorization={"record_digest": "sha256:" + "1" * 64},
        authorization_raw=b"fit-authorization",
        precommit={"record_digest": "sha256:" + "2" * 64},
        precommit_raw=b"fit-precommit",
        result={"record_digest": "sha256:" + "3" * 64},
        result_raw=b"fit-result",
        checkpoint_raw_sha256="sha256:" + "4" * 64,
        checkpoint_state_dict_sha256="sha256:" + "5" * 64,
        config_digest="sha256:" + "6" * 64,
        trainer_source_sha256=runner.FINAL_FIT_BINDING["trainer_source_sha256"],
    )


def _uniform_inference(
    stage: str,
    panel_ids: Sequence[str],
    raws: Sequence[bytes],
    _checkpoint_path: Path,
    _fit: runner.FitAuthority,
) -> Sequence[runner.InferenceRow]:
    assert stage in {"calibration", "evaluation"}
    assert len(panel_ids) == len(raws)
    logits10 = (0.0,) * 10
    probabilities10 = runner._softmax(logits10)
    logits3 = (0.0,) * 3
    probabilities3 = runner._softmax(logits3)
    return [
        runner.InferenceRow(
            panel_id=panel_id,
            straight_logits=logits10,
            straight_probabilities=probabilities10,
            arc_logits=logits10,
            arc_probabilities=probabilities10,
            catalog_logits=logits3,
            catalog_probabilities=probabilities3,
        )
        for panel_id in panel_ids
    ]


def _install_synthetic_boundaries(monkeypatch: pytest.MonkeyPatch) -> runner.FitAuthority:
    fit = _fit()
    authorities = {
        "calibration": _authority("calibration"),
        "evaluation": _authority("evaluation"),
    }
    monkeypatch.setattr(
        runner,
        "_load_v3_authority",
        lambda *, stage, plan_path, manifest_path: authorities[stage],
    )
    monkeypatch.setattr(
        runner,
        "_verify_fit_authority",
        lambda **_kwargs: fit,
    )

    def derive_labels(**kwargs: Any) -> dict[str, Any]:
        authority: runner.V3Authority = kwargs["authority"]
        prediction_path: Path = kwargs["prediction_path"]
        output_path: Path = kwargs["output_path"]
        prediction, prediction_raw = runner._load_record(
            prediction_path, label="synthetic durable prediction"
        )
        rows = []
        profiles = (
            "no_straight_actions",
            "normal_only",
            "decorated_only",
            "mixed_normal_and_decorated",
        )
        targets = (-1, 0, 1)
        for index, panel_id in enumerate(authority.panel_ids):
            target = targets[index % 3]
            row: dict[str, Any] = {
                "arc_action_count": 0,
                "catalog_convexity_class": runner.CATALOG_CLASS_ORDER[
                    runner.CATALOG_TARGET_TO_INDEX[target]
                ],
                "catalog_convexity_target": target,
                "catalog_match_kind": "synthetic",
                "panel_id": panel_id,
                "straight_action_count": 4 if index % 7 == 0 else 0,
            }
            if authority.stage == "evaluation":
                row["metric_strata"] = {
                    "crossing_task": index % 5 == 0,
                    "line_decoration": profiles[index % len(profiles)],
                    "thin_task": index % 6 == 0,
                }
            rows.append(row)
        body = {
            "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
            "config_digest": fit.config_digest,
            "custody": runner._custody(authority, fit),
            "label_authority_bindings": {},
            "panel_manifest_record_digest": authority.manifest["record_digest"],
            "prediction_record_digest": prediction["record_digest"],
            "prediction_source_sha256": runner._address(prediction_raw),
            "rows": rows,
            "schema": runner.LABEL_RECORD_SCHEMA,
            "stage": authority.stage,
        }
        final = runner._seal(body)
        runner._write_once(output_path, final)
        return final

    monkeypatch.setattr(runner, "_derive_and_freeze_labels", derive_labels)
    return fit


def _stage_paths(base: Path, stage: str) -> Mapping[str, Path]:
    return runner._stage_paths(base, stage)


def test_q96_is_exact_task_max_order_statistic_without_interpolation() -> None:
    task_ids = [f"hd_task_{index:04}" for index in range(100)]
    predicted = []
    labelled = []
    for index, task_id in enumerate(task_ids):
        score = index / 100.0
        for folder in (1, 0):
            for panel in range(7):
                panel_id = f"hd/{task_id}/{folder}/{panel}.png"
                straight = [score / 9.0] * 10
                straight[0] = 1.0 - score
                predicted.append(
                    {
                        "panel_id": panel_id,
                        "straight_probabilities": straight,
                        "arc_probabilities": [1.0] + [0.0] * 9,
                        "catalog_probabilities": [1.0, 0.0, 0.0],
                    }
                )
                labelled.append(
                    {
                        "panel_id": panel_id,
                        "straight_action_count": 0,
                        "arc_action_count": 0,
                        "catalog_convexity_target": -1,
                    }
                )
    scores = runner.calibration_scores(
        task_ids=task_ids, prediction_rows=predicted, label_rows=labelled
    )
    assert scores["sorted_joint_task_scores"][95] == pytest.approx(0.95)
    assert scores["task_scores_in_manifest_order"][95]["joint_score"] == pytest.approx(
        0.95
    )


def test_full_synthetic_chronology_and_zero_call_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_boundaries(monkeypatch)
    calibration_dir = tmp_path / "calibration"
    evaluation_dir = tmp_path / "evaluation"
    calibration = _stage_paths(calibration_dir, "calibration")
    evaluation = _stage_paths(evaluation_dir, "evaluation")
    events: list[str] = []

    def panel_reader_for(stage: str, authorization_path: Path):
        def read(panel_id: str) -> bytes:
            assert authorization_path.exists()
            events.append(f"{stage}:pixel")
            return (stage + "\0" + panel_id).encode()

        return read

    def inference(stage, panel_ids, raws, checkpoint_path, fit):
        expected = calibration if stage == "calibration" else evaluation
        assert expected["pixel_precommit"].exists()
        events.append(f"{stage}:inference")
        return _uniform_inference(stage, panel_ids, raws, checkpoint_path, fit)

    runner.run_calibration(
        plan_path=tmp_path / "plan.json",
        calibration_manifest_path=tmp_path / "calibration-manifest.json",
        fit_authorization_path=tmp_path / "fit-auth.json",
        fit_precommit_path=tmp_path / "fit-precommit.json",
        fit_result_path=tmp_path / "fit-result.json",
        checkpoint_path=tmp_path / "model.pt",
        authorization_output_path=calibration["authorization"],
        pixel_precommit_output_path=calibration["pixel_precommit"],
        prediction_output_path=calibration["prediction"],
        label_output_path=calibration["labels"],
        grant_output_path=calibration["terminal"],
        panel_reader=panel_reader_for("calibration", calibration["authorization"]),
        inference=inference,
        label_source_loader=lambda _barrier: (_ for _ in ()).throw(
            AssertionError("synthetic delayed-label boundary was patched")
        ),
    )
    grant, _ = runner._load_record(calibration["terminal"], label="synthetic grant")
    assert grant["q_rule"] == "sorted_scores[95]"
    assert grant["deployment_joint_q"] == pytest.approx(0.9)

    runner.run_evaluation(
        plan_path=tmp_path / "plan.json",
        calibration_manifest_path=tmp_path / "calibration-manifest.json",
        evaluation_manifest_path=tmp_path / "evaluation-manifest.json",
        fit_authorization_path=tmp_path / "fit-auth.json",
        fit_precommit_path=tmp_path / "fit-precommit.json",
        fit_result_path=tmp_path / "fit-result.json",
        checkpoint_path=tmp_path / "model.pt",
        calibration_authorization_path=calibration["authorization"],
        calibration_pixel_precommit_path=calibration["pixel_precommit"],
        calibration_prediction_path=calibration["prediction"],
        calibration_label_path=calibration["labels"],
        calibration_grant_path=calibration["terminal"],
        authorization_output_path=evaluation["authorization"],
        pixel_precommit_output_path=evaluation["pixel_precommit"],
        prediction_output_path=evaluation["prediction"],
        label_output_path=evaluation["labels"],
        result_output_path=evaluation["terminal"],
        panel_reader=panel_reader_for("evaluation", evaluation["authorization"]),
        inference=inference,
        label_source_loader=lambda _barrier: (_ for _ in ()).throw(
            AssertionError("synthetic delayed-label boundary was patched")
        ),
        metric_strata_loader=lambda _barrier: {},
    )
    prediction, _ = runner._load_record(evaluation["prediction"], label="eval prediction")
    assert prediction["joint_q_record_digest"] == grant["record_digest"]
    assert prediction["rows"][0]["straight_class_set"] == list(range(10))
    result, _ = runner._load_record(evaluation["terminal"], label="eval result")
    assert set(result["confusions_true_rows_predicted_columns"]) == {
        "arc_10x10",
        "catalog_3x3_unresolved_nonconvex_convex",
        "straight_10x10",
    }
    assert result["typed_outcome"]["disposition"] == "INDETERMINATE"
    assert "line_decoration:normal_only" in result["required_strata"]

    replay = runner.cold_replay(
        plan_path=tmp_path / "plan.json",
        calibration_manifest_path=tmp_path / "calibration-manifest.json",
        evaluation_manifest_path=tmp_path / "evaluation-manifest.json",
        fit_authorization_path=tmp_path / "fit-auth.json",
        fit_precommit_path=tmp_path / "fit-precommit.json",
        fit_result_path=tmp_path / "fit-result.json",
        checkpoint_path=tmp_path / "model.pt",
        calibration_authorization_path=calibration["authorization"],
        calibration_pixel_precommit_path=calibration["pixel_precommit"],
        calibration_prediction_path=calibration["prediction"],
        calibration_label_path=calibration["labels"],
        calibration_grant_path=calibration["terminal"],
        evaluation_authorization_path=evaluation["authorization"],
        evaluation_pixel_precommit_path=evaluation["pixel_precommit"],
        evaluation_prediction_path=evaluation["prediction"],
        evaluation_label_path=evaluation["labels"],
        evaluation_result_path=evaluation["terminal"],
        replay_output_path=tmp_path / "replay.json",
    )
    assert replay["model_training_calls"] == 0
    assert replay["inference_calls"] == 0
    assert replay["png_reads"] == 0
    assert events.index("calibration:inference") > events.index("calibration:pixel")
    assert events.index("evaluation:inference") > events.index("evaluation:pixel")


def test_eval_authorization_requires_frozen_grant_and_loader_construction_reads_nothing(
    tmp_path: Path
) -> None:
    with pytest.raises(runner.ActionCountCNNV3RunnerError, match="before grant freeze"):
        runner._authorize_stage(
            authority=_authority("evaluation"),
            fit=_fit(),
            output_path=tmp_path / "forbidden.json",
            calibration_grant=None,
        )
    loaders = runner.filesystem_delayed_loaders(
        catalog_audit_path=tmp_path / "does-not-exist-audit.json",
        shape_rows_path=tmp_path / "does-not-exist-shapes.tsv",
        attribute_rows_path=tmp_path / "does-not-exist-attributes.tsv",
        hd_programs_path=tmp_path / "does-not-exist-hd.json",
        bd_programs_path=tmp_path / "does-not-exist-bd.json",
    )
    assert all(callable(value) for value in loaders)


def test_failed_fit_gate_stops_before_authorization_or_any_calibration_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_v3_authority",
        lambda **_kwargs: _authority("calibration"),
    )
    monkeypatch.setattr(
        runner,
        "_verify_fit_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.ActionCountCNNV3RunnerError(
                "fit validation gate did not pass exactly; fresh pixels remain unauthorized"
            )
        ),
    )
    authorization = tmp_path / "must-not-exist.json"
    reads = 0

    def forbidden_reader(_panel_id: str) -> bytes:
        nonlocal reads
        reads += 1
        raise AssertionError("calibration pixel opened after failed fit gate")

    with pytest.raises(runner.ActionCountCNNV3RunnerError, match="remain unauthorized"):
        runner.run_calibration(
            plan_path=tmp_path / "plan.json",
            calibration_manifest_path=tmp_path / "manifest.json",
            fit_authorization_path=tmp_path / "fit-auth.json",
            fit_precommit_path=tmp_path / "fit-precommit.json",
            fit_result_path=tmp_path / "fit-result.json",
            checkpoint_path=tmp_path / "checkpoint.pt",
            authorization_output_path=authorization,
            pixel_precommit_output_path=tmp_path / "pixel-precommit.json",
            prediction_output_path=tmp_path / "predictions.json",
            label_output_path=tmp_path / "labels.json",
            grant_output_path=tmp_path / "grant.json",
            panel_reader=forbidden_reader,
            inference=_uniform_inference,
            label_source_loader=lambda _barrier: (_ for _ in ()).throw(
                AssertionError("label source opened after failed fit gate")
            ),
        )
    assert reads == 0
    assert not authorization.exists()


def test_cli_exposes_only_closed_whole_stage_commands() -> None:
    parser = runner._parser()
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {"calibrate", "evaluate", "replay"}
    help_text = parser.format_help()
    assert "precommit-calibration" not in help_text
    assert "infer-evaluation" not in help_text
    assert "V2 CAL/eval inputs" in parser.description
