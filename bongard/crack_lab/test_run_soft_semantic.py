"""Offline integration tests for the SEMANTIC-SOFT campaign runner."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_soft_semantic as R
import semantic_soft_pipeline as S
from soft_semantics import content_digest


CORPUS_DIGEST = content_digest({"fixture": "corpus"})
PANEL_SET_DIGEST = content_digest({"fixture": "panel-set"})
PRODUCER_DIGEST = content_digest({"fixture": "blind-scorer"})


def _spec(index: int) -> S.SoftPredicateSpec:
    return S.SoftPredicateSpec(
        hypothesis_id=f"closed-form-{index}",
        claim=f"A closed geometric form of fixture type {index} is present.",
        operational_definition=(
            "A continuous visible contour returns to its starting region "
            "without an open endpoint."),
        order="high_positive",
        comparison="absolute",
        aggregation="all",
        required_cues=(
            S.SoftCueSpec(
                f"closed-contour-{index}",
                "A continuous contour visibly encloses an interior region."),
        ),
        disqualifiers=(),
        preservation_morphisms=("translation", "rotation", "uniform_scale"),
    )


def _args(out_dir: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "out_dir": str(out_dir),
        "dataset_dir": str(out_dir.parent / "dataset"),
        "source": "basic",
        "limit": 1,
        "corpus_size": 1,
        "seed": 17,
        "condition": R.CONDITION_OBSERVED,
        "control_seed": 19,
        "control_replicate": 0,
        "model": "fixture-model",
        "reasoning_effort": "medium",
        "minutes": 2,
        "scorer_workers": 3,
        "prepare_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = object()
    monkeypatch.setattr(
        R.phase_d_protocol, "sample_corpus", lambda *args, **kwargs: [problem])
    monkeypatch.setattr(
        R.phase_d_protocol, "dataset_revision", lambda *args: "fixture-revision")
    monkeypatch.setattr(
        R.phase_d_protocol, "dataset_content_digest",
        lambda *args: content_digest({"fixture": "dataset"}))
    monkeypatch.setattr(
        R.phase_d_protocol, "build_corpus_manifest",
        lambda *args, **kwargs: {
            "schema": "fixture-corpus/v1",
            "corpus_digest": CORPUS_DIGEST,
            "problems": [{
                "category": "fixture",
                "panel_set_digest": PANEL_SET_DIGEST,
            }],
        })
    monkeypatch.setattr(
        R.phase_d_protocol, "build_corpus_bundle",
        lambda *args, **kwargs: {
            "schema": "fixture-panels/v1",
            "corpus_digest": CORPUS_DIGEST,
        })
    monkeypatch.setattr(
        R.phase_d_protocol, "validate_corpus_manifest", lambda manifest: None)


def _patch_bongard_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    fake_module = root / "bongard" / "crack_lab" / "run_soft_semantic.py"
    monkeypatch.setattr(R, "__file__", str(fake_module))


def test_mocked_campaign_runs_twelve_blind_turns_and_writes_replay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_corpus(monkeypatch)
    _patch_bongard_root(monkeypatch, tmp_path)
    expected_names = [
        f"{side}_{index}.png"
        for side in ("pos", "neg") for index in range(6)
    ]

    def write_fixture_panels(workspace: str, problem: object,
                             opaque_id: str) -> None:
        del problem
        directory = Path(workspace) / opaque_id
        directory.mkdir(parents=True)
        for name in expected_names:
            side, raw_index = name[:-4].split("_")
            presentation = np.full((16, 16), 255, dtype=np.uint8)
            presentation[int(raw_index), 0 if side == "pos" else 1] = 0
            Image.fromarray(presentation, mode="L").save(
                directory / name, format="PNG")

    proposer_calls: list[list[str]] = []
    scorer_calls: list[str] = []

    class FakeProposer:
        def __init__(self, model: str, *, minutes: int,
                     reasoning_effort: str) -> None:
            assert (model, minutes, reasoning_effort) == (
                "fixture-model", 2, "medium")

        def propose(self, problem_id: str,
                    panel_png_paths: list[str]) -> S.SoftProposalBundle:
            proposer_calls.append(
                [os.path.basename(path) for path in panel_png_paths])
            return S.SoftProposalBundle(
                problem_id=problem_id,
                hypotheses=tuple(_spec(index) for index in range(3)),
                analysis="Three frozen fixture rubrics.",
                receipt={
                    "receipt_digest": content_digest({"turn": "proposal"}),
                    "input_tokens": 5,
                    "cached_input_tokens": 1,
                    "output_tokens": 2,
                    "reasoning_output_tokens": 1,
                },
            )

    class FakeScorer:
        def __init__(self, model: str, *, minutes: int,
                     reasoning_effort: str) -> None:
            assert (model, minutes, reasoning_effort) == (
                "fixture-model", 2, "medium")

        def score_many(self, specs: tuple[S.SoftPredicateSpec, ...],
                       panel_png_path: str) -> tuple[S.PanelSoftScore, ...]:
            with Image.open(panel_png_path) as encoded:
                presentation = np.asarray(encoded.convert("L"))
            row, column = np.argwhere(presentation == 0)[0]
            name = f"{'pos' if column == 0 else 'neg'}_{int(row)}.png"
            scorer_calls.append(name)
            score = 0.9 if name.startswith("pos_") else 0.1
            receipt = {
                "receipt_digest": content_digest({"turn": name}),
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 1,
                "reasoning_output_tokens": 3,
            }
            return tuple(
                S.panel_soft_score_from_payload(
                    spec,
                    {
                        "atomic_scores": [{
                            "cue_id": cue_id,
                            "score": score,
                            "evidence": "The fixture contour is visible.",
                        } for cue_id in spec.cue_ids],
                        "uncertainty": 0.05,
                        "abstain": False,
                        "abstention_reason": "",
                    },
                    receipt,
                    producer_digest=PRODUCER_DIGEST,
                )
                for spec in specs
            )

    monkeypatch.setattr(R, "write_panels", write_fixture_panels)
    monkeypatch.setattr(R, "CodexSoftHypothesisProposer", FakeProposer)
    monkeypatch.setattr(R, "CodexBlindSoftBatchScorer", FakeScorer)

    out_dir = tmp_path / "bongard" / "runs" / "fixture"
    campaign = R.run(_args(out_dir))

    assert proposer_calls == [expected_names]
    assert sorted(scorer_calls) == sorted(expected_names)
    assert len(scorer_calls) == 12
    assert campaign["record_count"] == 1
    record = campaign["records"][0]
    assert record["status"] == "SOLVED_SEMANTIC_SOFT"
    assert record["infrastructure_valid"] is True
    assert record["scorer_error_measurements"] == 0
    assert record["candidate_count"] == 3
    assert record["information_boundary"]["scorer_workers"] == 3
    assert record["usage"] == {
        "input_tokens": 125,
        "cached_input_tokens": 25,
        "output_tokens": 14,
        "reasoning_output_tokens": 37,
        "turns": 13,
    }
    persisted = json.loads((out_dir / "campaign.json").read_text())
    assert persisted == json.loads(json.dumps(campaign))
    replay_report = R.replay_campaign_directory(str(out_dir))
    assert replay_report["valid"] is True
    assert replay_report["solved_count"] == 1
    unsigned = dict(campaign)
    campaign_digest = unsigned.pop("campaign_digest")
    assert campaign_digest == R.semantic_replay.canonical_json_digest(unsigned)

    tampered = copy.deepcopy(persisted)
    tampered["records"][0]["candidates"][0]["evidence"][0]["result"][
        "membership"] = 0.123
    tampered["campaign_digest"] = R.semantic_replay.canonical_json_digest({
        key: value for key, value in tampered.items()
        if key != "campaign_digest"})
    with pytest.raises(ValueError, match="membership|replay"):
        R.replay_campaign_artifact(tampered)

    with pytest.raises(RuntimeError, match="campaign.json already exists"):
        R.run(_args(out_dir))


def test_prepare_only_avoids_codex_and_worker_range_fails_early(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_corpus(monkeypatch)
    _patch_bongard_root(monkeypatch, tmp_path)

    class MustNotConstruct:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("prepare-only must not construct Codex clients")

    monkeypatch.setattr(R, "CodexSoftHypothesisProposer", MustNotConstruct)
    monkeypatch.setattr(R, "CodexBlindSoftBatchScorer", MustNotConstruct)

    out_dir = tmp_path / "bongard" / "runs" / "prepare"
    result = R.run(_args(out_dir, prepare_only=True))
    assert result == {"corpus_digest": CORPUS_DIGEST, "prepared": True}
    assert (out_dir / "corpus_manifest.json").is_file()
    assert (out_dir / "corpus_panels.json").is_file()
    assert not (out_dir / "campaign.json").exists()
    assert not (out_dir / "workspace").exists()

    def must_not_sample(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid worker count must fail before sampling")

    monkeypatch.setattr(R.phase_d_protocol, "sample_corpus", must_not_sample)
    with pytest.raises(SystemExit, match="scorer-workers in 1..12"):
        R.run(_args(
            tmp_path / "bongard" / "runs" / "invalid", scorer_workers=0))
    with pytest.raises(SystemExit, match="minutes in 1..120"):
        R.run(_args(
            tmp_path / "bongard" / "runs" / "invalid-minutes", minutes=121))


def test_total_scorer_failure_is_invalid_not_a_bad_benchmark(
        tmp_path: Path) -> None:
    paths = []
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            presentation = np.full((16, 16), 255, dtype=np.uint8)
            presentation[index, 0 if side == "pos" else 1] = 0
            Image.fromarray(presentation, mode="L").save(path, format="PNG")
            paths.append(str(path))

    class Proposer:
        def propose(self, problem_id: str, panel_png_paths: list[str]):
            return S.SoftProposalBundle(
                problem_id, tuple(_spec(index) for index in range(3)),
                "Fixture proposal.", {
                    "receipt_digest": content_digest({"turn": "proposal"}),
                })

    class BrokenScorer:
        def score_many(self, specs, panel_png_path):
            raise RuntimeError("fixture transport outage")

    record = R.evaluate_problem(
        "problem_00", paths, Proposer(), BrokenScorer(), scorer_workers=3)
    assert record["status"] == "INVALID_SEMANTIC_SOFT"
    assert record["solved"] is False
    assert record["infrastructure_valid"] is False
    assert record["scorer_error_measurements"] == 36


def test_cli_exposes_closed_reasoning_and_worker_options() -> None:
    args = R.parse_args([
        "--reasoning-effort", "ultra", "--scorer-workers", "12",
    ])
    assert args.reasoning_effort == "ultra"
    assert args.scorer_workers == 12

    with pytest.raises(SystemExit):
        R.parse_args(["--reasoning-effort", "unsupported"])
