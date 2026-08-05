"""Focused support/query and cold-replay tests for the grounded runner."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codex_proposer
import grounded_predicate_ir as G
import grounded_synthesis as S
import run_grounded_semantic as R
from grounded_observables import SMALL_GAP_ID
from grounded_proposer import (
    GroundedProposalBundle,
    GroundingIntent,
    grounded_catalog_digest,
)


def _args(out_dir: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[2]
    return SimpleNamespace(
        out_dir=str(out_dir),
        dataset_dir=str(repo_root / "downloads" / "Bongard-LOGO"),
        source="basic",
        limit=1,
        corpus_size=1,
        program_seed=20260805,
        support_seed=20260805,
        query_seed=20260806,
        model="fixture-model",
        reasoning_effort="medium",
        minutes=2,
    )


class _SmallGapProposer:
    def __init__(self, descriptors: tuple[object, ...]) -> None:
        self.descriptors = descriptors
        self.calls = 0

    def propose(
        self, problem_id: str, panel_png_paths: list[str]
    ) -> GroundedProposalBundle:
        self.calls += 1
        assert len(panel_png_paths) == 12
        assert [Path(path).name for path in panel_png_paths] == [
            f"{side}_{index}.png"
            for side in ("pos", "neg") for index in range(6)
        ]
        # Query materialization must happen strictly after this turn.
        assert not (Path(panel_png_paths[0]).parent.parent / "query").exists()
        return GroundedProposalBundle(
            problem_id=problem_id,
            analysis="The small exterior point-contact gap is side-selective.",
            intents=(GroundingIntent(
                "intent-00", SMALL_GAP_ID, "low",
                "Positive panels have a narrow exterior point-contact gap.",
            ),),
            catalog_digest=grounded_catalog_digest(self.descriptors),
            receipt={
                "source": "offline-fixture",
                "panel_set_digest": codex_proposer.semantic_panel_set_digest(
                    panel_png_paths),
            },
        )


def test_grounded_run_freezes_before_query_and_cold_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry, descriptors = R.default_grounded_observables()
    proposer = _SmallGapProposer(descriptors)
    out_dir = tmp_path / "grounded"
    original_synthesize = S.synthesize_grounded_predicate
    synthesis_calls = 0

    def checked_synthesis(intents, cases, registry):
        nonlocal synthesis_calls
        synthesis_calls += 1
        if synthesis_calls == 1:
            assert not (out_dir / "workspace" / "problem_00" / "query").exists()
        assert [case.label for case in cases] == [True] * 6 + [False] * 6
        return original_synthesize(intents, cases, registry)

    monkeypatch.setattr(R.S, "synthesize_grounded_predicate", checked_synthesis)
    campaign = R.run(_args(out_dir), proposer=proposer)

    assert proposer.calls == 1
    assert synthesis_calls == 2
    assert campaign["record_count"] == 1
    record = campaign["records"][0]
    assert record["status"] == "SOLVED_SEMANTIC_GROUNDED"
    assert record["support_evaluation"]["exact"] is True
    assert record["query_evaluation"]["exact"] is True
    assert record["support_evaluation"]["indeterminate_count"] == 0
    assert record["query_evaluation"]["error_count"] == 0
    assert record["support_panel_set"]["semantic_panel_set_digest"] != \
        record["query_panel_set"]["semantic_panel_set_digest"]
    assert record["formula"]["compiled"]["taint"] == "PURE"

    # Replay does no proposer call.  It deterministically re-synthesizes from
    # the stored support evidence to validate the complete frozen result.
    report = R.replay_campaign_directory(str(out_dir))
    assert report["valid"] is True
    assert report["solved_count"] == 1
    assert proposer.calls == 1
    assert synthesis_calls == 3


def test_grounded_replay_rejects_formula_and_panel_tampering(
    tmp_path: Path,
) -> None:
    _registry, descriptors = R.default_grounded_observables()
    out_dir = tmp_path / "grounded"
    campaign = R.run(
        _args(out_dir), proposer=_SmallGapProposer(descriptors))

    tampered = copy.deepcopy(campaign)
    predicate = tampered["records"][0]["formula"]["compiled"]["predicate"]
    if predicate["node"] == "all":
        predicate = predicate["children"][0]
    predicate["threshold"]["value"] = 999.0
    tampered["campaign_digest"] = G.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "campaign_digest"
    })
    with pytest.raises(ValueError, match="formula"):
        R.replay_campaign_artifact(tampered, str(out_dir))

    persisted = json.loads((out_dir / "campaign.json").read_text())
    relative = persisted["records"][0]["query_panel_set"]["panels"][0][
        "npy_path"]
    npy_path = out_dir / relative
    panel = np.load(npy_path, allow_pickle=False)
    panel[0, 0] = np.uint8(1 - int(panel[0, 0]))
    np.save(npy_path, panel, allow_pickle=False)
    with pytest.raises(ValueError, match="file digest"):
        R.replay_campaign_directory(str(out_dir))


def test_no_separator_is_an_unsolved_replayable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry, descriptors = R.default_grounded_observables()
    calls = 0

    def no_separator(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise S.NoGroundedSeparator("fixture has no positive conjunction")

    monkeypatch.setattr(R.S, "synthesize_grounded_predicate", no_separator)
    out_dir = tmp_path / "no-separator"
    campaign = R.run(
        _args(out_dir), proposer=_SmallGapProposer(descriptors))
    record = campaign["records"][0]

    assert record["status"] == "UNSOLVED_SEMANTIC_GROUNDED"
    assert record["solved"] is False
    assert record["formula"] is None
    assert record["support_evaluation"]["decisions"] == []
    assert (out_dir / "workspace" / "problem_00" / "query").is_dir()
    # Run-time cold replay and this explicit cold replay both reproduce the
    # deterministic no-separator condition without a proposer call.
    assert calls == 2
    assert R.replay_campaign_directory(str(out_dir))["valid"] is True
    assert calls == 3
