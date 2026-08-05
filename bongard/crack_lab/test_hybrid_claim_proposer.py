"""Single-claim headless proposer tests."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hybrid_claim_proposer as H  # noqa: E402


def test_codex_claim_proposer_commits_exactly_one_side_free_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    ordinal = 0
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            panel = np.full((128, 128), 255, dtype=np.uint8)
            panel[ordinal + 1, ordinal + 1] = 0
            Image.fromarray(panel, mode="L").save(path)
            paths.append(str(path))
            ordinal += 1
    calls = []

    def fake_run(task, panel_paths, schema, **kwargs):
        calls.append((task, tuple(panel_paths), schema, kwargs))
        payload = {
            "analysis": "A compact animal silhouette is stable.",
            "claim": "contains a bird-like outlined object",
        }
        output_schema_digest = H._raw_digest(schema)
        panel_view_digest = H.codex_proposer.ordered_panel_view_digest(
            panel_paths)
        causal = H.codex_proposer._causal_input_metadata(
            task, panel_paths, output_schema_digest, panel_view_digest, None)
        receipt = {
            "source": "codex-cli",
            **causal,
            "requested_model": kwargs["model"],
            "requested_reasoning_effort": kwargs["reasoning_effort"],
            "output_schema_digest": output_schema_digest,
            "structured_output_digest": H._raw_digest(payload),
            "proposed_source_digest": "",
            "proposed_log_digest": "",
        }
        return SimpleNamespace(
            payload=payload,
            receipt=SimpleNamespace(to_dict=lambda: receipt),
        )

    monkeypatch.setattr(H.codex_proposer, "run_codex_structured", fake_run)
    monkeypatch.setattr(H.codex_proposer, "validate_codex_receipt",
                        lambda _receipt: None)
    proposer = H.CodexHybridClaimProposer(
        model="fixture-model", minutes=2, reasoning_effort="medium")
    bundle = proposer.propose("problem_00", paths)

    assert len(calls) == 1
    assert bundle.claim == "contains a bird-like outlined object"
    stored = bundle.to_dict()
    assert H.ClaimProposalBundle.from_dict(stored) == bundle
    assert set(stored) == {
        "schema", "problem_id", "analysis", "claim", "receipt",
        "proposal_digest",
    }
    for field in (
        "requested_model", "requested_reasoning_effort", "prompt_digest",
        "output_schema_digest", "structured_output_digest", "input_digest",
        "panel_view_digest", "panel_set_digest",
    ):
        changed = copy.deepcopy(stored)
        changed["receipt"][field] = "tampered"
        unsigned = {key: item for key, item in changed.items()
                    if key != "proposal_digest"}
        changed["proposal_digest"] = H.canonical_digest(unsigned)
        changed_bundle = H.ClaimProposalBundle.from_dict(changed)
        with pytest.raises(ValueError, match=field):
            H.validate_claim_proposal_receipt(
                changed_bundle, paths,
                model="fixture-model", reasoning_effort="medium")


@pytest.mark.parametrize("claim", [
    "the positive panels contain birds",
    "the first six have the shape",
    "matches filename pos_0.png",
    "matches this exact pixel hash",
    "does not contain a circle",
    "an object without curves",
    "lacks a closed loop",
    "shows the absence of symmetry",
    "contains a non-bird object",
    "is missing a circular part",
    "fails to close its outline",
])
def test_claim_validator_rejects_side_and_identity_language(claim: str) -> None:
    bundle = H.ClaimProposalBundle(
        problem_id="problem_00",
        analysis="fixture",
        claim=claim,
        receipt=H.make_offline_fixture_receipt("claim-validator"),
    ).to_dict()
    with pytest.raises(ValueError, match="side-free"):
        H.ClaimProposalBundle.from_dict(bundle)
