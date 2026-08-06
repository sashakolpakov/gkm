from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.corpus import SplitIndex
from bongard.semantic_calibration import (
    CalibrationPanelSelection,
    SemanticCalibrationPlan,
    join_calibration_label,
)
from bongard.semantic_calibration_scoring import (
    SemanticCalibrationScoreAttempt,
    SemanticCalibrationScoreCommitment,
    SemanticCalibrationScoringError,
    score_semantic_calibration_panel,
)
from bongard.tests.test_semantic_commitment import _fixture
from bongard.tests.test_semantic_observation import _receipt as _soft_receipt
from bongard.transport import CodexStructuredResult


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _panel(path: Path) -> None:
    image = Image.new("L", (48, 48), color=255)
    draw = ImageDraw.Draw(image)
    draw.polygon(((8, 24), (22, 8), (40, 24), (22, 40)), fill=0)
    image.save(path, format="PNG")


def _score_fixture(tmp_path: Path):
    support, proposed, compiled = _fixture(tmp_path)
    panel = tmp_path / "opaque-calibration-panel.png"
    _panel(panel)
    selection = CalibrationPanelSelection(
        observation_id="development-0000",
        task_id="ff_nact2_5_0000",
        panel_id="calibration-panel-0000",
        panel_digest=hashlib.sha256(panel.read_bytes()).hexdigest(),
        split="train",
        dependence_cluster_id="task-ff_nact2_5_0000",
    )
    split = SplitIndex(
        groups=(
            ("test", ()),
            ("train", (selection.task_id,)),
            ("val", ()),
        ),
        source_digest="sha256:" + _digest("official split fixture"),
    )
    protocol = compiled.family.protocol
    plan = SemanticCalibrationPlan.create(
        protocol,
        split,
        (selection,),
        corpus_manifest_digest="sha256:" + _digest("complete corpus"),
        development_manifest_digest="sha256:" + _digest("development corpus"),
        label_reveal_protocol_digest=_digest("post-score label reveal"),
    )
    commitment = SemanticCalibrationScoreCommitment.from_panel(
        plan=plan,
        selection=selection,
        support=support,
        proposal_transport=proposed,
        protocol=protocol,
        panel=panel,
    )
    return panel, selection, protocol, plan, commitment


def test_pre_family_commitment_scores_then_joins_label_and_cold_replays(
    tmp_path: Path,
) -> None:
    panel, selection, protocol, plan, commitment = _score_fixture(tmp_path)
    calls: list[str] = []

    def scorer_transport(prompt, paths, names, schema, **kwargs):
        calls.append(hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest())
        claim = commitment.proposal_transport.proposal.soft_claim
        assert claim is not None
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue.cue_id,
                    "judgment": "supported",
                    "witness_ids": ["panel:geometry"],
                }
                for cue in claim.cues
            ]
        }
        return CodexStructuredResult(
            payload=payload,
            receipt=_soft_receipt(prompt, paths, names, schema, payload),
        )

    attempt = score_semantic_calibration_panel(
        panel,
        commitment,
        transport=scorer_transport,
    )
    assert calls == [selection.panel_digest]
    assert attempt.score_artifact.record.score == 1.0
    assert attempt.score_artifact.record.pre_observation_commitment_digest == (
        commitment.digest
    )
    assert attempt.score_artifact.record.outcome == "present"
    assert SemanticCalibrationScoreAttempt.from_data(
        attempt.to_data(),
        expected_digest=attempt.digest,
        panel=panel,
    ) == attempt

    measurement = join_calibration_label(
        plan,
        protocol,
        selection.observation_id,
        attempt.score_artifact,
        True,
        label_reveal_receipt_digest=_digest("label revealed after score"),
    )
    assert measurement.development_unit.affirmative_label is True
    assert measurement.score_artifact_digest == attempt.score_artifact.digest

    encoded = json.dumps(commitment.to_data(), sort_keys=True).lower()
    assert '"fitted_family_present": false' in encoded
    assert '"calibration_label_state": "withheld"' in encoded
    assert "affirmative_label" not in encoded
    assert '"python_predicate_authoritative": true' in encoded
    assert "lean" not in encoded


def test_calibration_score_commitment_rejects_tamper_wrong_pixels_and_direct_only(
    tmp_path: Path,
) -> None:
    panel, _selection, _protocol, _plan, commitment = _score_fixture(tmp_path)
    assert SemanticCalibrationScoreCommitment.from_data(
        commitment.to_data(), expected_digest=commitment.digest, panel=panel
    ) == commitment

    tampered = copy.deepcopy(commitment.to_data())
    tampered["calibration_label_state"] = "revealed"
    with pytest.raises(SemanticCalibrationScoringError, match="causal state"):
        SemanticCalibrationScoreCommitment.from_data(tampered)

    other = tmp_path / "other.png"
    image = Image.new("L", (48, 48), color=255)
    image.putpixel((1, 1), 0)
    image.save(other, format="PNG")
    with pytest.raises(SemanticCalibrationScoringError, match="panel bytes"):
        score_semantic_calibration_panel(other, commitment)

    object.__setattr__(
        commitment.proposal_transport.proposal,
        "soft_claim",
        None,
    )
    with pytest.raises(SemanticCalibrationScoringError, match="soft claim"):
        commitment.assert_untampered()
