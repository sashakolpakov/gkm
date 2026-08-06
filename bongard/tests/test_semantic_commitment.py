from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from PIL import Image
import pytest

from bongard.artifacts import BlobRef, SupportCommitment, SupportExample
from bongard.semantic_commitment import (
    REFERENCE_EXECUTION_SEMANTICS,
    SemanticCommitmentError,
    SemanticPreObservationCommitment,
)
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.semantic_synthesis import compile_visual_semantic_proposal
from bongard.tests.test_semantic_synthesis import _family
from bongard.tests.test_typed_visual_transport import _receipt
from bongard.transport import CodexStructuredResult
from bongard.typed_visual_transport import propose_typed_visual
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


def _paths(tmp_path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    sides: list[tuple[Path, ...]] = []
    for side, offset in (("positive", 0), ("negative", 6)):
        paths: list[Path] = []
        for index in range(6):
            path = tmp_path / f"private-{side}-{index}.png"
            image = Image.new("L", (16, 16), color=255)
            image.putpixel(((index + offset) % 16, (index * 3) % 16), 0)
            image.save(path, format="PNG")
            paths.append(path)
        sides.append(tuple(paths))
    return sides[0], sides[1]


def _payload() -> dict[str, object]:
    return {
        "positive_description": "one connected angular form with one enclosed region",
        "panel_descriptions": {
            **{
                f"pos_{index}": f"one angular loop, presentation {index}"
                for index in range(6)
            },
            **{
                f"neg_{index}": f"two separated marks, presentation {index}"
                for index in range(6)
            },
        },
        "view": "carrier_shape",
        "deterministic_atoms": [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 1},
            },
            {
                "catalog_key": "hole.owner_count",
                "comparison": "equal",
                "arguments": {"target_count": 1},
            },
        ],
        "soft_claim": {
            "positive_description": "a compact lamp-like angular silhouette",
            "cue_descriptions": [
                "a narrow central waist joins broader angular ends",
                "four outward tips surround the central waist",
            ],
        },
        "formula": {"kind": "all", "atom_indices": [0, 1, 2]},
    }


def _fixture(tmp_path: Path):
    family = _family()
    positive, negative = _paths(tmp_path)
    payload = _payload()

    def transport(prompt, paths, schema, **kwargs):
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(
                prompt,
                paths,
                schema,
                payload,
                model=kwargs["model"],
                effort=kwargs["reasoning_effort"],
            ),
        )

    proposed = propose_typed_visual(
        positive,
        negative,
        catalog=DIRECT_VISUAL_ATOM_CATALOG,
        protocol=family.protocol,
        transport=transport,
    )
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    compiled = compile_visual_semantic_proposal(
        proposed.proposal,
        policy=policy,
        expected_policy_digest=policy.digest(),
        family=family,
    )
    examples: list[SupportExample] = []
    for side, paths in (("negative", negative), ("positive", positive)):
        for index, path in enumerate(paths):
            payload_bytes = path.read_bytes()
            examples.append(
                SupportExample(
                    BlobRef.from_bytes(
                        f"support-{side}-{index}", payload_bytes, "image/png"
                    ),
                    side == "positive",
                )
            )
    support = SupportCommitment(
        run_id="run-semantic-commitment-fixture",
        issued_by="canonical-bongard-verifier",
        corpus_digest=hashlib.sha256(b"fixture corpus").hexdigest(),
        support=tuple(examples),
        verifier_nonce=hashlib.sha256(b"fixture nonce").hexdigest(),
    )
    return support, proposed, compiled


def test_pre_observation_commitment_binds_complete_python_first_preimages(
    tmp_path: Path,
) -> None:
    support, proposed, compiled = _fixture(tmp_path)
    artifact = SemanticPreObservationCommitment(support, proposed, compiled)
    data = artifact.to_data()

    assert data["reference_execution_semantics"] == REFERENCE_EXECUTION_SEMANTICS
    assert data["optional_checker_may_affect_result"] is False
    assert data["identities"] == artifact.identity_data()
    assert artifact.identity_data()["proposal_transport_digest"] == proposed.digest
    assert artifact.identity_data()["lowering_archive_digest"] == (
        compiled.lowering_archive.digest
    )
    assert data["proposal_transport"] == proposed.to_data()
    assert data["soft_scorer_family"] == compiled.family.to_data()
    assert data["compiled_formula"] == compiled.formula.to_data()
    assert len(artifact.digest) == 64
    artifact.assert_untampered()
    assert SemanticPreObservationCommitment.verify_data(
        data,
        support=support,
        proposal_transport=proposed,
        compiled=compiled,
        expected_digest=artifact.digest,
    ).digest == artifact.digest


def test_pre_observation_commitment_rejects_changed_archive_or_support_bytes(
    tmp_path: Path,
) -> None:
    support, proposed, compiled = _fixture(tmp_path)
    artifact = SemanticPreObservationCommitment(support, proposed, compiled)
    changed = copy.deepcopy(artifact.to_data())
    changed["identities"]["typed_proposal_digest"] = "0" * 64
    with pytest.raises(SemanticCommitmentError, match="differs from reconstructed"):
        SemanticPreObservationCommitment.verify_data(
            changed,
            support=support,
            proposal_transport=proposed,
            compiled=compiled,
        )

    first, *rest = support.support
    forged_panel = BlobRef(
        first.panel.blob_id,
        "f" * 64,
        first.panel.byte_count,
        first.panel.media_type,
    )
    forged_support = SupportCommitment(
        run_id=support.run_id,
        issued_by=support.issued_by,
        corpus_digest=support.corpus_digest,
        support=(SupportExample(forged_panel, first.positive), *rest),
        verifier_nonce=support.verifier_nonce,
    )
    with pytest.raises(SemanticCommitmentError, match="bytes differ"):
        SemanticPreObservationCommitment(forged_support, proposed, compiled)
