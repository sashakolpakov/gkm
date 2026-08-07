from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

from bongard.artifacts import canonical_digest
from bongard.benchmark import (
    EpisodeStatus,
    SupportGatePolicy,
    VISUAL_SEMANTIC_PREDICATE_MODE,
    prepare_episode,
    run_episode,
)
from bongard.semantic_checker import (
    OptionalCheckerDisagreement,
    OptionalCheckerProcess,
    OptionalCheckerResponse,
    OptionalCheckerStatus,
    SemanticCheckerProtocolError,
    audit_optional_semantic_checker,
    capture_python_semantic_authority,
)
from bongard.semantic_episode import VisualSemanticEpisode
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.tests.test_semantic_episode import _corpus, _proposal_payload
from bongard.tests.test_semantic_synthesis import _family
from bongard.tests.test_typed_visual_transport import _receipt
from bongard.transport import CodexStructuredResult


def _complete_python_episode(tmp_path: Path):
    corpus, task_id = _corpus(tmp_path)
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    plan = prepare_episode(
        corpus,
        task_id,
        seed="optional-checker-boundary",
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    payload = _proposal_payload()

    def proposer_transport(prompt, paths, schema, **kwargs):
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

    def forbidden_scorer_transport(*args, **kwargs):
        raise AssertionError("direct-only fixture attempted soft scoring")

    episode = VisualSemanticEpisode(
        task_id=task_id,
        support_commitment=plan.support,
        policy=policy,
        family=family,
        protocol=family.protocol,
        proposer_transport=proposer_transport,
        scorer_transport=forbidden_scorer_transport,
    )
    result = run_episode(
        plan,
        episode,
        episode,
        support_gate_policy=SupportGatePolicy.visual_semantic(),
    )
    assert result.status is EpisodeStatus.COMPLETE
    assert result.bundle is not None
    return episode, result


def _authoritative_projection(episode, result) -> dict[str, object]:
    authority = capture_python_semantic_authority(episode, result)
    assert episode.compiled is not None
    assert episode.pre_observation_commitment is not None
    assert result.bundle is not None
    assert result.proposal_freeze is not None
    assert result.support_gate is not None
    return {
        "authority": authority.to_data(),
        "formula": copy.deepcopy(episode.compiled.formula.to_data()),
        "registry": episode.compiled.registry.snapshot().to_data(),
        "attachment": episode.compiled.attachment_contract.to_data(),
        "precommit": episode.pre_observation_commitment.to_data(),
        "freeze": result.proposal_freeze.to_data(),
        "support_gate": result.support_gate.to_data(),
        "predictions": result.bundle.predictions.to_data(),
        "run_archive": result.bundle.to_archive_data(),
        "semantic_archive": copy.deepcopy(episode.artifact_data()),
        "episode_result": copy.deepcopy(result.to_data()),
    }


def _reply(request, *, agrees: bool, detail: str):
    return OptionalCheckerResponse(
        checker_id="fixture-proof-checker",
        checker_version="v1",
        request_digest=request["request_digest"],
        authority_digest=request["authority"]["authority_digest"],
        agrees=agrees,
        detail=detail,
    ).to_data()


_CHECKER_CHILD = r"""
import hashlib
import json
import sys

request = json.load(sys.stdin)
mode = sys.argv[1]
response = {
    "schema": "gkm.bongard-optional-semantic-checker-response.v1",
    "checker_id": "fixture-proof-checker",
    "checker_version": "v1",
    "request_digest": request["request_digest"],
    "authority_digest": request["authority"]["authority_digest"],
    "agrees": mode != "disagree",
    "detail": "independent process replay " + mode,
}
if mode == "foreign-authority":
    response["authority_digest"] = "0" * 64
payload = json.dumps(
    response,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
response["response_digest"] = hashlib.sha256(payload).hexdigest()
sys.stdout.buffer.write(json.dumps(
    response,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8") + b"\n")
"""


def _checker_process(mode: str = "agree") -> OptionalCheckerProcess:
    return OptionalCheckerProcess(
        checker_id="fixture-proof-checker",
        checker_version="v1",
        command=(str(Path(sys.executable).resolve()), "-c", _CHECKER_CHILD, mode),
    )


def test_checker_absent_agreeing_or_unavailable_cannot_change_python_authority(
    tmp_path: Path,
) -> None:
    episode, result = _complete_python_episode(tmp_path)
    before = _authoritative_projection(episode, result)
    authority = capture_python_semantic_authority(episode, result)

    absent = audit_optional_semantic_checker(episode, result)
    assert absent.status is OptionalCheckerStatus.ABSENT
    assert absent.authority_digest == authority.digest
    assert absent.to_data()["non_authoritative"] is True
    assert _authoritative_projection(episode, result) == before

    agreeing = audit_optional_semantic_checker(
        episode,
        result,
        checker=_checker_process(),
    )
    assert agreeing.status is OptionalCheckerStatus.AGREED
    assert agreeing.response is not None and agreeing.response.agrees
    assert agreeing.authority_digest == authority.digest
    assert _authoritative_projection(episode, result) == before

    unavailable_sidecar = audit_optional_semantic_checker(
        episode,
        result,
        checker=OptionalCheckerProcess(
            checker_id="fixture-proof-checker",
            checker_version="v1",
            command=("/definitely/missing/optional-checker",),
        ),
    )
    assert unavailable_sidecar.status is OptionalCheckerStatus.UNAVAILABLE
    assert unavailable_sidecar.response is None
    assert unavailable_sidecar.unavailability_reason == (
        "optional checker process could not complete"
    )
    assert _authoritative_projection(episode, result) == before

    # Checker results remain separate from both authoritative archives.
    semantic_archive = episode.artifact_data()
    assert "checker_sidecar" not in semantic_archive
    assert "checker_response" not in semantic_archive
    assert canonical_digest(semantic_archive) == authority.semantic_archive_digest
    assert result.bundle.to_archive_data()["archive_digest"] == (
        authority.run_archive_digest
    )


def test_checker_disagreement_is_explicit_and_cannot_rewrite_predictions(
    tmp_path: Path,
) -> None:
    episode, result = _complete_python_episode(tmp_path)
    before = _authoritative_projection(episode, result)

    with pytest.raises(OptionalCheckerDisagreement) as raised:
        audit_optional_semantic_checker(
            episode,
            result,
            checker=_checker_process("disagree"),
        )

    sidecar = raised.value.sidecar
    assert sidecar.status is OptionalCheckerStatus.DISAGREED
    assert sidecar.response is not None and not sidecar.response.agrees
    assert sidecar.to_data()["may_affect_python_result"] is False
    assert _authoritative_projection(episode, result) == before


def test_checker_response_must_bind_exact_python_authority(tmp_path: Path) -> None:
    episode, result = _complete_python_episode(tmp_path)
    before = _authoritative_projection(episode, result)

    with pytest.raises(
        SemanticCheckerProtocolError, match="another Python authority"
    ):
        audit_optional_semantic_checker(
            episode,
            result,
            checker=_checker_process("foreign-authority"),
        )
    assert _authoritative_projection(episode, result) == before


def test_in_process_checker_is_rejected_before_it_can_mutate_live_state(
    tmp_path: Path,
) -> None:
    episode, result = _complete_python_episode(tmp_path)
    before = _authoritative_projection(episode, result)
    invoked = False

    def mutating_checker(request):
        nonlocal invoked
        invoked = True
        episode.query_artifacts.clear()
        return _reply(request, agrees=True, detail="attempted to rewrite archive")

    with pytest.raises(
        SemanticCheckerProtocolError,
        match="in-process optional checker callables are forbidden",
    ):
        audit_optional_semantic_checker(
            episode,
            result,
            checker=mutating_checker,
        )
    assert invoked is False
    assert _authoritative_projection(episode, result) == before
