from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bongard.artifacts import (
    ArtifactTamperError,
    AtomReplayInput,
    ColdReplayInputs,
    QueryReplayInput,
    TruthEvidenceRecord,
)
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import Atom, Relation, StaticLegCall, formula_digest
from bongard.legs import BOOLEAN_WITNESS, PANEL, LegContract, LegRegistry, TypedValue
from bongard.predicate_backend import (
    PYTHON_PREDICATE_BACKEND,
    PredicateBackend,
    PythonPredicateBackend,
)


def origin(method: str) -> Provenance:
    return Provenance("test-python-leg", "1", method, input_digests=("panel",))


def witness_leg(panel: dict[str, str]) -> Evidence[bool]:
    state = panel["state"]
    if state == "present":
        return Evidence.present(True, origin(state))
    if state == "absent":
        return Evidence.certified_absent(origin(state), "constructive counter-witness")
    if state == "indeterminate":
        return Evidence.indeterminate(origin(state), "ambiguous raster")
    raise RuntimeError("detector crashed")


def predicate() -> tuple[LegRegistry, Atom]:
    registry = LegRegistry()
    reference = registry.register(
        LegContract(
            "python_witness",
            "1",
            (PANEL,),
            BOOLEAN_WITNESS,
            witness_leg,
        )
    )
    registry.freeze()
    return registry, Atom(
        StaticLegCall(reference, ("panel",)),
        Relation.PRESENT,
        "the registered positive witness is present",
    )


def truth_record(disposition: Disposition) -> TruthEvidenceRecord:
    provenance = origin(f"cold-{disposition.value}")
    if disposition is Disposition.PRESENT:
        evidence = Evidence.present(True, provenance)
    elif disposition is Disposition.CERTIFIED_ABSENT:
        evidence = Evidence.certified_absent(provenance, "cold counter-witness")
    elif disposition is Disposition.INDETERMINATE:
        evidence = Evidence.indeterminate(provenance, "cold ambiguity")
    else:
        evidence = Evidence.error(provenance, "ColdError", "cold failure")
    return TruthEvidenceRecord.from_evidence(evidence)


def cold_inputs(formula: Atom) -> ColdReplayInputs:
    return ColdReplayInputs(
        proposal_freeze_digest="0" * 64,
        query_release_digest="1" * 64,
        formula_digest=formula_digest(formula),
        registry_digest="2" * 64,
        queries=(
            QueryReplayInput(
                "query-a",
                "3" * 64,
                (AtomReplayInput((), truth_record(Disposition.PRESENT)),),
            ),
            QueryReplayInput(
                "query-b",
                "4" * 64,
                (
                    AtomReplayInput(
                        (), truth_record(Disposition.CERTIFIED_ABSENT)
                    ),
                ),
            ),
        ),
    )


def test_python_backend_is_the_complete_default_contract() -> None:
    assert isinstance(PYTHON_PREDICATE_BACKEND, PredicateBackend)
    assert isinstance(PythonPredicateBackend(), PredicateBackend)
    assert PYTHON_PREDICATE_BACKEND.backend_id == "python-closed-ir/v1"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("present", Disposition.PRESENT),
        ("absent", Disposition.CERTIFIED_ABSENT),
        ("indeterminate", Disposition.INDETERMINATE),
        ("error", Disposition.ERROR),
    ],
)
def test_python_backend_evaluates_all_four_dispositions(
    state: str, expected: Disposition
) -> None:
    registry, formula = predicate()
    PYTHON_PREDICATE_BACKEND.validate(formula, registry.snapshot(), {"panel": PANEL})
    result = PYTHON_PREDICATE_BACKEND.evaluate(
        formula,
        registry,
        {"panel": TypedValue(PANEL, {"state": state})},
    )
    assert result.disposition is expected


def test_python_backend_replays_plain_json_without_legs_or_backend_fields() -> None:
    _, formula = predicate()
    cold = cold_inputs(formula)
    formula_data = json.loads(json.dumps(formula.to_data()))
    cold_data = json.loads(json.dumps(cold.to_data()))

    assert "backend" not in formula_data
    assert "backend" not in cold_data
    replayed = PYTHON_PREDICATE_BACKEND.replay_payload(formula_data, cold_data)
    assert [(query_id, record.disposition) for query_id, record in replayed] == [
        ("query-a", Disposition.PRESENT),
        ("query-b", Disposition.CERTIFIED_ABSENT),
    ]


def test_python_backend_replay_fails_if_formula_changes_after_commitment() -> None:
    _, formula = predicate()
    cold = cold_inputs(formula)
    changed = replace(formula, claim="a changed claim")
    with pytest.raises(ArtifactTamperError, match="formula digest mismatch"):
        PYTHON_PREDICATE_BACKEND.replay_payload(
            changed.to_data(), cold.to_data()
        )
