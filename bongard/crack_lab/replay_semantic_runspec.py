"""Cold-process replay entry point for promoted semantic RunSpecs."""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cofibrations
import dataset
import semantic_compiler
import semantic_ir
import semantic_requirements
import semantic_selection
import soft_semantics
import visual_witnesses
from dataset import Problem
from semantic_ir import SemanticHypothesis
from semantic_legs import default_registry
from semantic_replay import (
    ReplayProvenanceMismatchError,
    ReplayValidationError,
    assert_replay_compatible,
    canonical_json_bytes,
    canonical_json_digest,
    callable_fingerprint,
    load_runspec,
    materialize_cold_inputs,
    semantic_cone_digest,
)
from semantic_verifier import verify_hypothesis


def verifier_related_sources() -> dict[str, object]:
    """Sources outside semantic_verifier.py that determine its verdict."""
    return {
        "cofibrations": cofibrations,
        "dataset": dataset,
        "semantic_compiler": semantic_compiler,
        "semantic_ir": semantic_ir,
        "semantic_requirements": semantic_requirements,
        "semantic_selection": semantic_selection,
        "soft_semantics": soft_semantics,
        "visual_witnesses": visual_witnesses,
    }


def replay_runspec(path: str) -> dict:
    spec = load_runspec(path)
    hash_runtime = dict(spec.provenance).get("python_hash_runtime", {})
    if hash_runtime:
        if not isinstance(hash_runtime, dict) \
                or set(hash_runtime) != {
                    "python_hash_seed_env", "python_hash_probes"}:
            raise ReplayValidationError(
                "RunSpec Python hash runtime provenance is malformed")
        seed = hash_runtime.get("python_hash_seed_env")
        probes = hash_runtime.get("python_hash_probes")
        if not isinstance(seed, str) or not isinstance(probes, list) \
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in probes):
            raise ReplayValidationError(
                "RunSpec Python hash runtime values are malformed")
        if seed != "random" and (
                os.environ.get("PYTHONHASHSEED") != seed
                or probes != [
                    hash(f"bongard-phase-d/v6/{index}")
                    for index in range(4)]):
            raise ReplayProvenanceMismatchError(
                "Python hash seed/probes differ from RunSpec provenance")
    registry = default_registry()
    assert_replay_compatible(
        spec,
        registry=registry,
        verifier=verify_hypothesis,
        verifier_sources=verifier_related_sources(),
    )

    cold = materialize_cold_inputs(spec)
    problem = Problem(
        problem_id=str(cold.problem["problem_id"]),
        category=str(cold.problem["category"]),
        concept="",
        pos=cold.positive_panels,
        neg=cold.negative_panels,
    )
    policy = cold.policy
    verdicts: list[dict] = []
    for cone_payload in cold.cones:
        hypothesis = SemanticHypothesis.from_dict(dict(cone_payload))
        expected = cold.expected_verifications.get(hypothesis.hypothesis_id)
        if expected is None:
            raise ReplayValidationError(
                f"cone {hypothesis.hypothesis_id!r} has no expected verification")
        observed = verify_hypothesis(
            hypothesis,
            registry,
            problem,
            max_support_errors=policy.max_support_errors,
            max_loo_errors=policy.max_loo_errors,
            max_rotated_loo_errors=policy.max_rotated_loo_errors,
        ).to_dict()
        expected_digest = canonical_json_digest(expected)
        observed_digest = canonical_json_digest(observed)
        if observed_digest != expected_digest:
            raise ReplayProvenanceMismatchError(
                f"verification mismatch for {hypothesis.hypothesis_id}: "
                f"recorded={expected_digest} observed={observed_digest}")
        if not observed.get("accepted"):
            raise ReplayValidationError(
                f"recorded cone {hypothesis.hypothesis_id!r} is not accepted")
        if policy.acceptance_mode == "exact" and any((
                observed.get("support_errors") != 0,
                observed.get("loo_errors") != 0,
                observed.get("rotated_loo_errors") != 0,
                observed.get("indeterminate_evaluations") != 0,
        )):
            raise ReplayValidationError(
                f"exact policy replayed nonzero error for "
                f"{hypothesis.hypothesis_id!r}")
        if (getattr(policy, "require_zero_unchecked_morphisms", True)
                and observed.get("unchecked_morphisms")):
            raise ReplayValidationError(
                f"cone {hypothesis.hypothesis_id!r} has unchecked morphisms")
        verdicts.append({
            "cone_id": hypothesis.hypothesis_id,
            "verification_digest": observed_digest,
            "accepted": True,
        })
    selection_receipt = None
    selection = cold.provenance.get("selection")
    if isinstance(selection, dict) and selection.get("candidates"):
        # Import lazily: run_semantic_cone imports this module's provenance
        # helper, while replay needs the live selector only after module load.
        from run_semantic_cone import (
            SELECTION_RISK_FIELDS,
            _candidate_manifest,
            _candidate_evaluations,
            _select,
            _selection_record,
        )

        recorded_fingerprint = selection.get("selector_fingerprint")
        current_fingerprint = callable_fingerprint(
            _select, require_source=True)
        if canonical_json_digest(recorded_fingerprint) != \
                canonical_json_digest(current_fingerprint):
            raise ReplayProvenanceMismatchError(
                "selection implementation fingerprint mismatch")
        if tuple(selection.get("risk_fields", ())) != SELECTION_RISK_FIELDS:
            raise ReplayValidationError(
                "selection risk fields do not match the live selector")
        recorded_unmeasured = selection.get("unmeasured_risks")
        expected_unmeasured = [
            name for name in semantic_selection.RISK_FIELDS
            if name not in SELECTION_RISK_FIELDS
        ]
        if recorded_unmeasured != expected_unmeasured:
            raise ReplayValidationError(
                "selection unmeasured risks do not match the live protocol")
        raw_lambda = selection.get("lambda")
        if isinstance(raw_lambda, bool) or not isinstance(
                raw_lambda, (int, float)):
            raise ReplayValidationError(
                "selection evidence has no numeric lambda")
        lambda_value = float(raw_lambda)
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise ReplayValidationError(
                "selection lambda must be finite and nonnegative")
        records = selection.get("candidates")
        if not isinstance(records, list) or not records:
            raise ReplayValidationError(
                "selection evidence must contain candidate records")

        candidate_ids: list[str] = []
        candidate_hypotheses: list[SemanticHypothesis] = []
        candidate_verifications = []
        candidate_origins: list[dict] = []
        for index, candidate in enumerate(records):
            if not isinstance(candidate, dict):
                raise ReplayValidationError(
                    f"selection candidate {index} is not an object")
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id \
                    or candidate_id in candidate_ids:
                raise ReplayValidationError(
                    "selection candidate IDs must be unique and non-empty")
            candidate_ids.append(candidate_id)
            try:
                hypothesis = SemanticHypothesis.from_dict(
                    dict(candidate["hypothesis"]))
            except Exception as exc:
                raise ReplayValidationError(
                    f"invalid selection hypothesis {candidate_id}: {exc}") from exc
            expected = candidate.get("expected_verification")
            if not isinstance(expected, dict):
                raise ReplayValidationError(
                    f"selection candidate {candidate_id} lacks verification")
            observed_verification = verify_hypothesis(
                hypothesis,
                registry,
                problem,
                max_support_errors=policy.max_support_errors,
                max_loo_errors=policy.max_loo_errors,
                max_rotated_loo_errors=policy.max_rotated_loo_errors,
            )
            if canonical_json_digest(observed_verification.to_dict()) != \
                    canonical_json_digest(expected):
                raise ReplayProvenanceMismatchError(
                    f"selection candidate verification mismatch for "
                    f"{candidate_id}")
            candidate_hypotheses.append(hypothesis)
            candidate_verifications.append(observed_verification)
            origin = candidate.get("origin")
            if not isinstance(origin, dict):
                raise ReplayValidationError(
                    f"selection candidate {candidate_id} lacks an origin")
            candidate_origins.append(origin)

        try:
            reproduced_manifest = _candidate_manifest(
                candidate_verifications,
                [hypothesis.to_dict()
                 for hypothesis in candidate_hypotheses],
                candidate_origins,
            )
        except ValueError as exc:
            raise ReplayValidationError(
                f"invalid selection candidate manifest: {exc}") from exc
        if canonical_json_digest(reproduced_manifest) != canonical_json_digest(
                selection.get("candidate_manifest")):
            raise ReplayProvenanceMismatchError(
                "selection candidate manifest does not reproduce")

        pairs = _candidate_evaluations(candidate_verifications, lambda_value)
        for candidate_id, candidate, (_verification, evaluation) in zip(
                candidate_ids, records, pairs):
            if candidate_id != evaluation.candidate_id \
                    or canonical_json_digest(candidate.get("evaluation")) != \
                    canonical_json_digest(evaluation.to_dict()):
                raise ReplayProvenanceMismatchError(
                    f"selection evaluation mismatch for {candidate_id}")
        selected = _select(candidate_verifications, lambda_value)
        if selected is None:
            raise ReplayValidationError(
                "recorded selection candidate set has no selected candidate")
        selected_index = next(
            index for index, candidate in enumerate(candidate_verifications)
            if candidate is selected)
        selected_candidate_id = candidate_ids[selected_index]
        if selected_candidate_id != selection.get("selected_candidate_id"):
            raise ReplayProvenanceMismatchError(
                "selected candidate ID does not reproduce")
        reproduced_record = _selection_record(
            selected, candidate_verifications, lambda_value)
        if canonical_json_digest(reproduced_record) != canonical_json_digest(
                selection.get("selected_record")):
            raise ReplayProvenanceMismatchError(
                "selected candidate record does not reproduce")
        if semantic_cone_digest(
                candidate_hypotheses[selected_index].to_dict()) != \
                spec.cones[0].cone_digest:
            raise ReplayProvenanceMismatchError(
                "selected candidate payload differs from certified cone")
        selection_receipt = {
            "selected_candidate_id": selected_candidate_id,
            "candidate_count": len(candidate_ids),
            "candidate_manifest_digest": canonical_json_digest(
                reproduced_manifest),
            "evidence_digest": canonical_json_digest(selection),
            "selected_record_digest": canonical_json_digest(
                reproduced_record),
        }
    return {
        "schema": "bongard.semantic-replay-receipt/v1",
        "status": "PASS",
        "process_mode": "fresh_python_subprocess",
        "spec_digest": spec.spec_digest,
        "panel_set_digest": spec.panel_set_digest,
        "cone_set_digest": spec.cone_set_digest,
        "verdicts": verdicts,
        "selection": selection_receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runspec")
    return parser.parse_args()


def main() -> None:
    receipt = replay_runspec(parse_args().runspec)
    sys.stdout.buffer.write(canonical_json_bytes(receipt) + b"\n")


if __name__ == "__main__":
    main()
