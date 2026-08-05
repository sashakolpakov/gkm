from __future__ import annotations

import base64
import copy
import json
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_replay as R


def _fake_leg(value):
    return value


def _second_fake_leg(left, right):
    return left if left is not None else right


def _fake_verifier(hypothesis, registry, problem, **policy):
    return hypothesis, registry, problem, policy


def _other_verifier(hypothesis, registry, problem, **policy):
    return problem, registry, hypothesis, policy


@dataclass(frozen=True)
class FakeContract:
    name: str = "identity"
    domain: tuple[str, ...] = ("Panel",)
    codomain: str = "Panel"
    implementation: object = _fake_leg
    complexity: int = 1
    invariances: frozenset[str] = frozenset({"translation"})
    equivariances: frozenset[str] = frozenset()
    failure_modes: tuple[str, ...] = ()
    indeterminate_modes: tuple[str, ...] = ()
    version: str = "1"
    proxy_for: tuple[str, ...] = ()
    measurement_kind: str | None = None
    proxy_directions: tuple[tuple[str, str], ...] = ()


class FakeRegistry:
    def __init__(self, contracts=None):
        self._contracts = tuple(contracts or (FakeContract(),))

    def contracts(self):
        return self._contracts


@dataclass(frozen=True)
class FakeProblem:
    problem_id: str
    category: str
    concept: str
    pos: tuple[np.ndarray, ...]
    neg: tuple[np.ndarray, ...]


def _problem() -> FakeProblem:
    pos0 = np.zeros((16, 16), dtype=np.uint8)
    pos0[2:14, 5] = 1
    pos1 = np.zeros((16, 16), dtype=np.uint8)
    np.fill_diagonal(pos1, 1)
    neg0 = np.zeros((16, 16), dtype=np.uint8)
    neg0[4:12, 4:12] = 1
    neg1 = np.zeros((16, 16), dtype=np.uint8)
    neg1[3:13, 3] = 1
    neg1[3:13, 12] = 1
    return FakeProblem(
        problem_id="fixture_problem",
        category="basic",
        concept="ground_truth_must_be_opt_in",
        pos=(pos0, pos1),
        neg=(neg0, neg1),
    )


def _cone(edge_parameters=None):
    return {
        "version": "0.1",
        "hypothesis_id": "H_fixture",
        "description": "one connected line",
        "diagram": {
            "edges": [{
                "target": "scene",
                "call": {
                    "leg_name": "parse_scene",
                    "args": ["panel"],
                    "parameters": edge_parameters or {},
                },
            }],
        },
        "score_node": "score",
        "order": "low_positive",
        "preservation_morphisms": [{"name": "rotate"}],
    }


def _spec(**overrides) -> R.SemanticRunSpec:
    args = {
        "opaque_id": "problem_00",
        "problem": _problem(),
        "cones": [_cone()],
        "registry": FakeRegistry(),
        "verifier": _fake_verifier,
        "policy": R.VerifierPolicy(
            unexecuted_checks=("contrast", "counterfactual", "archive_regression"),
        ),
        "expected_verifications": {
            "H_fixture": {
                "accepted": True,
                "support_errors": 0,
                "loo_errors": 0,
            },
        },
        "provenance": {
            "dataset": {"name": "fixture", "seed": 7, "revision": "deadbeef"},
            "harness_git_commit": "0123456789abcdef",
        },
        "dependency_distributions": (),
    }
    args.update(overrides)
    return R.build_runspec(**args)


def test_canonical_json_is_order_independent_and_strict():
    left = {"z": [3, {"b": 2, "a": 1}], "a": -0.0}
    right = {"a": 0.0, "z": [3, {"a": 1, "b": 2}]}
    assert R.canonical_json_bytes(left) == R.canonical_json_bytes(right)
    assert R.canonical_json_digest(left) == R.canonical_json_digest(right)
    assert R.canonical_json_bytes(left) == b'{"a":0.0,"z":[3,{"a":1,"b":2}]}'

    with pytest.raises(R.ReplayValidationError, match="non-finite"):
        R.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(R.ReplayValidationError, match="not a string"):
        R.canonical_json_bytes({1: "bad key"})


def test_binary_panel_is_canonical_bitpacked_and_round_trips():
    panel = np.zeros((19, 23), dtype=np.uint8)
    panel[1::3, 2::5] = 1
    record = R.PanelRecord.from_array(panel, "pos", 0)

    assert record.encoding == R.PACKED_BINARY_ENCODING
    assert len(base64.b64decode(record.data)) == (panel.size + 7) // 8
    np.testing.assert_array_equal(record.decode(), panel)
    record.validate()

    # Layout does not affect canonical C-order bytes or their digest.
    fortran = np.asfortranarray(panel)
    same = R.PanelRecord.from_array(fortran, "pos", 0)
    assert same.content_digest == record.content_digest
    assert same.data == record.data

    signed_zero = panel.astype(np.float32)
    signed_zero[0, 0] = -0.0
    signed_record = R.PanelRecord.from_array(signed_zero, "pos", 0)
    signed_record.validate()
    assert not np.signbit(signed_record.decode()[0, 0])


def test_nonbinary_panel_uses_canonical_little_endian_raw_bytes():
    native = np.arange(30, dtype=np.uint16).reshape(5, 6) * 3
    big_endian = native.astype(">u2")
    record = R.PanelRecord.from_array(big_endian, "neg", 2)

    assert record.encoding == R.RAW_ENCODING
    assert record.dtype == "<u2"
    decoded = record.decode()
    assert decoded.dtype.str == "<u2"
    np.testing.assert_array_equal(decoded, native)


def test_panel_validation_detects_payload_and_digest_corruption():
    record = R.PanelRecord.from_array(np.eye(8, dtype=np.uint8), "pos", 0)
    document = record.to_dict()
    payload = bytearray(base64.b64decode(document["data"]))
    payload[0] ^= 1
    document["data"] = base64.b64encode(payload).decode("ascii")
    with pytest.raises(R.ReplayValidationError, match="digest mismatch"):
        R.PanelRecord.from_dict(document)

    document = record.to_dict()
    document["data"] = "not base64!"
    with pytest.raises(R.ReplayValidationError, match="invalid base64"):
        R.PanelRecord.from_dict(document)

    document = record.to_dict()
    document["shape"][0] = 8.5
    with pytest.raises(R.ReplayValidationError, match="expected integer"):
        R.PanelRecord.from_dict(document)


def test_cone_digest_is_canonical_and_record_binds_expected_verdict():
    first = _cone({"alpha": 1, "beta": [2, 3]})
    second = json.loads(json.dumps(first))
    second["diagram"]["edges"][0]["call"]["parameters"] = {
        "beta": [2, 3], "alpha": 1,
    }
    assert R.semantic_cone_digest(first) == R.semantic_cone_digest(second)

    record = R.ConeRecord.from_cone(
        first,
        expected_verification={"accepted": True, "support_errors": 0},
    )
    assert record.cone_id == "H_fixture"
    assert record.expected_verification["accepted"] is True
    changed = record.to_dict()
    changed["cone"]["description"] = "different semantics"
    with pytest.raises(R.ReplayValidationError, match="digest mismatch"):
        R.ConeRecord.from_dict(changed)

    changed = record.to_dict()
    changed["cone_id"] = "H_other"
    with pytest.raises(R.ReplayValidationError, match="does not match"):
        R.ConeRecord.from_dict(changed)

    with pytest.raises(R.ReplayValidationError, match="hypothesis_id"):
        R.ConeRecord.from_cone({"description": "not replayable"})


def test_registry_fingerprint_covers_contract_order_and_implementation():
    a = FakeContract(name="a", domain=("Left", "Right"),
                     implementation=_second_fake_leg)
    b = FakeContract(name="b", implementation=_fake_leg)
    forward = R.registry_fingerprint(FakeRegistry((a, b)))
    reverse = R.registry_fingerprint(FakeRegistry((b, a)))
    assert forward == reverse

    swapped_domain = FakeContract(
        name="a", domain=("Right", "Left"), implementation=_second_fake_leg)
    changed = R.registry_fingerprint(FakeRegistry((swapped_domain, b)))
    assert changed["digest"] != forward["digest"]

    changed_code = FakeContract(name="b", implementation=_second_fake_leg)
    changed = R.registry_fingerprint(FakeRegistry((a, changed_code)))
    assert changed["digest"] != forward["digest"]

    changed_kind = FakeContract(name="b", measurement_kind="count")
    changed = R.registry_fingerprint(FakeRegistry((a, changed_kind)))
    assert changed["digest"] != forward["digest"]

    changed_direction = FakeContract(
        name="b", proxy_for=("open",),
        proxy_directions=(("open", "low"),))
    changed = R.registry_fingerprint(FakeRegistry((a, changed_direction)))
    assert changed["digest"] != forward["digest"]

    changed_disposition = FakeContract(
        name="b", indeterminate_modes=("poor_fit",))
    changed = R.registry_fingerprint(FakeRegistry((a, changed_disposition)))
    assert changed["digest"] != forward["digest"]
    with_manifest = R.registry_fingerprint(
        FakeRegistry((changed_disposition,)), include_manifest=True)
    assert with_manifest["schema"] == "bongard.semantic-registry/v3"
    assert with_manifest["contracts"][0]["failure_modes"] == []
    assert with_manifest["contracts"][0]["indeterminate_modes"] == ["poor_fit"]


def test_real_semantic_registry_has_stable_source_complete_fingerprint():
    from semantic_legs import default_registry

    first = R.registry_fingerprint(default_registry())
    second = R.registry_fingerprint(default_registry())
    assert first == second
    assert first["contract_count"] >= 50
    assert first["source_complete"] is True


def test_verifier_policy_cannot_launder_tolerant_thresholds_as_exact():
    exact = R.VerifierPolicy()
    tolerant = R.VerifierPolicy(max_support_errors=1, max_loo_errors=2)
    rotated_tolerant = R.VerifierPolicy(max_rotated_loo_errors=1)
    relaxed_gate = R.VerifierPolicy(require_zero_unchecked_morphisms=False)
    assert exact.to_dict()["acceptance_mode"] == "exact"
    assert exact.to_dict()["schema"] == "bongard.semantic-verifier-policy/v2"
    assert exact.to_dict()["require_zero_indeterminate_evaluations"] is True
    assert exact.require_zero_unchecked_morphisms is True
    with pytest.raises(R.ReplayValidationError, match="cannot admit indeterminate"):
        R.VerifierPolicy(
            require_zero_indeterminate_evaluations=False).validate()
    assert exact.require_threshold_overlap is False
    assert tolerant.to_dict()["acceptance_mode"] == "tolerant"
    assert rotated_tolerant.to_dict()["acceptance_mode"] == "tolerant"
    assert relaxed_gate.to_dict()["acceptance_mode"] == "tolerant"

    forged = tolerant.to_dict()
    forged["acceptance_mode"] = "exact"
    with pytest.raises(R.ReplayValidationError, match="does not match"):
        R.VerifierPolicy.from_dict(forged)

    forged = rotated_tolerant.to_dict()
    forged["acceptance_mode"] = "exact"
    with pytest.raises(R.ReplayValidationError, match="does not match"):
        R.VerifierPolicy.from_dict(forged)

    bad_span = exact.to_dict()
    bad_span["max_fold_threshold_span"] = "0.1"
    with pytest.raises(R.ReplayValidationError, match="finite and non-negative"):
        R.VerifierPolicy.from_dict(bad_span)

    with pytest.raises(R.ReplayValidationError, match="not an admission gate"):
        R.VerifierPolicy(max_fold_threshold_span=0.1).to_dict()
    with pytest.raises(R.ReplayValidationError, match="not enforced"):
        R.VerifierPolicy(require_threshold_overlap=True).to_dict()

    # The two fields added while v1 was still under development load with
    # fail-closed defaults for already-created v1 documents.
    earlier_v1 = exact.to_dict()
    del earlier_v1["max_rotated_loo_errors"]
    del earlier_v1["require_zero_unchecked_morphisms"]
    loaded = R.VerifierPolicy.from_dict(earlier_v1)
    assert loaded.max_rotated_loo_errors == 0
    assert loaded.require_zero_unchecked_morphisms is True

    malformed = exact.to_dict()
    malformed["unexecuted_checks"] = 7
    with pytest.raises(R.ReplayValidationError, match="string list"):
        R.VerifierPolicy.from_dict(malformed)

    malformed = exact.to_dict()
    malformed["threshold_policy"] = 7
    with pytest.raises(R.ReplayValidationError, match="must be a string"):
        R.VerifierPolicy.from_dict(malformed)


def test_build_validate_materialize_is_deterministic_and_ground_truth_safe():
    first = _spec()
    second = _spec()
    assert first.spec_digest == second.spec_digest
    assert R.canonical_json_bytes(first.to_dict()) == R.canonical_json_bytes(
        second.to_dict())
    first.validate()

    assert "concept" not in first.problem
    assert first.problem["opaque_id"] == "problem_00"
    assert first.problem["problem_id"] == "problem_00"
    assert "source_problem_id" not in first.problem
    assert first.panel_set_digest.startswith("sha256:")
    assert first.cone_set_digest.startswith("sha256:")
    assert first.registry["source_complete"] is True

    cold = R.materialize_cold_inputs(first)
    expected_problem = _problem()
    assert len(cold.positive_panels) == len(expected_problem.pos)
    assert len(cold.negative_panels) == len(expected_problem.neg)
    for observed, expected in zip(cold.positive_panels, expected_problem.pos):
        np.testing.assert_array_equal(observed, expected)
    for observed, expected in zip(cold.negative_panels, expected_problem.neg):
        np.testing.assert_array_equal(observed, expected)
    assert cold.cones[0]["hypothesis_id"] == "H_fixture"
    assert cold.expected_verifications["H_fixture"]["accepted"] is True
    assert cold.policy.acceptance_mode == "exact"
    assert set(cold.policy.unexecuted_checks) == {
        "contrast", "counterfactual", "archive_regression",
    }

    with pytest.raises(R.ReplayValidationError, match="unknown cones"):
        _spec(expected_verifications={"H_unknown": {"accepted": False}})

    disclosed = _spec(include_ground_truth=True)
    assert disclosed.problem["problem_id"] == "problem_00"
    assert disclosed.problem["source_problem_id"] == "fixture_problem"
    assert disclosed.problem["concept"] == "ground_truth_must_be_opt_in"


def test_runspec_save_load_is_canonical_atomic_and_write_bounded(tmp_path):
    spec = _spec()
    path = tmp_path / "nested" / "run.json"

    # The safe default refuses a path outside bongard/.
    with pytest.raises(R.ReplayWriteBoundaryError, match="escapes"):
        R.save_runspec(path, spec, create_parents=True)

    # The boundary directory itself is not a file target; accepting it would
    # create the atomic temp file in its parent, outside the allowed tree.
    with pytest.raises(R.ReplayWriteBoundaryError, match="strictly inside"):
        R.save_runspec(tmp_path, spec, allowed_root=tmp_path)

    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(R.ReplayWriteBoundaryError, match="escapes"):
        R.save_runspec(
            allowed / "escape" / "run.json",
            spec,
            allowed_root=allowed,
        )
    assert not list(outside.glob(".*.tmp"))

    saved = R.save_runspec(
        path,
        spec,
        allowed_root=tmp_path,
        create_parents=True,
    )
    assert saved == path
    assert not list(path.parent.glob(".*.tmp"))
    assert path.read_bytes() == R.canonical_json_bytes(spec.to_dict()) + b"\n"

    loaded = R.load_runspec(path)
    assert loaded.spec_digest == spec.spec_digest
    assert loaded.to_dict() == spec.to_dict()


def test_runspec_reports_missing_file_and_nested_corruption(tmp_path):
    with pytest.raises(R.ReplayDataMissingError, match="does not exist"):
        R.load_runspec(tmp_path / "absent.json")

    document = _spec().to_dict()
    document["cones"][0]["cone"]["description"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(R.ReplayValidationError, match="digest mismatch"):
        R.load_runspec(path)


def test_v1_records_reject_unknown_and_duplicate_json_keys(tmp_path):
    spec = _spec()

    document = spec.to_dict()
    document["injected"] = "not covered by the recorded spec digest"
    with pytest.raises(R.ReplayValidationError, match="unknown v1 keys"):
        R.SemanticRunSpec.from_dict(document)

    panel = spec.panels[0].to_dict()
    panel["injected"] = True
    with pytest.raises(R.ReplayValidationError, match="unknown v1 keys"):
        R.PanelRecord.from_dict(panel)

    cone = spec.cones[0].to_dict()
    cone["injected"] = True
    with pytest.raises(R.ReplayValidationError, match="unknown v1 keys"):
        R.ConeRecord.from_dict(cone)

    policy = spec.verifier["policy"].copy()
    policy["injected"] = True
    with pytest.raises(R.ReplayValidationError, match="unknown v1 keys"):
        R.VerifierPolicy.from_dict(policy)

    raw = R.canonical_json_bytes(spec.to_dict()).decode("utf-8")
    duplicate = '{"schema":"duplicate",' + raw[1:]
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(R.ReplayValidationError, match="duplicate JSON"):
        R.load_runspec(path)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"bad":NaN}', encoding="utf-8")
    with pytest.raises(R.ReplayValidationError, match="non-finite JSON"):
        R.load_runspec(nonfinite)


def test_replay_compatibility_detects_registry_and_verifier_drift():
    spec = _spec()
    R.assert_replay_compatible(
        spec,
        registry=FakeRegistry(),
        verifier=_fake_verifier,
    )

    changed_contract = FakeContract(complexity=2)
    with pytest.raises(R.ReplayProvenanceMismatchError, match="registry"):
        R.validate_registry_compatibility(spec, FakeRegistry((changed_contract,)))

    with pytest.raises(R.ReplayProvenanceMismatchError, match="verifier"):
        R.validate_verifier_compatibility(spec, _other_verifier)

    for key in ("implementation", "version", "cache_tag"):
        changed_verifier = copy.deepcopy(dict(spec.verifier))
        changed_verifier["provenance"]["python"][key] += "-different"
        changed_runtime = replace(spec, verifier=changed_verifier)
        with pytest.raises(R.ReplayProvenanceMismatchError, match=key):
            R.validate_verifier_compatibility(changed_runtime, _fake_verifier)

    for key in ("system", "machine", "byteorder"):
        changed_verifier = copy.deepcopy(dict(spec.verifier))
        changed_verifier["provenance"]["platform"][key] += "-different"
        changed_runtime = replace(spec, verifier=changed_verifier)
        with pytest.raises(R.ReplayProvenanceMismatchError, match=key):
            R.validate_verifier_compatibility(changed_runtime, _fake_verifier)


def test_verifier_related_sources_are_sealed_and_required_for_replay():
    sources = {
        "compiler": _fake_leg,
        "semantic_ir": FakeContract,
        "visual_witnesses": sys.modules[__name__],
    }
    spec = _spec(verifier_related_sources=sources)
    compatibility_alias = _spec(verifier_sources=sources)
    assert compatibility_alias.spec_digest == spec.spec_digest
    recorded = spec.verifier["provenance"]["related_sources"]
    assert list(recorded) == sorted(sources)
    assert all(item["source_complete"] for item in recorded.values())

    R.assert_replay_compatible(
        spec,
        registry=FakeRegistry(),
        verifier=_fake_verifier,
        verifier_related_sources=sources,
    )
    with pytest.raises(R.ReplayDataMissingError, match="verifier_sources"):
        R.assert_replay_compatible(
            spec,
            registry=FakeRegistry(),
            verifier=_fake_verifier,
        )

    changed = dict(sources)
    changed["compiler"] = _second_fake_leg
    with pytest.raises(R.ReplayProvenanceMismatchError, match="compiler"):
        R.assert_replay_compatible(
            spec,
            registry=FakeRegistry(),
            verifier=_fake_verifier,
            verifier_related_sources=changed,
        )

    with pytest.raises(R.ReplayProvenanceMismatchError, match="source set"):
        R.assert_replay_compatible(
            spec,
            registry=FakeRegistry(),
            verifier=_fake_verifier,
            verifier_related_sources={"compiler": _fake_leg},
        )

    with pytest.raises(R.ReplayValidationError, match="only one"):
        _spec(
            verifier_sources=sources,
            verifier_related_sources=sources,
        )


def test_verifier_related_source_manifest_rejects_malformed_digests():
    spec = _spec(verifier_sources={"compiler": _fake_leg})
    document = copy.deepcopy(spec.to_dict())
    document["verifier"]["provenance"]["related_sources"]["compiler"][
        "source_digest"
    ] = "not-a-digest"
    with pytest.raises(R.ReplayValidationError, match="source_digest"):
        R.SemanticRunSpec.from_dict(document)


def test_missing_sources_and_dependencies_are_explicit():
    with pytest.raises(R.ReplayDataMissingError, match="source"):
        R.callable_fingerprint(len, require_source=True)

    missing_name = "definitely-not-an-installed-bongard-replay-package"
    observed = R.capture_dependency_versions((missing_name,), strict=False)
    assert observed == [{"distribution": missing_name, "status": "missing"}]
    with pytest.raises(R.ReplayMissingDependencyError, match="not installed"):
        R.capture_dependency_versions((missing_name,), strict=True)


def test_spec_digest_rejects_provenance_rewriting_even_when_payloads_are_intact():
    spec = _spec()
    document = copy.deepcopy(spec.to_dict())
    document["provenance"]["harness_git_commit"] = "rewritten"
    with pytest.raises(R.ReplayValidationError, match="RunSpec digest mismatch"):
        R.SemanticRunSpec.from_dict(document)
