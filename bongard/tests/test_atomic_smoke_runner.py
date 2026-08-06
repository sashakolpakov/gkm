"""Offline end-to-end and adversarial tests for the 29-call atomic smoke.

The transports below mint fully validated synthetic Codex receipts.  They do
not invoke Codex, a model API, a subprocess, or the network.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bongard import benchmark
from bongard.artifacts import canonical_digest
import bongard.atomic_semantic_synthesis as S
from bongard.atomic_semantic_synthesis import (
    OPERATIONAL_SELECTION_SCOPE,
    AtomicSelectionArchive,
    OperationalNonmatchRecord,
)
import bongard.atomic_smoke_precommit as P
from bongard.atomic_smoke_precommit import (
    AtomicSmokePrecommit,
    AtomicSmokeSelection,
    ExposurePersistenceReceipt,
)
import bongard.atomic_smoke_runner as R
from bongard.atomic_smoke_runner import (
    ATOMIC_SMOKE_SUCCESS_CALL_COUNT,
    AtomicSmokeRun,
    AtomicSmokeRunError,
    atomic_smoke_run_protocol_digest,
    cold_decode_and_replay_atomic_smoke_run,
    run_atomic_smoke,
)
from bongard.cohorts import parse_official_task_id
from bongard.corpus import BongardTask, ShapeBongardCorpus, SplitIndex
from bongard.exposure import ExposureLedger
from bongard.semantic_calibration_campaign import semantic_generator_cluster_id
import bongard.transport as T
import bongard.typed_visual_proposal as V


TASK_ID = "bd_mismatch_triangle_rec1_0000"
SOURCE_DEPENDENCY_DIGEST = hashlib.sha256(
    b"synthetic atomic runner source dependencies"
).hexdigest()
LABEL_NONCE = "a" * 64
LAUNCHER_DIGEST = "b" * 64
COMMAND_CONFIG_DIGEST = "sha256:" + hashlib.sha256(
    b"synthetic atomic smoke command config"
).hexdigest()
MODEL = "gpt-test"
EFFORT = "medium"
LIVE_OBSERVER_QUESTIONS = (
    "Are hollow circles arranged in a row along an edge?",
    "Are hollow squares arranged in a row along an edge?",
    "Are outlined triangles arranged in a row along an edge?",
    "Does a polygonal loop have symbol-lined edges?",
    "Is a chain of small symbols attached to a larger contour?",
    "Does a smaller loop meet a larger loop at one junction?",
    "Is a cluster of repeated marks attached at a corner?",
    "Does a contour include a jagged black edge?",
    "Is there a tilted quadrilateral loop?",
    "Is there a triangular arrangement of repeated small marks?",
)


@dataclass(frozen=True)
class _Fixture:
    precommit: AtomicSmokePrecommit
    prediction_store: Path
    journal_store: Path


def _synthetic_precommit(root: Path, *, source_digest: str = SOURCE_DEPENDENCY_DIGEST) -> _Fixture:
    """Build a real typed plan/precommit over fourteen distinct local panels."""

    task_root = root / "bd" / "images" / TASK_ID
    sides: dict[str, tuple[Path, ...]] = {}
    for label in ("1", "0"):
        directory = task_root / label
        directory.mkdir(parents=True)
        paths: list[Path] = []
        for index in range(7):
            path = directory / f"{index}.png"
            # The benchmark boundary hashes opaque PNG bytes; image decoding is
            # intentionally the responsibility of the fake observer transport.
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + f"{TASK_ID}:{label}:{index}".encode("ascii")
            )
            paths.append(path)
        sides[label] = tuple(paths)
    task = BongardTask(
        task_id=TASK_ID,
        family="bd",
        root=task_root,
        positive=sides["1"],
        negative=sides["0"],
    )
    corpus = ShapeBongardCorpus(
        root,
        (task,),
        layout="archive",
        split=SplitIndex(
            groups=(("test", ()), ("train", (TASK_ID,)), ("val", ())),
            source_digest=P.OFFICIAL_SPLIT_SOURCE_DIGEST,
        ),
    )
    manifest = corpus.build_manifest()
    plan = benchmark.prepare_episode(
        corpus,
        TASK_ID,
        seed="synthetic episode seed",
        corpus_manifest=manifest,
        verifier_id="offline-fixture",
        label_seal_nonce=LABEL_NONCE,
    )
    parsed = parse_official_task_id(TASK_ID)
    selection = AtomicSmokeSelection.create(
        source_corpus_manifest_digest=P.OFFICIAL_CORPUS_MANIFEST_DIGEST,
        split_source_digest=P.OFFICIAL_SPLIT_SOURCE_DIGEST,
        exposure_predecessor_digest=P.OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST,
        historical_seed_digest=P.OFFICIAL_HISTORICAL_SEED_DIGEST,
        resolver_policy_digest=P.OFFICIAL_RESOLVER_POLICY_DIGEST,
        blocked_morphology_policy_digest=(
            P.OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST
        ),
        seed="synthetic post-freeze selection seed",
        selected_task_id=TASK_ID,
        selected_generator_cluster_id=semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        ),
    )
    successor = ExposureLedger.create(P.OFFICIAL_CORPUS_MANIFEST_DIGEST).record(
        phase=P.ATOMIC_SMOKE_EXPOSURE_PHASE,
        actor="offline-fixture",
        purpose=P.ATOMIC_SMOKE_EXPOSURE_PURPOSE,
        task_ids=(TASK_ID,),
        source="atomic-smoke-selection:" + selection.digest,
        observed_at="2026-08-06T12:00:00Z",
        known_task_ids=(TASK_ID,),
    )
    exposure_payload = successor.to_json().encode("utf-8")
    exposure_filename = (
        successor.digest.removeprefix("sha256:") + ".exposure.json"
    )
    exposure_receipt = ExposurePersistenceReceipt.create(
        ledger=successor,
        filename=exposure_filename,
        payload=exposure_payload,
    )
    precommit = AtomicSmokePrecommit.create(
        selection=selection,
        exposure_successor=successor,
        exposure_persistence_receipt=exposure_receipt,
        source_dependency_digest=source_digest,
        development_manifest=manifest,
        episode_plan=plan,
    )
    prediction_store = root / "prediction-store"
    prediction_store.mkdir()
    journal_store = root / "journal-store"
    journal_store.mkdir(mode=0o700)
    return _Fixture(
        precommit=precommit,
        prediction_store=prediction_store,
        journal_store=journal_store,
    )


def _receipt(
    *,
    prompt: str,
    paths: Sequence[str],
    names: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    ordinal: int,
    domain: str,
) -> T.CodexReceipt:
    """Mint one genuine, causally bound synthetic transport receipt."""

    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(dict(schema))
    if domain == "named-image":
        identities = [
            {
                "name": name,
                "byte_count": len(Path(path).read_bytes()),
                "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path, name in zip(paths, names, strict=True)
        ]
        input_schema = T.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
        view_digest = canonical_digest(identities)
        set_digest = "sha256:" + canonical_digest(
            {"schema": input_schema, "images": identities}
        )
        envelope = {
            "schema": input_schema,
            "task": prompt,
            "ordered_image_identities": identities,
            "image_view_digest": view_digest,
            "image_set_digest": set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
        }
    elif domain == "text":
        input_schema = T.TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA
        view_digest = canonical_digest([])
        set_digest = "sha256:" + canonical_digest(
            {"schema": input_schema, "images": []}
        )
        envelope = {
            "schema": input_schema,
            "task": prompt,
            "image_count": 0,
            "image_view_digest": view_digest,
            "image_set_digest": set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
        }
    else:  # pragma: no cover - test helper contract
        raise AssertionError(domain)
    body: dict[str, Any] = {
        "schema": T.CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": EFFORT,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 5,
        "reasoning_output_tokens": 1,
        "thread_id": f"00000000-0000-4000-8000-{ordinal:012d}",
        "codex_cli_version": "codex-cli offline-test",
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": "absent",
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": input_schema,
        "input_digest": canonical_digest(envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": hashlib.sha256(
            f"offline-observer-stream-{ordinal}".encode("ascii")
        ).hexdigest(),
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": T.CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    T.validate_codex_receipt(body)
    return T.CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _redigest_receipt(
    receipt: T.CodexReceipt, **changes: Any
) -> T.CodexReceipt:
    body = {**receipt.to_dict(), **changes}
    body["receipt_digest"] = canonical_digest(
        {key: value for key, value in body.items() if key != "receipt_digest"}
    )
    T.validate_codex_receipt(body)
    return T.CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


class _OfflineCodex:
    """Deterministic visual observer with optional receipt-boundary faults."""

    def __init__(
        self,
        *,
        fault_ordinal: int | None = None,
        fault: str | None = None,
        proposal_phrases: Sequence[str] = (LIVE_OBSERVER_QUESTIONS[0],),
    ):
        self.calls = 0
        self.fault_ordinal = fault_ordinal
        self.fault = fault
        self.proposal_phrases = tuple(proposal_phrases)

    def _next(self) -> int:
        self.calls += 1
        return self.calls

    def named(
        self,
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **_kwargs: Any,
    ) -> T.CodexStructuredResult:
        ordinal = self._next()
        if "description" in schema["properties"]:
            description = (
                "A sharp triangular form is visible."
                if Path(paths[0]).parent.name == "1"
                else "A rounded circular form is visible."
            )
            payload: dict[str, Any] = {"description": description}
        else:
            scorer_input = json.loads(prompt.rsplit("\n", 1)[1])
            present = "sharp triangular" in scorer_input["panel"]["description"]
            disposition = "present" if present else "operational_nonmatch"
            payload = {
                "results": [
                    {
                        "atom_id": atom["atom_id"],
                        "disposition": disposition,
                        "explanation": (
                            "visible match" if present else "clear visible nonmatch"
                        ),
                    }
                    for atom in scorer_input["atoms"]
                ]
            }
        receipt_prompt = (
            prompt + " stale" if self._fault(ordinal, "stale") else prompt
        )
        receipt_domain = (
            "text" if self._fault(ordinal, "cross-domain") else "named-image"
        )
        receipt = _receipt(
            prompt=receipt_prompt,
            paths=() if receipt_domain == "text" else paths,
            names=() if receipt_domain == "text" else names,
            schema=schema,
            payload=payload,
            ordinal=ordinal,
            domain=receipt_domain,
        )
        if self._fault(ordinal, "rerolled"):
            receipt = _redigest_receipt(
                receipt, thread_id="00000000-0000-4000-8000-000000000001"
            )
        return T.CodexStructuredResult(payload=payload, receipt=receipt)

    def text(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        **_kwargs: Any,
    ) -> T.CodexStructuredResult:
        ordinal = self._next()
        payload = {
            "atoms": [{"phrase": phrase} for phrase in self.proposal_phrases]
        }
        receipt = _receipt(
            prompt=prompt,
            paths=(),
            names=(),
            schema=schema,
            payload=payload,
            ordinal=ordinal,
            domain="text",
        )
        return T.CodexStructuredResult(payload=payload, receipt=receipt)

    def _fault(self, ordinal: int, name: str) -> bool:
        return self.fault_ordinal == ordinal and self.fault == name


def _run(fixture: _Fixture, observer: _OfflineCodex) -> AtomicSmokeRun:
    return run_atomic_smoke(
        fixture.precommit,
        source_dependency_digest=fixture.precommit.source_dependency_digest,
        command_config_digest=COMMAND_CONFIG_DIGEST,
        expected_protocol_digest=atomic_smoke_run_protocol_digest(),
        expected_launcher_digest=LAUNCHER_DIGEST,
        prediction_store_dir=fixture.prediction_store,
        journal_store_dir=fixture.journal_store,
        model=MODEL,
        reasoning_effort=EFFORT,
        named_image_transport=observer.named,
        text_transport=observer.text,
    )


def _cold_kwargs(run: AtomicSmokeRun, fixture: _Fixture) -> dict[str, Any]:
    return {
        "value": run.to_data(),
        "expected_run_digest": run.digest,
        "expected_source_dependency_digest": run.source_dependency_digest,
        "expected_precommit_digest": run.precommit_digest,
        "expected_command_config_digest": run.command_config_digest,
        "expected_protocol_digest": run.protocol_digest,
        "expected_launcher_digest": run.expected_launcher_digest,
        "expected_evidence_digest": run.evidence_digest,
        "precommit_public_data": fixture.precommit.to_data(),
        "label_seal_nonce": LABEL_NONCE,
        "prediction_store_dir": fixture.prediction_store,
        "journal_store_dir": fixture.journal_store,
    }


def test_exact_29_call_python_end_to_end_and_operational_scope(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex()
    run = _run(fixture, observer)

    assert run.status == "complete"
    assert run.terminal_phase == "cold-replay-verified"
    assert observer.calls == len(run.calls) == ATOMIC_SMOKE_SUCCESS_CALL_COUNT == 29
    assert [call.phase for call in run.calls] == (
        ["support-description"] * 12
        + ["atom-proposal"]
        + ["support-scoring"] * 12
        + ["query-description"] * 2
        + ["query-scoring"] * 2
    )
    assert [call.domain for call in run.calls] == (
        ["named-image"] * 12
        + ["text"]
        + ["named-image"] * 16
    )
    assert run.score == {
        "image_total": 2,
        "determinate": 2,
        "abstentions": 0,
        "image_correct": 2,
        "puzzle_correct": True,
    }
    assert all(
        getattr(run, field) is False
        for field in (
            "dependence_design_authorized",
            "calibration_authorized",
            "benchmark_claim_authorized",
            "official_test_authorized",
        )
    )
    assert run.exploratory_uncalibrated_nonmatch is True

    archive = AtomicSelectionArchive.from_data(
        run.to_data()["selection_archive_data"]
    )
    assert archive.matrix.atoms[0].positive_description == LIVE_OBSERVER_QUESTIONS[0]
    assert archive.selection_scope == OPERATIONAL_SELECTION_SCOPE
    claim_authority = run.selection_archive_data["claim_authority"]
    assert claim_authority == {
        "calibration_authorized": False,
        "benchmark_claim_authorized": False,
        "semantic_truth_claim": False,
    }
    negative_records = [
        cell.evidence
        for cell in archive.matrix.cells
        if not dict(archive.support_labels)[cell.panel_id]
    ]
    assert negative_records
    assert all(isinstance(record, OperationalNonmatchRecord) for record in negative_records)
    encoded = json.dumps(run.to_data(), sort_keys=True)
    assert '"disposition": "operational_nonmatch"' in encoded
    assert "certified_absent" not in encoded
    assert sorted(
        row["predicted_positive"]
        for row in run.prediction_commitment_data["queries"]
    ) == [False, True]
    assert cold_decode_and_replay_atomic_smoke_run(
        **_cold_kwargs(run, fixture)
    ).to_data() == run.to_data()


def test_question_contract_is_identical_in_schema_prompt_and_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = R._OBSERVER_QUESTION_CONTRACT
    collection_contract = R._OBSERVER_QUESTION_COLLECTION_CONTRACT
    schema = R._proposal_schema()
    prompt = R._proposal_prompt((), {})
    protocol_digest = atomic_smoke_run_protocol_digest()

    atoms_schema = schema["properties"]["atoms"]
    assert atoms_schema["description"] == collection_contract
    assert atoms_schema["items"]["properties"]["phrase"]["description"] == contract
    assert collection_contract in prompt

    monkeypatch.setattr(
        R,
        "_OBSERVER_QUESTION_COLLECTION_CONTRACT",
        collection_contract + " Revised.",
    )
    assert R._proposal_schema() != schema
    assert R._proposal_prompt((), {}) != prompt
    assert atomic_smoke_run_protocol_digest() != protocol_digest


def test_all_ten_live_question_phrases_parse_byte_identically(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex(proposal_phrases=LIVE_OBSERVER_QUESTIONS)
    run = _run(fixture, observer)

    assert run.status == "complete"
    raw_phrases = tuple(
        item["phrase"] for item in run.calls[12].payload["atoms"]
    )
    assert tuple(phrase.encode("ascii") for phrase in raw_phrases) == tuple(
        phrase.encode("ascii") for phrase in LIVE_OBSERVER_QUESTIONS
    )
    assert R.validate_atomic_smoke_proposal_payload(run.calls[12].payload) == (
        LIVE_OBSERVER_QUESTIONS
    )
    archive = AtomicSelectionArchive.from_data(
        run.to_data()["selection_archive_data"]
    )
    assert {atom.positive_description for atom in archive.matrix.atoms} == set(
        LIVE_OBSERVER_QUESTIONS
    )


def test_contract_discloses_and_protocol_binds_every_downstream_surface_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = R._OBSERVER_QUESTION_CONTRACT
    atomic_policy = S.atomic_affirmative_surface_policy_data()
    prose_policy = V.affirmative_prose_surface_policy_data()

    assert {item["family"] for item in atomic_policy["closed_families"]} == {
        "disjunction",
        "negation-laundering",
        "bundling",
    }
    assert {item["family"] for item in prose_policy["closed_families"]} == {
        "explicit-negation",
        "support-relative",
        "support-index-count",
        "control-text",
    }
    for policy in (atomic_policy, prose_policy):
        for family in policy["closed_families"]:
            assert family["family"] in contract
            for pattern in family["patterns"]:
                assert pattern["name"] in contract
                assert isinstance(pattern["regex"], str) and pattern["regex"]
                assert isinstance(pattern["flags"], int)

    protocol_digest = atomic_smoke_run_protocol_digest()
    monkeypatch.setattr(
        R,
        "affirmative_prose_surface_policy_data",
        lambda: {**prose_policy, "audit_mutation": True},
    )
    assert atomic_smoke_run_protocol_digest() != protocol_digest


def test_duplicate_question_phrases_are_rejected_exactly() -> None:
    phrase = LIVE_OBSERVER_QUESTIONS[0]
    with pytest.raises(AtomicSmokeRunError, match="pairwise-distinct"):
        R.validate_atomic_smoke_proposal_payload(
            {"atoms": [{"phrase": phrase}, {"phrase": phrase}]}
        )


def test_observer_question_soft_cue_byte_limit_is_exact() -> None:
    prefix = "Is "
    overlength = (
        prefix
        + "a" * (V.MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES - len(prefix))
        + "?"
    )
    assert len(overlength.encode("utf-8")) == (
        V.MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES + 1
    )
    with pytest.raises(AtomicSmokeRunError, match="soft-cue limit"):
        R.validate_atomic_smoke_proposal_payload(
            {"atoms": [{"phrase": overlength}]}
        )


@pytest.mark.parametrize(
    "phrase",
    (
        "Are hollow circles, arranged along an edge?",
        " Are hollow circles arranged along an edge?",
        "Are hollow circles arranged along an edge? ",
        "Are hollow  circles arranged along an edge?",
        "Are hollow_circles arranged along an edge?",
    ),
)
def test_illegal_punctuation_whitespace_and_non_surface_characters_are_rejected(
    phrase: str,
) -> None:
    with pytest.raises(AtomicSmokeRunError):
        R.validate_atomic_smoke_proposal_payload(
            {"atoms": [{"phrase": phrase}]}
        )


@pytest.mark.parametrize(
    ("family", "phrase"),
    (
        ("disjunction", "Does the shape contain circles or squares?"),
        ("negation-laundering", "Does the shape avoid circles?"),
        ("bundling", "Does the shape contain circles and squares?"),
    ),
)
def test_every_atomicity_pattern_family_is_rejected(
    family: str, phrase: str
) -> None:
    with pytest.raises(AtomicSmokeRunError, match=family):
        R.validate_atomic_smoke_proposal_payload(
            {"atoms": [{"phrase": phrase}]}
        )


@pytest.mark.parametrize(
    ("surface", "phrase"),
    (
        ("not", "Is the shape not circular?"),
        ("labelled support item", "Is this a positive support panel?"),
        ("support label", "Is the shape in class A?"),
        ("indexed support item", "Does the first panel have circles?"),
        ("support-set cardinality", "Do all six panels have circles?"),
        ("threshold", "Does the threshold mark a circle?"),
        ("prompt vocabulary", "Is prompt-injection visible?"),
        ("code instruction", "Does code describe a circle?"),
    ),
)
def test_every_reachable_downstream_affirmative_surface_is_rejected(
    surface: str, phrase: str
) -> None:
    with pytest.raises(AtomicSmokeRunError, match=surface):
        R.validate_atomic_smoke_proposal_payload(
            {"atoms": [{"phrase": phrase}]}
        )


@pytest.mark.parametrize("count", (0, 13))
def test_proposal_count_contract_is_exactly_one_through_twelve(count: int) -> None:
    payload = {
        "atoms": [
            {"phrase": f"Is there visible shape {index}?"}
            for index in range(count)
        ]
    }
    with pytest.raises(AtomicSmokeRunError, match="1..12"):
        R.validate_atomic_smoke_proposal_payload(payload)


def test_public_preflight_schemas_and_scorer_surface_are_runner_exact() -> None:
    assert R.atomic_smoke_proposal_schema() == R._proposal_schema()
    assert R.atomic_smoke_scorer_schema() == R._scorer_schema()
    atom_id = "a" * 64
    payload = {
        "results": [
            {
                "atom_id": atom_id,
                "disposition": "indeterminate",
                "explanation": "The visible geometry is ambiguous.",
            }
        ]
    }
    assert R.validate_atomic_smoke_scorer_payload(
        payload, expected_atom_ids=(atom_id,)
    ) == ((atom_id, "indeterminate", "The visible geometry is ambiguous."),)


@pytest.mark.parametrize(
    "phrase",
    (
        "Hollow circles are arranged in a row along an edge.",
        "Are hollow circles arranged in a row? along an edge?",
        "Are hollow circles arranged in a row along an edge??",
    ),
)
def test_atom_proposal_rejects_noncanonical_or_internal_question_marks(
    tmp_path: Path, phrase: str
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex(proposal_phrases=(phrase,))
    run = _run(fixture, observer)

    assert run.status == "failed"
    assert run.terminal_phase == "atom-proposal"
    assert observer.calls == 13
    assert len(run.calls) == 13
    assert "canonical observer-question form" in run.failure["reason"]
    assert run.selection_archive_data is None
    assert run.prediction_commitment_data is None


def test_query_pixels_are_unavailable_until_formula_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    plan = fixture.precommit.episode_plan
    query_source_ids = {id(source) for source in plan._query_sources}
    state = {"formula_frozen": False, "query_reads": 0}
    original_synthesize = R.synthesize_atomic_conjunction
    original_read = type(plan._query_sources[0]).read_verified

    def synthesize_then_freeze(*args: Any, **kwargs: Any):
        archive = original_synthesize(*args, **kwargs)
        state["formula_frozen"] = True
        return archive

    def guarded_read(source: Any) -> bytes:
        if id(source) in query_source_ids:
            assert state["formula_frozen"] is True
            state["query_reads"] += 1
        return original_read(source)

    monkeypatch.setattr(R, "synthesize_atomic_conjunction", synthesize_then_freeze)
    monkeypatch.setattr(type(plan._query_sources[0]), "read_verified", guarded_read)
    run = _run(fixture, _OfflineCodex())
    assert run.status == "complete"
    # Each of two query sources is verified before and after description and scoring.
    assert state == {"formula_frozen": True, "query_reads": 8}


def test_prediction_is_durable_before_labels_are_revealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex()
    plan = fixture.precommit.episode_plan
    original_reveal = type(plan)._revealed_labels
    observed: list[str] = []

    def guarded_reveal(live_plan: benchmark.EpisodePlan):
        files = tuple(fixture.prediction_store.glob("*.predictions.json"))
        assert len(files) == 1
        assert files[0].stat().st_size > 0
        assert observer.calls == 29
        observed.append(files[0].name)
        return original_reveal(live_plan)

    monkeypatch.setattr(type(plan), "_revealed_labels", guarded_reveal)
    run = _run(fixture, observer)
    assert run.status == "complete"
    assert observed == [run.prediction_persistence_receipt.filename]


def test_failure_terminal_preserves_exact_prefix_and_cold_replays(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex(fault_ordinal=14, fault="stale")
    run = _run(fixture, observer)

    assert run.status == "failed"
    assert run.terminal_phase == "support-scoring"
    assert observer.calls == 14
    assert len(run.calls) == 13
    assert run.calls[-1].phase == "atom-proposal"
    assert run.label_reveal_data is None
    assert run.score is None
    assert cold_decode_and_replay_atomic_smoke_run(
        **_cold_kwargs(run, fixture)
    ).to_data() == run.to_data()


@pytest.mark.parametrize(
    ("fault_ordinal", "fault", "prefix_length"),
    ((14, "cross-domain", 13), (14, "stale", 13), (2, "rerolled", 1)),
)
def test_cross_domain_stale_and_rerolled_receipts_are_rejected(
    tmp_path: Path,
    fault_ordinal: int,
    fault: str,
    prefix_length: int,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    run = _run(
        fixture, _OfflineCodex(fault_ordinal=fault_ordinal, fault=fault)
    )
    assert run.status == "failed"
    assert len(run.calls) == prefix_length
    assert run.label_reveal_data is None
    assert run.prediction_commitment_data is None
    assert run.failure["reason"]


def test_cold_replay_rejects_nonce_precommit_transcript_formula_and_store_tamper(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path / "primary")
    run = _run(fixture, _OfflineCodex())
    assert run.status == "complete"
    cold = _cold_kwargs(run, fixture)

    with pytest.raises(AtomicSmokeRunError, match="label seal"):
        cold_decode_and_replay_atomic_smoke_run(
            **{**cold, "label_seal_nonce": "c" * 64}
        )

    other = _synthetic_precommit(
        tmp_path / "other", source_digest=hashlib.sha256(b"other source").hexdigest()
    )
    with pytest.raises(AtomicSmokeRunError, match="external precommit"):
        cold_decode_and_replay_atomic_smoke_run(
            **{**cold, "precommit_public_data": other.precommit.to_data()}
        )

    transcript_tamper = deepcopy(run.to_data())
    transcript_tamper["calls"][0]["payload"]["description"] = "Altered vision."
    with pytest.raises(AtomicSmokeRunError):
        cold_decode_and_replay_atomic_smoke_run(
            **{**cold, "value": transcript_tamper}
        )

    formula_tamper = deepcopy(run.to_data())
    formula_tamper["selection_archive_data"]["formula"]["atom_ids"] = []
    with pytest.raises(AtomicSmokeRunError):
        cold_decode_and_replay_atomic_smoke_run(
            **{**cold, "value": formula_tamper}
        )

    persisted = (
        fixture.prediction_store / run.prediction_persistence_receipt.filename
    )
    persisted.write_bytes(b"{}\n")
    with pytest.raises(AtomicSmokeRunError, match="persisted prediction"):
        cold_decode_and_replay_atomic_smoke_run(**cold)


def test_complete_journal_has_exact_header_29_intents_29_results_and_terminal(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    run = _run(fixture, _OfflineCodex())

    assert run.status == "complete"
    assert run.command_config_digest == COMMAND_CONFIG_DIGEST
    assert run.journal_receipt.intent_count == 29
    assert run.journal_receipt.result_count == 29
    assert run.journal_receipt.state == "result-closed"
    assert run.journal_receipt.open_intent_ordinal is None
    assert len(tuple(fixture.journal_store.glob("*.intent.json"))) == 29
    result_paths = tuple(sorted(fixture.journal_store.glob("*.result.json")))
    assert len(result_paths) == 29
    assert len(tuple(fixture.journal_store.glob("*.header.json"))) == 1
    assert len(tuple(fixture.journal_store.glob("*.terminal-run.json"))) == 1
    for ordinal, path in enumerate(result_paths, 1):
        result = R.AtomicSmokeCallJournalResult.from_data(
            json.loads(path.read_bytes())
        )
        assert result.call.ordinal == ordinal
        assert result.call.to_data() == run.calls[ordinal - 1].to_data()
    header_path = next(fixture.journal_store.glob("*.header.json"))
    header = R.AtomicSmokeJournalHeader.from_data(json.loads(header_path.read_bytes()))
    assert header.command_config_digest == COMMAND_CONFIG_DIGEST
    assert header.header_digest == run.journal_receipt.header_digest


def test_transport_failure_leaves_first_intent_attempted_with_unknown_completion(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex()

    def failed_named(*args: Any, **kwargs: Any) -> T.CodexStructuredResult:
        observer._next()
        raise RuntimeError("synthetic unknown transport completion")

    run = run_atomic_smoke(
        fixture.precommit,
        source_dependency_digest=fixture.precommit.source_dependency_digest,
        command_config_digest=COMMAND_CONFIG_DIGEST,
        expected_protocol_digest=atomic_smoke_run_protocol_digest(),
        expected_launcher_digest=LAUNCHER_DIGEST,
        prediction_store_dir=fixture.prediction_store,
        journal_store_dir=fixture.journal_store,
        model=MODEL,
        reasoning_effort=EFFORT,
        named_image_transport=failed_named,
        text_transport=observer.text,
    )

    assert run.status == "failed"
    assert observer.calls == 1
    assert len(run.calls) == 0
    assert run.journal_receipt.state == "intent-open"
    assert run.journal_receipt.open_intent_ordinal == 1
    assert run.journal_receipt.intent_count == 1
    assert run.journal_receipt.result_count == 0
    assert len(tuple(fixture.journal_store.glob("*.intent.json"))) == 1
    assert not tuple(fixture.journal_store.glob("*.result.json"))
    assert len(tuple(fixture.journal_store.glob("*.terminal-run.json"))) == 1
    assert cold_decode_and_replay_atomic_smoke_run(
        **_cold_kwargs(run, fixture)
    ).to_data() == run.to_data()


@pytest.mark.parametrize(
    ("boundary", "observer_calls", "intent_count", "result_count"),
    (
        ("before-intent", 0, 0, 0),
        ("before-result", 1, 1, 0),
        ("after-result", 1, 1, 1),
    ),
)
def test_journal_boundary_failures_preserve_exact_durable_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    observer_calls: int,
    intent_count: int,
    result_count: int,
) -> None:
    fixture = _synthetic_precommit(tmp_path)
    observer = _OfflineCodex()
    if boundary in {"before-intent", "before-result"}:
        original = R._persist_journal_json
        failed = {"value": False}

        def inject(directory_fd: int, filename: str, value: Mapping[str, Any]) -> bool:
            suffix = ".001.intent.json" if boundary == "before-intent" else ".001.result.json"
            if filename.endswith(suffix) and not failed["value"]:
                failed["value"] = True
                raise OSError("synthetic persistence boundary fault")
            return original(directory_fd, filename, value)

        monkeypatch.setattr(R, "_persist_journal_json", inject)
    else:
        original_result = R._AtomicSmokeCallJournal.persist_result
        failed = {"value": False}

        def inject_after(
            journal: Any, intent: Any, call: Any
        ) -> Any:
            durable = original_result(journal, intent, call)
            if not failed["value"]:
                failed["value"] = True
                raise OSError("synthetic post-result durability fault")
            return durable

        monkeypatch.setattr(R._AtomicSmokeCallJournal, "persist_result", inject_after)

    run = _run(fixture, observer)
    assert run.status == "failed"
    assert observer.calls == observer_calls
    assert len(run.calls) == result_count
    assert run.journal_receipt.intent_count == intent_count
    assert run.journal_receipt.result_count == result_count
    assert len(tuple(fixture.journal_store.glob("*.intent.json"))) == intent_count
    assert len(tuple(fixture.journal_store.glob("*.result.json"))) == result_count
    assert len(tuple(fixture.journal_store.glob("*.terminal-run.json"))) == 1
    assert cold_decode_and_replay_atomic_smoke_run(
        **_cold_kwargs(run, fixture)
    ).to_data() == run.to_data()


def test_journal_tamper_cross_run_and_second_attempt_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path / "primary")
    first_observer = _OfflineCodex()
    run = _run(fixture, first_observer)
    assert run.status == "complete"

    retry_observer = _OfflineCodex()
    with pytest.raises(AtomicSmokeRunError, match="resume/retry"):
        _run(fixture, retry_observer)
    assert retry_observer.calls == 0

    other = _synthetic_precommit(tmp_path / "other")
    with pytest.raises(AtomicSmokeRunError, match="journal"):
        cold_decode_and_replay_atomic_smoke_run(
            **{**_cold_kwargs(run, fixture), "journal_store_dir": other.journal_store}
        )

    intent_path = sorted(fixture.journal_store.glob("*.intent.json"))[0]
    intent_data = json.loads(intent_path.read_bytes())
    intent_data["prompt"] = intent_data["prompt"] + " tampered"
    intent_path.write_bytes(
        json.dumps(intent_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    with pytest.raises(AtomicSmokeRunError, match="journal"):
        cold_decode_and_replay_atomic_smoke_run(**_cold_kwargs(run, fixture))


def test_journal_rejects_bool_coercion_and_non_private_directory(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_precommit(tmp_path / "typed")
    run = _run(fixture, _OfflineCodex())
    receipt = deepcopy(run.journal_receipt.to_data())
    receipt["result_count"] = True
    with pytest.raises(AtomicSmokeRunError, match="literal integer"):
        R.AtomicSmokeJournalReceipt.from_data(receipt)

    intent_path = sorted(fixture.journal_store.glob("*.intent.json"))[0]
    intent = json.loads(intent_path.read_bytes())
    intent["ordinal"] = True
    with pytest.raises(AtomicSmokeRunError, match="literal integer"):
        R.AtomicSmokeCallIntent.from_data(intent)

    insecure = _synthetic_precommit(tmp_path / "insecure")
    insecure.journal_store.chmod(0o755)
    observer = _OfflineCodex()
    with pytest.raises(AtomicSmokeRunError, match="exact mode 0700"):
        _run(insecure, observer)
    assert observer.calls == 0
