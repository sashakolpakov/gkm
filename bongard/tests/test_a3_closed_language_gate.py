from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import bongard.a3_closed_language_gate as gate
from bongard.artifacts import canonical_digest, canonical_json
from bongard.closed_visual_predicates import (
    ClosedPanelPredicate,
    ClosedPredicateKind,
    SymmetryMetric,
    SymmetryThresholdPredicate,
    freeze_closed_predicate_library,
)
from bongard.composite_visual_packet import (
    extract_exact_panel_witness_packet,
    verify_exact_panel_witness_packet,
)


def _png(seed: int) -> bytes:
    panel = np.full((40, 40), 255, dtype=np.uint8)
    row = 3 + seed % 12
    column = 4 + (seed * 3) % 15
    panel[row : row + 8, column : column + 9] = 0
    panel[30, 2 + seed] = 0
    encoded = BytesIO()
    Image.fromarray(panel, mode="L").save(encoded, format="PNG")
    return encoded.getvalue()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_mapping(root: Path) -> gate.A3SupportMapping:
    supports: list[gate.SupportPanelAuthority] = []
    for ordinal, label in enumerate((True,) * 6 + (False,) * 6):
        source_index = ordinal if label else ordinal - 6
        relative = f"synthetic/images/task/{1 if label else 0}/{source_index}.png"
        payload = _png(ordinal)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        supports.append(
            gate.SupportPanelAuthority(
                label=label,
                source_index=source_index,
                relative_path=relative,
                png_sha256=hashlib.sha256(payload).hexdigest(),
                historical_loop_packet_digest=_digest(f"historical-{ordinal}"),
            )
        )
    heldouts = tuple(
        gate.HeldoutPanelAuthority(
            label=label,
            source_index=6,
            relative_path=f"synthetic/images/task/{1 if label else 0}/heldout.png",
            png_sha256=_digest(f"forbidden-{label}"),
        )
        for label in (False, True)
    )
    return gate.A3SupportMapping(
        authority_record_digest=_digest("synthetic-record"),
        authority_file_sha256=_digest("synthetic-file"),
        task_id="synthetic-task",
        split="train",
        corpus_manifest_digest="sha256:" + _digest("synthetic-corpus"),
        split_source_digest="sha256:" + _digest("synthetic-split"),
        supports=tuple(supports),
        heldouts=heldouts,
    )


def test_canonical_a3_mapping_is_pinned_and_contains_only_twelve_supports(
    tmp_path: Path,
) -> None:
    mapping = gate.load_canonical_a3_support_mapping()

    assert mapping.authority_record_digest == (
        gate.CANONICAL_FORENSICS_RECORD_DIGEST
    )
    assert mapping.support_mapping_digest == gate.CANONICAL_SUPPORT_MAPPING_DIGEST
    assert len(mapping.supports) == 12
    assert [(item.label, item.source_index) for item in mapping.supports] == [
        *((True, index) for index in (0, 1, 2, 3, 5, 6)),
        *((False, index) for index in (0, 1, 2, 3, 4, 6)),
    ]
    assert {(item.label, item.source_index) for item in mapping.heldouts} == {
        (True, 4),
        (False, 5),
    }
    assert not ({item.relative_path for item in mapping.supports} & {
        item.relative_path for item in mapping.heldouts
    })

    original = json.loads(gate.CANONICAL_RECORD_PATH.read_bytes())
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(original, indent=2), encoding="utf-8")
    with pytest.raises(gate.A3ClosedLanguageGateError, match="canonical encoding"):
        gate.load_canonical_a3_support_mapping(pretty)

    changed = dict(original)
    changed["source_binding"] = dict(changed["source_binding"])
    changed["source_binding"]["task_id"] = "some-other-task"
    content = dict(changed)
    content.pop("record_digest")
    changed["record_digest"] = canonical_digest(content)
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(canonical_json(changed) + b"\n")
    with pytest.raises(gate.A3ClosedLanguageGateError, match="not the pin"):
        gate.load_canonical_a3_support_mapping(tampered)


def test_secure_support_reader_checks_digest_escape_and_every_symlink_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    relative = "bd/images/task/1/0.png"
    payload = _png(0)
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    authority = gate.SupportPanelAuthority(
        label=True,
        source_index=0,
        relative_path=relative,
        png_sha256=hashlib.sha256(payload).hexdigest(),
        historical_loop_packet_digest=_digest("packet"),
    )
    assert gate._read_authenticated_support_png(
        corpus_root=root, authority=authority
    ) == payload

    wrong = gate.SupportPanelAuthority(
        label=True,
        source_index=0,
        relative_path=relative,
        png_sha256="0" * 64,
        historical_loop_packet_digest=_digest("packet"),
    )
    with pytest.raises(gate.A3ClosedLanguageGateError, match="SHA-256"):
        gate._read_authenticated_support_png(corpus_root=root, authority=wrong)

    with pytest.raises(gate.A3ClosedLanguageGateError, match="unsafe"):
        gate.SupportPanelAuthority(
            label=True,
            source_index=0,
            relative_path="../outside.png",
            png_sha256=hashlib.sha256(payload).hexdigest(),
            historical_loop_packet_digest=_digest("packet"),
        )

    outside = tmp_path / "outside.png"
    outside.write_bytes(payload)
    link = root / "bd" / "images" / "linked"
    link.symlink_to(outside.parent, target_is_directory=True)
    linked = gate.SupportPanelAuthority(
        label=True,
        source_index=0,
        relative_path="bd/images/linked/outside.png",
        png_sha256=hashlib.sha256(payload).hexdigest(),
        historical_loop_packet_digest=_digest("packet"),
    )
    with pytest.raises(gate.A3ClosedLanguageGateError, match="no-follow"):
        gate._read_authenticated_support_png(corpus_root=root, authority=linked)


def test_small_synthetic_library_uses_exact_composites_and_tagged_summary() -> None:
    packets = tuple(
        extract_exact_panel_witness_packet(_png(index)) for index in range(4)
    )
    for index, packet in enumerate(packets):
        assert verify_exact_panel_witness_packet(
            packet, expected_png_bytes=_png(index)
        ) == packet

    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST,
            500_000,
        )
    )
    library = freeze_closed_predicate_library((predicate,))
    frozen = gate._bind_frozen_library(
        library,
        expected_member_count=1,
        require_complete=False,
        expected_kind_counts=None,
    )
    result = gate.support_only_expressibility_oracle(
        frozen.library,
        positive_support_packets=packets[:2],
        negative_support_packets=packets[2:],
        model_predicate=None,
    )
    summary = gate._separator_counts_by_kind(
        result.separator_digests,
        frozen.member_kind_by_digest,
    )
    assert set(summary) == {item.value for item in ClosedPredicateKind}
    assert sum(summary.values()) == len(result.separator_digests)
    assert frozen.member_count == 1
    assert frozen.member_counts_by_kind == {
        "direct_counts": 0,
        "relational": 0,
        "symmetry": 1,
    }

    fake_digests = tuple(_digest(f"member-{index}") for index in range(3))
    assert gate._separator_counts_by_kind(
        fake_digests,
        {
            fake_digests[0]: ClosedPredicateKind.RELATIONAL,
            fake_digests[1]: ClosedPredicateKind.DIRECT_COUNTS,
            fake_digests[2]: ClosedPredicateKind.SYMMETRY,
        },
    ) == {"direct_counts": 1, "relational": 1, "symmetry": 1}


def test_synthetic_end_to_end_reads_only_supports_and_writes_compact_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    mapping = _synthetic_mapping(corpus_root)
    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(
            SymmetryMetric.COVERAGE_AT_LEAST,
            500_000,
        )
    )
    frozen = gate._bind_frozen_library(
        freeze_closed_predicate_library((predicate,)),
        expected_member_count=1,
        require_complete=False,
        expected_kind_counts=None,
    )

    # The production function rejects this mapping.  This explicit replacement
    # opens only the private synthetic execution seam for this unit test.
    monkeypatch.setattr(gate, "_require_canonical_mapping", lambda _: None)
    read_paths: list[str] = []

    def read_support(
        *, corpus_root: str | Path, authority: gate.SupportPanelAuthority
    ) -> bytes:
        read_paths.append(authority.relative_path)
        return gate._read_authenticated_support_png(
            corpus_root=corpus_root, authority=authority
        )

    outcome = gate._execute_frozen_support_gate(
        mapping=mapping,
        frozen=frozen,
        corpus_root=corpus_root,
        output_dir=tmp_path / "results",
        support_reader=read_support,
    )

    assert read_paths == [item.relative_path for item in mapping.supports]
    assert not set(read_paths) & {item.relative_path for item in mapping.heldouts}
    payload = outcome.report_path.read_bytes()
    decoded = json.loads(payload)
    assert payload == canonical_json(decoded) + b"\n"
    assert len(payload) < 64 * 1024
    content = dict(decoded)
    declared = content.pop("record_digest")
    assert declared == "sha256:" + canonical_digest(content)
    assert decoded["support"]["count"] == 12
    assert decoded["frozen_library"]["member_count"] == 1
    assert sum(
        decoded["oracle"]["separator_counts_by_tagged_kind"].values()
    ) == decoded["oracle"]["exact_forward_separator_count"]
    assert decoded["claim_boundary"] == {
        "action_program_json_authorized": False,
        "benchmark_or_generalization_claim_authorized": False,
        "canonical_attempt3_support_mapping_only": True,
        "evaluation_kind": (
            "already-exposed-support-only-closed-language-coverage"
        ),
        "heldout_pixels_read": False,
        "model_or_proposer_called": False,
        "negation_rescue_authorized": False,
        "new_exposure_event_created": False,
        "new_pixels_opened": False,
        "official_test_pixels_read": False,
        "polarity_flip_authorized": False,
        "query_pixels_read": False,
    }


def test_public_runner_freezes_before_entering_png_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    mapping = object()
    frozen = object()
    expected = gate.A3ClosedLanguageGateResult({}, tmp_path / "unused.json")

    def load(_: str | Path) -> object:
        events.append("load-control-json")
        return mapping

    def freeze() -> object:
        events.append("freeze-65678")
        return frozen

    def execute(**kwargs: object) -> gate.A3ClosedLanguageGateResult:
        assert kwargs["mapping"] is mapping
        assert kwargs["frozen"] is frozen
        events.append("png-phase")
        return expected

    monkeypatch.setattr(gate, "load_canonical_a3_support_mapping", load)
    monkeypatch.setattr(gate, "_freeze_complete_library_before_pixels", freeze)
    monkeypatch.setattr(gate, "_execute_frozen_support_gate", execute)

    assert gate.run_a3_closed_language_gate(
        corpus_root=tmp_path,
        output_dir=tmp_path,
        forensics_record_path=tmp_path / "control.json",
    ) is expected
    assert events == ["load-control-json", "freeze-65678", "png-phase"]


def test_production_execution_rejects_noncanonical_mapping_before_reader(
    tmp_path: Path,
) -> None:
    mapping = _synthetic_mapping(tmp_path / "corpus")
    predicate = ClosedPanelPredicate.symmetry(
        SymmetryThresholdPredicate(SymmetryMetric.COVERAGE_AT_LEAST, 500_000)
    )
    frozen = gate._bind_frozen_library(
        freeze_closed_predicate_library((predicate,)),
        expected_member_count=1,
        require_complete=False,
        expected_kind_counts=None,
    )
    called = False

    def forbidden_reader(**_: object) -> bytes:
        nonlocal called
        called = True
        raise AssertionError("reader must not be called")

    with pytest.raises(
        gate.A3ClosedLanguageGateError,
        match="only the pinned canonical A3 support mapping",
    ):
        gate._execute_frozen_support_gate(
            mapping=mapping,
            frozen=frozen,
            corpus_root=tmp_path,
            output_dir=tmp_path,
            support_reader=forbidden_reader,
        )
    assert called is False
