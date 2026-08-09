"""Bounded affirmative atom-slate and heterogeneous-contrast tests."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from bongard import panel_positive_atom_slate as atom_module
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard.panel_positive_atom_slate import (
    ATOM_COUNT,
    ATOM_IDS,
    FORMULA_COUNT,
    PROPOSER_IMAGE_NAMES,
    AffirmativeAtomSlate,
    AtomFormula,
    AtomPanelScoreArtifact,
    AtomPanelScoreRequest,
    AtomPanelScoreRow,
    AtomPanelStatus,
    AtomScoreInterval,
    AtomSlateProposerArtifact,
    AtomSlateProposerRequest,
    AtomSupportInventory,
    PositiveAtomSlateError,
    atom_panel_score_output_schema,
    atom_panel_score_prompt,
    atom_slate_proposer_output_schema,
    atom_slate_proposer_prompt,
    enumerate_affirmative_atom_formulas,
    observe_affirmative_atom_panel,
    propose_affirmative_atom_slate,
    verify_atom_panel_score_artifact,
    verify_atom_slate_proposer_artifact,
)
from bongard.panel_typed_codex_observer import build_panel_only_observation_context
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _groups() -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    return (
        tuple(_png(270 + index) for index in range(6)),
        tuple(_png(280 + index) for index in range(6)),
    )


def _runtime(first_png: bytes):
    return build_panel_only_observation_context(
        first_png,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    ).runtime


def _slate_payload() -> dict[str, object]:
    atoms = (
        "convex carrier",
        "four straight structural runs",
        "single closed contour",
        "bilateral symmetry",
        "oblique corners",
        "central point contact",
        "curved outer boundary",
        "nested figure",
    )
    return dict(zip(ATOM_IDS, atoms, strict=True))


def _transport(payload, expected_names, expected_bytes, calls):
    def call(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == tuple(expected_names)
        assert tuple(Path(path).read_bytes() for path in paths) == tuple(expected_bytes)
        calls.append((prompt, tuple(names), deepcopy(schema)))
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return call


def _proposer_artifact():
    first, second = _groups()
    request = AtomSlateProposerRequest.build(first, second, runtime=_runtime(first[0]))
    calls: list[object] = []
    artifact = propose_affirmative_atom_slate(
        first,
        second,
        request=request,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_transport(
            _slate_payload(), PROPOSER_IMAGE_NAMES, (*first, *second), calls
        ),
    )
    assert len(calls) == 1
    return first, second, artifact


def _score_payload(default: tuple[int, int] = (3, 4)) -> dict[str, object]:
    payload: dict[str, object] = {}
    for atom_id in ATOM_IDS:
        payload[f"{atom_id}_lower"] = default[0]
        payload[f"{atom_id}_upper"] = default[1]
    return payload


def test_proposer_has_fixed_slots_exact_support_boundary_and_no_formula_output() -> None:
    first, second = _groups()
    request = AtomSlateProposerRequest.build(first, second, runtime=_runtime(first[0]))
    assert AtomSlateProposerRequest.from_data(request.to_data()) == request
    assert len(request.presentation) == 12
    assert tuple(item.name for item in request.presentation) == PROPOSER_IMAGE_NAMES
    assert request.to_data()["query_image_count"] == 0

    prompt = atom_slate_proposer_prompt(request)
    schema = atom_slate_proposer_output_schema(request)
    assert "Group B may be heterogeneous" in prompt
    assert "never infer or describe one shared Group B concept" in prompt
    assert "Do not select or combine atoms" in prompt
    assert set(schema["properties"]) == set(ATOM_IDS)
    assert schema["required"] == list(ATOM_IDS)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "bad_atom",
    (
        "closed and convex carrier",
        "closed or convex carrier",
        "carrier without a dent",
        "score above three",
        "python predicate",
    ),
)
def test_slate_rejects_composition_negation_thresholds_and_code(bad_atom: str) -> None:
    payload = _slate_payload()
    payload["atom_03"] = bad_atom
    first, second = _groups()
    request = AtomSlateProposerRequest.build(first, second, runtime=_runtime(first[0]))
    with pytest.raises(PositiveAtomSlateError):
        propose_affirmative_atom_slate(
            first,
            second,
            request=request,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=_transport(
                payload, PROPOSER_IMAGE_NAMES, (*first, *second), []
            ),
        )


def test_proposer_one_call_roundtrip_and_pixel_bound_cold_replay() -> None:
    first, second, artifact = _proposer_artifact()
    assert artifact.benchmark_sealable is False
    assert AtomSlateProposerArtifact.from_data(artifact.to_data()) == artifact
    assert verify_atom_slate_proposer_artifact(
        artifact,
        first,
        second,
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact

    changed = list(second)
    changed[-1] = _png(399)
    with pytest.raises(PositiveAtomSlateError, match="pixels"):
        verify_atom_slate_proposer_artifact(
            artifact,
            first,
            tuple(changed),
            expected_artifact_digest=artifact.artifact_digest,
        )


def test_panel_scores_all_eight_atoms_in_one_batch_and_cold_replays() -> None:
    first, _second, proposer = _proposer_artifact()
    panel = first[2]
    request = AtomPanelScoreRequest.build_from_proposer(
        panel,
        2,
        proposer,
        expected_proposer_artifact_digest=proposer.artifact_digest,
    )
    with pytest.raises(PositiveAtomSlateError, match="exact exposed support"):
        AtomPanelScoreRequest.build_from_proposer(
            _png(777),
            2,
            proposer,
            expected_proposer_artifact_digest=proposer.artifact_digest,
        )
    assert AtomPanelScoreRequest.from_data(request.to_data()) == request
    prompt = atom_panel_score_prompt(request)
    schema = atom_panel_score_output_schema(request)
    assert "Independently judge" in prompt
    assert "Never let one atom's score affect another" in prompt
    assert len(schema["properties"]) == 2 * ATOM_COUNT
    assert list(schema["properties"])[0:2] == ["atom_00_lower", "atom_00_upper"]

    calls: list[object] = []
    payload = _score_payload()
    artifact = observe_affirmative_atom_panel(
        panel,
        request=request,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=_transport(payload, ("panel.png",), (panel,), calls),
    )
    assert len(calls) == 1
    assert artifact.status is AtomPanelStatus.SUCCESS
    assert artifact.row.dispositions == (artifact.row.intervals[0].disposition,) * 8
    assert AtomPanelScoreArtifact.from_data(artifact.to_data()) == artifact
    assert verify_atom_panel_score_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
    ) == artifact
    assert len(calls) == 1

    with pytest.raises(PositiveAtomSlateError, match="pixels"):
        verify_atom_panel_score_artifact(
            artifact,
            _png(398),
            expected_artifact_digest=artifact.artifact_digest,
            source_proposer_artifact=proposer,
            expected_source_proposer_artifact_digest=proposer.artifact_digest,
        )


def test_parser_and_transport_failures_are_error_rows_not_absence() -> None:
    first, _second, proposer = _proposer_artifact()
    panel = first[0]
    request = AtomPanelScoreRequest.build_from_proposer(
        panel,
        0,
        proposer,
        expected_proposer_artifact_digest=proposer.artifact_digest,
    )
    invalid = _score_payload()
    invalid["atom_00_lower"] = 4
    invalid["atom_00_upper"] = 1
    parser = observe_affirmative_atom_panel(
        panel,
        request=request,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=_transport(invalid, ("panel.png",), (panel,), []),
    )
    assert parser.status is AtomPanelStatus.PARSER_ERROR
    assert parser.row.error_code == "atom_payload_rejected"
    assert all(item.value == "error" for item in parser.row.dispositions)

    def broken(*args, **kwargs):
        raise RuntimeError("fixture transport failed")

    failed = observe_affirmative_atom_panel(
        panel,
        request=request,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=broken,
    )
    assert failed.status is AtomPanelStatus.TRANSPORT_ERROR
    assert failed.row.error_code == "atom_transport_failed"
    assert all(item.value == "error" for item in failed.row.dispositions)


def _heterogeneous_rows(
    slate: AffirmativeAtomSlate,
) -> tuple[AtomPanelScoreRow, ...]:
    rows = []
    for ordinal in range(12):
        intervals = []
        for atom_index in range(8):
            if atom_index >= 2:
                bounds = (1, 3)
            elif ordinal < 6:
                bounds = (3, 4)
            elif atom_index == 0:
                bounds = (3, 4) if ordinal < 9 else (0, 1)
            else:
                bounds = (0, 1) if ordinal < 9 else (3, 4)
            intervals.append(AtomScoreInterval(*bounds))
        rows.append(AtomPanelScoreRow(ordinal, slate.slate_digest, tuple(intervals)))
    return tuple(rows)


def test_python_pair_search_handles_heterogeneous_negatives_without_a_foil() -> None:
    slate = AffirmativeAtomSlate(tuple(_slate_payload()[name] for name in ATOM_IDS))
    rows = _heterogeneous_rows(slate)
    inventory = AtomSupportInventory.create(slate, rows)

    assert len(enumerate_affirmative_atom_formulas()) == FORMULA_COUNT == 36
    assert tuple(profile.formula for profile in inventory.profiles) == (
        enumerate_affirmative_atom_formulas()
    )
    assert inventory.admitted_formulas == (AtomFormula((0, 1)),)
    pair = next(profile for profile in inventory.profiles if profile.formula == AtomFormula((0, 1)))
    atom_0 = next(profile for profile in inventory.profiles if profile.formula == AtomFormula((0,)))
    atom_1 = next(profile for profile in inventory.profiles if profile.formula == AtomFormula((1,)))
    assert pair.native_present == 6 and pair.contrast_absent == 6 and pair.admitted
    assert atom_0.contrast_present == 3 and not atom_0.admitted
    assert atom_1.contrast_present == 3 and not atom_1.admitted
    assert inventory.gap is None and inventory.query_release_allowed is True
    assert AtomSupportInventory.from_data(inventory.to_data()) == inventory


def test_preregistered_gate_allows_one_indeterminate_but_no_error_or_contradiction() -> None:
    slate = AffirmativeAtomSlate(tuple(_slate_payload()[name] for name in ATOM_IDS))
    rows = list(_heterogeneous_rows(slate))
    uncertain = list(rows[0].intervals)
    uncertain[0] = AtomScoreInterval(1, 3)
    uncertain[1] = AtomScoreInterval(1, 3)
    rows[0] = AtomPanelScoreRow(0, slate.slate_digest, tuple(uncertain))
    allowed = AtomSupportInventory.create(slate, rows)
    assert allowed.admitted_formulas == (AtomFormula((0, 1)),)

    rows[0] = AtomPanelScoreRow.error(0, slate.slate_digest, "fixture_error")
    error_gap = AtomSupportInventory.create(slate, rows)
    assert error_gap.admitted_formulas == ()
    assert error_gap.gap is not None
    assert error_gap.gap.error_row_ordinals == (0,)

    rows = list(_heterogeneous_rows(slate))
    contradiction = list(rows[0].intervals)
    contradiction[0] = AtomScoreInterval(0, 1)
    rows[0] = AtomPanelScoreRow(0, slate.slate_digest, tuple(contradiction))
    contradiction_gap = AtomSupportInventory.create(slate, rows)
    assert contradiction_gap.admitted_formulas == ()
    assert contradiction_gap.gap is not None


def test_no_survivor_is_a_typed_gap_and_tampering_fails_closed() -> None:
    slate = AffirmativeAtomSlate(tuple(_slate_payload()[name] for name in ATOM_IDS))
    rows = tuple(
        AtomPanelScoreRow(
            ordinal,
            slate.slate_digest,
            (AtomScoreInterval(3, 4),) * ATOM_COUNT,
        )
        for ordinal in range(12)
    )
    inventory = AtomSupportInventory.create(slate, rows)
    assert inventory.admitted_formulas == ()
    assert inventory.gap is not None
    assert inventory.gap.to_data()["code"] == "no_admissible_affirmative_singleton_or_pair"
    assert inventory.query_release_allowed is False

    with pytest.raises(PositiveAtomSlateError, match="before formula enumeration"):
        AtomSupportInventory.create(slate, rows[:-1])

    tampered = deepcopy(inventory.to_data())
    tampered["threshold_selected_after_observations"] = True
    with pytest.raises(PositiveAtomSlateError):
        AtomSupportInventory.from_data(tampered)


def test_exact_journal_terminals_bind_proposer_and_panel_cold_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _groups()
    request = AtomSlateProposerRequest.build(first, second, runtime=_runtime(first[0]))
    proposer_payload = _slate_payload()
    calls = 0

    def proposer_physical(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        return CodexStructuredResult(
            deepcopy(proposer_payload),
            _receipt(prompt, paths, names, schema, proposer_payload),
        )

    monkeypatch.setattr(atom_module, "run_codex_named_images_structured", proposer_physical)
    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=NO_TOOLS_KWARGS["model_catalog_snapshot"],
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_KWARGS["no_tools_attestation"],
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "proposer-journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_atom_slate_fixture_0001",
        turn_kind="positive_atom_slate_proposer",
        expected_prompt=atom_slate_proposer_prompt(request),
        expected_images=tuple(zip(PROPOSER_IMAGE_NAMES, (*first, *second), strict=True)),
        expected_output_schema=atom_slate_proposer_output_schema(request),
        runtime=runtime,
        underlying_transport=proposer_physical,
    )
    proposer = propose_affirmative_atom_slate(
        first,
        second,
        request=request,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=proposer_journal,
    )
    proposer_terminal = proposer_journal.verify()
    assert proposer.benchmark_sealable is True and calls == 1
    with pytest.raises(PositiveAtomSlateError, match="terminal"):
        verify_atom_slate_proposer_artifact(
            proposer,
            first,
            second,
            expected_artifact_digest=proposer.artifact_digest,
        )
    assert verify_atom_slate_proposer_artifact(
        proposer,
        first,
        second,
        expected_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
    ) == proposer

    panel = first[0]
    panel_request = AtomPanelScoreRequest.build_from_proposer(
        panel,
        0,
        proposer,
        expected_proposer_artifact_digest=proposer.artifact_digest,
    )
    panel_payload = _score_payload()

    def panel_physical(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        return CodexStructuredResult(
            deepcopy(panel_payload),
            _receipt(prompt, paths, names, schema, panel_payload),
        )

    monkeypatch.setattr(atom_module, "run_codex_named_images_structured", panel_physical)
    panel_journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "panel-journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "d" * 64,
        task_id="hd_atom_slate_fixture_0001",
        turn_kind="positive_atom_panel_00",
        expected_prompt=atom_panel_score_prompt(panel_request),
        expected_images=(("panel.png", panel),),
        expected_output_schema=atom_panel_score_output_schema(panel_request),
        runtime=runtime,
        underlying_transport=panel_physical,
    )
    observed = observe_affirmative_atom_panel(
        panel,
        request=panel_request,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=panel_journal,
    )
    panel_terminal = panel_journal.verify()
    assert observed.benchmark_sealable is True and calls == 2
    with pytest.raises(PositiveAtomSlateError, match="terminal"):
        verify_atom_panel_score_artifact(
            observed,
            panel,
            expected_artifact_digest=observed.artifact_digest,
            source_proposer_artifact=proposer,
            expected_source_proposer_artifact_digest=proposer.artifact_digest,
            proposer_journal_terminal=proposer_terminal,
        )
    assert verify_atom_panel_score_artifact(
        observed,
        panel,
        expected_artifact_digest=observed.artifact_digest,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
        panel_journal_terminal=panel_terminal,
    ) == observed
    assert calls == 2

    wrong = ObjectBongardTurnJournalSummary(
        panel_terminal.manifest_digest,
        panel_terminal.turn_key,
        panel_terminal.terminal_status,
        panel_terminal.claim_digest,
        panel_terminal.result_digest,
        panel_terminal.outcome_digest,
        "sha256:" + "b" * 64,
    )
    with pytest.raises(PositiveAtomSlateError, match="terminal"):
        verify_atom_panel_score_artifact(
            observed,
            panel,
            expected_artifact_digest=observed.artifact_digest,
            source_proposer_artifact=proposer,
            expected_source_proposer_artifact_digest=proposer.artifact_digest,
            proposer_journal_terminal=proposer_terminal,
            panel_journal_terminal=wrong,
        )


def test_module_has_no_lean_dependency() -> None:
    source = Path(atom_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "lean" or name.startswith("lean.") for name in imports)
