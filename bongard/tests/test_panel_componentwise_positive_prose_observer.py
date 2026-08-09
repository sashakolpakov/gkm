"""Focused tests for the raw-panel component-wise positive prose observer."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from bongard import panel_positive_prose_observer as v1_observer_module
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard.panel_componentwise_positive_prose_observer import (
    ComponentScoreInterval,
    ComponentwiseMatchStatus,
    ComponentwisePositiveCue,
    ComponentwisePositiveProseError,
    ComponentwisePositiveProseObservation,
    ComponentwisePositiveProsePanelArtifact,
    ComponentwisePositiveProsePanelRequest,
    classify_component_interval,
    combine_component_dispositions,
    componentwise_positive_prose_output_schema,
    componentwise_positive_prose_prompt,
    observe_componentwise_positive_prose_panel,
    verify_componentwise_positive_prose_panel_artifact,
)
from bongard.panel_support_positive_proposer import (
    SupportPositiveProposerArtifact,
    SupportPositiveTransportProvenance,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_panel_positive_prose_observer import (
    _context,
    _source_proposer,
    _support_groups,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _terminal(seed: str = "a") -> ObjectBongardTurnJournalSummary:
    address = "sha256:" + seed * 64
    return ObjectBongardTurnJournalSummary(
        address, address, "success", address, address, address, address
    )


def _fixture(panel: bytes):
    context = _context(panel)
    first, second = _support_groups()
    terminal = _terminal()
    base = _source_proposer(context.runtime)
    source = replace(
        base,
        transport_provenance=SupportPositiveTransportProvenance.create(
            "production_exactly_once_journal", journal_summary=terminal
        ),
    )
    request = ComponentwisePositiveProsePanelRequest.build_from_proposer(
        context,
        source,
        first,
        second,
        expected_artifact_digest=source.artifact_digest,
        proposer_journal_terminal=terminal,
    )
    return request, source, first, second, terminal


def _observe(
    payload: dict[str, object] | None = None,
    *,
    fail: bool = False,
):
    panel = _png(211)
    request, source, first, second, terminal = _fixture(panel)
    body = payload or {
        "component_1_lower": 3,
        "component_1_upper": 4,
        "component_2_lower": 3,
        "component_2_upper": 3,
    }
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert tuple(names) == ("panel.png",)
        assert Path(paths[0]).read_bytes() == panel
        if fail:
            raise RuntimeError("synthetic component transport failure")
        return CodexStructuredResult(
            deepcopy(body), _receipt(prompt, paths, names, schema, body)
        )

    artifact = observe_componentwise_positive_prose_panel(
        panel,
        request=request,
        source_proposer_artifact=source,
        group_a_pngs=first,
        group_b_pngs=second,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        proposer_journal_terminal=terminal,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == 1
    return artifact, panel, request, source, first, second, terminal, lambda: calls


def test_request_requires_exact_admitted_proposer_pixels_and_external_terminal() -> None:
    panel = _png(212)
    context = _context(panel)
    first, second = _support_groups()
    base = _source_proposer(context.runtime)
    with pytest.raises(ComponentwisePositiveProseError, match="external terminal"):
        ComponentwisePositiveProsePanelRequest.build_from_proposer(
            context,
            base,
            first,
            second,
            expected_artifact_digest=base.artifact_digest,
            proposer_journal_terminal=_terminal(),
        )

    request, source, first, second, terminal = _fixture(panel)
    changed = list(first)
    changed[0] = _png(250)
    with pytest.raises(ComponentwisePositiveProseError, match="pixels"):
        ComponentwisePositiveProsePanelRequest.build_from_proposer(
            context,
            source,
            tuple(changed),
            second,
            expected_artifact_digest=source.artifact_digest,
            proposer_journal_terminal=terminal,
        )
    assert ComponentwisePositiveProsePanelRequest.from_data(request.to_data()) == request


def test_cue_preserves_two_exact_components_and_terminal_custody() -> None:
    request, source, _, _, terminal = _fixture(_png(213))
    cue = request.cue
    assert cue.component_1 == source.rubric.component_1
    assert cue.component_2 == source.rubric.component_2
    assert cue.component_digest(1) != cue.component_digest(2)
    assert cue.proposer_terminal.matches(terminal)
    assert ComponentwisePositiveCue.from_data(cue.to_data()) == cue
    assert cue.to_data()["foil_or_complement_present"] is False


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    (
        (0, 0, Disposition.CERTIFIED_ABSENT),
        (0, 1, Disposition.CERTIFIED_ABSENT),
        (1, 1, Disposition.CERTIFIED_ABSENT),
        (1, 2, Disposition.INDETERMINATE),
        (2, 2, Disposition.INDETERMINATE),
        (2, 4, Disposition.INDETERMINATE),
        (3, 3, Disposition.PRESENT),
        (3, 4, Disposition.PRESENT),
        (4, 4, Disposition.PRESENT),
    ),
)
def test_fixed_component_projection_and_score_one_contradiction(
    lower: int, upper: int, expected: Disposition
) -> None:
    assert classify_component_interval(ComponentScoreInterval(lower, upper)) is expected


def test_python_conjunction_projection_has_error_then_absence_precedence() -> None:
    states = tuple(Disposition)
    for first in states:
        for second in states:
            disposition, status = combine_component_dispositions(first, second)
            if Disposition.ERROR in (first, second):
                assert (disposition, status) == (
                    Disposition.ERROR,
                    ComponentwiseMatchStatus.ERROR,
                )
            elif Disposition.CERTIFIED_ABSENT in (first, second):
                assert (disposition, status) == (
                    Disposition.CERTIFIED_ABSENT,
                    ComponentwiseMatchStatus.NONMATCH,
                )
            elif first is second is Disposition.PRESENT:
                assert (disposition, status) == (
                    Disposition.PRESENT,
                    ComponentwiseMatchStatus.MATCH,
                )
            else:
                assert (disposition, status) == (
                    Disposition.INDETERMINATE,
                    ComponentwiseMatchStatus.INDETERMINATE,
                )


def test_successful_artifact_round_trip_and_zero_call_replay() -> None:
    result = _observe()
    artifact, panel, request, source, first, second, terminal, calls = result
    assert artifact.observation.component_1_disposition is Disposition.PRESENT
    assert artifact.observation.component_2_disposition is Disposition.PRESENT
    assert artifact.observation.match_status is ComponentwiseMatchStatus.MATCH
    assert artifact.benchmark_sealable is False
    assert ComponentwisePositiveProsePanelArtifact.from_data(artifact.to_data()) == artifact
    assert verify_componentwise_positive_prose_panel_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=source,
        group_a_pngs=first,
        group_b_pngs=second,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        proposer_journal_terminal=terminal,
        expected_request_digest=request.request_digest,
    ) == artifact
    assert calls() == 1


def test_one_score_one_component_makes_conjunction_certified_nonmatch() -> None:
    artifact, *_ = _observe(
        {
            "component_1_lower": 4,
            "component_1_upper": 4,
            "component_2_lower": 1,
            "component_2_upper": 1,
        }
    )
    assert artifact.observation.component_1_disposition is Disposition.PRESENT
    assert artifact.observation.component_2_disposition is Disposition.CERTIFIED_ABSENT
    assert artifact.observation.conjunction_disposition is Disposition.CERTIFIED_ABSENT
    assert artifact.observation.match_status is ComponentwiseMatchStatus.NONMATCH


def test_transport_and_parser_failures_are_error_not_absence() -> None:
    transport_error, *_ = _observe(fail=True)
    parser_error, *_ = _observe(
        {
            "component_1_lower": 3,
            "component_1_upper": 4,
            "component_2_lower": 4,
            "component_2_upper": 2,
        }
    )
    for artifact in (transport_error, parser_error):
        assert artifact.observation.conjunction_disposition is Disposition.ERROR
        assert artifact.observation.match_status is ComponentwiseMatchStatus.ERROR
        assert artifact.benchmark_sealable is False


def test_receipt_blocks_resealed_component_payload() -> None:
    artifact, *_ = _observe()
    raw = deepcopy(artifact.to_data())
    raw["model_payload"] = {
        "component_1_lower": 0,
        "component_1_upper": 1,
        "component_2_lower": 0,
        "component_2_upper": 1,
    }
    cue = artifact.request.cue
    raw["observation"] = ComponentwisePositiveProseObservation.from_intervals(
        cue, ComponentScoreInterval(0, 1), ComponentScoreInterval(0, 1)
    ).to_data()
    raw["artifact_digest"] = canonical_digest(
        {key: value for key, value in raw.items() if key != "artifact_digest"}
    )
    with pytest.raises(ComponentwisePositiveProseError, match="receipt"):
        ComponentwisePositiveProsePanelArtifact.from_data(raw)


def test_prompt_schema_and_artifact_state_raw_panel_limitations() -> None:
    request, *_ = _fixture(_png(214))
    prompt = componentwise_positive_prose_prompt(request)
    schema = componentwise_positive_prose_output_schema(request)
    assert f"COMPONENT 1\n{request.cue.component_1}\n" in prompt
    assert f"COMPONENT 2\n{request.cue.component_2}\n" in prompt
    assert "Score 1 requires a decisive visible contradiction" in prompt
    assert set(schema["properties"]) == {
        "component_1_lower", "component_1_upper",
        "component_2_lower", "component_2_upper",
    }
    data = request.to_data()
    assert data["raw_panel_only"] is True
    assert data["candidate_independent_transformed_view_supported"] is False
    assert data["crop_or_context_adapter_present"] is False
    assert data["observer_calibrated"] is False


def test_query_journal_is_exactly_once_and_requires_external_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _png(215)
    request, source, first, second, proposer_terminal = _fixture(panel)
    payload = {
        "component_1_lower": 3,
        "component_1_upper": 4,
        "component_2_lower": 3,
        "component_2_upper": 4,
    }
    calls = 0

    def physical_transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        v1_observer_module, "run_codex_named_images_structured", physical_transport
    )
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
    journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "component-query-journal",
        authorization_digest="sha256:" + "b" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_componentwise_fixture_0001",
        turn_kind="componentwise_query_00",
        expected_prompt=componentwise_positive_prose_prompt(request),
        expected_images=(("panel.png", panel),),
        expected_output_schema=componentwise_positive_prose_output_schema(request),
        runtime=runtime,
        underlying_transport=physical_transport,
    )

    def observe():
        return observe_componentwise_positive_prose_panel(
            panel,
            request=request,
            source_proposer_artifact=source,
            group_a_pngs=first,
            group_b_pngs=second,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            proposer_journal_terminal=proposer_terminal,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=journal,
        )

    artifact = observe()
    assert observe() == artifact
    assert calls == 1
    assert artifact.benchmark_sealable is True
    with pytest.raises(ComponentwisePositiveProseError, match="query journal"):
        verify_componentwise_positive_prose_panel_artifact(
            artifact,
            panel,
            expected_artifact_digest=artifact.artifact_digest,
            source_proposer_artifact=source,
            group_a_pngs=first,
            group_b_pngs=second,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            proposer_journal_terminal=proposer_terminal,
        )
    assert verify_componentwise_positive_prose_panel_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=source,
        group_a_pngs=first,
        group_b_pngs=second,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
        query_journal_terminal=journal.verify(),
    ) == artifact
    assert calls == 1


def test_module_imports_no_lean_or_foil_observer() -> None:
    path = Path(__file__).parents[1] / "panel_componentwise_positive_prose_observer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any("lean" in item for item in lowered)
    assert "bongard.object_bongard_panel_rubric_observer" not in lowered
