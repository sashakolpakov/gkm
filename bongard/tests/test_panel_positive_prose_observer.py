"""Focused tests for the one-sided positive-prose panel observer."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from bongard import panel_positive_prose_observer as observer_module
from bongard import panel_support_positive_proposer as proposer_module
from bongard.canonical import canonical_digest
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard.panel_positive_prose_observer import (
    POSITIVE_ORIENTATION,
    PositiveProseCue,
    PositiveProseDisposition,
    PositiveProseObservation,
    PositiveProseObserverError,
    PositiveProsePanelArtifact,
    PositiveProsePanelContext,
    PositiveProsePanelRequest,
    PositiveProseScoreInterval,
    PositiveProseTransportProvenance,
    classify_positive_prose_interval,
    observe_positive_prose_panel,
    positive_prose_panel_output_schema,
    positive_prose_panel_prompt,
    verify_positive_prose_panel_artifact,
)
from bongard.panel_support_positive_proposer import (
    SUPPORT_POSITIVE_PRESENTATION_NAMES,
    SupportPositiveProposerArtifact,
    SupportPositiveProposerRequest,
    propose_support_positive_rubric,
    support_positive_proposer_output_schema,
    support_positive_proposer_prompt,
    verify_support_positive_proposer_artifact,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.prototype_scene_observer import (
    prototype_scene_transport_source_digest,
)
from bongard.transport import CodexStructuredResult


PANEL_ID = "hd/hd_positive_prose_fixture_0001/1/6.png"


def _context(panel: bytes) -> PositiveProsePanelContext:
    return PositiveProsePanelContext.build(
        panel,
        panel_id=PANEL_ID,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
    )


def _support_groups() -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    return (
        tuple(_png(170 + index) for index in range(6)),
        tuple(_png(180 + index) for index in range(6)),
    )


def _support_payload(*, admitted: bool = True) -> dict[str, object]:
    result: dict[str, object] = {
        "cue_text": "convex carrier and four straight structural runs",
        "component_1": "convex carrier",
        "component_2": "four straight structural runs",
    }
    for index, name in enumerate(SUPPORT_POSITIVE_PRESENTATION_NAMES):
        result[name.removesuffix(".png") + "_estimate"] = (
            "supports" if index < 6 else "does_not_support"
        )
    if not admitted:
        result["group_a_05_estimate"] = "unclear"
    return result


def _source_proposer(
    runtime, *, admitted: bool = True
) -> SupportPositiveProposerArtifact:
    first, second = _support_groups()
    request = SupportPositiveProposerRequest.build(first, second, runtime=runtime)
    payload = _support_payload(admitted=admitted)

    def transport(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == SUPPORT_POSITIVE_PRESENTATION_NAMES
        assert tuple(Path(path).read_bytes() for path in paths) == (*first, *second)
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    return propose_support_positive_rubric(
        first,
        second,
        request=request,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )


def _request_and_source(
    panel: bytes, *, admitted: bool = True
) -> tuple[PositiveProsePanelRequest, SupportPositiveProposerArtifact]:
    context = _context(panel)
    source = _source_proposer(context.runtime, admitted=admitted)
    request = PositiveProsePanelRequest.build_from_proposer(
        context,
        source,
        expected_artifact_digest=source.artifact_digest,
    )
    return request, source


def _cue() -> PositiveProseCue:
    return _request_and_source(_png(96))[0].cue


def _request(panel: bytes) -> PositiveProsePanelRequest:
    return _request_and_source(panel)[0]


def _observe(
    lower: int = 3,
    upper: int = 4,
    *,
    fail: bool = False,
    malformed: bool = False,
):
    panel = _png(97)
    request, source = _request_and_source(panel)
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert names == ("panel.png",)
        assert len(paths) == 1
        assert Path(paths[0]).read_bytes() == panel
        assert request.cue.text in prompt
        assert "second description" in prompt
        if fail:
            raise RuntimeError("synthetic transport failure")
        payload = (
            {"lower": 1, "upper": 7}
            if malformed
            else {"lower": lower, "upper": upper}
        )
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    artifact = observe_positive_prose_panel(
        panel,
        request=request,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    assert calls == 1
    return artifact, panel, request, source


def test_frozen_positive_conjunction_has_no_foil_or_negative_field() -> None:
    cue = _cue()
    assert " and " in cue.text
    assert cue.to_data()["conjunction_allowed"] is True
    assert cue.to_data()["positive_orientation"] == POSITIVE_ORIENTATION
    assert cue.to_data()["foil_field_present"] is False
    assert cue.to_data()["complement_field_present"] is False
    assert cue.to_data()["negative_class_description_field_present"] is False
    assert PositiveProseCue.from_data(cue.to_data()) == cue

    tampered = {**cue.to_data(), "foil": "anything"}
    with pytest.raises(PositiveProseObserverError, match="fields differ"):
        PositiveProseCue.from_data(tampered)


def test_request_requires_admitted_proposer_and_rejects_gap_or_bare_cue() -> None:
    panel = _png(95)
    context = _context(panel)
    gap = _source_proposer(context.runtime, admitted=False)
    assert gap.rubric is None
    assert gap.proposal_gap is not None
    with pytest.raises(PositiveProseObserverError, match="proposal gap"):
        PositiveProsePanelRequest.build_from_proposer(
            context,
            gap,
            expected_artifact_digest=gap.artifact_digest,
        )
    assert not hasattr(PositiveProsePanelRequest, "build")
    assert not hasattr(PositiveProseCue, "freeze")


def test_request_binds_exact_proposer_commitment_and_runtime() -> None:
    panel = _png(94)
    context = _context(panel)
    source = _source_proposer(context.runtime)
    with pytest.raises(PositiveProseObserverError, match="commitment"):
        PositiveProsePanelRequest.build_from_proposer(
            context,
            source,
            expected_artifact_digest="0" * 64,
        )


def test_live_observe_replays_source_and_rejects_forged_sealability_before_call() -> None:
    panel = _png(92)
    request, source = _request_and_source(panel)
    assert source.benchmark_sealable is False
    raw = deepcopy(request.to_data())
    cue = raw["cue"]
    assert isinstance(cue, dict)
    cue["source_proposer_benchmark_sealable"] = True
    cue["cue_digest"] = canonical_digest(
        {key: value for key, value in cue.items() if key != "cue_digest"}
    )
    raw["request_digest"] = canonical_digest(
        {key: value for key, value in raw.items() if key != "request_digest"}
    )
    forged = PositiveProsePanelRequest.from_data(raw)
    calls = 0

    def forbidden_transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("forged request reached transport")

    with pytest.raises(PositiveProseObserverError, match="proposer lineage"):
        observe_positive_prose_panel(
            panel,
            request=forged,
            source_proposer_artifact=source,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=forbidden_transport,
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    (
        (3, 3, PositiveProseDisposition.PRESENT),
        (3, 4, PositiveProseDisposition.PRESENT),
        (4, 4, PositiveProseDisposition.PRESENT),
        (0, 0, PositiveProseDisposition.CERTIFIED_ABSENT),
        (0, 1, PositiveProseDisposition.CERTIFIED_ABSENT),
        (1, 1, PositiveProseDisposition.CERTIFIED_ABSENT),
        (2, 2, PositiveProseDisposition.INDETERMINATE),
        (1, 3, PositiveProseDisposition.INDETERMINATE),
        (0, 4, PositiveProseDisposition.INDETERMINATE),
        (2, 4, PositiveProseDisposition.INDETERMINATE),
    ),
)
def test_fixed_python_interval_projection(
    lower: int, upper: int, expected: PositiveProseDisposition
) -> None:
    interval = PositiveProseScoreInterval(lower, upper)
    assert classify_positive_prose_interval(interval) is expected
    observation = PositiveProseObservation.from_interval(
        _cue().cue_digest, interval
    )
    assert observation.disposition is expected
    assert PositiveProseObservation.from_data(observation.to_data()) == observation


def test_one_panel_receipt_round_trip_and_model_free_cold_replay() -> None:
    artifact, panel, request, source = _observe()
    assert artifact.status.value == "success"
    assert artifact.physical_call_count == 1
    assert tuple(item.name for item in artifact.presentation) == ("panel.png",)
    assert artifact.observation.disposition is PositiveProseDisposition.PRESENT
    assert artifact.observation.interval == PositiveProseScoreInterval(3, 4)
    assert artifact.transport_provenance.kind == "injected_unverified"
    assert artifact.benchmark_sealable is False
    assert PositiveProsePanelArtifact.from_data(artifact.to_data()) == artifact
    assert verify_positive_prose_panel_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        expected_request_digest=request.request_digest,
    ) == artifact


def test_transport_and_parser_failures_are_error_not_absence() -> None:
    transport_error, _, _, _ = _observe(fail=True)
    parser_error, _, _, _ = _observe(malformed=True)
    for artifact in (transport_error, parser_error):
        assert artifact.observation.disposition is PositiveProseDisposition.ERROR
        assert artifact.observation.interval is None
        assert artifact.observation.error_code == artifact.failure_code
        assert artifact.benchmark_sealable is False
    assert transport_error.receipt is None
    assert parser_error.receipt is not None


def test_receipt_blocks_resealed_payload_and_disposition_tamper() -> None:
    artifact, _, _, _ = _observe(0, 1)
    data = deepcopy(artifact.to_data())
    data["model_payload"] = {"lower": 4, "upper": 4}
    data["observation"] = PositiveProseObservation.from_interval(
        artifact.request.cue.cue_digest, PositiveProseScoreInterval(4, 4)
    ).to_data()
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    with pytest.raises(PositiveProseObserverError, match="receipt"):
        PositiveProsePanelArtifact.from_data(data)


def test_prompt_and_schema_expose_only_one_cue_and_one_neutral_panel() -> None:
    panel = _png(98)
    request = _request(panel)
    prompt = positive_prose_panel_prompt(request)
    schema = positive_prose_panel_output_schema(request)
    assert prompt.count(request.cue.text) == 1
    assert "panel.png" in prompt
    lowered = prompt.lower()
    assert "foil" not in lowered
    assert "negative class" not in lowered
    assert POSITIVE_ORIENTATION not in prompt
    assert set(schema["properties"]) == {"lower", "upper"}
    assert schema["properties"]["lower"]["enum"] == [0, 1, 2, 3, 4]


def test_transport_provenance_distinguishes_journal_from_injection() -> None:
    direct = PositiveProseTransportProvenance.create("production_direct")
    address = "sha256:" + "a" * 64
    summary = ObjectBongardTurnJournalSummary(
        address, address, "success", address, address, address, address
    )
    journal = PositiveProseTransportProvenance.create(
        "production_exactly_once_journal", journal_summary=summary
    )
    injected = PositiveProseTransportProvenance.create("injected_unverified")
    assert direct.production_transport_chain_verified is True
    assert direct.benchmark_sealable is False
    assert journal.production_transport_chain_verified is True
    assert journal.benchmark_sealable is True
    assert journal.journal_terminal_record_digest == address
    assert injected.production_transport_chain_verified is False
    assert injected.benchmark_sealable is False


def test_exact_query_journal_replays_once_and_unsealed_source_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _png(99)
    request, source = _request_and_source(panel)
    calls = 0

    def physical_transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        payload = {"lower": 4, "upper": 4}
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        observer_module, "run_codex_named_images_structured", physical_transport
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
        tmp_path / "journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_positive_prose_fixture_0001",
        turn_kind="positive_prose_panel_00",
        expected_prompt=positive_prose_panel_prompt(request),
        expected_images=(("panel.png", panel),),
        expected_output_schema=positive_prose_panel_output_schema(request),
        runtime=runtime,
        underlying_transport=physical_transport,
    )
    first = observe_positive_prose_panel(
        panel,
        request=request,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=journal,
    )
    second = observe_positive_prose_panel(
        panel,
        request=request,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
        transport=journal,
    )
    assert calls == 1
    assert first == second
    assert first.benchmark_sealable is False
    assert request.cue.source_proposer_benchmark_sealable is False
    assert first.transport_provenance.kind == "production_exactly_once_journal"
    assert journal.verify().terminal_status == "success"
    with pytest.raises(PositiveProseObserverError, match="journal terminal"):
        verify_positive_prose_panel_artifact(
            first,
            panel,
            expected_artifact_digest=first.artifact_digest,
            source_proposer_artifact=source,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            expected_request_digest=request.request_digest,
        )
    terminal = journal.verify()
    wrong_terminal = ObjectBongardTurnJournalSummary(
        terminal.manifest_digest,
        terminal.turn_key,
        terminal.terminal_status,
        terminal.claim_digest,
        terminal.result_digest,
        terminal.outcome_digest,
        "sha256:" + "b" * 64,
    )
    with pytest.raises(PositiveProseObserverError, match="journal terminal"):
        verify_positive_prose_panel_artifact(
            first,
            panel,
            expected_artifact_digest=first.artifact_digest,
            source_proposer_artifact=source,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            query_journal_terminal=wrong_terminal,
            expected_request_digest=request.request_digest,
        )
    assert verify_positive_prose_panel_artifact(
        first,
        panel,
        expected_artifact_digest=first.artifact_digest,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        query_journal_terminal=terminal,
        expected_request_digest=request.request_digest,
    ) == first


def test_exact_query_journal_binds_durable_failure_and_replays_zero_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _png(91)
    request, source = _request_and_source(panel)
    calls = 0

    def failing_transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic durable query failure")

    monkeypatch.setattr(
        observer_module, "run_codex_named_images_structured", failing_transport
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
        tmp_path / "failure-journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_positive_prose_fixture_0001",
        turn_kind="positive_prose_failure_00",
        expected_prompt=positive_prose_panel_prompt(request),
        expected_images=(("panel.png", panel),),
        expected_output_schema=positive_prose_panel_output_schema(request),
        runtime=runtime,
        underlying_transport=failing_transport,
    )

    def observe():
        return observe_positive_prose_panel(
            panel,
            request=request,
            source_proposer_artifact=source,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=journal,
        )

    artifact = observe()
    assert observe() == artifact
    terminal = journal.verify()
    assert calls == 1
    assert terminal.terminal_status == "failure"
    assert artifact.status.value == "transport_error"
    assert artifact.observation.disposition is PositiveProseDisposition.ERROR
    assert artifact.transport_provenance.kind == "production_exactly_once_journal"
    assert artifact.transport_provenance.journal_terminal_status == "failure"
    assert artifact.transport_provenance.benchmark_sealable is False
    assert artifact.benchmark_sealable is False
    assert verify_positive_prose_panel_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        query_journal_terminal=terminal,
        expected_request_digest=request.request_digest,
    ) == artifact


def test_exact_source_and_query_journals_form_one_sealable_cold_replay_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _png(93)
    context = _context(panel)
    first_group, second_group = _support_groups()
    support_request = SupportPositiveProposerRequest.build(
        first_group, second_group, runtime=context.runtime
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
    support_calls = 0
    support_payload = _support_payload()

    def support_transport(prompt, paths, names, schema, **kwargs):
        nonlocal support_calls
        support_calls += 1
        return CodexStructuredResult(
            deepcopy(support_payload),
            _receipt(prompt, paths, names, schema, support_payload),
        )

    monkeypatch.setattr(
        proposer_module, "run_codex_named_images_structured", support_transport
    )
    support_journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "support-journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_positive_prose_fixture_0001",
        turn_kind="positive_prose_support",
        expected_prompt=support_positive_proposer_prompt(support_request),
        expected_images=tuple(
            zip(
                SUPPORT_POSITIVE_PRESENTATION_NAMES,
                (*first_group, *second_group),
                strict=True,
            )
        ),
        expected_output_schema=support_positive_proposer_output_schema(
            support_request
        ),
        runtime=runtime,
        underlying_transport=support_transport,
    )

    def propose():
        return propose_support_positive_rubric(
            first_group,
            second_group,
            request=support_request,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=support_journal,
        )

    source = propose()
    assert propose() == source
    assert support_calls == 1
    assert source.benchmark_sealable is True
    request = PositiveProsePanelRequest.build_from_proposer(
        context,
        source,
        expected_artifact_digest=source.artifact_digest,
    )
    query_calls = 0

    def query_transport(prompt, paths, names, schema, **kwargs):
        nonlocal query_calls
        query_calls += 1
        payload = {"lower": 3, "upper": 4}
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        observer_module, "run_codex_named_images_structured", query_transport
    )
    query_journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "query-journal",
        authorization_digest="sha256:" + "a" * 64,
        execution_precommit_digest="sha256:" + "c" * 64,
        task_id="hd_positive_prose_fixture_0001",
        turn_kind="positive_prose_query_00",
        expected_prompt=positive_prose_panel_prompt(request),
        expected_images=(("panel.png", panel),),
        expected_output_schema=positive_prose_panel_output_schema(request),
        runtime=runtime,
        underlying_transport=query_transport,
    )

    def observe():
        return observe_positive_prose_panel(
            panel,
            request=request,
            source_proposer_artifact=source,
            expected_source_proposer_artifact_digest=source.artifact_digest,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=query_journal,
        )

    result = observe()
    assert observe() == result
    assert query_calls == 1
    assert result.benchmark_sealable is True
    assert verify_support_positive_proposer_artifact(
        source,
        first_group,
        second_group,
        expected_artifact_digest=source.artifact_digest,
    ) == source
    assert verify_positive_prose_panel_artifact(
        result,
        panel,
        expected_artifact_digest=result.artifact_digest,
        source_proposer_artifact=source,
        expected_source_proposer_artifact_digest=source.artifact_digest,
        query_journal_terminal=query_journal.verify(),
        expected_request_digest=request.request_digest,
    ) == result


def test_path_imports_no_lean_and_contains_no_foil_spec_dependency() -> None:
    path = Path(__file__).parents[1] / "panel_positive_prose_observer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any("lean" in item for item in lowered)
    assert "bongard.object_bongard_rubric_language" not in lowered
    assert "bongard.object_bongard_panel_rubric_observer" not in lowered
