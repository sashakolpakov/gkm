from __future__ import annotations

from dataclasses import dataclass
import threading

import bongard.prototype_pair_campaign as campaign


@dataclass(frozen=True)
class _Artifact:
    index: int


def test_completed_observer_is_persisted_before_slow_schedule_head(
    monkeypatch,
) -> None:
    release_slow = threading.Event()
    slow_started = threading.Event()
    persistence_order: list[int] = []

    def turn(index: int) -> _Artifact:
        if index == 0:
            slow_started.set()
            assert release_slow.wait(timeout=2)
        else:
            assert slow_started.wait(timeout=2)
        return _Artifact(index)

    def persist(
        state: campaign._CampaignState,
        _store: object,
        _clock: object,
        *,
        artifact: _Artifact,
        **_kwargs: object,
    ) -> None:
        persistence_order.append(artifact.index)
        state.stored_objects.append(f"object-{artifact.index}")  # type: ignore[arg-type]
        state.call_terminals.append({"index": artifact.index})
        if artifact.index == 1:
            release_slow.set()

    monkeypatch.setattr(campaign, "PrototypeSceneObserverArtifact", _Artifact)
    monkeypatch.setattr(campaign, "_persist_observer_result", persist)
    state = campaign._CampaignState()
    tickets = (
        campaign._CallTicket(claim="claim-0", fresh=True, terminal_outcome=None),
        campaign._CallTicket(claim="claim-1", fresh=True, terminal_outcome=None),
    )

    results = campaign._run_fresh_observer_batch(
        state,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        fresh_indices=(0, 1),
        tickets=tickets,
        subject_ids=("panel-0", "panel-1"),
        turn=turn,  # type: ignore[arg-type]
        phase="calibration",
        kind="observer_artifact",
        precommit=object(),  # type: ignore[arg-type]
        max_workers=2,
    )

    assert persistence_order == [1, 0]
    assert tuple(results) == (0, 1)
    assert state.stored_objects == ["object-0", "object-1"]
    assert state.call_terminals == [{"index": 0}, {"index": 1}]
