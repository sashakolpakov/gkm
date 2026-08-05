"""Mock-first chronology, typing, outcome, and replay tests for HYBRID."""
from __future__ import annotations

import copy
import hashlib
import os
import sys
import threading
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_predicate_ir as G  # noqa: E402
import grounded_contrastive_oracle as C  # noqa: E402
import run_hybrid_contrastive as R  # noqa: E402
from hybrid_claim_proposer import (  # noqa: E402
    ClaimProposalBundle,
    make_offline_fixture_receipt,
)
from hybrid_program_split import sample_basic_program_splits  # noqa: E402


def _args(out_dir: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[2]
    return SimpleNamespace(
        out_dir=str(out_dir),
        dataset_dir=str(repo_root / "downloads" / "Bongard-LOGO"),
        limit=1,
        pool_size=64,
        sampling_seed=20260805,
        support_seed=20260805,
        query_seed=20260806,
        model="fixture-model",
        reasoning_effort="medium",
        minutes=2,
        scorer_workers=4,
        executable="fixture-codex",
        verbose_oracle=False,
    )


def _file_sha(path: str) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _affirmative_query_array_digests() -> frozenset[str]:
    repo_root = Path(__file__).resolve().parents[2]
    latent = sample_basic_program_splits(
        str(repo_root / "downloads" / "Bongard-LOGO"),
        limit=1, seed=20260805, pool_size=64)[0]
    query = latent.render("query", 20260806)
    return frozenset(R._array_digest(panel) for panel in query.pos)


def _target_array_digest(path: str) -> str:
    with Image.open(path) as encoded:
        presentation = np.asarray(encoded.convert("L"))
    panel = np.ascontiguousarray((presentation == 0).astype(np.uint8))
    return R._array_digest(panel)


@dataclass(frozen=True)
class _FixtureContract:
    claim: str
    positive_digests: tuple[str, ...]
    foil_digests: tuple[str, ...]
    model: str
    reasoning_effort: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fixture-contrastive-contract/v1",
            "claim": self.claim,
            "positive_digests": list(self.positive_digests),
            "foil_digests": list(self.foil_digests),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "decoder": "fixture-direct-categorical/no-threshold",
        }

    def digest(self) -> str:
        return R.canonical_digest(self.to_dict())


def _observation_from_dict(value: Mapping[str, Any]) -> G.Observation:
    if value.get("status") == "present":
        return G.Present(
            value["value"], value["unit"], tuple(value["provenance"]))
    if value.get("status") == "indeterminate":
        return G.Indeterminate(
            value["mode"], value["detail"], tuple(value["provenance"]))
    if value.get("status") == "error":
        return G.Error(
            value["code"], value["detail"], tuple(value["provenance"]))
    raise ValueError("fixture observation is malformed")


@dataclass(frozen=True)
class _FixtureEvaluation:
    contract_digest: str
    target_digest: str
    observation: G.Observation
    receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "fixture-contrastive-evaluation/v1",
            "contract_digest": self.contract_digest,
            "target_digest": self.target_digest,
            "observation": self.observation.to_dict(),
            "receipt": dict(self.receipt),
        }
        body["evaluation_digest"] = R.canonical_digest(body)
        return body


class _FixtureOracle:
    def __init__(self, backend: "_FixtureBackend", contract: _FixtureContract):
        self.backend = backend
        self.contract = contract

    def observable_contract(self, observable_id: str) -> G.ObservableContract:
        return G.ObservableContract(
            observable_id=observable_id,
            value_type=G.ValueType.BOOLEAN,
            unit=G.Unit.BOOLEAN,
            referent="panel.fixture-open-vocabulary-claim",
            reducer=G.Reducer.IDENTITY,
            evaluator=lambda context: self.evaluate(context).observation,
            indeterminate_modes=("fixture-abstain",),
            source=G.ObservableSource.ORACLE,
            version="v1",
        )

    def evaluate(self, target_png_path: str) -> _FixtureEvaluation:
        name = Path(target_png_path).name
        assert name.startswith("target_") and name.endswith(".png")
        assert "pos" not in name and "neg" not in name
        self.backend.worker_arguments.append((target_png_path,))
        ordinal = int(name.removeprefix("target_").removesuffix(".png"))
        provenance = (self.contract.digest(), _file_sha(target_png_path))
        if ordinal in self.backend.error_ordinals:
            observation: G.Observation = G.Error(
                "fixture-oracle-error", "fixture transport", provenance)
        elif ordinal in self.backend.abstain_ordinals:
            observation = G.Indeterminate(
                "fixture-abstain", "fixture ambiguity", provenance)
        else:
            observation = G.Present(
                _target_array_digest(target_png_path) in
                _affirmative_query_array_digests(),
                G.Unit.BOOLEAN, provenance)
        receipt = {
            "schema": "fixture-receipt/v1",
            "source": "offline-fixture",
            "target_digest": _file_sha(target_png_path),
            "contract_digest": self.contract.digest(),
        }
        receipt["receipt_digest"] = R.canonical_digest(receipt)
        return _FixtureEvaluation(
            self.contract.digest(), _file_sha(target_png_path),
            observation, receipt)


class _FixtureBackend:
    def __init__(self, out_dir: Path, *, abstain=(), error=()) -> None:
        self.out_dir = out_dir
        self.abstain_ordinals = set(abstain)
        self.error_ordinals = set(error)
        self.worker_arguments: list[tuple[str]] = []
        self.oracle_creations = 0
        self.replay_calls = 0

    def create_contract(
        self, claim: str, positive_paths: Sequence[str], foil_paths: Sequence[str],
        *, model: str, reasoning_effort: str,
    ) -> _FixtureContract:
        assert len(positive_paths) == len(foil_paths) == 6
        return _FixtureContract(
            claim,
            tuple(sorted(_file_sha(path) for path in positive_paths)),
            tuple(sorted(_file_sha(path) for path in foil_paths)),
            model,
            reasoning_effort,
        )

    def restore_contract(self, value: Mapping[str, Any]) -> _FixtureContract:
        if set(value) != {
            "schema", "claim", "positive_digests", "foil_digests", "model",
            "reasoning_effort", "decoder",
        } or value["schema"] != "fixture-contrastive-contract/v1" \
                or value["decoder"] != "fixture-direct-categorical/no-threshold":
            raise ValueError("fixture contract differs")
        return _FixtureContract(
            value["claim"], tuple(value["positive_digests"]),
            tuple(value["foil_digests"]), value["model"],
            value["reasoning_effort"])

    def create_oracle(
        self, contract: _FixtureContract, positive_paths: Sequence[str],
        foil_paths: Sequence[str], *, minutes: int, executable: str,
        verbose: bool,
    ) -> _FixtureOracle:
        self.oracle_creations += 1
        if self.oracle_creations == 1:
            assert not (self.out_dir / "workspace" / "problem_00" / "query").exists()
            assert (self.out_dir / "workspace" / "problem_00" /
                    "query_latent_commitment.json").exists()
        assert self.create_contract(
            contract.claim, positive_paths, foil_paths,
            model=contract.model,
            reasoning_effort=contract.reasoning_effort) == contract
        return _FixtureOracle(self, contract)

    def replay_evaluation(
        self, contract: _FixtureContract, value: Mapping[str, Any],
        target_png_path: str,
    ) -> _FixtureEvaluation:
        self.replay_calls += 1
        if not isinstance(value, Mapping) or set(value) != {
            "schema", "contract_digest", "target_digest", "observation",
            "receipt", "evaluation_digest",
        }:
            raise ValueError("fixture evaluation fields differ")
        unsigned = {key: item for key, item in value.items()
                    if key != "evaluation_digest"}
        if value["schema"] != "fixture-contrastive-evaluation/v1" \
                or value["evaluation_digest"] != R.canonical_digest(unsigned) \
                or value["contract_digest"] != contract.digest() \
                or value["target_digest"] != _file_sha(target_png_path):
            raise ValueError("fixture evaluation binding differs")
        receipt = value["receipt"]
        receipt_unsigned = {key: item for key, item in receipt.items()
                            if key != "receipt_digest"}
        if receipt["receipt_digest"] != R.canonical_digest(receipt_unsigned) \
                or receipt["target_digest"] != value["target_digest"] \
                or receipt["contract_digest"] != contract.digest():
            raise ValueError("fixture receipt binding differs")
        return _FixtureEvaluation(
            contract.digest(), value["target_digest"],
            _observation_from_dict(value["observation"]), dict(receipt))


class _FixtureProposer:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.calls = 0

    def propose(
        self, problem_id: str, support_png_paths: Sequence[str],
    ) -> ClaimProposalBundle:
        self.calls += 1
        assert len(support_png_paths) == 12
        record_dir = self.out_dir / "workspace" / problem_id
        assert (record_dir / "query_latent_commitment.json").exists()
        assert not (record_dir / "query").exists()
        assert not (record_dir / "oracle_freeze.json").exists()
        return ClaimProposalBundle(
            problem_id=problem_id,
            analysis="A stable articulated animal silhouette separates the sets.",
            claim="contains a bird-like articulated silhouette",
            receipt=make_offline_fixture_receipt("fixture-proposal"),
        )


def test_hybrid_run_commits_freezes_scores_neutral_targets_and_replays(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "hybrid"
    proposer = _FixtureProposer(out_dir)
    backend = _FixtureBackend(out_dir)
    campaign = R.run(
        _args(out_dir), proposer=proposer, backend=backend)

    assert proposer.calls == 1
    assert len(backend.worker_arguments) == 12
    assert all(len(arguments) == 1 for arguments in backend.worker_arguments)
    assert sorted(Path(arguments[0]).name for arguments in backend.worker_arguments) == [
        f"target_{index:02d}.png" for index in range(12)]
    record = campaign["records"][0]
    assert [decision["ordinal"] for decision in
            record["query_evaluation"]["decisions"]] == list(range(12))
    assert sorted(decision["slot"] for decision in
                  record["query_evaluation"]["decisions"]) == sorted(
                      f"{side}_{index}" for side in ("pos", "neg")
                      for index in range(6))
    # Neutral target ordinals are deliberately not a side encoding.
    labels_by_ordinal = [decision["label"] for decision in
                         record["query_evaluation"]["decisions"]]
    assert labels_by_ordinal != [True] * 6 + [False] * 6
    assert record["status"] == "SOLVED_HYBRID_EXPLORATORY"
    assert record["query_evaluation"]["exact"] is True
    assert record["query_evaluation"]["abstention_count"] == 0
    assert record["formula"]["compiled"]["taint"] == "HYBRID"
    assert "support_evaluation" not in record
    assert record["oracle_freeze"]["polarity"] == \
        "literal-affirmative-eq-true/no-reversal"
    assert record["program_split"]["program_split_digest"] == \
        record["query_latent_commitment"]["program_split_digest"]

    report = R.replay_campaign_directory(str(out_dir), backend=backend)
    assert report["status"] == "LIVE_EVIDENCE_REPLAY_VALID"
    assert report["perception_reexecuted"] is False
    assert report["live_oracle_calls"] == 0


@pytest.mark.parametrize(("abstain", "error", "expected"), [
    ((0,), (), "UNSOLVED_HYBRID_EXPLORATORY"),
    ((), (0,), "INVALID_HYBRID_EXPLORATORY"),
])
def test_hybrid_abstention_and_error_never_become_false_or_solved(
    tmp_path: Path, abstain: tuple[int, ...], error: tuple[int, ...],
    expected: str,
) -> None:
    out_dir = tmp_path / expected.lower()
    backend = _FixtureBackend(out_dir, abstain=abstain, error=error)
    campaign = R.run(
        _args(out_dir), proposer=_FixtureProposer(out_dir), backend=backend)
    record = campaign["records"][0]
    assert record["status"] == expected
    assert record["solved"] is False
    assert record["query_evaluation"]["decisions"][0]["predicted"] is None


def test_hybrid_replay_rejects_downstream_and_panel_tampering(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "tamper"
    backend = _FixtureBackend(out_dir)
    campaign = R.run(
        _args(out_dir), proposer=_FixtureProposer(out_dir), backend=backend)

    changed = copy.deepcopy(campaign)
    changed["records"][0]["query_evaluation"]["decisions"][0][
        "predicted"] = False
    changed["campaign_digest"] = R.canonical_digest({
        key: item for key, item in changed.items() if key != "campaign_digest"})
    with pytest.raises(ValueError, match="decision"):
        R.replay_campaign_artifact(changed, str(out_dir), backend=backend)

    relative = campaign["records"][0]["query_panel_set"]["panels"][0][
        "png_path"]
    (out_dir / relative).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file digest"):
        R.replay_campaign_directory(str(out_dir), backend=backend)


def test_hybrid_replay_resamples_once_and_rejects_cherry_picked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "resample"
    backend = _FixtureBackend(out_dir)
    campaign = R.run(
        _args(out_dir), proposer=_FixtureProposer(out_dir), backend=backend)
    original_sampler = R.sample_basic_program_splits
    calls = 0

    def counted_sampler(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(R, "sample_basic_program_splits", counted_sampler)
    report = R.replay_campaign_directory(str(out_dir), backend=backend)
    assert report["status"] == "LIVE_EVIDENCE_REPLAY_VALID"
    assert calls == 1

    changed = copy.deepcopy(campaign)
    changed["records"][0]["program_split"]["concept"] = "cherry-picked"
    changed["campaign_digest"] = R.canonical_digest({
        key: item for key, item in changed.items() if key != "campaign_digest"})
    with pytest.raises(ValueError, match="deterministic sampler"):
        R.replay_campaign_artifact(changed, str(out_dir), backend=backend)
    assert calls == 2


def test_default_v2_backend_uses_two_fresh_swapped_turns_and_cold_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production backend with a deterministic fake transport."""
    out_dir = tmp_path / "v2-backend"
    calls: list[dict[str, Any]] = []
    calls_lock = threading.Lock()

    def fake_transport(prompt, paths, names, schema, **kwargs):
        target_is_affirmative = (
            _target_array_digest(paths[0]) in
            _affirmative_query_array_digests())
        comparisons = []
        presentation = []
        for index, pair_id in enumerate(C.PAIR_IDS):
            left_path = paths[1 + 2 * index]
            right_path = paths[2 + 2 * index]
            desired_prefix = "pos_" if target_is_affirmative else "neg_"
            left_matches = Path(left_path).name.startswith(desired_prefix)
            right_matches = Path(right_path).name.startswith(desired_prefix)
            assert left_matches is not right_matches
            comparisons.append({
                "pair_id": pair_id,
                "choice": "left" if left_matches else "right",
                "evidence": f"fixture visible resemblance for {pair_id}",
            })
            presentation.append(C.PresentationPair(
                pair_id, f"fixture-{index}",
                C.ImageBinding.from_path(left_path),
                C.ImageBinding.from_path(right_path),
                "anchor",
            ))
        payload = {"comparisons": comparisons}
        target = C.ImageBinding.from_path(paths[0])
        binding = C._named_binding(prompt, schema, target, presentation)
        receipt = {
            "source": "offline-fixture",
            "requested_model": kwargs["model"],
            "requested_reasoning_effort": kwargs["reasoning_effort"],
            "input_digest_schema": C.codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "task_digest": binding["prompt_digest"],
            "structured_output_digest": C._raw_digest(payload),
            **binding,
        }
        with calls_lock:
            calls.append({
                "prompt": prompt, "names": tuple(names),
                "target": Path(paths[0]).name,
            })
        return SimpleNamespace(
            payload=payload,
            receipt=SimpleNamespace(to_dict=lambda: dict(receipt)),
        )

    monkeypatch.setattr(
        C.codex_proposer, "run_codex_named_images_structured", fake_transport)
    campaign = R.run(
        _args(out_dir), proposer=_FixtureProposer(out_dir),
        backend=R.CodexBackend())

    record = campaign["records"][0]
    assert record["status"] == "SOLVED_HYBRID_EXPLORATORY"
    assert record["oracle_contract"]["contract"]["protocol_status"] == \
        "HYBRID-EXPLORATORY"
    assert record["oracle_contract"]["contract"]["calibrator"] is None
    assert len(record["oracle_contract"]["contract"]["pairs"]) == 3
    assert len(calls) == 24
    assert all(call["target"].startswith("target_") for call in calls)
    expected_names = ("target.png",) + tuple(
        name for pair_id in C.PAIR_IDS
        for name in (f"{pair_id}_left.png", f"{pair_id}_right.png"))
    assert all(call["names"] == expected_names for call in calls)
    assert all("positive" not in call["prompt"].lower()
               and "negative" not in call["prompt"].lower()
               and "anchor" not in call["prompt"].lower()
               and "foil" not in call["prompt"].lower()
               for call in calls)
    report = R.replay_campaign_directory(
        str(out_dir), backend=R.CodexBackend())
    assert report["status"] == "LIVE_EVIDENCE_REPLAY_VALID"
    assert report["live_oracle_calls"] == 0
