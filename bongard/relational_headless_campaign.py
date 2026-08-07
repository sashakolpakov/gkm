"""Atomic full-current-DEV campaigns for the standalone relational runner.

Campaign preparation is metadata-only: it selects the complete maximum
disclosure-disjoint set of historically-clean, semantic-unseen DEV tasks and
freezes one :class:`~bongard.relational_headless_runner.RelationalHeadlessPlan`
per task.  Execution persists that complete plan and every embedded task plan,
then records the entire cohort in one exposure-ledger transition before the
first corpus path is resolved.  Tasks run serially in the frozen order, once
each, and every terminal outcome remains in the aggregate denominator.

The module deliberately has no import or dependency on ``arc/codex_proposer``.
Python is the predicate, aggregation, and cold-replay authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import fcntl
from functools import lru_cache
from pathlib import Path, PurePosixPath
import json
import os
import secrets
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import canonical_digest, canonical_json
from bongard.cohorts import classify_task, parse_official_task_id
from bongard.closed_visual_predicates import (
    ClosedPanelPredicate,
    evaluate_closed_predicate,
)
from bongard.composite_visual_packet import (
    ExactPanelWitnessPacket,
    extract_exact_panel_witness_packet,
    verify_exact_panel_witness_packet,
)
from bongard.corpus import PNG_SIGNATURE, SplitIndex
from bongard.evidence import Disposition
from bongard.exposure import (
    ExposureLedger,
    ExposureViolation,
    basic_morphology_cluster_id,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.loop_scene_witnesses import (
    LoopScenePacket,
    extract_loop_scene_witnesses,
    verify_loop_scene_packet,
)
from bongard.relational_headless_runner import (
    EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID,
    EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS,
    ENGINEERING_CAMPAIGN_AUTHORIZATION_ACTOR,
    ENGINEERING_CAMPAIGN_AUTHORIZATION_PHASE,
    ENGINEERING_CAMPAIGN_AUTHORIZATION_PURPOSE,
    EXPLICITLY_SEALED_ENGINEERING_TASK_ID,
    FAILURE_SCHEMA,
    PLAN_SCHEMA,
    PROTOCOL_ID,
    RUN_SCHEMA,
    STRICT_DEV_ADMISSION_POLICY_ID,
    STRICT_DEV_MODE,
    PacketExtractor,
    PacketVerifier,
    PngReader,
    RelationalHeadlessOutcome,
    RelationalHeadlessPlan,
    RelationalHeadlessRunError,
    ReleaseArchiveAuthenticator,
    StructuredTransport,
    _persist_artifact,
    _persist_exposure,
    _read_authenticated_release_panel,
    _read_png_no_follow,
    _raw_digest,
    _require_address,
    _require_sha256,
    _seal,
    _stable_read,
    _terminal_failure,
    _verify_seal,
    _write_once_durable,
    cold_replay_relational_headless_run,
    load_relational_artifact,
    prepare_relational_headless_plan,
    relational_headless_runner_source_digest,
    run_relational_headless,
    verify_relational_predictions,
    verify_relational_proposal_freeze,
)
from bongard.relational_visual_query import (
    RelationalVisualQuery,
    evaluate_relational_query,
)
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CloudPolicyCacheSnapshot,
    run_codex_structured,
    snapshot_cloud_policy_cache,
)


CAMPAIGN_PLAN_SCHEMA = "gkm.bongard-relational-headless-dev-campaign-plan.v4"
CAMPAIGN_RUN_SCHEMA = "gkm.bongard-relational-headless-dev-campaign-run.v4"
CAMPAIGN_REPLAY_SCHEMA = "gkm.bongard-relational-headless-dev-campaign-replay.v4"
ATTEMPT_CLAIM_SCHEMA = "gkm.bongard-relational-campaign-attempt-claim.v1"
ATTEMPT_TERMINAL_SCHEMA = "gkm.bongard-relational-campaign-attempt-terminal.v1"
CAMPAIGN_PROTOCOL_ID = "bongard.relational-headless/full-current-dev-v4"
CAMPAIGN_SELECTION_ALGORITHM_ID = (
    "bongard.relational-headless/full-current-dev-maximum-token-packing-v4"
)
CAMPAIGN_EXPOSURE_PHASE = "relational-headless-full-current-dev-campaign-v4"
CAMPAIGN_EXPOSURE_ACTOR = "headless-codex-relational-campaign"
CAMPAIGN_EXPOSURE_PURPOSE = (
    "atomic full-cohort disclosure before serial one-shot DEV execution"
)
ENGINEERING_CAMPAIGN_PROTOCOL_ID = (
    "bongard.closed-visual/exact-unused-train-semantics-reused-engineering-v2"
)
ENGINEERING_CAMPAIGN_SELECTION_ALGORITHM_ID = (
    "bongard.closed-visual/fixed-five-train-allowlist-v2"
)
ARTIFACT_STORE_NORMALIZATION = "expanduser-abspath-realpath/v1"


class RelationalHeadlessCampaignError(RelationalHeadlessRunError):
    """A full-current-DEV campaign violated its frozen batch protocol."""


def _normalize_artifact_store_path(value: str | Path) -> str:
    """Return the one public canonical root used for artifacts and journals."""

    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise RelationalHeadlessCampaignError(
            "artifact/attempt store path must be path-like"
        ) from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise RelationalHeadlessCampaignError(
            "artifact/attempt store path must be non-empty text without NUL"
        )
    return os.path.realpath(os.path.abspath(os.path.expanduser(raw)))


def _require_bound_artifact_store(
    plan: "RelationalHeadlessCampaignPlan", value: str | Path
) -> Path:
    actual = _normalize_artifact_store_path(value)
    if actual != plan.artifact_store_path:
        raise RelationalHeadlessCampaignError(
            "artifact/attempt store differs from the campaign plan binding"
        )
    return Path(actual)


def relational_headless_campaign_source_digest() -> str:
    """Return the exact campaign executor/aggregator/replay source identity."""

    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rank(seed: str, domain: str, value: object) -> str:
    return canonical_digest(
        {
            "algorithm_id": CAMPAIGN_SELECTION_ALGORITHM_ID,
            "seed": seed,
            "domain": domain,
            "value": value,
        }
    )


def _derived_secret(
    secret: str,
    domain: str,
    task_id: str,
    *,
    algorithm_id: str = CAMPAIGN_SELECTION_ALGORITHM_ID,
) -> str:
    return canonical_digest(
        {
            "algorithm_id": algorithm_id,
            "secret": secret,
            "domain": domain,
            "task_id": task_id,
        }
    )


def _read_private_schedule_secret(path: Path) -> str:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise RelationalHeadlessCampaignError(
            "cannot inspect private schedule secret"
        ) from exc
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_mode & 0o077
    ):
        raise RelationalHeadlessCampaignError(
            "schedule secret file must be a private regular file"
        )
    try:
        text = _stable_read(path, maximum=4096).decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise RelationalHeadlessCampaignError(
            "cannot read private schedule secret"
        ) from exc
    if text.endswith("\n"):
        text = text[:-1]
    _require_sha256(text, "private schedule secret")
    return text


def _disclosure_tokens(family: str, concepts: Sequence[str]) -> tuple[str, ...]:
    if family == "bd":
        return tuple(
            sorted(
                {
                    token
                    for concept in concepts
                    for token in (
                        "basic_family:" + concept,
                        "basic_morphology:"
                        + basic_morphology_cluster_id(concept),
                    )
                }
            )
        )
    if family == "hd":
        return tuple(
            sorted(
                {"abstract_pair:" + "\0".join(concepts)}
                | {"abstract_attribute:" + item for item in concepts}
            )
        )
    return ("freeform_family:" + "\0".join(concepts),)


def _collision_tokens(family: str, concepts: Sequence[str]) -> frozenset[str]:
    return frozenset(
        token
        for token in _disclosure_tokens(family, concepts)
        if token.startswith("basic_morphology:")
        or token.startswith("abstract_attribute:")
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    task_id: str
    family: str
    concepts: tuple[str, ...]
    split: str


def _maximum_disjoint_candidates(
    candidates: Sequence[_Candidate], *, seed: str, family: str
) -> tuple[_Candidate, ...]:
    """Exact maximum-cardinality token packing with a public hash tie-break."""

    frozen = tuple(candidates)
    if not frozen:
        return ()
    token_sets = tuple(
        _collision_tokens(item.family, item.concepts) for item in frozen
    )
    vocabulary = tuple(sorted({token for tokens in token_sets for token in tokens}))
    if not vocabulary:
        return tuple(
            sorted(
                frozen,
                key=lambda item: (
                    _rank(seed, "maximum-disjoint", item.task_id),
                    item.task_id,
                ),
            )
        )
    bit_of = {token: 1 << index for index, token in enumerate(vocabulary)}
    masks = tuple(
        sum((bit_of[token] for token in tokens), start=0)
        for tokens in token_sets
    )
    ranks = tuple(
        (
            _rank(
                seed,
                "maximum-disjoint-candidate",
                {
                    "family": family,
                    "task_id": item.task_id,
                    "concepts": list(item.concepts),
                },
            ),
            item.task_id,
        )
        for item in frozen
    )

    def tie_key(indices: tuple[int, ...]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(ranks[index] for index in indices))

    def better(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        if len(left) != len(right):
            return left if len(left) > len(right) else right
        return left if tie_key(left) < tie_key(right) else right

    @lru_cache(maxsize=None)
    def solve(available: int) -> tuple[int, ...]:
        if available == 0:
            return ()
        first = available & -available
        best = solve(available & ~first)
        for index, mask in enumerate(masks):
            if mask & first and mask & available == mask:
                proposal = tuple(
                    sorted((index, *solve(available & ~mask)))
                )
                best = better(best, proposal)
        return best

    chosen = solve((1 << len(vocabulary)) - 1)
    return tuple(frozen[index] for index in sorted(chosen, key=lambda i: ranks[i]))


def _predecessor_hd_attributes(
    predecessor: ExposureLedger, historical: Any
) -> frozenset[str]:
    result: set[str] = set()
    for task_id in predecessor.exposed_task_ids:
        parsed = parse_official_task_id(task_id, historical)
        if parsed.family == "hd":
            result.update(parsed.concepts)
    return frozenset(result)


def _select_full_current_dev(
    *, split_index: SplitIndex, predecessor: ExposureLedger, seed: str
) -> tuple[tuple[_Candidate, ...], dict[str, object]]:
    historical = load_historical_exposure()
    resolver = semantic_resolver_policy_digest(historical)
    groups = split_index.canonical_groups
    allowed = tuple(sorted(set(groups["train"]) | set(groups["val"])))
    exact_unused = predecessor.unseen_task_ids(allowed)
    exposed_hd = _predecessor_hd_attributes(predecessor, historical)
    viable: list[_Candidate] = []
    for task_id in exact_unused:
        assignment = split_index.assignment(task_id)
        record = classify_task(
            task_id,
            historical,
            split=assignment.split,
            regime=assignment.regime,
        )
        if not record.historically_clean or record.semantic_cohort != "dev":
            continue
        if record.family == "hd" and set(record.parsed.concepts) & exposed_hd:
            continue
        try:
            predecessor.assert_semantically_unseen(
                task_ids=(task_id,),
                historical_seed=historical,
                expected_historical_seed_digest=historical.seed_digest,
                expected_resolver_policy_digest=resolver,
            )
        except ExposureViolation:
            continue
        viable.append(
            _Candidate(
                task_id,
                record.family,
                record.parsed.concepts,
                assignment.split,
            )
        )

    # Multiple HD instances represent one semantic generator.  Basic pair
    # expressions are likewise one generator, even though current BD has one
    # official instance.  Freeze one public seed-ranked representative first.
    grouped: dict[tuple[str, tuple[str, ...]], list[_Candidate]] = defaultdict(list)
    for item in viable:
        grouped[(item.family, item.concepts)].append(item)
    representatives = tuple(
        min(
            siblings,
            key=lambda item: (
                _rank(seed, "within-semantic-generator", item.task_id),
                item.task_id,
            ),
        )
        for _key, siblings in sorted(grouped.items())
    )
    by_family = {
        family: tuple(item for item in representatives if item.family == family)
        for family in ("bd", "hd")
    }
    packed = tuple(
        item
        for family in ("bd", "hd")
        for item in _maximum_disjoint_candidates(
            by_family[family], seed=seed, family=family
        )
    )
    selected = tuple(
        sorted(
            packed,
            key=lambda item: (
                _rank(seed, "serial-campaign-order", item.task_id),
                item.task_id,
            ),
        )
    )
    selected_tokens: set[str] = set()
    for item in selected:
        tokens = set(_disclosure_tokens(item.family, item.concepts))
        if tokens & selected_tokens:
            raise RelationalHeadlessCampaignError(
                "full DEV selector emitted a disclosure-token collision"
            )
        selected_tokens.update(tokens)
    metadata: dict[str, object] = {
        "exact_unused_train_val_count": len(exact_unused),
        "individually_viable_count": len(viable),
        "semantic_generator_representative_count": len(representatives),
        "maximum_capacity_by_family": {
            family: len(
                _maximum_disjoint_candidates(
                    by_family[family], seed=seed, family=family
                )
            )
            for family in ("bd", "hd")
        },
        "selected_disclosure_tokens_digest": canonical_digest(
            sorted(selected_tokens)
        ),
        "historical_seed_digest": historical.seed_digest,
        "semantic_resolver_policy_digest": resolver,
    }
    return selected, metadata


@dataclass(frozen=True, slots=True)
class RelationalHeadlessCampaignPlan:
    campaign_mode: str
    artifact_store_path: str
    official_release_descriptor_digest: str
    corpus_digest: str
    split_source_digest: str
    exposure_predecessor_digest: str
    selection_seed_provenance: str
    selection_seed_digest: str
    schedule_secret_digest: str
    campaign_python_source_digest: str
    runner_python_source_digest: str
    exposure_observed_at: str
    expected_task_count: int
    selection_metadata: Mapping[str, object]
    task_plans: tuple[RelationalHeadlessPlan, ...]
    _seed: str = field(repr=False, compare=False)
    _schedule_secret: str = field(repr=False, compare=False)
    _cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if _normalize_artifact_store_path(self.artifact_store_path) != (
            self.artifact_store_path
        ):
            raise RelationalHeadlessCampaignError(
                "campaign artifact/attempt store path is not canonical"
            )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.task_plans)

    @property
    def campaign_protocol_id(self) -> str:
        return (
            ENGINEERING_CAMPAIGN_PROTOCOL_ID
            if self.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else CAMPAIGN_PROTOCOL_ID
        )

    @property
    def selection_algorithm_id(self) -> str:
        return (
            ENGINEERING_CAMPAIGN_SELECTION_ALGORITHM_ID
            if self.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else CAMPAIGN_SELECTION_ALGORITHM_ID
        )

    @property
    def task_admission_policy_id(self) -> str:
        return (
            EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID
            if self.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else STRICT_DEV_ADMISSION_POLICY_ID
        )

    @property
    def exposure_authorization(self) -> tuple[str, str, str]:
        return (
            (
                ENGINEERING_CAMPAIGN_AUTHORIZATION_PHASE,
                ENGINEERING_CAMPAIGN_AUTHORIZATION_ACTOR,
                ENGINEERING_CAMPAIGN_AUTHORIZATION_PURPOSE,
            )
            if self.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else (
                CAMPAIGN_EXPOSURE_PHASE,
                CAMPAIGN_EXPOSURE_ACTOR,
                CAMPAIGN_EXPOSURE_PURPOSE,
            )
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_PLAN_SCHEMA,
            "campaign_protocol_id": self.campaign_protocol_id,
            "task_protocol_id": PROTOCOL_ID,
            "campaign_mode": self.campaign_mode,
            "artifact_attempt_store_binding": {
                "normalized_absolute_path": self.artifact_store_path,
                "normalization": ARTIFACT_STORE_NORMALIZATION,
                "covers": [
                    "campaign-and-task-artifacts",
                    "relational-headless-attempt-journal",
                ],
                "path_substitution_before_execution": "rejected",
                "residual_operator_filesystem_trust": (
                    "deletion or copying of the bound store remains under "
                    "operator/filesystem control"
                ),
            },
            "selection_algorithm_id": self.selection_algorithm_id,
            "task_admission_policy_id": self.task_admission_policy_id,
            "source_identities": {
                "campaign_python_source_digest": (
                    self.campaign_python_source_digest
                ),
                "runner_python_source_digest": (
                    self.runner_python_source_digest
                ),
            },
            "official_release_descriptor_digest": (
                self.official_release_descriptor_digest
            ),
            "release_authentication": self.task_plans[0].to_data()[
                "release_authentication"
            ],
            "corpus_digest": self.corpus_digest,
            "split_source_digest": self.split_source_digest,
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "selection_seed": self._seed,
            "selection_seed_provenance": self.selection_seed_provenance,
            "selection_seed_digest": self.selection_seed_digest,
            "schedule_secret_digest": self.schedule_secret_digest,
            "schedule_secret_publicly_disclosed": False,
            "exposure_observed_at": self.exposure_observed_at,
            "expected_task_count": self.expected_task_count,
            "runtime": {
                "model": self.task_plans[0].model,
                "reasoning_effort": self.task_plans[0].reasoning_effort,
                "minutes_per_task": self.task_plans[0].minutes,
                "expected_launcher_digest": (
                    self.task_plans[0].expected_launcher_digest
                ),
                "cloud_policy_cache_binding": (
                    self.task_plans[0].cloud_policy_cache_binding
                ),
            },
            "selection": dict(self.selection_metadata),
            "execution_policy": {
                "cohort_exposure": "one-atomic-ledger-edge-before-task-1",
                "order": "serial-frozen-order",
                "proposals_per_task": 1,
                "rerolls": 0,
                "all_terminal_outcomes_remain_in_denominator": True,
                "attempt_journal": (
                    "exclusive-claimed-before-transport-terminal-after; "
                    "a-preexisting-claim-is-never-model-eligible"
                ),
                "official_test_authorized": False,
            },
            "tasks": [
                {
                    "ordinal": ordinal,
                    "task_id": task.task_id,
                    "family": task.family,
                    "headless_plan_digest": task.digest,
                    "headless_plan": task.to_data(),
                }
                for ordinal, task in enumerate(self.task_plans)
            ],
        }


def prepare_full_current_dev_campaign(
    *,
    artifact_store: str | Path,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    expected_release_descriptor_digest: str,
    release_authenticator: ReleaseArchiveAuthenticator,
    expected_corpus_digest: str,
    expected_split_source_digest: str,
    expected_exposure_predecessor_digest: str,
    campaign_seed: str,
    selection_seed_provenance: str,
    schedule_secret: str,
    exposure_observed_at: str,
    expected_task_count: int,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
) -> RelationalHeadlessCampaignPlan:
    """Freeze the complete current strict-DEV cohort without resolving paths."""

    normalized_artifact_store = _normalize_artifact_store_path(artifact_store)
    if not isinstance(expected_task_count, int) or isinstance(
        expected_task_count, bool
    ) or expected_task_count < 1:
        raise RelationalHeadlessCampaignError(
            "expected_task_count must be a positive integer"
        )
    if (
        not isinstance(selection_seed_provenance, str)
        or not selection_seed_provenance.strip()
        or selection_seed_provenance != selection_seed_provenance.strip()
    ):
        raise RelationalHeadlessCampaignError(
            "selection_seed_provenance must be non-empty trimmed text"
        )
    _require_sha256(schedule_secret, "private schedule secret")
    schedule_secret_digest = canonical_digest(
        {
            "schema": "gkm.bongard-relational-campaign-schedule-secret.v1",
            "schedule_secret": schedule_secret,
        }
    )
    release_digest = _require_address(
        expected_release_descriptor_digest, "official release descriptor digest"
    )
    if (
        not isinstance(release_authenticator, ReleaseArchiveAuthenticator)
        or release_authenticator.release_descriptor_digest != release_digest
    ):
        raise RelationalHeadlessCampaignError(
            "release archive authenticator differs from descriptor pin"
        )
    if predecessor.digest != expected_exposure_predecessor_digest:
        raise RelationalHeadlessCampaignError("campaign predecessor differs from pin")
    predecessor.assert_corpus(expected_corpus_digest)
    if split_index.source_digest != expected_split_source_digest:
        raise RelationalHeadlessCampaignError("campaign split differs from pin")
    groups = split_index.canonical_groups
    split_index.validate(
        set(groups["train"]) | set(groups["val"]) | set(groups["test"]),
        official_counts=False,
    )
    selected, metadata = _select_full_current_dev(
        split_index=split_index,
        predecessor=predecessor,
        seed=campaign_seed,
    )
    if len(selected) != expected_task_count:
        raise RelationalHeadlessCampaignError(
            "full current strict-DEV capacity differs from precommit: "
            f"{len(selected)} != {expected_task_count}"
        )
    cache = cloud_policy_cache_snapshot or CloudPolicyCacheSnapshot(None)
    task_plans = tuple(
        prepare_relational_headless_plan(
            task_id=item.task_id,
            split_index=split_index,
            predecessor=predecessor,
            expected_exposure_predecessor_digest=predecessor.digest,
            expected_corpus_digest=expected_corpus_digest,
            expected_split_source_digest=expected_split_source_digest,
            seed=_derived_secret(schedule_secret, "task-seed", item.task_id),
            exposure_observed_at=exposure_observed_at,
            expected_launcher_digest=expected_launcher_digest,
            release_authenticator=release_authenticator,
            cloud_policy_cache_snapshot=cache,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            label_nonce=_derived_secret(
                schedule_secret, "task-label-nonce", item.task_id
            ),
            support_selection_key=_derived_secret(
                schedule_secret, "support-selection-key", item.task_id
            ),
        )
        for item in selected
    )
    return RelationalHeadlessCampaignPlan(
        campaign_mode=STRICT_DEV_MODE,
        artifact_store_path=normalized_artifact_store,
        official_release_descriptor_digest=release_digest,
        corpus_digest=expected_corpus_digest,
        split_source_digest=expected_split_source_digest,
        exposure_predecessor_digest=predecessor.digest,
        selection_seed_provenance=selection_seed_provenance,
        selection_seed_digest=canonical_digest(campaign_seed),
        schedule_secret_digest=schedule_secret_digest,
        campaign_python_source_digest=(
            relational_headless_campaign_source_digest()
        ),
        runner_python_source_digest=relational_headless_runner_source_digest(),
        exposure_observed_at=exposure_observed_at,
        expected_task_count=expected_task_count,
        selection_metadata=metadata,
        task_plans=task_plans,
        _seed=campaign_seed,
        _schedule_secret=schedule_secret,
        _cloud_policy_cache_snapshot=cache,
    )


def prepare_exact_unused_train_engineering_campaign(
    *,
    artifact_store: str | Path,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    expected_release_descriptor_digest: str,
    release_authenticator: ReleaseArchiveAuthenticator,
    expected_corpus_digest: str,
    expected_split_source_digest: str,
    expected_exposure_predecessor_digest: str,
    campaign_seed: str,
    selection_seed_provenance: str,
    schedule_secret: str,
    exposure_observed_at: str,
    expected_task_count: int,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
) -> RelationalHeadlessCampaignPlan:
    """Freeze the labelled fixed-five TRAIN engineering campaign, metadata-only."""

    normalized_artifact_store = _normalize_artifact_store_path(artifact_store)
    if expected_task_count != len(EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS):
        raise RelationalHeadlessCampaignError(
            "engineering campaign task count must equal the fixed five-task allowlist"
        )
    if (
        not isinstance(selection_seed_provenance, str)
        or not selection_seed_provenance.strip()
        or selection_seed_provenance != selection_seed_provenance.strip()
    ):
        raise RelationalHeadlessCampaignError(
            "selection_seed_provenance must be non-empty trimmed text"
        )
    _require_sha256(schedule_secret, "private schedule secret")
    release_digest = _require_address(
        expected_release_descriptor_digest, "official release descriptor digest"
    )
    if (
        not isinstance(release_authenticator, ReleaseArchiveAuthenticator)
        or release_authenticator.release_descriptor_digest != release_digest
    ):
        raise RelationalHeadlessCampaignError(
            "release archive authenticator differs from descriptor pin"
        )
    if predecessor.digest != expected_exposure_predecessor_digest:
        raise RelationalHeadlessCampaignError("campaign predecessor differs from pin")
    predecessor.assert_corpus(expected_corpus_digest)
    if split_index.source_digest != expected_split_source_digest:
        raise RelationalHeadlessCampaignError("campaign split differs from pin")
    groups = split_index.canonical_groups
    split_index.validate(
        set(groups["train"]) | set(groups["val"]) | set(groups["test"]),
        official_counts=False,
    )
    if EXPLICITLY_SEALED_ENGINEERING_TASK_ID in EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS:
        raise RelationalHeadlessCampaignError(
            "sealed task contaminated the engineering allowlist"
        )
    for task_id in EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS:
        assignment = split_index.assignment(task_id)
        if assignment.split != "train" or assignment.regime is not None:
            raise RelationalHeadlessCampaignError(
                "engineering allowlist task is not exact TRAIN"
            )
    predecessor.assert_unseen(task_ids=EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS)
    cache = cloud_policy_cache_snapshot or CloudPolicyCacheSnapshot(None)
    task_plans = tuple(
        prepare_relational_headless_plan(
            task_id=task_id,
            split_index=split_index,
            predecessor=predecessor,
            expected_exposure_predecessor_digest=predecessor.digest,
            expected_corpus_digest=expected_corpus_digest,
            expected_split_source_digest=expected_split_source_digest,
            seed=_derived_secret(
                schedule_secret,
                "task-seed",
                task_id,
                algorithm_id=ENGINEERING_CAMPAIGN_SELECTION_ALGORITHM_ID,
            ),
            exposure_observed_at=exposure_observed_at,
            expected_launcher_digest=expected_launcher_digest,
            release_authenticator=release_authenticator,
            cloud_policy_cache_snapshot=cache,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            label_nonce=_derived_secret(
                schedule_secret,
                "task-label-nonce",
                task_id,
                algorithm_id=ENGINEERING_CAMPAIGN_SELECTION_ALGORITHM_ID,
            ),
            support_selection_key=_derived_secret(
                schedule_secret,
                "support-selection-key",
                task_id,
                algorithm_id=ENGINEERING_CAMPAIGN_SELECTION_ALGORITHM_ID,
            ),
            benchmark_mode=EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
        )
        for task_id in EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
    )
    historical = load_historical_exposure()
    metadata: dict[str, object] = {
        "selection_kind": "fixed-explicit-allowlist",
        "fixed_allowlist": list(EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS),
        "fixed_allowlist_digest": canonical_digest(
            list(EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS)
        ),
        "exact_unused_active_ledger": True,
        "required_split": "train",
        "required_historical_semantics": "historically_exposed",
        "semantic_unseen_asserted": False,
        "explicit_sealed_task_rejected": EXPLICITLY_SEALED_ENGINEERING_TASK_ID,
        "historical_seed_digest": historical.seed_digest,
        "semantic_resolver_policy_digest": (
            semantic_resolver_policy_digest(historical)
        ),
        "closed_predicate_binding": dict(
            task_plans[0].closed_predicate_binding or {}
        ),
    }
    return RelationalHeadlessCampaignPlan(
        campaign_mode=EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
        artifact_store_path=normalized_artifact_store,
        official_release_descriptor_digest=release_digest,
        corpus_digest=expected_corpus_digest,
        split_source_digest=expected_split_source_digest,
        exposure_predecessor_digest=predecessor.digest,
        selection_seed_provenance=selection_seed_provenance,
        selection_seed_digest=canonical_digest(campaign_seed),
        schedule_secret_digest=canonical_digest(
            {
                "schema": "gkm.bongard-relational-campaign-schedule-secret.v1",
                "schedule_secret": schedule_secret,
            }
        ),
        campaign_python_source_digest=(
            relational_headless_campaign_source_digest()
        ),
        runner_python_source_digest=relational_headless_runner_source_digest(),
        exposure_observed_at=exposure_observed_at,
        expected_task_count=expected_task_count,
        selection_metadata=metadata,
        task_plans=task_plans,
        _seed=campaign_seed,
        _schedule_secret=schedule_secret,
        _cloud_policy_cache_snapshot=cache,
    )


@dataclass(frozen=True, slots=True)
class RelationalHeadlessCampaignOutcome:
    plan: RelationalHeadlessCampaignPlan
    exposure_successor: ExposureLedger
    plan_path: Path
    exposure_path: Path
    task_outcomes: tuple[RelationalHeadlessOutcome, ...]
    report: Mapping[str, Any]
    report_path: Path

    def to_data(self) -> dict[str, object]:
        return {
            "campaign_plan_digest": self.plan.digest,
            "exposure_successor_digest": self.exposure_successor.digest,
            "task_count": len(self.task_outcomes),
            "report_digest": self.report["digest"],
            "plan_path": str(self.plan_path),
            "exposure_path": str(self.exposure_path),
            "report_path": str(self.report_path),
        }


def _report_content(
    *,
    plan: RelationalHeadlessCampaignPlan,
    successor: ExposureLedger,
    outcomes: Sequence[RelationalHeadlessOutcome],
    journal_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if tuple(item.plan.task_id for item in outcomes) != plan.task_ids:
        raise RelationalHeadlessCampaignError("campaign outcome order differs")
    if len(journal_records) != len(outcomes):
        raise RelationalHeadlessCampaignError("campaign journal order differs")
    statuses = Counter(item.status for item in outcomes)
    task_correct = 0
    released_query_count = 0
    correct_query_count = 0
    abstentions = 0
    errors = 0
    tasks: list[dict[str, object]] = []
    for ordinal, (outcome, journal) in enumerate(
        zip(outcomes, journal_records, strict=True)
    ):
        artifact = outcome.artifact
        score = artifact.get("score") if outcome.status == "complete" else None
        jointly_correct = False
        if isinstance(score, Mapping):
            released_query_count += int(score["total"])
            correct_query_count += int(score["correct"])
            abstentions += int(score["abstentions"])
            errors += int(score["errors"])
            jointly_correct = (
                score["correct"] == score["total"] == 2
                and score["abstentions"] == 0
                and score["errors"] == 0
            )
            task_correct += int(jointly_correct)
        tasks.append(
            {
                "ordinal": ordinal,
                "task_id": outcome.plan.task_id,
                "plan_digest": outcome.plan.digest,
                "status": outcome.status,
                "terminal_digest": artifact["digest"],
                "freeze_digest": artifact.get("freeze_digest"),
                "prediction_digest": artifact.get("prediction_digest"),
                "jointly_correct": jointly_correct,
                "reroll_attempted": artifact.get("reroll_attempted"),
                "attempt_claim_digest": journal.get("claim_digest"),
                "attempt_terminal_digest": journal.get("terminal_digest"),
                "attempt_resumed": journal.get("resumed"),
                "attempt_journal_persistence_error": journal.get(
                    "persistence_error"
                ),
            }
        )
    return {
        "schema": CAMPAIGN_RUN_SCHEMA,
        "campaign_protocol_id": plan.campaign_protocol_id,
        "task_protocol_id": PROTOCOL_ID,
        "campaign_mode": plan.campaign_mode,
        "campaign_plan_digest": plan.digest,
        "exposure_predecessor_digest": plan.exposure_predecessor_digest,
        "exposure_successor_digest": successor.digest,
        "cohort_exposed_atomically_before_task_1": True,
        "serial_frozen_order": True,
        "rerolls_attempted": False,
        "all_terminal_outcomes_in_denominator": True,
        "attempt_journal": {
            "claim_state": "CLAIMED",
            "claim_precedes_transport": True,
            "preexisting_claim_model_eligible": False,
            "terminal_state_recorded_after_attempt": True,
        },
        "task_count": len(plan.task_ids),
        "status_counts": {
            key: statuses.get(key, 0)
            for key in ("complete", "support_rejected", "terminal_failure")
        },
        "joint_task_accuracy": {
            "correct": task_correct,
            "denominator": len(plan.task_ids),
        },
        "fixed_denominator_query_score": {
            "correct": correct_query_count,
            "denominator": 2 * len(plan.task_ids),
            "unreleased_or_incorrect": (
                2 * len(plan.task_ids) - correct_query_count
            ),
        },
        "released_query_accuracy_diagnostic": {
            "correct": correct_query_count,
            "denominator": released_query_count,
            "abstentions": abstentions,
            "errors": errors,
            "headline_metric": False,
        },
        "tasks": tasks,
    }


def _verify_campaign_plan_reproduction(
    *,
    plan: RelationalHeadlessCampaignPlan,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
) -> None:
    if not plan.task_plans:
        raise RelationalHeadlessCampaignError("campaign plan is empty")
    first = plan.task_plans[0]
    prepare = (
        prepare_exact_unused_train_engineering_campaign
        if plan.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else prepare_full_current_dev_campaign
    )
    reproduced = prepare(
        artifact_store=plan.artifact_store_path,
        split_index=split_index,
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            plan.official_release_descriptor_digest
        ),
        release_authenticator=plan.task_plans[0]._release_authenticator,
        expected_corpus_digest=plan.corpus_digest,
        expected_split_source_digest=plan.split_source_digest,
        expected_exposure_predecessor_digest=plan.exposure_predecessor_digest,
        campaign_seed=plan._seed,
        selection_seed_provenance=plan.selection_seed_provenance,
        schedule_secret=plan._schedule_secret,
        exposure_observed_at=plan.exposure_observed_at,
        expected_task_count=plan.expected_task_count,
        expected_launcher_digest=first.expected_launcher_digest,
        cloud_policy_cache_snapshot=plan._cloud_policy_cache_snapshot,
        model=first.model,
        reasoning_effort=first.reasoning_effort,
        minutes=first.minutes,
    )
    if reproduced.to_data() != plan.to_data():
        raise RelationalHeadlessCampaignError(
            "campaign plan does not reproduce metadata-only"
        )


def _assert_campaign_sources_current(
    plan: RelationalHeadlessCampaignPlan,
) -> None:
    if (
        relational_headless_campaign_source_digest()
        != plan.campaign_python_source_digest
        or relational_headless_runner_source_digest()
        != plan.runner_python_source_digest
    ):
        raise RelationalHeadlessCampaignError(
            "campaign or runner source changed after plan preparation"
        )


class _AttemptAlreadyClaimed(RelationalHeadlessCampaignError):
    pass


def _secure_attempt_directory(
    artifact_store: str | Path, plan: RelationalHeadlessCampaignPlan
) -> Path:
    parent = _require_bound_artifact_store(plan, artifact_store)
    parent.mkdir(parents=True, exist_ok=True)
    current = parent
    for component in (
        "relational-headless-attempt-journal",
        plan.digest.removeprefix("sha256:"),
    ):
        current = current / component
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            pass
        try:
            status = os.lstat(current)
        except OSError as exc:
            raise RelationalHeadlessCampaignError(
                f"cannot inspect attempt journal directory: {current}"
            ) from exc
        if (
            not stat.S_ISDIR(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or status.st_mode & 0o077
        ):
            raise RelationalHeadlessCampaignError(
                "attempt journal directory must be private and non-symlink"
            )
    return current


def _acquire_campaign_lock(
    artifact_store: str | Path, plan: RelationalHeadlessCampaignPlan
) -> tuple[int, Path]:
    root = _secure_attempt_directory(artifact_store, plan)
    path = root / ".campaign.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_mode & 0o077
        ):
            raise RelationalHeadlessCampaignError(
                "campaign lock must be a private regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, RelationalHeadlessRunError) as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise RelationalHeadlessCampaignError(
            "campaign is already running or its lock is invalid"
        ) from exc
    return descriptor, path


def _attempt_paths(
    artifact_store: str | Path,
    *,
    plan: RelationalHeadlessCampaignPlan,
    ordinal: int,
    task_plan: RelationalHeadlessPlan,
) -> tuple[Path, Path]:
    root = _secure_attempt_directory(artifact_store, plan)
    stem = f"{ordinal:04d}-{task_plan.digest.removeprefix('sha256:')}"
    return root / f"{stem}.claimed.json", root / f"{stem}.terminal.json"


def _exclusive_create_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise _AttemptAlreadyClaimed(f"attempt journal already exists: {path}") from exc
    except OSError as exc:
        raise RelationalHeadlessCampaignError(
            f"cannot create attempt journal: {path}"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RelationalHeadlessCampaignError(
                    f"short attempt journal write: {path}"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if _stable_read(path) != payload:
        raise RelationalHeadlessCampaignError(
            f"attempt journal reload differs: {path}"
        )


def _load_canonical_journal(path: Path, schema: str) -> dict[str, Any]:
    payload = _stable_read(path, maximum=32 * 1024 * 1024)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RelationalHeadlessCampaignError(
            f"attempt journal is malformed: {path}"
        ) from exc
    if (
        not isinstance(value, dict)
        or canonical_json(value) + b"\n" != payload
    ):
        raise RelationalHeadlessCampaignError(
            f"attempt journal is not canonical: {path}"
        )
    return _verify_seal(value, schema)


def _claim_content(
    *,
    plan: RelationalHeadlessCampaignPlan,
    successor: ExposureLedger,
    ordinal: int,
    task_plan: RelationalHeadlessPlan,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": ATTEMPT_CLAIM_SCHEMA,
            "campaign_protocol_id": plan.campaign_protocol_id,
            "state": "CLAIMED",
            "campaign_plan_digest": plan.digest,
            "exposure_successor_digest": successor.digest,
            "ordinal": ordinal,
            "task_id": task_plan.task_id,
            "task_plan_digest": task_plan.digest,
            "transport_authorizations": 1,
            "preexisting_claim_model_eligible": False,
        }
    )


def _persist_attempt_claim(path: Path, claim: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_json(dict(claim)) + b"\n"
    _exclusive_create_durable(path, payload)
    return _load_canonical_journal(path, ATTEMPT_CLAIM_SCHEMA)


def _load_expected_attempt_claim(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    loaded = _load_canonical_journal(path, ATTEMPT_CLAIM_SCHEMA)
    if loaded != dict(expected):
        raise RelationalHeadlessCampaignError(
            "preexisting attempt claim differs from frozen campaign task"
        )
    return loaded


def _terminal_journal_content(
    *,
    plan: RelationalHeadlessCampaignPlan,
    ordinal: int,
    claim: Mapping[str, Any],
    outcome: RelationalHeadlessOutcome,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": ATTEMPT_TERMINAL_SCHEMA,
            "campaign_protocol_id": plan.campaign_protocol_id,
            "state": "TERMINAL",
            "campaign_plan_digest": plan.digest,
            "ordinal": ordinal,
            "task_id": outcome.plan.task_id,
            "task_plan_digest": outcome.plan.digest,
            "claim_digest": claim["digest"],
            "status": outcome.status,
            "outcome": outcome.to_data(),
            "terminal_artifact": dict(outcome.artifact),
        }
    )


def _persist_attempt_terminal(
    path: Path, terminal: Mapping[str, Any]
) -> dict[str, Any]:
    payload = canonical_json(dict(terminal)) + b"\n"
    try:
        _exclusive_create_durable(path, payload)
    except _AttemptAlreadyClaimed:
        loaded = _load_canonical_journal(path, ATTEMPT_TERMINAL_SCHEMA)
        if loaded != dict(terminal):
            raise RelationalHeadlessCampaignError(
                "preexisting attempt terminal differs"
            )
        return loaded
    return _load_canonical_journal(path, ATTEMPT_TERMINAL_SCHEMA)


def _resume_attempt_outcome(
    *,
    path: Path,
    plan: RelationalHeadlessCampaignPlan,
    ordinal: int,
    claim: Mapping[str, Any],
    task_plan: RelationalHeadlessPlan,
    successor: ExposureLedger,
    plan_path: Path,
    exposure_path: Path,
    artifact_store: str | Path,
) -> RelationalHeadlessOutcome | None:
    try:
        terminal = _load_canonical_journal(path, ATTEMPT_TERMINAL_SCHEMA)
    except RelationalHeadlessRunError:
        if not path.exists():
            return None
        raise
    expected_common = {
        "campaign_protocol_id": plan.campaign_protocol_id,
        "state": "TERMINAL",
        "campaign_plan_digest": plan.digest,
        "ordinal": ordinal,
        "task_id": task_plan.task_id,
        "task_plan_digest": task_plan.digest,
        "claim_digest": claim["digest"],
    }
    if any(terminal.get(key) != value for key, value in expected_common.items()):
        raise RelationalHeadlessCampaignError(
            "attempt terminal differs from frozen campaign task"
        )
    status = terminal.get("status")
    outcome_data = terminal.get("outcome")
    artifact = terminal.get("terminal_artifact")
    if (
        status not in {"complete", "support_rejected", "terminal_failure"}
        or not isinstance(outcome_data, Mapping)
        or not isinstance(artifact, Mapping)
        or outcome_data.get("status") != status
        or outcome_data.get("plan_digest") != task_plan.digest
        or outcome_data.get("exposure_successor_digest") != successor.digest
        or outcome_data.get("terminal_digest") != artifact.get("digest")
    ):
        raise RelationalHeadlessCampaignError(
            "attempt terminal outcome is malformed"
        )
    schema = FAILURE_SCHEMA if status == "terminal_failure" else RUN_SCHEMA
    verified_artifact = _verify_seal(artifact, schema)
    suffix = (
        "relational-headless-failure"
        if status == "terminal_failure"
        else "relational-headless-run"
    )
    terminal_path = (
        Path(artifact_store)
        / f"{verified_artifact['digest']}.{suffix}.json"
    )
    if str(terminal_path) != outcome_data.get("terminal_path"):
        raise RelationalHeadlessCampaignError(
            "attempt terminal artifact path differs"
        )
    persisted_path, persisted = _persist_artifact(
        artifact_store, verified_artifact, suffix=suffix
    )
    if persisted_path != terminal_path or persisted != verified_artifact:
        raise RelationalHeadlessCampaignError(
            "attempt terminal artifact cold reload differs"
        )
    freeze_path_value = outcome_data.get("freeze_path")
    prediction_path_value = outcome_data.get("prediction_path")
    return RelationalHeadlessOutcome(
        status,
        task_plan,
        successor,
        plan_path,
        exposure_path,
        terminal_path,
        verified_artifact,
        freeze_path=(
            None if freeze_path_value is None else Path(freeze_path_value)
        ),
        prediction_path=(
            None
            if prediction_path_value is None
            else Path(prediction_path_value)
        ),
    )


def _run_relational_headless_campaign_locked(
    *,
    corpus_root: str | Path,
    plan: RelationalHeadlessCampaignPlan,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    exposure_store: str | Path,
    artifact_store: str | Path,
    executable: str = "codex",
    verbose: bool = False,
    transport: StructuredTransport = run_codex_structured,
    png_reader: PngReader = _read_png_no_follow,
    extractor: PacketExtractor | None = None,
    packet_verifier: PacketVerifier | None = None,
) -> RelationalHeadlessCampaignOutcome:
    """Atomically expose the cohort and consume each task claim at most once."""

    artifact_store = _require_bound_artifact_store(plan, artifact_store)
    _verify_campaign_plan_reproduction(
        plan=plan, split_index=split_index, predecessor=predecessor
    )
    if extractor is None:
        extractor = (
            extract_exact_panel_witness_packet
            if plan.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else extract_loop_scene_witnesses
        )
    if packet_verifier is None:
        packet_verifier = (
            verify_exact_panel_witness_packet
            if plan.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else verify_loop_scene_packet
        )
    _assert_campaign_sources_current(plan)
    plan_artifact = _seal(plan.to_data())
    if plan_artifact["digest"] != plan.digest:
        raise RelationalHeadlessCampaignError("campaign plan digest differs")
    plan_path, reloaded_plan = _persist_artifact(
        artifact_store,
        plan_artifact,
        suffix="relational-headless-campaign-plan",
    )
    _verify_seal(reloaded_plan, CAMPAIGN_PLAN_SCHEMA)

    task_plan_paths: dict[str, Path] = {}
    for task_plan in plan.task_plans:
        artifact = _seal(task_plan.to_data())
        path, reloaded = _persist_artifact(
            artifact_store, artifact, suffix="relational-headless-plan"
        )
        _verify_seal(reloaded, PLAN_SCHEMA)
        task_plan_paths[task_plan.task_id] = path

    # No corpus root or panel path has been resolved above this line.
    _assert_campaign_sources_current(plan)
    groups = split_index.canonical_groups
    source = f"{plan.campaign_protocol_id}:plan:{plan.digest}"
    exposure_phase, exposure_actor, exposure_purpose = plan.exposure_authorization
    successor = predecessor.record(
        phase=exposure_phase,
        actor=exposure_actor,
        purpose=exposure_purpose,
        task_ids=plan.task_ids,
        source=source,
        observed_at=plan.exposure_observed_at,
        known_task_ids=(
            set(groups["train"]) | set(groups["val"]) | set(groups["test"])
        ),
        sealed_task_ids=groups["test"],
        require_unseen=True,
    )
    exposure_path, cold_successor = _persist_exposure(successor, exposure_store)
    if cold_successor != successor:
        raise RelationalHeadlessCampaignError("campaign exposure cold reload differs")

    cache = plan._cloud_policy_cache_snapshot
    outcomes: list[RelationalHeadlessOutcome] = []
    journal_records: list[dict[str, Any]] = []
    for ordinal, task_plan in enumerate(plan.task_plans):
        claim_path, terminal_journal_path = _attempt_paths(
            artifact_store,
            plan=plan,
            ordinal=ordinal,
            task_plan=task_plan,
        )
        expected_claim = _claim_content(
            plan=plan,
            successor=successor,
            ordinal=ordinal,
            task_plan=task_plan,
        )
        resumed = False
        claim: dict[str, Any] | None = None
        outcome: RelationalHeadlessOutcome | None = None
        persistence_error: str | None = None
        try:
            claim = _persist_attempt_claim(claim_path, expected_claim)
        except _AttemptAlreadyClaimed:
            # CLAIMED permanently consumes the proposer authorization.  A
            # corrupt or incomplete claim can only become infrastructure
            # failure; it can never cause another model call.
            resumed = True
            try:
                claim = _load_expected_attempt_claim(claim_path, expected_claim)
                outcome = _resume_attempt_outcome(
                    path=terminal_journal_path,
                    plan=plan,
                    ordinal=ordinal,
                    claim=claim,
                    task_plan=task_plan,
                    successor=successor,
                    plan_path=task_plan_paths[task_plan.task_id],
                    exposure_path=exposure_path,
                    artifact_store=artifact_store,
                )
                if outcome is None:
                    outcome = _terminal_failure(
                        plan=task_plan,
                        successor=successor,
                        phase="campaign-resume-incomplete-claimed-attempt",
                        error=RelationalHeadlessCampaignError(
                            "claimed task has no durable terminal; proposer "
                            "authorization is consumed"
                        ),
                        artifact_store=artifact_store,
                        plan_path=task_plan_paths[task_plan.task_id],
                        exposure_path=exposure_path,
                    )
            except Exception as exc:
                outcome = _terminal_failure(
                    plan=task_plan,
                    successor=successor,
                    phase="campaign-resume-journal-verification",
                    error=exc,
                    artifact_store=artifact_store,
                    plan_path=task_plan_paths[task_plan.task_id],
                    exposure_path=exposure_path,
                )
        except Exception as exc:
            persistence_error = type(exc).__name__ + ": " + str(exc)
            outcome = _terminal_failure(
                plan=task_plan,
                successor=successor,
                phase="campaign-attempt-claim-persistence",
                error=exc,
                artifact_store=artifact_store,
                plan_path=task_plan_paths[task_plan.task_id],
                exposure_path=exposure_path,
            )

        if claim is not None and outcome is None:
            try:
                # The fsynced CLAIMED record above is the last boundary before
                # support extraction and the sole proposer call.
                _assert_campaign_sources_current(plan)
                outcome = run_relational_headless(
                    corpus_root=corpus_root,
                    task_id=task_plan.task_id,
                    split_index=split_index,
                    predecessor=predecessor,
                    expected_corpus_digest=plan.corpus_digest,
                    expected_split_source_digest=plan.split_source_digest,
                    expected_exposure_predecessor_digest=(
                        plan.exposure_predecessor_digest
                    ),
                    seed=task_plan._seed,
                    exposure_observed_at=plan.exposure_observed_at,
                    exposure_store=exposure_store,
                    artifact_store=artifact_store,
                    expected_launcher_digest=task_plan.expected_launcher_digest,
                    release_authenticator=task_plan._release_authenticator,
                    cloud_policy_cache_snapshot=cache,
                    model=task_plan.model,
                    reasoning_effort=task_plan.reasoning_effort,
                    minutes=task_plan.minutes,
                    executable=executable,
                    verbose=verbose,
                    transport=transport,
                    png_reader=png_reader,
                    extractor=extractor,
                    packet_verifier=packet_verifier,
                    label_nonce=task_plan._label_nonce,
                    support_selection_key=task_plan._support_selection_key,
                    precommitted_exposure_successor=successor,
                    precommitted_exposure_path=exposure_path,
                    precommitted_campaign_task_ids=plan.task_ids,
                    precommitted_campaign_source=source,
                    precommitted_campaign_task_plan_digest=task_plan.digest,
                    benchmark_mode=task_plan.benchmark_mode,
                    closed_library=task_plan._closed_library,
                )
                if outcome.plan.digest != task_plan.digest:
                    raise RelationalHeadlessCampaignError(
                        "executed task plan differs from campaign freeze"
                    )
            except Exception as exc:  # preserve denominator after atomic edge
                outcome = _terminal_failure(
                    plan=task_plan,
                    successor=successor,
                    phase="campaign-task-dispatch",
                    error=exc,
                    artifact_store=artifact_store,
                    plan_path=task_plan_paths[task_plan.task_id],
                    exposure_path=exposure_path,
                )

        if outcome is None:
            raise RelationalHeadlessCampaignError(
                "campaign task did not produce a terminal outcome"
            )
        terminal_journal: dict[str, Any] | None = None
        if claim is not None:
            try:
                terminal_journal = _persist_attempt_terminal(
                    terminal_journal_path,
                    _terminal_journal_content(
                        plan=plan,
                        ordinal=ordinal,
                        claim=claim,
                        outcome=outcome,
                    ),
                )
            except Exception as exc:
                persistence_error = type(exc).__name__ + ": " + str(exc)
                try:
                    recovered = _resume_attempt_outcome(
                        path=terminal_journal_path,
                        plan=plan,
                        ordinal=ordinal,
                        claim=claim,
                        task_plan=task_plan,
                        successor=successor,
                        plan_path=task_plan_paths[task_plan.task_id],
                        exposure_path=exposure_path,
                        artifact_store=artifact_store,
                    )
                except Exception:
                    recovered = None
                if recovered is not None:
                    outcome = recovered
                    terminal_journal = _load_canonical_journal(
                        terminal_journal_path, ATTEMPT_TERMINAL_SCHEMA
                    )
                else:
                    try:
                        outcome = _terminal_failure(
                            plan=task_plan,
                            successor=successor,
                            phase="campaign-attempt-terminal-persistence",
                            error=exc,
                            artifact_store=artifact_store,
                            plan_path=task_plan_paths[task_plan.task_id],
                            exposure_path=exposure_path,
                            freeze=(
                                None
                                if outcome.artifact.get("freeze_digest") is None
                                else {
                                    "digest": outcome.artifact["freeze_digest"]
                                }
                            ),
                            freeze_path=outcome.freeze_path,
                            predictions=(
                                None
                                if outcome.artifact.get("prediction_digest") is None
                                else {
                                    "digest": outcome.artifact[
                                        "prediction_digest"
                                    ]
                                }
                            ),
                            prediction_path=outcome.prediction_path,
                            labels_revealed=(
                                outcome.status == "complete"
                                or outcome.artifact.get("query_labels_revealed")
                                is True
                            ),
                        )
                        terminal_journal = _persist_attempt_terminal(
                            terminal_journal_path,
                            _terminal_journal_content(
                                plan=plan,
                                ordinal=ordinal,
                                claim=claim,
                                outcome=outcome,
                            ),
                        )
                    except Exception as retry_exc:
                        persistence_error += (
                            "; retry "
                            + type(retry_exc).__name__
                            + ": "
                            + str(retry_exc)
                        )
        outcomes.append(outcome)
        journal_records.append(
            {
                "claim_digest": None if claim is None else claim["digest"],
                "terminal_digest": (
                    None
                    if terminal_journal is None
                    else terminal_journal["digest"]
                ),
                "resumed": resumed,
                "persistence_error": persistence_error,
            }
        )

    content = _report_content(
        plan=plan,
        successor=successor,
        outcomes=outcomes,
        journal_records=journal_records,
    )
    report = _seal(content)
    report_path, reloaded_report = _persist_artifact(
        artifact_store, report, suffix="relational-headless-campaign-run"
    )
    _verify_seal(reloaded_report, CAMPAIGN_RUN_SCHEMA)
    return RelationalHeadlessCampaignOutcome(
        plan,
        successor,
        plan_path,
        exposure_path,
        tuple(outcomes),
        reloaded_report,
        report_path,
    )


def run_relational_headless_campaign(
    *,
    corpus_root: str | Path,
    plan: RelationalHeadlessCampaignPlan,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    exposure_store: str | Path,
    artifact_store: str | Path,
    executable: str = "codex",
    verbose: bool = False,
    transport: StructuredTransport = run_codex_structured,
    png_reader: PngReader = _read_png_no_follow,
    extractor: PacketExtractor | None = None,
    packet_verifier: PacketVerifier | None = None,
) -> RelationalHeadlessCampaignOutcome:
    """Hold one OS lock across the complete campaign and all task journals."""

    artifact_store = _require_bound_artifact_store(plan, artifact_store)
    descriptor, _lock_path = _acquire_campaign_lock(artifact_store, plan)
    try:
        return _run_relational_headless_campaign_locked(
            corpus_root=corpus_root,
            plan=plan,
            split_index=split_index,
            predecessor=predecessor,
            exposure_store=exposure_store,
            artifact_store=artifact_store,
            executable=executable,
            verbose=verbose,
            transport=transport,
            png_reader=png_reader,
            extractor=extractor,
            packet_verifier=packet_verifier,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class RelationalCampaignTaskReplayInput:
    freeze: Mapping[str, Any] | None = None
    predictions: Mapping[str, Any] | None = None
    support_png_bytes: Mapping[str, bytes] = field(default_factory=dict)
    query_png_bytes: Mapping[str, bytes] = field(default_factory=dict)
    attempt_claim: Mapping[str, Any] | None = None
    attempt_terminal: Mapping[str, Any] | None = None


def _replay_support_bytes(
    freeze: Mapping[str, Any],
    support_png_bytes: Mapping[str, bytes],
    *,
    plan: RelationalHeadlessPlan,
    exposure_successor: ExposureLedger,
    packet_cache: dict[tuple[str, str], object] | None = None,
) -> None:
    verified = verify_relational_proposal_freeze(
        freeze,
        plan=plan,
        exposure_successor=exposure_successor,
    )
    entries = verified["support_entries"]
    if set(support_png_bytes) != {
        item["presentation_name"] for item in entries
    }:
        raise RelationalHeadlessCampaignError("campaign support byte set differs")
    query = (
        ClosedPanelPredicate.from_data(verified["query"])
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else RelationalVisualQuery.from_data(verified["query"])
    )
    for entry in entries:
        payload = support_png_bytes[entry["presentation_name"]]
        if (
            not isinstance(payload, bytes)
            or not payload.startswith(PNG_SIGNATURE)
            or _raw_digest(payload) != entry["source_sha256"]
            or len(payload) != entry["byte_count"]
        ):
            raise RelationalHeadlessCampaignError("campaign support bytes differ")
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
            packet = ExactPanelWitnessPacket.from_data(entry["packet"])
            cache_key = (plan.benchmark_mode, entry["source_sha256"])
            cached = None if packet_cache is None else packet_cache.get(cache_key)
            if cached is None:
                verify_exact_panel_witness_packet(packet, expected_png_bytes=payload)
                if packet_cache is not None:
                    packet_cache[cache_key] = packet
            elif cached != packet:
                raise RelationalHeadlessCampaignError(
                    "same support bytes produced different composite packets"
                )
            replay = evaluate_closed_predicate(query, packet)
        else:
            packet = LoopScenePacket.from_data(entry["packet"])
            cache_key = (plan.benchmark_mode, entry["source_sha256"])
            cached = None if packet_cache is None else packet_cache.get(cache_key)
            if cached is None:
                verify_loop_scene_packet(packet, expected_png_bytes=payload)
                if packet_cache is not None:
                    packet_cache[cache_key] = packet
            elif cached != packet:
                raise RelationalHeadlessCampaignError(
                    "same support bytes produced different loop packets"
                )
            replay = evaluate_relational_query(query, packet)
        if replay.to_data() != entry["query_result"]:
            raise RelationalHeadlessCampaignError(
                "campaign support predicate replay differs"
            )
        receipt = entry["release_panel_receipt"]
        if plan._release_authenticator.authenticate(
            receipt["relative_path"], payload
        ) != receipt:
            raise RelationalHeadlessCampaignError(
                "campaign support release receipt differs"
            )


def _replay_query_bytes(
    predictions: Mapping[str, Any],
    query_png_bytes: Mapping[str, bytes],
    *,
    plan: RelationalHeadlessPlan,
    packet_cache: dict[tuple[str, str], object] | None = None,
) -> None:
    entries = predictions["entries"]
    if set(query_png_bytes) != {item["query_id"] for item in entries}:
        raise RelationalHeadlessCampaignError("campaign query byte set differs")
    for slot, entry in enumerate(entries):
        payload = query_png_bytes[entry["query_id"]]
        if (
            not isinstance(payload, bytes)
            or not payload.startswith(PNG_SIGNATURE)
            or _raw_digest(payload) != entry["source_sha256"]
            or len(payload) != entry["byte_count"]
        ):
            raise RelationalHeadlessCampaignError("campaign query bytes differ")
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
            packet = ExactPanelWitnessPacket.from_data(entry["packet"])
            cache_key = (plan.benchmark_mode, entry["source_sha256"])
            cached = None if packet_cache is None else packet_cache.get(cache_key)
            if cached is None:
                verify_exact_panel_witness_packet(packet, expected_png_bytes=payload)
                if packet_cache is not None:
                    packet_cache[cache_key] = packet
            elif cached != packet:
                raise RelationalHeadlessCampaignError(
                    "same query bytes produced different composite packets"
                )
        else:
            packet = LoopScenePacket.from_data(entry["packet"])
            cache_key = (plan.benchmark_mode, entry["source_sha256"])
            cached = None if packet_cache is None else packet_cache.get(cache_key)
            if cached is None:
                verify_loop_scene_packet(packet, expected_png_bytes=payload)
                if packet_cache is not None:
                    packet_cache[cache_key] = packet
            elif cached != packet:
                raise RelationalHeadlessCampaignError(
                    "same query bytes produced different loop packets"
                )
        receipt = entry["release_panel_receipt"]
        if plan._release_authenticator.authenticate(
            receipt["relative_path"], payload
        ) != receipt:
            raise RelationalHeadlessCampaignError(
                "campaign query release receipt differs"
            )


def _verify_replay_attempt_journal(
    *,
    plan: RelationalHeadlessCampaignPlan,
    successor: ExposureLedger,
    ordinal: int,
    task_plan: RelationalHeadlessPlan,
    outcome: RelationalHeadlessOutcome,
    report_task: Mapping[str, Any],
    replay_input: RelationalCampaignTaskReplayInput,
) -> dict[str, Any]:
    claim_digest = report_task.get("attempt_claim_digest")
    terminal_digest = report_task.get("attempt_terminal_digest")
    resumed = report_task.get("attempt_resumed")
    persistence_error = report_task.get("attempt_journal_persistence_error")
    if not isinstance(resumed, bool) or (
        persistence_error is not None and not isinstance(persistence_error, str)
    ):
        raise RelationalHeadlessCampaignError(
            "campaign attempt journal report is malformed"
        )
    if claim_digest is None:
        if (
            terminal_digest is not None
            or replay_input.attempt_claim is not None
            or replay_input.attempt_terminal is not None
            or persistence_error is None
        ):
            raise RelationalHeadlessCampaignError(
                "campaign missing-claim journal chain differs"
            )
    else:
        _require_sha256(claim_digest, "campaign attempt claim digest")
        if replay_input.attempt_claim is None:
            raise RelationalHeadlessCampaignError(
                "campaign attempt claim replay is absent"
            )
        claim = _verify_seal(replay_input.attempt_claim, ATTEMPT_CLAIM_SCHEMA)
        expected_claim = _claim_content(
            plan=plan,
            successor=successor,
            ordinal=ordinal,
            task_plan=task_plan,
        )
        if claim != expected_claim or claim["digest"] != claim_digest:
            raise RelationalHeadlessCampaignError(
                "campaign attempt claim replay differs"
            )
        if terminal_digest is None:
            if replay_input.attempt_terminal is not None or persistence_error is None:
                raise RelationalHeadlessCampaignError(
                    "campaign missing-terminal journal chain differs"
                )
        else:
            _require_sha256(terminal_digest, "campaign attempt terminal digest")
            if replay_input.attempt_terminal is None:
                raise RelationalHeadlessCampaignError(
                    "campaign attempt terminal replay is absent"
                )
            terminal = _verify_seal(
                replay_input.attempt_terminal, ATTEMPT_TERMINAL_SCHEMA
            )
            expected_terminal = _terminal_journal_content(
                plan=plan,
                ordinal=ordinal,
                claim=claim,
                outcome=outcome,
            )
            if terminal != expected_terminal or terminal["digest"] != terminal_digest:
                raise RelationalHeadlessCampaignError(
                    "campaign attempt terminal replay differs"
                )
    return {
        "claim_digest": claim_digest,
        "terminal_digest": terminal_digest,
        "resumed": resumed,
        "persistence_error": persistence_error,
    }


def cold_replay_relational_headless_campaign(
    *,
    plan: RelationalHeadlessCampaignPlan,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    exposure_successor: ExposureLedger,
    task_outcomes: Sequence[RelationalHeadlessOutcome],
    campaign_run: Mapping[str, Any],
    replay_inputs: Mapping[str, RelationalCampaignTaskReplayInput],
) -> Mapping[str, Any]:
    """Reproduce selection, batch edge, task terminals, and aggregate in Python."""

    _verify_campaign_plan_reproduction(
        plan=plan, split_index=split_index, predecessor=predecessor
    )
    source = f"{plan.campaign_protocol_id}:plan:{plan.digest}"
    if (
        len(exposure_successor.events) != len(predecessor.events) + 1
        or exposure_successor.events[:-1] != predecessor.events
        or exposure_successor.corpus_digest != predecessor.corpus_digest
    ):
        raise RelationalHeadlessCampaignError("campaign exposure chain differs")
    event = exposure_successor.events[-1]
    exposure_phase, exposure_actor, exposure_purpose = plan.exposure_authorization
    if (
        event.phase != exposure_phase
        or event.actor != exposure_actor
        or event.purpose != exposure_purpose
        or event.task_ids != tuple(sorted(plan.task_ids))
        or event.panel_ids
        or event.source != source
        or event.observed_at != plan.exposure_observed_at
    ):
        raise RelationalHeadlessCampaignError("campaign atomic edge differs")
    if tuple(item.plan.task_id for item in task_outcomes) != plan.task_ids:
        raise RelationalHeadlessCampaignError("campaign replay order differs")
    if set(replay_inputs) != set(plan.task_ids):
        raise RelationalHeadlessCampaignError("campaign replay inventory differs")
    verified_run = _verify_seal(campaign_run, CAMPAIGN_RUN_SCHEMA)
    report_tasks = verified_run.get("tasks")
    if not isinstance(report_tasks, list) or len(report_tasks) != len(task_outcomes):
        raise RelationalHeadlessCampaignError(
            "campaign attempt journal inventory differs"
        )

    complete_replayed = 0
    support_rejections_replayed = 0
    terminal_integrity = 0
    packet_cache: dict[tuple[str, str], object] = {}
    journal_records: list[dict[str, Any]] = []
    for ordinal, (task_plan, outcome, report_task) in enumerate(
        zip(plan.task_plans, task_outcomes, report_tasks, strict=True)
    ):
        if (
            outcome.plan.digest != task_plan.digest
            or outcome.exposure_successor.digest != exposure_successor.digest
        ):
            raise RelationalHeadlessCampaignError("campaign task chain differs")
        replay_input = replay_inputs[task_plan.task_id]
        if not isinstance(report_task, Mapping):
            raise RelationalHeadlessCampaignError(
                "campaign attempt journal record is malformed"
            )
        journal_records.append(
            _verify_replay_attempt_journal(
                plan=plan,
                successor=exposure_successor,
                ordinal=ordinal,
                task_plan=task_plan,
                outcome=outcome,
                report_task=report_task,
                replay_input=replay_input,
            )
        )
        if outcome.status == "complete":
            _verify_seal(outcome.artifact, RUN_SCHEMA)
            if replay_input.freeze is None or replay_input.predictions is None:
                raise RelationalHeadlessCampaignError(
                    "complete campaign task lacks replay artifacts"
                )
            cold_replay_relational_headless_run(
                plan=task_plan,
                exposure_successor=exposure_successor,
                freeze=replay_input.freeze,
                predictions=replay_input.predictions,
                final_run=outcome.artifact,
                support_png_bytes=replay_input.support_png_bytes,
                query_png_bytes=replay_input.query_png_bytes,
                release_authenticator=task_plan._release_authenticator,
            )
            complete_replayed += 1
        elif outcome.status == "support_rejected":
            final = _verify_seal(outcome.artifact, RUN_SCHEMA)
            if (
                replay_input.freeze is None
                or replay_input.predictions is not None
                or replay_input.query_png_bytes
            ):
                raise RelationalHeadlessCampaignError(
                    "support rejection replay inventory differs"
                )
            freeze = verify_relational_proposal_freeze(
                replay_input.freeze,
                plan=task_plan,
                exposure_successor=exposure_successor,
            )
            if (
                final.get("status") != "support_rejected"
                or final.get("plan_digest") != task_plan.digest
                or final.get("exposure_successor_digest")
                != exposure_successor.digest
                or freeze["support_gate_accepted"] is not False
                or final.get("freeze_digest") != freeze["digest"]
                or final.get("prediction_digest") is not None
                or final.get("query_paths_resolved") is not False
                or final.get("query_pixels_opened") is not False
                or final.get("query_labels_revealed") is not False
                or final.get("reroll_attempted") is not False
            ):
                raise RelationalHeadlessCampaignError(
                    "support rejection chain differs"
                )
            _replay_support_bytes(
                freeze,
                replay_input.support_png_bytes,
                plan=task_plan,
                exposure_successor=exposure_successor,
                packet_cache=packet_cache,
            )
            support_rejections_replayed += 1
        elif outcome.status == "terminal_failure":
            failure = _verify_seal(outcome.artifact, FAILURE_SCHEMA)
            if (
                failure.get("status") != "terminal_failure"
                or failure.get("plan_digest") != task_plan.digest
                or failure.get("exposure_successor_digest")
                != exposure_successor.digest
                or failure.get("reroll_attempted") is not False
                or not isinstance(failure.get("query_labels_revealed"), bool)
            ):
                raise RelationalHeadlessCampaignError(
                    "terminal task chain or policy differs"
                )
            frozen_digest = failure.get("freeze_digest")
            prediction_digest = failure.get("prediction_digest")
            if (replay_input.freeze is None) != (frozen_digest is None):
                raise RelationalHeadlessCampaignError(
                    "terminal freeze presence differs"
                )
            if (replay_input.predictions is None) != (
                prediction_digest is None
            ):
                raise RelationalHeadlessCampaignError(
                    "terminal prediction presence differs"
                )
            if prediction_digest is not None and frozen_digest is None:
                raise RelationalHeadlessCampaignError(
                    "terminal prediction lacks a frozen proposal"
                )
            if failure.get("query_labels_revealed") and prediction_digest is None:
                raise RelationalHeadlessCampaignError(
                    "terminal label reveal lacks committed predictions"
                )
            if replay_input.freeze is None and (
                replay_input.support_png_bytes or replay_input.query_png_bytes
            ):
                raise RelationalHeadlessCampaignError(
                    "pre-freeze terminal contains replay pixel inventory"
                )
            if replay_input.predictions is None and replay_input.query_png_bytes:
                raise RelationalHeadlessCampaignError(
                    "pre-prediction terminal contains query byte inventory"
                )
            if replay_input.freeze is not None:
                freeze = verify_relational_proposal_freeze(
                    replay_input.freeze,
                    plan=task_plan,
                    exposure_successor=exposure_successor,
                )
                if failure.get("freeze_digest") != freeze["digest"]:
                    raise RelationalHeadlessCampaignError(
                        "terminal freeze chain differs"
                    )
                _replay_support_bytes(
                    freeze,
                    replay_input.support_png_bytes,
                    plan=task_plan,
                    exposure_successor=exposure_successor,
                    packet_cache=packet_cache,
                )
                if replay_input.predictions is not None:
                    predictions = verify_relational_predictions(
                        replay_input.predictions,
                        freeze=freeze,
                        plan=task_plan,
                        exposure_successor=exposure_successor,
                    )
                    if prediction_digest != predictions["digest"]:
                        raise RelationalHeadlessCampaignError(
                            "terminal prediction chain differs"
                        )
                    _replay_query_bytes(
                        predictions,
                        replay_input.query_png_bytes,
                        plan=task_plan,
                        packet_cache=packet_cache,
                    )
            terminal_integrity += 1
        else:
            raise RelationalHeadlessCampaignError(
                f"unknown campaign task status {outcome.status!r}"
            )

    expected_content = _report_content(
        plan=plan,
        successor=exposure_successor,
        outcomes=task_outcomes,
        journal_records=journal_records,
    )
    if dict(verified_run) != _seal(expected_content):
        raise RelationalHeadlessCampaignError("campaign aggregate does not reproduce")
    return _seal(
        {
            "schema": CAMPAIGN_REPLAY_SCHEMA,
            "campaign_protocol_id": plan.campaign_protocol_id,
            "campaign_plan_digest": plan.digest,
            "campaign_run_digest": verified_run["digest"],
            "exposure_successor_digest": exposure_successor.digest,
            "task_count": len(plan.task_ids),
            "complete_runs_replayed": complete_replayed,
            "support_rejections_replayed": support_rejections_replayed,
            "terminal_failures_integrity_verified": terminal_integrity,
            "all_tasks_accounted_for": True,
            "proposer_or_model_called_during_replay": False,
            "source_identities": {
                "campaign_python_source_digest": (
                    plan.campaign_python_source_digest
                ),
                "runner_python_source_digest": plan.runner_python_source_digest,
                "task_protocol_digest_set": sorted(
                    {item.protocol_digest for item in plan.task_plans}
                ),
                "python_only": True,
                "lean_required": False,
                "semantic_checker_imported": False,
            },
            "predicate_authority": (
                "pure-python-closed-visual-predicate-union"
                if plan.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
                else "pure-python-relational-query"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class DurableRelationalCampaignReplay:
    """Persisted model-free replay receipt for one durable campaign outcome."""

    receipt: Mapping[str, Any]
    receipt_path: Path

    def to_data(self) -> dict[str, object]:
        return {
            "cold_replay_digest": self.receipt["digest"],
            "cold_replay_path": str(self.receipt_path),
            "proposer_or_model_called_during_replay": False,
        }


def _authenticated_replay_png(
    *,
    corpus_root: str | Path,
    task_plan: RelationalHeadlessPlan,
    receipt: object,
    png_reader: PngReader,
) -> bytes:
    if not isinstance(receipt, Mapping):
        raise RelationalHeadlessCampaignError(
            "durable replay panel receipt is not an object"
        )
    relative_path = receipt.get("relative_path")
    if not isinstance(relative_path, str):
        raise RelationalHeadlessCampaignError(
            "durable replay panel receipt lacks a relative path"
        )
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RelationalHeadlessCampaignError(
            "durable replay panel relative path is unsafe"
        )
    root = Path(os.path.abspath(os.path.expanduser(str(corpus_root))))
    payload, authenticated = _read_authenticated_release_panel(
        corpus_root=root,
        path=root.joinpath(*relative.parts),
        authenticator=task_plan._release_authenticator,
        observer_reader=png_reader,
    )
    if authenticated != dict(receipt):
        raise RelationalHeadlessCampaignError(
            "durable replay panel receipt differs from exact release bytes"
        )
    return payload


def cold_replay_durable_relational_headless_campaign(
    *,
    corpus_root: str | Path,
    artifact_store: str | Path,
    campaign_outcome: RelationalHeadlessCampaignOutcome,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    png_reader: PngReader = _read_png_no_follow,
) -> DurableRelationalCampaignReplay:
    """Cold-reload a campaign and exact PNGs, replay it, and persist a receipt.

    This API has deliberately no transport, executable, model, retry, or
    proposer parameter. Every artifact and panel is reopened through the
    hardened no-follow readers before the model-free replay is invoked.
    """

    if not isinstance(campaign_outcome, RelationalHeadlessCampaignOutcome):
        raise TypeError("campaign_outcome must be a campaign outcome")
    if not callable(png_reader):
        raise TypeError("png_reader must be callable")
    plan = campaign_outcome.plan
    artifact_store = _require_bound_artifact_store(plan, artifact_store)
    plan_artifact = load_relational_artifact(campaign_outcome.plan_path)
    if plan_artifact != _seal(plan.to_data()):
        raise RelationalHeadlessCampaignError(
            "durable campaign plan differs before cold replay"
        )
    successor = ExposureLedger.load(campaign_outcome.exposure_path)
    if successor != campaign_outcome.exposure_successor:
        raise RelationalHeadlessCampaignError(
            "durable campaign exposure differs before cold replay"
        )
    campaign_run = load_relational_artifact(campaign_outcome.report_path)
    if campaign_run != dict(campaign_outcome.report):
        raise RelationalHeadlessCampaignError(
            "durable campaign report differs before cold replay"
        )
    report_tasks = campaign_run.get("tasks")
    if not isinstance(report_tasks, list) or len(report_tasks) != len(
        plan.task_plans
    ):
        raise RelationalHeadlessCampaignError(
            "durable campaign report task inventory differs"
        )

    replay_inputs: dict[str, RelationalCampaignTaskReplayInput] = {}
    cold_outcomes: list[RelationalHeadlessOutcome] = []
    for ordinal, (task_plan, outcome, report_task) in enumerate(
        zip(
            plan.task_plans,
            campaign_outcome.task_outcomes,
            report_tasks,
            strict=True,
        )
    ):
        if not isinstance(report_task, Mapping):
            raise RelationalHeadlessCampaignError(
                "durable campaign report task is malformed"
            )
        task_plan_artifact = load_relational_artifact(outcome.plan_path)
        if task_plan_artifact != _seal(task_plan.to_data()):
            raise RelationalHeadlessCampaignError(
                "durable task plan differs before cold replay"
            )
        terminal = load_relational_artifact(outcome.terminal_path)
        if terminal != dict(outcome.artifact):
            raise RelationalHeadlessCampaignError(
                "durable task terminal differs before cold replay"
            )
        freeze_digest = terminal.get("freeze_digest")
        prediction_digest = terminal.get("prediction_digest")
        freeze = None
        predictions = None
        if freeze_digest is not None:
            if outcome.freeze_path is None:
                raise RelationalHeadlessCampaignError(
                    "durable task terminal names an unavailable freeze"
                )
            freeze = load_relational_artifact(outcome.freeze_path)
            if freeze.get("digest") != freeze_digest:
                raise RelationalHeadlessCampaignError(
                    "durable task freeze digest differs"
                )
        elif outcome.freeze_path is not None:
            raise RelationalHeadlessCampaignError(
                "durable task has an unreferenced freeze path"
            )
        if prediction_digest is not None:
            if outcome.prediction_path is None or freeze is None:
                raise RelationalHeadlessCampaignError(
                    "durable task terminal names unavailable predictions"
                )
            predictions = load_relational_artifact(outcome.prediction_path)
            if predictions.get("digest") != prediction_digest:
                raise RelationalHeadlessCampaignError(
                    "durable task prediction digest differs"
                )
        elif outcome.prediction_path is not None:
            raise RelationalHeadlessCampaignError(
                "durable task has an unreferenced prediction path"
            )

        support_png_bytes: dict[str, bytes] = {}
        if freeze is not None:
            support_entries = freeze.get("support_entries")
            if not isinstance(support_entries, list):
                raise RelationalHeadlessCampaignError(
                    "durable freeze support inventory is malformed"
                )
            for entry in support_entries:
                if not isinstance(entry, Mapping) or not isinstance(
                    entry.get("presentation_name"), str
                ):
                    raise RelationalHeadlessCampaignError(
                        "durable freeze support entry is malformed"
                    )
                support_png_bytes[entry["presentation_name"]] = (
                    _authenticated_replay_png(
                        corpus_root=corpus_root,
                        task_plan=task_plan,
                        receipt=entry.get("release_panel_receipt"),
                        png_reader=png_reader,
                    )
                )
        query_png_bytes: dict[str, bytes] = {}
        if predictions is not None:
            prediction_entries = predictions.get("entries")
            if not isinstance(prediction_entries, list):
                raise RelationalHeadlessCampaignError(
                    "durable prediction inventory is malformed"
                )
            for entry in prediction_entries:
                if not isinstance(entry, Mapping) or not isinstance(
                    entry.get("query_id"), str
                ):
                    raise RelationalHeadlessCampaignError(
                        "durable prediction entry is malformed"
                    )
                query_png_bytes[entry["query_id"]] = _authenticated_replay_png(
                    corpus_root=corpus_root,
                    task_plan=task_plan,
                    receipt=entry.get("release_panel_receipt"),
                    png_reader=png_reader,
                )

        claim_path, terminal_journal_path = _attempt_paths(
            artifact_store,
            plan=plan,
            ordinal=ordinal,
            task_plan=task_plan,
        )
        claim = None
        terminal_journal = None
        if report_task.get("attempt_claim_digest") is not None:
            claim = _load_canonical_journal(claim_path, ATTEMPT_CLAIM_SCHEMA)
        if report_task.get("attempt_terminal_digest") is not None:
            terminal_journal = _load_canonical_journal(
                terminal_journal_path, ATTEMPT_TERMINAL_SCHEMA
            )
        replay_inputs[task_plan.task_id] = RelationalCampaignTaskReplayInput(
            freeze=freeze,
            predictions=predictions,
            support_png_bytes=support_png_bytes,
            query_png_bytes=query_png_bytes,
            attempt_claim=claim,
            attempt_terminal=terminal_journal,
        )
        cold_outcomes.append(
            RelationalHeadlessOutcome(
                outcome.status,
                task_plan,
                successor,
                outcome.plan_path,
                campaign_outcome.exposure_path,
                outcome.terminal_path,
                terminal,
                freeze_path=outcome.freeze_path,
                prediction_path=outcome.prediction_path,
            )
        )

    receipt = cold_replay_relational_headless_campaign(
        plan=plan,
        split_index=split_index,
        predecessor=predecessor,
        exposure_successor=successor,
        task_outcomes=tuple(cold_outcomes),
        campaign_run=campaign_run,
        replay_inputs=replay_inputs,
    )
    receipt_path, cold_receipt = _persist_artifact(
        artifact_store,
        receipt,
        suffix="relational-headless-campaign-replay",
    )
    _verify_seal(cold_receipt, CAMPAIGN_REPLAY_SCHEMA)
    return DurableRelationalCampaignReplay(cold_receipt, receipt_path)


def verify_relational_headless_campaign_plan_artifact(
    artifact: Mapping[str, Any],
    *,
    plan: RelationalHeadlessCampaignPlan,
) -> Mapping[str, Any]:
    """Require one committed plan artifact to equal fresh metadata selection."""

    verified = _verify_seal(artifact, CAMPAIGN_PLAN_SCHEMA)
    if dict(verified) != _seal(plan.to_data()):
        raise RelationalHeadlessCampaignError(
            "committed campaign plan differs from metadata-only reproduction"
        )
    return artifact


def write_relational_headless_campaign_plan(
    path: str | Path, *, plan: RelationalHeadlessCampaignPlan
) -> Path:
    """Durably create one canonical public plan; never serialize its secret."""

    artifact = _seal(plan.to_data())
    payload = canonical_json(artifact) + b"\n"
    if plan._schedule_secret.encode("ascii") in payload:
        raise RelationalHeadlessCampaignError(
            "private schedule secret leaked into public campaign plan"
        )
    return _write_once_durable(Path(path), payload)


def export_cloud_policy_cache_snapshot(path: str | Path) -> Mapping[str, object]:
    """Freeze the current signed cache outside git for later plan execution."""

    snapshot = snapshot_cloud_policy_cache()
    if snapshot.data is None:
        raise RelationalHeadlessCampaignError(
            "no live signed cloud-policy cache is available to export"
        )
    destination = _write_once_durable(Path(path), snapshot.data)
    if _stable_read(destination, maximum=1024 * 1024) != snapshot.data:
        raise RelationalHeadlessCampaignError("exported policy cache differs")
    return {
        "path": str(destination),
        "binding": snapshot.binding,
        "byte_count": len(snapshot.data),
    }


def generate_private_schedule_secret(path: str | Path) -> Mapping[str, object]:
    """Generate one 256-bit schedule root in a mode-0600 write-once file."""

    secret = secrets.token_hex(32)
    payload = (secret + "\n").encode("ascii")
    destination = _write_once_durable(Path(path), payload)
    if _stable_read(destination, maximum=4096) != payload:
        raise RelationalHeadlessCampaignError(
            "private schedule secret cold reload differs"
        )
    return {
        "path": str(destination),
        "schedule_secret_digest": canonical_digest(
            {
                "schema": "gkm.bongard-relational-campaign-schedule-secret.v1",
                "schedule_secret": secret,
            }
        ),
        "byte_count": len(payload),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bongard.relational_headless_campaign",
        description=(
            "Verify, or explicitly execute, a committed full-current strict-DEV "
            "relational campaign. Verification is metadata-only."
        ),
    )
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--ledger-in", required=True, type=Path)
    plan_mode = parser.add_mutually_exclusive_group(required=True)
    plan_mode.add_argument("--plan-file", type=Path)
    plan_mode.add_argument("--write-plan", type=Path)
    parser.add_argument("--expected-plan-digest")
    parser.add_argument("--expected-release-digest", required=True)
    parser.add_argument("--release-descriptor-file", required=True, type=Path)
    parser.add_argument("--release-archive", required=True, type=Path)
    parser.add_argument("--expected-corpus-digest", required=True)
    parser.add_argument("--expected-split-digest", required=True)
    parser.add_argument("--expected-ledger-digest", required=True)
    parser.add_argument("--campaign-seed", required=True)
    parser.add_argument(
        "--campaign-mode",
        choices=(STRICT_DEV_MODE, EXACT_UNUSED_TRAIN_ENGINEERING_MODE),
        default=STRICT_DEV_MODE,
        help=(
            "explicit campaign admission/predicate mode; strict DEV remains "
            "the default"
        ),
    )
    parser.add_argument("--selection-seed-provenance", required=True)
    parser.add_argument("--schedule-secret-file", required=True, type=Path)
    parser.add_argument("--exposure-observed-at", required=True)
    parser.add_argument("--expected-task-count", required=True, type=int)
    parser.add_argument("--expected-codex-launcher-sha256", required=True)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", default=15, type=int)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--cloud-policy-cache-policy",
        choices=("absent", "snapshot"),
        default="absent",
    )
    cache_mode.add_argument("--cloud-policy-cache-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--exposure-store", type=Path)
    parser.add_argument(
        "--artifact-store",
        required=True,
        type=Path,
        help=(
            "artifact and attempt-journal root precommitted into every public plan"
        ),
    )
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    split_index = SplitIndex.load(args.split_file)
    predecessor = ExposureLedger.load(args.ledger_in)
    if args.cloud_policy_cache_file is not None:
        cache = CloudPolicyCacheSnapshot(
            _stable_read(args.cloud_policy_cache_file, maximum=1024 * 1024)
        )
    else:
        cache = (
            CloudPolicyCacheSnapshot(None)
            if args.cloud_policy_cache_policy == "absent"
            else snapshot_cloud_policy_cache()
        )
    schedule_secret = _read_private_schedule_secret(args.schedule_secret_file)
    release_authenticator = ReleaseArchiveAuthenticator.load(
        release_descriptor_path=args.release_descriptor_file,
        expected_release_descriptor_digest=args.expected_release_digest,
        archive_path=args.release_archive,
    )
    prepare_campaign = (
        prepare_exact_unused_train_engineering_campaign
        if args.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else prepare_full_current_dev_campaign
    )
    plan = prepare_campaign(
        artifact_store=args.artifact_store,
        split_index=split_index,
        predecessor=predecessor,
        expected_release_descriptor_digest=args.expected_release_digest,
        release_authenticator=release_authenticator,
        expected_corpus_digest=args.expected_corpus_digest,
        expected_split_source_digest=args.expected_split_digest,
        expected_exposure_predecessor_digest=args.expected_ledger_digest,
        campaign_seed=args.campaign_seed,
        selection_seed_provenance=args.selection_seed_provenance,
        schedule_secret=schedule_secret,
        exposure_observed_at=args.exposure_observed_at,
        expected_task_count=args.expected_task_count,
        expected_launcher_digest=args.expected_codex_launcher_sha256,
        cloud_policy_cache_snapshot=cache,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        minutes=args.minutes,
    )
    if args.write_plan is not None:
        artifact = _seal(plan.to_data())
        if (
            args.expected_plan_digest is not None
            and artifact["digest"] != args.expected_plan_digest
        ):
            raise RelationalHeadlessCampaignError(
                "generated campaign plan differs from optional digest pin"
            )
        path = write_relational_headless_campaign_plan(args.write_plan, plan=plan)
        print(
            json.dumps(
                {
                    "status": "written-metadata-only",
                    "campaign_plan_digest": artifact["digest"],
                    "task_count": len(plan.task_ids),
                    "path": str(path),
                    "pixels_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.expected_plan_digest is None:
        raise RelationalHeadlessCampaignError(
            "verification/execution requires --expected-plan-digest"
        )
    try:
        plan_bytes = _stable_read(args.plan_file)
        raw_plan = json.loads(plan_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RelationalHeadlessCampaignError(
            "cannot load committed campaign plan"
        ) from exc
    if (
        not isinstance(raw_plan, dict)
        or canonical_json(raw_plan) + b"\n" != plan_bytes
    ):
        raise RelationalHeadlessCampaignError(
            "committed campaign plan is not canonical JSON"
        )
    verify_relational_headless_campaign_plan_artifact(raw_plan, plan=plan)
    if raw_plan["digest"] != args.expected_plan_digest:
        raise RelationalHeadlessCampaignError(
            "committed campaign plan digest differs from command pin"
        )
    _require_bound_artifact_store(plan, args.artifact_store)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "verified-metadata-only",
                    "campaign_plan_digest": plan.digest,
                    "task_count": len(plan.task_ids),
                    "pixels_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if any(
        item is None
        for item in (args.corpus_root, args.exposure_store)
    ):
        raise RelationalHeadlessCampaignError(
            "--execute requires --corpus-root and --exposure-store"
        )
    outcome = run_relational_headless_campaign(
        corpus_root=args.corpus_root,
        plan=plan,
        split_index=split_index,
        predecessor=predecessor,
        exposure_store=args.exposure_store,
        artifact_store=args.artifact_store,
        executable=args.executable,
        verbose=args.verbose,
    )
    durable_replay = cold_replay_durable_relational_headless_campaign(
        corpus_root=args.corpus_root,
        artifact_store=args.artifact_store,
        campaign_outcome=outcome,
        split_index=split_index,
        predecessor=predecessor,
    )
    print(
        json.dumps(
            {**outcome.to_data(), "cold_replay": durable_replay.to_data()},
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ARTIFACT_STORE_NORMALIZATION",
    "CAMPAIGN_PLAN_SCHEMA",
    "CAMPAIGN_PROTOCOL_ID",
    "CAMPAIGN_REPLAY_SCHEMA",
    "CAMPAIGN_RUN_SCHEMA",
    "ENGINEERING_CAMPAIGN_PROTOCOL_ID",
    "DurableRelationalCampaignReplay",
    "RelationalCampaignTaskReplayInput",
    "RelationalHeadlessCampaignError",
    "RelationalHeadlessCampaignOutcome",
    "RelationalHeadlessCampaignPlan",
    "cold_replay_relational_headless_campaign",
    "cold_replay_durable_relational_headless_campaign",
    "export_cloud_policy_cache_snapshot",
    "generate_private_schedule_secret",
    "main",
    "prepare_full_current_dev_campaign",
    "prepare_exact_unused_train_engineering_campaign",
    "relational_headless_campaign_source_digest",
    "run_relational_headless_campaign",
    "verify_relational_headless_campaign_plan_artifact",
    "write_relational_headless_campaign_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
