"""Stage-B end-to-end validation of the frozen visual-semantic pipeline.

Stage A estimates the soft scorer conditional on a proposer having emitted a
soft claim.  This module measures the missing deployment estimand on a fresh,
metadata-only ``v2-clean`` DEV cohort: proposal attrition, exact 12/12 support
gate coverage, and query behaviour conditional on passing that gate.

The public plan is created before selected PNGs or transports are touched.  It
contains the raw seed, requested count, exact semantic-cluster-disjoint task
selection, the complete Stage-A campaign and execution identities, the source
corpus/split commitments, the exposure-ledger predecessor, and all acceptance
thresholds.  Execution first records *all* selected tasks in one append-only
ledger transition, then uses the deployed ``prepare_episode`` ->
``VisualSemanticEpisode`` -> ``run_episode`` path with
``SupportGatePolicy.visual_semantic()``.

Python is the sole reference implementation for selection, predicate
evaluation, statistics, replay, and artifact identity.  A detached optional
checker may be added by a caller, but it is deliberately absent from every
digest and can never change a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import canonical_digest, canonical_json, verify_archive_data
from bongard.benchmark import (
    EpisodeStatus,
    SupportGatePolicy,
    SupportGateResult,
    VISUAL_SEMANTIC_PREDICATE_MODE,
    _derive_hex,
    prepare_episode,
    run_episode,
)
from bongard.cohorts import build_cohort_report, parse_official_task_id
from bongard.corpus import CorpusManifest, FAMILIES, ShapeBongardCorpus
from bongard.exposure import (
    ExposureLedger,
    ExposureViolation,
    SemanticExposureResolution,
    basic_morphology_cluster_id,
    semantic_policy_blocked_keys,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_SELECTION_ALGORITHM,
    SemanticCalibrationCampaignArtifact,
    SemanticCalibrationExecutionConfig,
    SemanticCalibrationProposalArchive,
    select_semantic_calibration_tasks,
    semantic_generator_cluster_id,
    verify_semantic_campaign_against_corpus,
)
from bongard.semantic_calibration_command import (
    STAGE_A_COMMAND_RECEIPT_SCHEMA,
    StageACommandReceipt,
    freeze_stage_a_source_dependencies,
)
from bongard.semantic_episode import VisualSemanticEpisode
from bongard.semantic_policy import VisualSemanticPolicy
from bongard.semantic_run_verification import (
    VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA,
    VisualSemanticCalibrationCampaignAnchor,
    VisualSemanticRunVerification,
    VisualSemanticRunVerificationError,
    _build_visual_semantic_run_record_from_verified_anchor,
    _verify_visual_semantic_run_data_with_verified_anchor,
)
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_structured,
)


GATED_DEV_PLAN_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-plan.v2"
GATED_DEV_ARTIFACT_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-validation.v2"
GATED_DEV_POLICY_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-policy.v1"
GATED_DEV_SELECTION_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-selection.v1"
GATED_DEV_SUMMARY_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-summary.v1"
GATED_DEV_TASK_RUN_SCHEMA = "gkm.bongard-visual-semantic-gated-dev-task-run.v2"
GATED_DEV_TASK_REPLAY_RECEIPT_SCHEMA = (
    "gkm.bongard-visual-semantic-gated-dev-task-replay-receipt.v1"
)
GATED_DEV_BOUND_SCHEMA = "gkm.bongard-raw-simultaneous-hoeffding-bound.v1"

REFERENCE_SEMANTICS = "pure-python-gated-dev-validation/v2"
CENTRAL_CAMPAIGN_LAYOUT = (
    "one-full-stage-a-campaign-plus-successful-receipt-digest-and-"
    "per-task-campaign-calibration-digest-references/v1"
)
ESTIMAND = (
    "selected-cluster gate coverage plus query outcomes conditional on an "
    "aligned exact visual-semantic support gate/v1"
)
BOUND_METHOD = "raw-two-sided-bonferroni-hoeffding-over-generator-clusters/v1"
GATED_DEV_SELECTION_ALGORITHM = (
    CAMPAIGN_SELECTION_ALGORITHM
    + "+ledger-and-batch-constituent-attribute-disjoint-hd/v2"
)
METRIC_NAMES = (
    "selected_gate_coverage",
    "gated_both_query_correct",
    "gated_fully_determinate",
    "gated_any_abstention",
    "gated_any_error",
)
REPLAYABLE_TERMINAL_STATUSES = (
    EpisodeStatus.COMPLETE.value,
    EpisodeStatus.SUPPORT_REJECTED.value,
    EpisodeStatus.PROPOSAL_ERROR.value,
)
DEFAULT_POWER_NOTE = (
    "Against the complete A2 exposure ledger, the remaining full-ledger-"
    "disjoint DEV maximum is N=16 (BD=16, HD=0), below the frozen "
    "minimum_selected_clusters=24 gate.  Any N=16 execution is a descriptive "
    "pilot only and cannot authorize SEALED."
)
INFERENCE_MODE = "descriptive-only-pending-family-stratified-power-audit/v1"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")

StructuredTransport = Callable[..., Any]


class GatedDevValidationError(ValueError):
    """A Stage-B plan, run, or cold replay failed closed."""


class GatedDevTransportIdentityError(GatedDevValidationError):
    """A successful Codex receipt differs from the Stage-A environment."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise GatedDevValidationError(f"{label} must be an object")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> Mapping[str, Any]:
    actual = set(value)
    if actual != expected:
        raise GatedDevValidationError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GatedDevValidationError(f"{label} must be a list")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise GatedDevValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise GatedDevValidationError(f"{label} must be a sha256: content address")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise GatedDevValidationError(f"invalid {label} {value!r}")
    return value


def _probability(value: object, label: str, *, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GatedDevValidationError(f"{label} must be numeric")
    frozen = float(value)
    lower_ok = frozen > 0.0 if strict else frozen >= 0.0
    upper_ok = frozen < 1.0 if strict else frozen <= 1.0
    if not math.isfinite(frozen) or not lower_ok or not upper_ok:
        brackets = "(0, 1)" if strict else "[0, 1]"
        raise GatedDevValidationError(f"{label} must lie in {brackets}")
    return frozen


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GatedDevValidationError(f"{label} must be a positive integer")
    return value


def _manifest_digest(manifest: CorpusManifest) -> str:
    if not isinstance(manifest, CorpusManifest):
        raise TypeError("source_corpus_manifest must be CorpusManifest")
    expected = "sha256:" + canonical_digest(manifest.content_dict())
    if manifest.digest != expected:
        raise GatedDevValidationError("source corpus manifest digest is invalid")
    return expected


def _disclosure_tokens(family: str, concepts: Sequence[str]) -> tuple[str, ...]:
    if family == "bd":
        return tuple(
            sorted(
                {
                    token
                    for concept in concepts
                    for token in (
                        "basic_family:" + concept,
                        "basic_morphology:" + basic_morphology_cluster_id(concept),
                    )
                }
            )
        )
    if family == "hd":
        return tuple(
            sorted(
                {"abstract_pair:" + "\0".join(concepts)}
                | {"abstract_attribute:" + concept for concept in concepts}
            )
        )
    return ("freeform_family:" + "\0".join(concepts),)


def _blocked_policy_digest(
    historical_seed_digest: str,
    resolver_digest: str,
    blocked_clusters: Sequence[str],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-stage-b-basic-morphology-block-policy.v1",
            "historical_seed_digest": historical_seed_digest,
            "resolver_policy_digest": resolver_digest,
            "blocked_clusters": list(blocked_clusters),
        }
    )


def _rank(seed: str, domain: str, value: object) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-semantic-campaign-ranking.v1",
            "seed": seed,
            "domain": domain,
            "value": value,
        }
    )


def _maximum_disjoint_candidates(
    candidates: Sequence[Any], *, seed: str, family: str
) -> tuple[Any, ...]:
    """Exact maximum-cardinality token packing with a public hash tie-break.

    DEV HD has fewer than twenty constituent attributes.  A memoized bitmask
    matching is therefore both simpler and safer than a seed-ranked greedy
    pass: changing the seed can choose another maximum matching but can never
    manufacture a larger admissible reservoir.
    """

    frozen = tuple(candidates)
    if not frozen:
        return ()
    token_sets = tuple(
        frozenset(_disclosure_tokens(item.family, item.concepts)) for item in frozen
    )
    # Pair-identity tokens are candidate-unique and do not constrain matching;
    # only constituent/morphology collision domains enter the bitmask.
    collision_sets = tuple(
        frozenset(
            token
            for token in tokens
            if not token.startswith("abstract_pair:")
            and not token.startswith("basic_family:")
        )
        for tokens in token_sets
    )
    vocabulary = tuple(sorted({token for tokens in collision_sets for token in tokens}))
    bit_of = {token: 1 << index for index, token in enumerate(vocabulary)}
    masks = tuple(
        sum((bit_of[token] for token in tokens), start=0)
        for tokens in collision_sets
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

    memo: dict[int, tuple[int, ...]] = {}

    def solve(available: int) -> tuple[int, ...]:
        cached = memo.get(available)
        if cached is not None:
            return cached
        if available == 0:
            return ()
        first = available & -available
        best = solve(available & ~first)
        for index, mask in enumerate(masks):
            if mask & first and mask & available == mask:
                candidate = tuple(sorted((index, *solve(available & ~mask))))
                best = better(best, candidate)
        memo[available] = best
        return best

    full_mask = (1 << len(vocabulary)) - 1
    chosen = solve(full_mask)
    return tuple(
        frozen[index]
        for index in sorted(chosen, key=lambda item: ranks[item])
    )


def _ledger_exposed_hd_constituent_attributes(
    exposure_predecessor: ExposureLedger,
) -> tuple[str, ...]:
    """Return every HD attribute opened anywhere in the predecessor ledger."""

    if not isinstance(exposure_predecessor, ExposureLedger):
        raise TypeError("exposure_predecessor must be ExposureLedger")
    attributes: set[str] = set()
    for task_id in sorted(exposure_predecessor.exposed_task_ids):
        parsed = parse_official_task_id(task_id)
        if parsed.family == "hd":
            attributes.update(parsed.concepts)
    return tuple(sorted(attributes))


def _select_strict_dev_tasks(
    corpus: ShapeBongardCorpus,
    *,
    report: Any,
    families: tuple[str, ...],
    candidate_count: int,
    seed: str,
    exposure_predecessor: ExposureLedger,
    historical: Any,
    resolver_digest: str,
    blocked_clusters: frozenset[str],
) -> tuple[tuple[Any, ...], tuple[tuple[str, int], ...]]:
    """Select after ledger-wide and within-batch HD attribute blocking."""

    grouped: dict[str, dict[str, list[Any]]] = {family: {} for family in families}
    exposed_hd_attributes = set(
        _ledger_exposed_hd_constituent_attributes(exposure_predecessor)
    )
    for record in report.records:
        parsed = record.parsed
        if (
            parsed.family not in grouped
            or record.split not in {"train", "val"}
            or not record.historically_clean
            or record.semantic_cohort != "dev"
        ):
            continue
        if parsed.family == "bd" and any(
            basic_morphology_cluster_id(item) in blocked_clusters
            for item in parsed.concepts
        ):
            continue
        if parsed.family == "hd" and (
            set(parsed.concepts) & exposed_hd_attributes
        ):
            continue
        try:
            exposure_predecessor.assert_unseen(task_ids=(parsed.task_id,))
            exposure_predecessor.assert_semantically_unseen(
                task_ids=(parsed.task_id,),
                historical_seed=historical,
                expected_historical_seed_digest=historical.seed_digest,
                expected_resolver_policy_digest=resolver_digest,
            )
        except ExposureViolation:
            continue
        cluster = semantic_generator_cluster_id(parsed.family, parsed.concepts)
        grouped[parsed.family].setdefault(cluster, []).append(parsed)

    queues: dict[str, list[Any]] = {}
    for family in families:
        representatives = [
            min(
                siblings,
                key=lambda item: (
                    _rank(seed, "within-generator-cluster", item.task_id),
                    item.task_id,
                ),
            )
            for siblings in grouped[family].values()
        ]
        ranked = sorted(
            representatives,
            key=lambda item: (
                _rank(
                    seed,
                    "generator-cluster",
                    {"family": item.family, "concepts": list(item.concepts)},
                ),
                item.concepts,
                item.task_id,
            ),
        )
        queues[family] = list(
            _maximum_disjoint_candidates(ranked, seed=seed, family=family)
        )
    availability = tuple((family, len(queues[family])) for family in families)
    maximum_admissible = sum(count for _, count in availability)
    if candidate_count > maximum_admissible:
        breakdown = ", ".join(
            f"{family.upper()}={count}" for family, count in availability
        )
        raise GatedDevValidationError(
            f"requested {candidate_count} tasks but ledger-wide and within-batch "
            f"strict DEV semantics permit {maximum_admissible} ({breakdown}); "
            "HD candidates reusing any predecessor-ledger constituent attribute "
            "are inadmissible"
        )
    selected: list[Any] = []
    used_tokens: set[str] = set()
    offsets = {family: 0 for family in families}
    while len(selected) < candidate_count:
        advanced = False
        for family in families:
            queue = queues[family]
            while offsets[family] < len(queue):
                candidate = queue[offsets[family]]
                offsets[family] += 1
                tokens = set(_disclosure_tokens(candidate.family, candidate.concepts))
                if tokens & used_tokens:
                    continue
                selected.append(candidate)
                used_tokens.update(tokens)
                advanced = True
                break
            if len(selected) == candidate_count:
                break
        if not advanced:
            raise GatedDevValidationError(
                f"requested {candidate_count} tasks but strict DEV semantics permit "
                f"only {len(selected)} for this seed/order"
            )
    return tuple(selected), availability


@dataclass(frozen=True, slots=True)
class GatedDevAcceptancePolicy:
    """All numbers fixed before task selection, pixels, or transport calls."""

    confidence_level: float = 0.80
    minimum_selected_clusters: int = 24
    minimum_gate_passed_clusters: int = 20
    minimum_gate_coverage_lower: float = 0.54
    minimum_both_query_correct_lower: float = 0.65
    minimum_fully_determinate_lower: float = 0.65
    maximum_any_abstention_upper: float = 0.35
    maximum_any_error_upper: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level", strict=True),
        )
        for name in ("minimum_selected_clusters", "minimum_gate_passed_clusters"):
            _positive_int(getattr(self, name), name)
        for name in (
            "minimum_gate_coverage_lower",
            "minimum_both_query_correct_lower",
            "minimum_fully_determinate_lower",
            "maximum_any_abstention_upper",
            "maximum_any_error_upper",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))

    def to_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_POLICY_SCHEMA,
            "confidence_level": self.confidence_level,
            "simultaneous_metric_count": len(METRIC_NAMES),
            "bound_method": BOUND_METHOD,
            "inference_mode": INFERENCE_MODE,
            "minimum_selected_clusters": self.minimum_selected_clusters,
            "minimum_gate_passed_clusters": self.minimum_gate_passed_clusters,
            "thresholds": {
                "minimum_gate_coverage_lower": self.minimum_gate_coverage_lower,
                "minimum_both_query_correct_lower": self.minimum_both_query_correct_lower,
                "minimum_fully_determinate_lower": self.minimum_fully_determinate_lower,
                "maximum_any_abstention_upper": self.maximum_any_abstention_upper,
                "maximum_any_error_upper": self.maximum_any_error_upper,
            },
            "selected_denominator": (
                "all preregistered tasks including proposal errors and support rejections"
            ),
            "conditional_denominator": (
                "tasks whose exact 12-panel visual-semantic gate is aligned"
            ),
            "panel_dependence_rule": (
                "reduce each task to one bounded cluster-level outcome before inference"
            ),
            "power_note": DEFAULT_POWER_NOTE,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "GatedDevAcceptancePolicy":
        data = _fields(
            _mapping(value, "gated DEV acceptance policy"),
            {
                "schema",
                "confidence_level",
                "simultaneous_metric_count",
                "bound_method",
                "inference_mode",
                "minimum_selected_clusters",
                "minimum_gate_passed_clusters",
                "thresholds",
                "selected_denominator",
                "conditional_denominator",
                "panel_dependence_rule",
                "power_note",
            },
            "gated DEV acceptance policy",
        )
        thresholds = _fields(
            _mapping(data["thresholds"], "acceptance thresholds"),
            {
                "minimum_gate_coverage_lower",
                "minimum_both_query_correct_lower",
                "minimum_fully_determinate_lower",
                "maximum_any_abstention_upper",
                "maximum_any_error_upper",
            },
            "acceptance thresholds",
        )
        fixed = {
            "schema": GATED_DEV_POLICY_SCHEMA,
            "simultaneous_metric_count": len(METRIC_NAMES),
            "bound_method": BOUND_METHOD,
            "inference_mode": INFERENCE_MODE,
            "selected_denominator": (
                "all preregistered tasks including proposal errors and support rejections"
            ),
            "conditional_denominator": (
                "tasks whose exact 12-panel visual-semantic gate is aligned"
            ),
            "panel_dependence_rule": (
                "reduce each task to one bounded cluster-level outcome before inference"
            ),
            "power_note": DEFAULT_POWER_NOTE,
        }
        for name, expected in fixed.items():
            if data[name] != expected or type(data[name]) is not type(expected):
                raise GatedDevValidationError(f"acceptance policy changed {name}")
        result = cls(
            confidence_level=data["confidence_level"],
            minimum_selected_clusters=data["minimum_selected_clusters"],
            minimum_gate_passed_clusters=data["minimum_gate_passed_clusters"],
            minimum_gate_coverage_lower=thresholds["minimum_gate_coverage_lower"],
            minimum_both_query_correct_lower=thresholds[
                "minimum_both_query_correct_lower"
            ],
            minimum_fully_determinate_lower=thresholds[
                "minimum_fully_determinate_lower"
            ],
            maximum_any_abstention_upper=thresholds[
                "maximum_any_abstention_upper"
            ],
            maximum_any_error_upper=thresholds["maximum_any_error_upper"],
        )
        if result.to_data() != dict(data):
            raise GatedDevValidationError("acceptance policy is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class GatedDevSelection:
    task_id: str
    family: str
    concepts: tuple[str, ...]
    split: str
    dependence_cluster_id: str
    disclosure_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        parsed = parse_official_task_id(self.task_id)
        if (
            parsed.family != self.family
            or parsed.concepts != self.concepts
            or self.family not in {"bd", "hd"}
        ):
            raise GatedDevValidationError("DEV selection differs from official parser")
        if self.split not in {"train", "val"}:
            raise GatedDevValidationError("DEV selection must be train/val, never test")
        expected_cluster = semantic_generator_cluster_id(self.family, self.concepts)
        if self.dependence_cluster_id != expected_cluster:
            raise GatedDevValidationError("DEV dependence cluster differs")
        if self.disclosure_tokens != _disclosure_tokens(self.family, self.concepts):
            raise GatedDevValidationError("DEV disclosure tokens differ")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_SELECTION_SCHEMA,
            "task_id": self.task_id,
            "family": self.family,
            "concepts": list(self.concepts),
            "split": self.split,
            "semantic_cohort": "dev",
            "dependence_cluster_id": self.dependence_cluster_id,
            "disclosure_tokens": list(self.disclosure_tokens),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "GatedDevSelection":
        data = _fields(
            _mapping(value, "gated DEV selection"),
            {
                "schema",
                "task_id",
                "family",
                "concepts",
                "split",
                "semantic_cohort",
                "dependence_cluster_id",
                "disclosure_tokens",
            },
            "gated DEV selection",
        )
        concepts = _list(data["concepts"], "selection concepts")
        tokens = _list(data["disclosure_tokens"], "selection disclosure tokens")
        if (
            data["schema"] != GATED_DEV_SELECTION_SCHEMA
            or data["semantic_cohort"] != "dev"
            or any(not isinstance(item, str) for item in concepts + tokens)
        ):
            raise GatedDevValidationError("gated DEV selection schema differs")
        result = cls(
            task_id=data["task_id"],
            family=data["family"],
            concepts=tuple(concepts),
            split=data["split"],
            dependence_cluster_id=data["dependence_cluster_id"],
            disclosure_tokens=tuple(tokens),
        )
        if result.to_data() != dict(data):
            raise GatedDevValidationError("gated DEV selection is not canonical")
        return result


def _ledger_extends(successor: ExposureLedger, predecessor: ExposureLedger) -> bool:
    return (
        successor.corpus_digest == predecessor.corpus_digest
        and successor.events[: len(predecessor.events)] == predecessor.events
    )


def _stable_read_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise GatedDevValidationError(f"cannot read {label} {path}") from exc
    identity = lambda item: (  # noqa: E731 - compact immutable stat identity.
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if not path.is_file() or identity(before) != identity(after):
        raise GatedDevValidationError(f"{label} changed while being read")
    return payload


def _fsync_gated_dev_precommit(path: Path) -> None:
    """Make one verified regular-file precommit and its directory durable."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise GatedDevValidationError(
            "platform cannot durably verify the Stage-B exposure precommit"
        )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, file_flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise GatedDevValidationError(
                    "Stage-B exposure precommit is not a regular file"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        parent_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise GatedDevValidationError(
            f"cannot fsync Stage-B exposure precommit {path}"
        ) from exc


def _persist_gated_dev_exposure_precommit(
    successor: ExposureLedger,
    directory: str | Path,
) -> Path:
    """Persist, fsync, and exactly reload the full DEV disclosure transition."""

    if not isinstance(successor, ExposureLedger):
        raise TypeError("successor must be ExposureLedger")
    try:
        path = successor.write_content_addressed(directory).resolve()
    except Exception as exc:  # noqa: BLE001 - normalize the persistence boundary.
        raise GatedDevValidationError(
            "cannot persist Stage-B exposure successor before DEV access"
        ) from exc
    _fsync_gated_dev_precommit(path)
    payload = _stable_read_bytes(path, "Stage-B exposure precommit")
    expected_payload = successor.to_json().encode("utf-8")
    if payload != expected_payload:
        raise GatedDevValidationError(
            "durable Stage-B exposure precommit bytes differ from the exact successor"
        )
    try:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("exposure ledger root must be an object")
        reloaded = ExposureLedger.from_dict(decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GatedDevValidationError(
            "cannot reload durable Stage-B exposure successor"
        ) from exc
    if reloaded != successor:
        raise GatedDevValidationError(
            "durable Stage-B exposure successor differs from the in-memory precommit"
        )
    return path


def _payload_address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_current_stage_a_selection_algorithm(
    archive: SemanticCalibrationProposalArchive,
) -> None:
    """Keep legacy Stage-A archives decodable but unable to authorize Stage B."""

    if not isinstance(archive, SemanticCalibrationProposalArchive):
        raise TypeError("Stage-A proposal archive must be typed")
    archive.assert_untampered()
    if archive.selection_algorithm != CAMPAIGN_SELECTION_ALGORITHM:
        raise GatedDevValidationError(
            "Stage B requires the current constituent-disjoint Stage-A selection "
            "algorithm; legacy v1 campaigns remain audit-only"
        )


def _authenticate_stage_a_command_receipt(
    receipt: StageACommandReceipt,
    *,
    campaign: SemanticCalibrationCampaignArtifact,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
) -> StageACommandReceipt:
    """Re-decode and bind the durable Stage-A command boundary.

    A direct campaign object is insufficient: this also authenticates the
    descriptive command design, exact terminal campaign bytes, exact exposure
    successor, execution configuration, launcher identity, and private cache
    handoff retained by the successful Stage-A command.
    """

    if not isinstance(receipt, StageACommandReceipt):
        raise TypeError("stage_a_command_receipt must be StageACommandReceipt")
    if not isinstance(campaign, SemanticCalibrationCampaignArtifact):
        raise TypeError("stage_a_campaign must be SemanticCalibrationCampaignArtifact")
    campaign.assert_untampered()
    archive = campaign.score_batch.commitment_batch.proposal_archive
    _require_current_stage_a_selection_algorithm(archive)
    try:
        trusted = StageACommandReceipt.from_bytes(
            receipt.receipt_payload,
            expected_receipt_digest=receipt.receipt_digest,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GatedDevValidationError(
            f"Stage-A command receipt is not strictly decodable: {exc}"
        ) from exc
    if trusted != receipt or trusted.to_data() != receipt.to_data():
        raise GatedDevValidationError(
            "Stage-A command receipt object differs from its canonical bytes"
        )
    if trusted.status != "succeeded" or trusted.cold_verified is not True:
        raise GatedDevValidationError(
            "Stage B requires a successful cold-verified Stage-A command receipt"
        )
    if (
        trusted.to_data().get("schema") != STAGE_A_COMMAND_RECEIPT_SCHEMA
        or trusted.source_dependencies is None
    ):
        raise GatedDevValidationError(
            "Stage B requires a successful v2 Stage-A receipt with frozen source "
            "dependencies; v1 remains audit-only"
        )
    observed_sources = freeze_stage_a_source_dependencies()
    if observed_sources != trusted.source_dependencies:
        raise GatedDevValidationError(
            "current Stage-B Python sources differ from the successful Stage-A "
            "source dependency identity"
        )

    terminal_payload = _stable_read_bytes(
        trusted.terminal_artifact_path, "Stage-A terminal artifact"
    )
    expected_terminal = canonical_json(campaign.to_data()) + b"\n"
    if (
        trusted.terminal_internal_digest != campaign.digest
        or trusted.terminal_artifact_file_sha256
        != _payload_address(terminal_payload)
        or terminal_payload != expected_terminal
    ):
        raise GatedDevValidationError(
            "Stage-A receipt terminal bytes differ from the authenticated campaign"
        )

    ledger_payload = _stable_read_bytes(
        trusted.exposure_ledger_path, "Stage-A exposure successor"
    )
    if trusted.exposure_ledger_file_sha256 != _payload_address(ledger_payload):
        raise GatedDevValidationError(
            "Stage-A exposure successor file hash differs from command receipt"
        )
    try:
        raw_ledger = json.loads(ledger_payload)
        if not isinstance(raw_ledger, Mapping):
            raise TypeError("exposure ledger root must be an object")
        durable_successor = ExposureLedger.from_dict(raw_ledger)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GatedDevValidationError(
            f"Stage-A exposure successor is invalid: {exc}"
        ) from exc

    config = trusted.command_config
    execution = archive.execution_config
    identity_joins = {
        "exposure_successor": durable_successor == archive.exposure_successor,
        "exposure_successor_digest": (
            trusted.exposure_ledger_digest == archive.exposure_successor.digest
        ),
        "exposure_predecessor_digest": (
            config.expected_exposure_ledger_digest
            == archive.exposure_predecessor.digest
        ),
        "protocol_digest": trusted.protocol_digest == archive.protocol.digest(),
        "protocol": config.build_protocol().to_data() == archive.protocol.to_data(),
        "selection_seed": config.selection_seed == archive.selection_seed,
        "candidate_count": config.candidate_count == archive.candidate_count,
        "semantic_cohort": config.semantic_cohort == archive.semantic_cohort,
        "families": config.families == archive.families,
        "proposer_minutes": config.proposer_minutes == execution.proposer_minutes,
        "scorer_minutes": config.scorer_minutes == execution.scorer_minutes,
        "proposer_max_workers": (
            config.proposer_max_workers == execution.proposer_max_workers
        ),
        "scorer_max_workers": (
            config.scorer_max_workers == execution.scorer_max_workers
        ),
        "executable": config.executable == execution.executable,
        "receipt_launcher": (
            trusted.launcher_digest == execution.expected_codex_launcher_digest
        ),
        "config_launcher": (
            config.expected_codex_launcher_digest
            == execution.expected_codex_launcher_digest
        ),
        "cloud_policy_cache_binding": (
            trusted.cloud_policy_cache_binding
            == execution.cloud_policy_cache_binding
        ),
    }
    mismatched_joins = tuple(
        name for name, matches in identity_joins.items() if not matches
    )
    if mismatched_joins:
        raise GatedDevValidationError(
            "Stage-A command design, protocol, exposure, or execution identity "
            "differs from the authenticated campaign: "
            + ", ".join(mismatched_joins)
        )
    try:
        durable_snapshot = trusted.load_cache_snapshot()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GatedDevValidationError(
            f"Stage-A cache handoff is not authentic: {exc}"
        ) from exc
    if cloud_policy_cache_snapshot is not None:
        if not isinstance(cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
            raise TypeError(
                "cloud_policy_cache_snapshot must be CloudPolicyCacheSnapshot"
            )
        if durable_snapshot != cloud_policy_cache_snapshot:
            raise GatedDevValidationError(
                "supplied Stage-B cache snapshot differs from exact Stage-A handoff"
            )
    return trusted


def _campaign_anchor_from_verified_stage_a(
    campaign: SemanticCalibrationCampaignArtifact,
) -> VisualSemanticCalibrationCampaignAnchor:
    """Project one already-verified full campaign to its per-task authority."""

    execution = (
        campaign.score_batch.commitment_batch.proposal_archive.execution_config
    )
    return VisualSemanticCalibrationCampaignAnchor(
        campaign_digest=campaign._sealed_digest,
        calibration=campaign.calibration,
        expected_codex_launcher_digest=(
            execution.expected_codex_launcher_digest
        ),
        cloud_policy_cache_binding=execution.cloud_policy_cache_binding,
    )


@dataclass(frozen=True)
class GatedDevValidationPlan:
    """Public Stage-B preregistration, complete before selected bytes open."""

    public_seed: str
    selection_seed_provenance: str
    requested_task_count: int
    families: tuple[str, ...]
    selections: tuple[GatedDevSelection, ...]
    available_clusters_by_family: tuple[tuple[str, int], ...]
    maximum_admissible_task_count: int
    acceptance_policy: GatedDevAcceptancePolicy
    source_corpus_manifest_digest: str
    split_source_digest: str
    historical_seed_digest: str
    semantic_resolver_policy_digest: str
    dev_cohort_report_digest: str
    blocked_morphology_clusters: tuple[str, ...]
    blocked_policy_digest: str
    exposed_hd_constituent_attributes: tuple[str, ...]
    exposure_predecessor_digest: str
    semantic_disclosure_keys: tuple[tuple[str, tuple[str, ...]], ...]
    stage_a_command_receipt_digest: str
    stage_a_campaign_digest: str
    stage_a_calibration_digest: str
    stage_a_family_digest: str
    stage_a_protocol_digest: str
    visual_semantic_policy_digest: str
    execution_config: SemanticCalibrationExecutionConfig
    task_max_workers: int
    exposure_observed_at: str
    exposure_actor: str = "visual-semantic-stage-b"
    exposure_purpose: str = "precommitted gated DEV validation batch"
    exposure_source: str = "gkm.bongard-stage-b/v1"
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.public_seed, "externally supplied public seed")
        if (
            not isinstance(self.selection_seed_provenance, str)
            or not self.selection_seed_provenance.strip()
            or "\x00" in self.selection_seed_provenance
            or len(self.selection_seed_provenance.encode("utf-8")) > 4_096
        ):
            raise GatedDevValidationError(
                "selection_seed_provenance must be bounded non-empty text"
            )
        _positive_int(self.requested_task_count, "requested_task_count")
        if (
            not isinstance(self.families, tuple)
            or not self.families
            or len(self.families) != len(set(self.families))
            or any(item not in {"bd", "hd"} for item in self.families)
        ):
            raise GatedDevValidationError("Stage-B families must be unique bd/hd IDs")
        if (
            not isinstance(self.selections, tuple)
            or len(self.selections) != self.requested_task_count
            or any(not isinstance(item, GatedDevSelection) for item in self.selections)
        ):
            raise GatedDevValidationError("Stage-B selection does not cover raw N")
        for name in ("task_id", "dependence_cluster_id"):
            values = tuple(getattr(item, name) for item in self.selections)
            if len(values) != len(set(values)):
                raise GatedDevValidationError(f"Stage-B selection repeats {name}")
        if any(item.family not in self.families for item in self.selections):
            raise GatedDevValidationError("selection contains a family outside scope")
        if (
            self.available_clusters_by_family
            != tuple(
                (family, dict(self.available_clusters_by_family).get(family, -1))
                for family in self.families
            )
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for _, count in self.available_clusters_by_family
            )
        ):
            raise GatedDevValidationError("available family-cluster inventory differs")
        if (
            isinstance(self.maximum_admissible_task_count, bool)
            or not isinstance(self.maximum_admissible_task_count, int)
            or self.maximum_admissible_task_count
            != sum(count for _, count in self.available_clusters_by_family)
            or self.requested_task_count > self.maximum_admissible_task_count
        ):
            raise GatedDevValidationError("maximum admissible task count differs")
        seen_tokens: set[str] = set()
        for item in self.selections:
            overlap = seen_tokens & set(item.disclosure_tokens)
            if overlap:
                raise GatedDevValidationError(
                    f"Stage-B selections share semantic disclosure keys {sorted(overlap)}"
                )
            seen_tokens.update(item.disclosure_tokens)
        if not isinstance(self.acceptance_policy, GatedDevAcceptancePolicy):
            raise TypeError("acceptance_policy must be GatedDevAcceptancePolicy")
        _address(self.source_corpus_manifest_digest, "source corpus manifest digest")
        _address(self.split_source_digest, "split source digest")
        _address(self.historical_seed_digest, "historical seed digest")
        _address(self.semantic_resolver_policy_digest, "semantic resolver digest")
        _address(self.dev_cohort_report_digest, "DEV cohort report digest")
        _address(self.exposure_predecessor_digest, "exposure predecessor digest")
        for name in (
            "blocked_policy_digest",
            "stage_a_command_receipt_digest",
            "stage_a_campaign_digest",
            "stage_a_calibration_digest",
            "stage_a_family_digest",
            "stage_a_protocol_digest",
            "visual_semantic_policy_digest",
        ):
            _digest(getattr(self, name), name)
        historical = load_historical_exposure()
        resolver = semantic_resolver_policy_digest(historical)
        current_blocked = tuple(
            sorted(
                key.concepts[0]
                for key in semantic_policy_blocked_keys(historical)
                if key.kind == "basic_morphology_cluster"
            )
        )
        if (
            historical.seed_digest != self.historical_seed_digest
            or resolver != self.semantic_resolver_policy_digest
            or current_blocked != self.blocked_morphology_clusters
            or self.blocked_policy_digest
            != _blocked_policy_digest(
                historical.seed_digest, resolver, current_blocked
            )
        ):
            raise GatedDevValidationError("historical morphology policy drifted")
        if any(
            item.family == "bd"
            and any(
                basic_morphology_cluster_id(concept) in set(current_blocked)
                for concept in item.concepts
            )
            for item in self.selections
        ):
            raise GatedDevValidationError("selection contains blocked morphology")
        if (
            not isinstance(self.exposed_hd_constituent_attributes, tuple)
            or tuple(sorted(self.exposed_hd_constituent_attributes))
            != self.exposed_hd_constituent_attributes
            or len(self.exposed_hd_constituent_attributes)
            != len(set(self.exposed_hd_constituent_attributes))
            or any(
                not isinstance(item, str) or not item
                for item in self.exposed_hd_constituent_attributes
            )
        ):
            raise GatedDevValidationError(
                "ledger-exposed HD constituent inventory is not canonical"
            )
        exposed_hd = set(self.exposed_hd_constituent_attributes)
        if any(
            item.family == "hd" and set(item.concepts) & exposed_hd
            for item in self.selections
        ):
            raise GatedDevValidationError(
                "selection reuses a predecessor-ledger HD constituent attribute"
            )
        if (
            not isinstance(self.semantic_disclosure_keys, tuple)
            or tuple(sorted(self.semantic_disclosure_keys))
            != self.semantic_disclosure_keys
            or len(self.semantic_disclosure_keys)
            != len(set(self.semantic_disclosure_keys))
        ):
            raise GatedDevValidationError("semantic disclosure receipt is not canonical")
        if not isinstance(self.execution_config, SemanticCalibrationExecutionConfig):
            raise TypeError("execution_config must be the Stage-A execution config")
        if (
            isinstance(self.task_max_workers, bool)
            or not isinstance(self.task_max_workers, int)
            or not 1 <= self.task_max_workers <= 32
        ):
            raise GatedDevValidationError("task_max_workers must lie in [1, 32]")
        for name in (
            "exposure_observed_at",
            "exposure_actor",
            "exposure_purpose",
            "exposure_source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise GatedDevValidationError(f"{name} must be non-empty bounded text")
        object.__setattr__(self, "_sealed_digest", self.digest)

    def content_data(self) -> dict[str, object]:
        gate = SupportGatePolicy.visual_semantic()
        return {
            "schema": GATED_DEV_PLAN_SCHEMA,
            "reference_execution_semantics": REFERENCE_SEMANTICS,
            "estimand": ESTIMAND,
            "stage_a_authority_layout": CENTRAL_CAMPAIGN_LAYOUT,
            "experimental_unit_qualification": (
                "exact generator-pair/task experimental units; simultaneous bounds "
                "assume the preregistered design-based cross-unit condition, which "
                "corpus metadata alone does not prove"
            ),
            "public_seed": self.public_seed,
            "public_seed_digest": hashlib.sha256(self.public_seed.encode("utf-8")).hexdigest(),
            "selection_seed_provenance": self.selection_seed_provenance,
            "requested_task_count": self.requested_task_count,
            "semantic_cohort": "dev",
            "families": list(self.families),
            "selection_algorithm": GATED_DEV_SELECTION_ALGORITHM,
            "selections": [item.to_data() for item in self.selections],
            "available_clusters_by_family": dict(
                self.available_clusters_by_family
            ),
            "maximum_admissible_task_count": self.maximum_admissible_task_count,
            "acceptance_policy": self.acceptance_policy.to_data(),
            "acceptance_policy_digest": self.acceptance_policy.digest,
            "source_corpus_manifest_digest": self.source_corpus_manifest_digest,
            "split_source_digest": self.split_source_digest,
            "historical_seed_digest": self.historical_seed_digest,
            "semantic_resolver_policy_digest": self.semantic_resolver_policy_digest,
            "dev_cohort_report_digest": self.dev_cohort_report_digest,
            "blocked_morphology_clusters": list(self.blocked_morphology_clusters),
            "blocked_policy_digest": self.blocked_policy_digest,
            "ledger_exposed_hd_constituent_attributes": list(
                self.exposed_hd_constituent_attributes
            ),
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "semantic_unseen_receipt": {
                "task_ids": [item.task_id for item in self.selections],
                "semantic_keys": [
                    {"kind": kind, "concepts": list(concepts)}
                    for kind, concepts in self.semantic_disclosure_keys
                ],
                "historical_seed_digest": self.historical_seed_digest,
                "resolver_policy_digest": self.semantic_resolver_policy_digest,
                "ledger_digest": self.exposure_predecessor_digest,
            },
            "stage_a": {
                "command_receipt_digest": self.stage_a_command_receipt_digest,
                "campaign_digest": self.stage_a_campaign_digest,
                "calibration_digest": self.stage_a_calibration_digest,
                "family_digest": self.stage_a_family_digest,
                "protocol_digest": self.stage_a_protocol_digest,
                "execution_config": self.execution_config.to_data(),
                "execution_config_digest": self.execution_config.digest,
            },
            "visual_semantic_policy_digest": self.visual_semantic_policy_digest,
            "support_gate_policy": gate.to_data(),
            "support_gate_policy_digest": canonical_digest(gate.to_data()),
            "scheduling": {
                "semantics": "ordered-thread-map-after-batch-exposure/v1",
                "task_max_workers": self.task_max_workers,
                "all_exposures_recorded_before_executor": True,
            },
            "exposure_event": {
                "phase": "dev-validation",
                "actor": self.exposure_actor,
                "purpose": self.exposure_purpose,
                "source": self.exposure_source,
                "observed_at": self.exposure_observed_at,
            },
            "python_predicate_authoritative": True,
            "python_statistics_authoritative": True,
            "optional_checker_in_artifact_identity": False,
            "optional_checker_may_affect_result": False,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        content = self.content_data()
        return {**content, "plan_digest": canonical_digest(content)}

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise GatedDevValidationError("gated DEV plan changed after sealing")

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "GatedDevValidationPlan":
        data = _fields(
            _mapping(value, "gated DEV plan"),
            {
                "schema",
                "reference_execution_semantics",
                "estimand",
                "stage_a_authority_layout",
                "experimental_unit_qualification",
                "public_seed",
                "public_seed_digest",
                "selection_seed_provenance",
                "requested_task_count",
                "semantic_cohort",
                "families",
                "selection_algorithm",
                "selections",
                "available_clusters_by_family",
                "maximum_admissible_task_count",
                "acceptance_policy",
                "acceptance_policy_digest",
                "source_corpus_manifest_digest",
                "split_source_digest",
                "historical_seed_digest",
                "semantic_resolver_policy_digest",
                "dev_cohort_report_digest",
                "blocked_morphology_clusters",
                "blocked_policy_digest",
                "ledger_exposed_hd_constituent_attributes",
                "exposure_predecessor_digest",
                "semantic_unseen_receipt",
                "stage_a",
                "visual_semantic_policy_digest",
                "support_gate_policy",
                "support_gate_policy_digest",
                "scheduling",
                "exposure_event",
                "python_predicate_authoritative",
                "python_statistics_authoritative",
                "optional_checker_in_artifact_identity",
                "optional_checker_may_affect_result",
                "plan_digest",
            },
            "gated DEV plan",
        )
        fixed = {
            "schema": GATED_DEV_PLAN_SCHEMA,
            "reference_execution_semantics": REFERENCE_SEMANTICS,
            "estimand": ESTIMAND,
            "stage_a_authority_layout": CENTRAL_CAMPAIGN_LAYOUT,
            "experimental_unit_qualification": (
                "exact generator-pair/task experimental units; simultaneous bounds "
                "assume the preregistered design-based cross-unit condition, which "
                "corpus metadata alone does not prove"
            ),
            "semantic_cohort": "dev",
            "selection_algorithm": GATED_DEV_SELECTION_ALGORITHM,
            "python_predicate_authoritative": True,
            "python_statistics_authoritative": True,
            "optional_checker_in_artifact_identity": False,
            "optional_checker_may_affect_result": False,
        }
        for name, expected in fixed.items():
            if data[name] != expected or type(data[name]) is not type(expected):
                raise GatedDevValidationError(f"gated DEV plan changed {name}")
        if data["public_seed_digest"] != hashlib.sha256(
            str(data["public_seed"]).encode("utf-8")
        ).hexdigest():
            raise GatedDevValidationError("public seed digest differs")
        policy = GatedDevAcceptancePolicy.from_data(
            _mapping(data["acceptance_policy"], "acceptance policy")
        )
        if data["acceptance_policy_digest"] != policy.digest:
            raise GatedDevValidationError("acceptance policy digest differs")
        family_values = _list(data["families"], "Stage-B families")
        selections = tuple(
            GatedDevSelection.from_data(_mapping(item, "Stage-B selection"))
            for item in _list(data["selections"], "Stage-B selections")
        )
        blocked = tuple(_list(data["blocked_morphology_clusters"], "blocked clusters"))
        exposed_hd = tuple(
            _list(
                data["ledger_exposed_hd_constituent_attributes"],
                "ledger-exposed HD constituent attributes",
            )
        )
        if any(
            not isinstance(item, str)
            for item in (*family_values, *blocked, *exposed_hd)
        ):
            raise GatedDevValidationError("family or blocked-cluster list is malformed")
        available_map = _mapping(
            data["available_clusters_by_family"], "available family clusters"
        )
        if set(available_map) != set(family_values):
            raise GatedDevValidationError("available family cluster keys differ")
        available = tuple((family, available_map[family]) for family in family_values)
        stage_a = _fields(
            _mapping(data["stage_a"], "Stage-A anchor"),
            {
                "command_receipt_digest",
                "campaign_digest",
                "calibration_digest",
                "family_digest",
                "protocol_digest",
                "execution_config",
                "execution_config_digest",
            },
            "Stage-A anchor",
        )
        execution = SemanticCalibrationExecutionConfig.from_data(
            _mapping(stage_a["execution_config"], "Stage-A execution config")
        )
        if stage_a["execution_config_digest"] != execution.digest:
            raise GatedDevValidationError("Stage-A execution config digest differs")
        scheduling = _fields(
            _mapping(data["scheduling"], "Stage-B scheduling"),
            {"semantics", "task_max_workers", "all_exposures_recorded_before_executor"},
            "Stage-B scheduling",
        )
        if scheduling != {
            "semantics": "ordered-thread-map-after-batch-exposure/v1",
            "task_max_workers": scheduling["task_max_workers"],
            "all_exposures_recorded_before_executor": True,
        }:
            raise GatedDevValidationError("Stage-B scheduling semantics differ")
        event = _fields(
            _mapping(data["exposure_event"], "Stage-B exposure event"),
            {"phase", "actor", "purpose", "source", "observed_at"},
            "Stage-B exposure event",
        )
        if event["phase"] != "dev-validation":
            raise GatedDevValidationError("Stage-B exposure phase differs")
        gate = SupportGatePolicy.visual_semantic()
        if (
            data["support_gate_policy"] != gate.to_data()
            or data["support_gate_policy_digest"] != canonical_digest(gate.to_data())
        ):
            raise GatedDevValidationError("support gate policy differs")
        receipt = _fields(
            _mapping(data["semantic_unseen_receipt"], "semantic unseen receipt"),
            {
                "task_ids",
                "semantic_keys",
                "historical_seed_digest",
                "resolver_policy_digest",
                "ledger_digest",
            },
            "semantic unseen receipt",
        )
        raw_keys = _list(receipt["semantic_keys"], "semantic unseen keys")
        keys: list[tuple[str, tuple[str, ...]]] = []
        for raw in raw_keys:
            entry = _fields(
                _mapping(raw, "semantic unseen key"),
                {"kind", "concepts"},
                "semantic unseen key",
            )
            concepts = _list(entry["concepts"], "semantic key concepts")
            if not isinstance(entry["kind"], str) or any(
                not isinstance(item, str) for item in concepts
            ):
                raise GatedDevValidationError("semantic unseen key is malformed")
            keys.append((entry["kind"], tuple(concepts)))
        if (
            receipt["task_ids"] != [item.task_id for item in selections]
            or receipt["historical_seed_digest"] != data["historical_seed_digest"]
            or receipt["resolver_policy_digest"]
            != data["semantic_resolver_policy_digest"]
            or receipt["ledger_digest"] != data["exposure_predecessor_digest"]
        ):
            raise GatedDevValidationError("semantic unseen receipt parents differ")
        result = cls(
            public_seed=data["public_seed"],
            selection_seed_provenance=data["selection_seed_provenance"],
            requested_task_count=data["requested_task_count"],
            families=tuple(family_values),
            selections=selections,
            available_clusters_by_family=available,
            maximum_admissible_task_count=data["maximum_admissible_task_count"],
            acceptance_policy=policy,
            source_corpus_manifest_digest=data["source_corpus_manifest_digest"],
            split_source_digest=data["split_source_digest"],
            historical_seed_digest=data["historical_seed_digest"],
            semantic_resolver_policy_digest=data["semantic_resolver_policy_digest"],
            dev_cohort_report_digest=data["dev_cohort_report_digest"],
            blocked_morphology_clusters=blocked,
            blocked_policy_digest=data["blocked_policy_digest"],
            exposed_hd_constituent_attributes=exposed_hd,
            exposure_predecessor_digest=data["exposure_predecessor_digest"],
            semantic_disclosure_keys=tuple(keys),
            stage_a_command_receipt_digest=stage_a["command_receipt_digest"],
            stage_a_campaign_digest=stage_a["campaign_digest"],
            stage_a_calibration_digest=stage_a["calibration_digest"],
            stage_a_family_digest=stage_a["family_digest"],
            stage_a_protocol_digest=stage_a["protocol_digest"],
            visual_semantic_policy_digest=data["visual_semantic_policy_digest"],
            execution_config=execution,
            task_max_workers=scheduling["task_max_workers"],
            exposure_observed_at=event["observed_at"],
            exposure_actor=event["actor"],
            exposure_purpose=event["purpose"],
            exposure_source=event["source"],
        )
        if result.digest != _digest(data["plan_digest"], "plan digest"):
            raise GatedDevValidationError("gated DEV plan digest differs")
        if result.to_data() != dict(data):
            raise GatedDevValidationError("gated DEV plan is not canonical")
        return result


def _validate_campaign_policy(
    campaign: SemanticCalibrationCampaignArtifact,
    policy: VisualSemanticPolicy,
    *,
    source_manifest_digest: str,
) -> None:
    if not isinstance(campaign, SemanticCalibrationCampaignArtifact):
        raise TypeError("stage_a_campaign must be SemanticCalibrationCampaignArtifact")
    campaign.assert_untampered()
    calibration = campaign.calibration
    calibration.assert_untampered()
    if not isinstance(policy, VisualSemanticPolicy):
        raise TypeError("visual_semantic_policy must be VisualSemanticPolicy")
    if VisualSemanticPolicy.from_data(policy.to_data()).to_data() != policy.to_data():
        raise GatedDevValidationError("visual-semantic policy does not round-trip")
    archive = campaign.score_batch.commitment_batch.proposal_archive
    expected = {
        "campaign source corpus": (
            archive.source_corpus_manifest_digest,
            source_manifest_digest,
        ),
        "calibration source corpus": (
            calibration.plan.corpus_manifest_digest,
            source_manifest_digest,
        ),
        "policy scorer protocol": (
            policy.soft_scorer_protocol_digest,
            calibration.protocol.digest(),
        ),
        "policy scorer family": (
            policy.soft_scorer_family_digest,
            calibration.family.digest(),
        ),
        "policy development manifest": (
            policy.soft_family_development_manifest_digest,
            calibration.family.development_manifest_digest,
        ),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise GatedDevValidationError(f"{label} differs")


def _validate_full_source_commitment(
    corpus: ShapeBongardCorpus,
    manifest: CorpusManifest,
    *,
    expected_manifest_digest: str,
    expected_split_source_digest: str,
) -> str:
    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be ShapeBongardCorpus")
    actual = _manifest_digest(manifest)
    if actual != _address(expected_manifest_digest, "expected source manifest digest"):
        raise GatedDevValidationError("source manifest differs from precommit")
    split_digest = _address(expected_split_source_digest, "expected split digest")
    if (
        corpus.split.source_digest != split_digest
        or manifest.split.source_digest != split_digest
        or corpus.split.to_manifest_dict() != manifest.split.to_manifest_dict()
    ):
        raise GatedDevValidationError("source split differs from precommit")
    manifest_ids = tuple(item.task_id for item in manifest.tasks)
    if manifest_ids != corpus.task_ids or len(manifest_ids) != len(set(manifest_ids)):
        raise GatedDevValidationError(
            "source manifest is not the complete supplied corpus inventory"
        )
    if dict(manifest.family_counts) != dict(corpus.family_counts):
        raise GatedDevValidationError("source manifest family counts differ")
    return actual


def _selection_from_corpus(
    corpus: ShapeBongardCorpus,
    parsed: Any,
) -> GatedDevSelection:
    """Project one parsed candidate to the exact public selection record."""

    return GatedDevSelection(
        task_id=parsed.task_id,
        family=parsed.family,
        concepts=parsed.concepts,
        split=corpus.assignment(parsed.task_id).split or "",
        dependence_cluster_id=semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        ),
        disclosure_tokens=_disclosure_tokens(parsed.family, parsed.concepts),
    )


def _audit_gated_dev_selection_against_corpus(
    corpus: ShapeBongardCorpus,
    *,
    plan: GatedDevValidationPlan,
    predecessor: ExposureLedger,
) -> None:
    """Recompute the complete metadata-only Stage-B selection authority.

    A plan digest authenticates only the bytes it contains.  This replay joins
    those bytes back to the trusted corpus, historical policy, live ledger,
    public seed, exact ordered selection, and maximum admissible capacity.
    """

    if predecessor.digest != plan.exposure_predecessor_digest:
        raise GatedDevValidationError(
            "Stage-B selection predecessor differs from public plan"
        )
    historical = load_historical_exposure()
    resolution = predecessor.assert_semantically_unseen(
        task_ids=(item.task_id for item in plan.selections),
        historical_seed=historical,
        expected_historical_seed_digest=plan.historical_seed_digest,
        expected_resolver_policy_digest=plan.semantic_resolver_policy_digest,
    )
    if tuple((key.kind, key.concepts) for key in resolution.semantic_keys) != (
        plan.semantic_disclosure_keys
    ):
        raise GatedDevValidationError(
            "Stage-B semantic-unseen receipt differs from predecessor"
        )
    report = build_cohort_report(corpus, historical, cohort="dev")
    if report.digest != plan.dev_cohort_report_digest:
        raise GatedDevValidationError(
            "Stage-B DEV cohort report differs from public plan"
        )
    if _ledger_exposed_hd_constituent_attributes(predecessor) != (
        plan.exposed_hd_constituent_attributes
    ):
        raise GatedDevValidationError(
            "Stage-B predecessor HD constituent inventory differs from public plan"
        )
    selected, availability = _select_strict_dev_tasks(
        corpus,
        report=report,
        families=plan.families,
        candidate_count=plan.requested_task_count,
        seed=plan.public_seed,
        exposure_predecessor=predecessor,
        historical=historical,
        resolver_digest=plan.semantic_resolver_policy_digest,
        blocked_clusters=frozenset(plan.blocked_morphology_clusters),
    )
    expected_selections = tuple(
        _selection_from_corpus(corpus, item) for item in selected
    )
    if expected_selections != plan.selections:
        raise GatedDevValidationError(
            "Stage-B selection/order differs from public-seed corpus replay"
        )
    maximum = sum(count for _, count in availability)
    if (
        availability != plan.available_clusters_by_family
        or maximum != plan.maximum_admissible_task_count
    ):
        raise GatedDevValidationError(
            "Stage-B availability/maximum differs from full corpus replay"
        )


def plan_gated_dev_validation(
    corpus: ShapeBongardCorpus,
    *,
    source_corpus_manifest: CorpusManifest,
    expected_source_corpus_manifest_digest: str,
    expected_split_source_digest: str,
    stage_a_campaign: SemanticCalibrationCampaignArtifact,
    stage_a_command_receipt: StageACommandReceipt,
    visual_semantic_policy: VisualSemanticPolicy,
    exposure_predecessor: ExposureLedger,
    expected_exposure_predecessor_digest: str,
    public_seed: str,
    selection_seed_provenance: str,
    requested_task_count: int,
    exposure_observed_at: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    acceptance_policy: GatedDevAcceptancePolicy = GatedDevAcceptancePolicy(),
    families: Sequence[str] = ("bd", "hd"),
    task_max_workers: int = 4,
) -> GatedDevValidationPlan:
    """Freeze the complete Stage-B plan without reading a selected PNG.

    ``source_corpus_manifest`` is a previously built, trusted full-corpus
    snapshot.  This function checks its already committed metadata but never
    calls ``build_manifest`` or opens a panel.
    """

    if not isinstance(acceptance_policy, GatedDevAcceptancePolicy):
        raise TypeError("acceptance_policy must be frozen before selection")
    # Force exact policy reconstruction before selection begins.
    acceptance_policy = GatedDevAcceptancePolicy.from_data(
        acceptance_policy.to_data()
    )
    _digest(public_seed, "externally supplied public seed")
    if (
        not isinstance(selection_seed_provenance, str)
        or not selection_seed_provenance.strip()
        or "\x00" in selection_seed_provenance
    ):
        raise GatedDevValidationError(
            "selection_seed_provenance must be non-empty text"
        )
    _positive_int(requested_task_count, "requested_task_count")
    source_digest = _validate_full_source_commitment(
        corpus,
        source_corpus_manifest,
        expected_manifest_digest=expected_source_corpus_manifest_digest,
        expected_split_source_digest=expected_split_source_digest,
    )
    # Authenticate every Stage-A score/label parent against its already-exposed
    # source bytes before selecting any fresh DEV task.  Digest consistency
    # alone is not evidence that the revealed labels came from this corpus.
    verified_campaign, _ = verify_semantic_campaign_against_corpus(
        stage_a_campaign.to_data(),
        corpus=corpus,
        corpus_manifest=source_corpus_manifest,
    )
    if verified_campaign.digest != stage_a_campaign.digest:
        raise GatedDevValidationError("verified Stage-A campaign identity differs")
    stage_a_campaign = verified_campaign
    stage_a_command_receipt = _authenticate_stage_a_command_receipt(
        stage_a_command_receipt,
        campaign=stage_a_campaign,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
    )
    _validate_campaign_policy(
        stage_a_campaign,
        visual_semantic_policy,
        source_manifest_digest=source_digest,
    )
    if not isinstance(exposure_predecessor, ExposureLedger):
        raise TypeError("exposure_predecessor must be ExposureLedger")
    if exposure_predecessor.digest != _address(
        expected_exposure_predecessor_digest,
        "expected exposure predecessor digest",
    ):
        raise GatedDevValidationError("exposure predecessor differs from precommit")
    if exposure_predecessor.corpus_digest != source_digest:
        raise GatedDevValidationError("exposure predecessor belongs to another corpus")
    proposal_archive = (
        stage_a_campaign.score_batch.commitment_batch.proposal_archive
    )
    if not _ledger_extends(
        exposure_predecessor, proposal_archive.exposure_successor
    ):
        raise GatedDevValidationError(
            "Stage-B exposure predecessor does not extend the full Stage-A campaign"
        )
    if not isinstance(cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("cloud_policy_cache_snapshot must be frozen before planning")
    config = proposal_archive.execution_config
    if (
        stage_a_command_receipt.launcher_digest
        != config.expected_codex_launcher_digest
        or cloud_policy_cache_snapshot.binding != config.cloud_policy_cache_binding
    ):
        raise GatedDevValidationError(
            "Stage-B launcher/cache identity differs from Stage-A"
        )
    scope = tuple(families)
    if (
        not scope
        or len(scope) != len(set(scope))
        or any(item not in {"bd", "hd"} for item in scope)
    ):
        raise GatedDevValidationError("families must be unique bd/hd IDs")

    historical = load_historical_exposure()
    resolver_digest = semantic_resolver_policy_digest(historical)
    report = build_cohort_report(corpus, historical, cohort="dev")
    blocked = tuple(
        sorted(
            key.concepts[0]
            for key in semantic_policy_blocked_keys(historical)
            if key.kind == "basic_morphology_cluster"
        )
    )
    exposed_hd_attributes = _ledger_exposed_hd_constituent_attributes(
        exposure_predecessor
    )
    parsed, availability = _select_strict_dev_tasks(
        corpus,
        report=report,
        families=scope,
        candidate_count=requested_task_count,
        seed=public_seed,
        exposure_predecessor=exposure_predecessor,
        historical=historical,
        resolver_digest=resolver_digest,
        blocked_clusters=frozenset(blocked),
    )
    selections = tuple(_selection_from_corpus(corpus, item) for item in parsed)
    resolution = exposure_predecessor.assert_semantically_unseen(
        task_ids=(item.task_id for item in selections),
        historical_seed=historical,
        expected_historical_seed_digest=historical.seed_digest,
        expected_resolver_policy_digest=resolver_digest,
    )
    keys = tuple((key.kind, key.concepts) for key in resolution.semantic_keys)
    plan = GatedDevValidationPlan(
        public_seed=public_seed,
        selection_seed_provenance=selection_seed_provenance,
        requested_task_count=requested_task_count,
        families=scope,
        selections=selections,
        available_clusters_by_family=availability,
        maximum_admissible_task_count=sum(count for _, count in availability),
        acceptance_policy=acceptance_policy,
        source_corpus_manifest_digest=source_digest,
        split_source_digest=expected_split_source_digest,
        historical_seed_digest=historical.seed_digest,
        semantic_resolver_policy_digest=resolver_digest,
        dev_cohort_report_digest=report.digest,
        blocked_morphology_clusters=blocked,
        blocked_policy_digest=_blocked_policy_digest(
            historical.seed_digest, resolver_digest, blocked
        ),
        exposed_hd_constituent_attributes=exposed_hd_attributes,
        exposure_predecessor_digest=exposure_predecessor.digest,
        semantic_disclosure_keys=keys,
        stage_a_command_receipt_digest=stage_a_command_receipt.receipt_digest,
        stage_a_campaign_digest=stage_a_campaign.digest,
        stage_a_calibration_digest=stage_a_campaign.calibration.digest,
        stage_a_family_digest=stage_a_campaign.calibration.family.digest(),
        stage_a_protocol_digest=stage_a_campaign.calibration.protocol.digest(),
        visual_semantic_policy_digest=visual_semantic_policy.digest(),
        execution_config=config,
        task_max_workers=task_max_workers,
        exposure_observed_at=exposure_observed_at,
    )
    # The exact round trip is part of the pre-pixel freeze boundary.
    return GatedDevValidationPlan.from_data(plan.to_data())


@dataclass(frozen=True, slots=True)
class RawSimultaneousHoeffdingBound:
    metric: str
    successes: int
    cluster_count: int
    confidence_level: float
    simultaneous_metric_count: int = len(METRIC_NAMES)

    def __post_init__(self) -> None:
        if self.metric not in METRIC_NAMES:
            raise GatedDevValidationError("unknown Stage-B metric")
        if (
            isinstance(self.successes, bool)
            or not isinstance(self.successes, int)
            or isinstance(self.cluster_count, bool)
            or not isinstance(self.cluster_count, int)
            or not 0 <= self.successes <= self.cluster_count
        ):
            raise GatedDevValidationError("invalid Hoeffding successes/count")
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence level", strict=True),
        )
        if self.simultaneous_metric_count != len(METRIC_NAMES):
            raise GatedDevValidationError("simultaneous metric family changed")

    @property
    def mean(self) -> float | None:
        return self.successes / self.cluster_count if self.cluster_count else None

    @property
    def radius(self) -> float | None:
        if not self.cluster_count:
            return None
        alpha = 1.0 - self.confidence_level
        return math.sqrt(
            math.log(2.0 * self.simultaneous_metric_count / alpha)
            / (2.0 * self.cluster_count)
        )

    @property
    def raw_lower(self) -> float | None:
        return None if self.mean is None else self.mean - (self.radius or 0.0)

    @property
    def raw_upper(self) -> float | None:
        return None if self.mean is None else self.mean + (self.radius or 0.0)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_BOUND_SCHEMA,
            "metric": self.metric,
            "successes": self.successes,
            "cluster_count": self.cluster_count,
            "mean": self.mean,
            "confidence_level": self.confidence_level,
            "simultaneous_metric_count": self.simultaneous_metric_count,
            "method": BOUND_METHOD,
            "radius": self.radius,
            "raw_lower": self.raw_lower,
            "raw_upper": self.raw_upper,
            "clipped_for_acceptance": False,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "RawSimultaneousHoeffdingBound":
        data = _fields(
            _mapping(value, "Hoeffding bound"),
            {
                "schema",
                "metric",
                "successes",
                "cluster_count",
                "mean",
                "confidence_level",
                "simultaneous_metric_count",
                "method",
                "radius",
                "raw_lower",
                "raw_upper",
                "clipped_for_acceptance",
            },
            "Hoeffding bound",
        )
        if (
            data["schema"] != GATED_DEV_BOUND_SCHEMA
            or data["method"] != BOUND_METHOD
            or data["clipped_for_acceptance"] is not False
        ):
            raise GatedDevValidationError("Hoeffding method changed")
        result = cls(
            metric=data["metric"],
            successes=data["successes"],
            cluster_count=data["cluster_count"],
            confidence_level=data["confidence_level"],
            simultaneous_metric_count=data["simultaneous_metric_count"],
        )
        if result.to_data() != dict(data):
            raise GatedDevValidationError("Hoeffding bound does not reproduce")
        return result


def _episode_seed(plan: GatedDevValidationPlan, selection: GatedDevSelection) -> str:
    return "stage-b-episode-" + canonical_digest(
        {
            "schema": "gkm.bongard-stage-b-episode-seed.v1",
            "plan_digest": plan.digest,
            "public_seed": plan.public_seed,
            "task_id": selection.task_id,
            "dependence_cluster_id": selection.dependence_cluster_id,
        }
    )


@dataclass(frozen=True)
class GatedDevTaskRun:
    selection: GatedDevSelection
    episode_seed: str
    outer_record: Mapping[str, Any]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, GatedDevSelection):
            raise TypeError("selection must be GatedDevSelection")
        if not isinstance(self.episode_seed, str) or not self.episode_seed:
            raise GatedDevValidationError("episode seed must be non-empty")
        record = _mapping(self.outer_record, "visual-semantic outer run")
        if record.get("schema") != VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA:
            raise GatedDevValidationError(
                "Stage-B task must use one central campaign calibration reference"
            )
        _digest(
            record.get("calibration_campaign_digest"),
            "outer calibration campaign digest",
        )
        _digest(record.get("calibration_digest"), "outer calibration digest")
        if "calibration" in record:
            raise GatedDevValidationError(
                "Stage-B task duplicated the central calibration payload"
            )
        archived = _digest(record.get("record_digest"), "outer record digest")
        content = {key: value for key, value in record.items() if key != "record_digest"}
        if canonical_digest(content) != archived:
            raise GatedDevValidationError("visual-semantic outer record digest differs")
        episode = _mapping(record.get("episode"), "outer episode result")
        public_plan = _mapping(record.get("plan"), "outer episode plan")
        if (
            episode.get("task_id") != self.selection.task_id
            or public_plan.get("task_id") != self.selection.task_id
            or episode.get("plan_digest") != canonical_digest(dict(public_plan))
        ):
            raise GatedDevValidationError("task run belongs to another selection")
        if public_plan.get("seed_digest") != _derive_hex(
            "seed", self.episode_seed
        ):
            raise GatedDevValidationError(
                "Stage-B wrapper episode seed differs from outer public plan"
            )
        object.__setattr__(self, "_sealed_digest", self.digest)

    @property
    def status(self) -> str:
        return str(_mapping(self.outer_record["episode"], "episode")["status"])

    @property
    def score(self) -> Mapping[str, Any]:
        return _mapping(
            _mapping(self.outer_record["episode"], "episode")["score"],
            "episode score",
        )

    @property
    def gate_result(self) -> str | None:
        semantic = _mapping(self.outer_record["visual_semantic"], "semantic run")
        gate = semantic.get("support_gate")
        return None if gate is None else str(_mapping(gate, "support gate")["result"])

    @property
    def gate_passed(self) -> bool:
        return self.gate_result == SupportGateResult.ALIGNED.value

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_TASK_RUN_SCHEMA,
            "selection": self.selection.to_data(),
            "episode_seed": self.episode_seed,
            "status": self.status,
            "gate_result": self.gate_result,
            "gate_passed": self.gate_passed,
            "outer_record": dict(self.outer_record),
            "outer_record_digest": self.outer_record["record_digest"],
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        content = self.content_data()
        return {**content, "task_run_digest": canonical_digest(content)}

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise GatedDevValidationError("Stage-B task run changed after sealing")

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "GatedDevTaskRun":
        data = _fields(
            _mapping(value, "Stage-B task run"),
            {
                "schema",
                "selection",
                "episode_seed",
                "status",
                "gate_result",
                "gate_passed",
                "outer_record",
                "outer_record_digest",
                "task_run_digest",
            },
            "Stage-B task run",
        )
        if data["schema"] != GATED_DEV_TASK_RUN_SCHEMA:
            raise GatedDevValidationError("unsupported Stage-B task-run schema")
        result = cls(
            selection=GatedDevSelection.from_data(
                _mapping(data["selection"], "task-run selection")
            ),
            episode_seed=data["episode_seed"],
            outer_record=_mapping(data["outer_record"], "outer run record"),
        )
        if (
            data["outer_record_digest"] != result.outer_record["record_digest"]
            or data["status"] != result.status
            or data["gate_result"] != result.gate_result
            or data["gate_passed"] is not result.gate_passed
            or result.digest != _digest(data["task_run_digest"], "task run digest")
            or result.to_data() != dict(data)
        ):
            raise GatedDevValidationError("Stage-B task run summary differs")
        return result


def _score_int(score: Mapping[str, Any], name: str, maximum: int = 2) -> int:
    value = score.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise GatedDevValidationError(f"invalid episode score {name}")
    return value


_SCORE_FIELDS = {
    "image_correct",
    "image_total",
    "image_accuracy",
    "puzzle_correct",
    "puzzle_accuracy",
    "determinate",
    "abstentions",
    "errors",
}
_RUN_VERIFICATION_FIELDS = {
    "run_id",
    "status",
    "record_digest",
    "plan_digest",
    "calibration_campaign_digest",
    "calibration_digest",
    "expected_codex_launcher_digest",
    "cloud_policy_cache_binding",
    "policy_digest",
    "proposal_transport_digest",
    "proposal_receipt_digest",
    "pre_observation_commitment_digest",
    "lowering_archive_digest",
    "support_gate_digest",
    "proposal_freeze_digest",
    "prediction_commitment_digest",
    "archive_digest",
    "verified_blob_ids",
    "support_observation_digests",
    "query_observation_digests",
    "registered_atom_replays",
    "optional_checker_required",
}


def _canonical_episode_score(value: object) -> dict[str, object]:
    score = _fields(_mapping(value, "episode score"), _SCORE_FIELDS, "episode score")
    image_correct = _score_int(score, "image_correct")
    image_total = _score_int(score, "image_total")
    determinate = _score_int(score, "determinate")
    abstentions = _score_int(score, "abstentions")
    errors = _score_int(score, "errors")
    puzzle_correct = score["puzzle_correct"]
    image_accuracy = score["image_accuracy"]
    puzzle_accuracy = score["puzzle_accuracy"]
    if image_total != 2:
        raise GatedDevValidationError("Stage-B score must cover exactly two queries")
    if type(puzzle_correct) is not bool:
        raise GatedDevValidationError("episode puzzle_correct must be Boolean")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in (image_accuracy, puzzle_accuracy)
    ):
        raise GatedDevValidationError("episode score accuracies must be numeric")
    if (
        float(image_accuracy) != image_correct / image_total
        or puzzle_correct is not (image_correct == image_total)
        or float(puzzle_accuracy) != float(puzzle_correct)
        or determinate + abstentions != image_total
        or errors > abstentions
    ):
        raise GatedDevValidationError("episode score fields are internally inconsistent")
    return {
        "image_correct": image_correct,
        "image_total": image_total,
        "image_accuracy": image_correct / image_total,
        "puzzle_correct": puzzle_correct,
        "puzzle_accuracy": float(puzzle_correct),
        "determinate": determinate,
        "abstentions": abstentions,
        "errors": errors,
    }


@dataclass(frozen=True)
class GatedDevTaskReplayReceipt:
    """A metric-bearing outcome emitted only by full model-free replay.

    The lower verifier authenticates pixels, observations, gates, freezes,
    predictions, scores, and successful Codex environment receipts.  Stage B
    persists that verifier's exact report and derives all denominators from
    these receipts rather than from self-authenticating episode dictionaries.
    """

    task_id: str
    task_run_digest: str
    outer_record_digest: str
    status: str
    gate_result: str | None
    score: Mapping[str, Any]
    verification: Mapping[str, Any]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _identifier(self.task_id, "replay receipt task id")
        _digest(self.task_run_digest, "replay receipt task-run digest")
        _digest(self.outer_record_digest, "replay receipt outer-record digest")
        if self.status not in REPLAYABLE_TERMINAL_STATUSES:
            raise GatedDevValidationError(
                f"non-replayable Stage-B terminal status {self.status!r}"
            )
        if self.gate_result is not None and self.gate_result not in {
            item.value for item in SupportGateResult
        }:
            raise GatedDevValidationError("unknown replayed support-gate result")
        score = _canonical_episode_score(self.score)
        verification = _fields(
            _mapping(self.verification, "visual-semantic replay verification"),
            _RUN_VERIFICATION_FIELDS,
            "visual-semantic replay verification",
        )
        if (
            verification["status"] != self.status
            or verification["record_digest"] != self.outer_record_digest
            or verification["optional_checker_required"] is not False
        ):
            raise GatedDevValidationError(
                "replay verification identity differs from task outcome"
            )
        proposal_bound = (
            verification["proposal_transport_digest"] is not None
            and verification["proposal_receipt_digest"] is not None
        )
        if not proposal_bound:
            raise GatedDevValidationError(
                "replayable Stage-B outcome lacks a successful typed proposer receipt"
            )
        if self.status == EpisodeStatus.COMPLETE.value:
            if (
                self.gate_result != SupportGateResult.ALIGNED.value
                or verification["support_gate_digest"] is None
                or verification["proposal_freeze_digest"] is None
                or verification["prediction_commitment_digest"] is None
                or verification["archive_digest"] is None
            ):
                raise GatedDevValidationError(
                    "completed replay receipt lacks aligned gate/freeze/archive"
                )
        elif self.status == EpisodeStatus.SUPPORT_REJECTED.value:
            if (
                self.gate_result in {None, SupportGateResult.ALIGNED.value}
                or verification["support_gate_digest"] is None
                or verification["proposal_freeze_digest"] is None
                or verification["prediction_commitment_digest"] is not None
                or verification["archive_digest"] is not None
            ):
                raise GatedDevValidationError(
                    "support-rejected replay receipt lacks reconstructed gate/freeze"
                )
        else:
            # The lower verifier only reaches this branch for a retained
            # RejectedTypedVisualProposalAttempt with a successful Codex
            # receipt.  Generic transport failures never receive a receipt.
            if (
                self.gate_result is not None
                or verification["pre_observation_commitment_digest"] is not None
                or verification["support_gate_digest"] is not None
                or verification["proposal_freeze_digest"] is not None
                or verification["prediction_commitment_digest"] is not None
                or verification["archive_digest"] is not None
            ):
                raise GatedDevValidationError(
                    "typed proposal rejection contains post-proposal artifacts"
                )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "verification", dict(verification))
        object.__setattr__(self, "_sealed_digest", self.digest)

    @property
    def gate_passed(self) -> bool:
        return self.gate_result == SupportGateResult.ALIGNED.value

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_TASK_REPLAY_RECEIPT_SCHEMA,
            "task_id": self.task_id,
            "task_run_digest": self.task_run_digest,
            "outer_record_digest": self.outer_record_digest,
            "status": self.status,
            "gate_result": self.gate_result,
            "gate_passed": self.gate_passed,
            "score": dict(self.score),
            "verification": dict(self.verification),
            "verification_digest": canonical_digest(dict(self.verification)),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        content = self.content_data()
        return {
            **content,
            "replay_receipt_digest": canonical_digest(content),
        }

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise GatedDevValidationError("Stage-B replay receipt changed after sealing")

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "GatedDevTaskReplayReceipt":
        data = _fields(
            _mapping(value, "Stage-B replay receipt"),
            {
                "schema",
                "task_id",
                "task_run_digest",
                "outer_record_digest",
                "status",
                "gate_result",
                "gate_passed",
                "score",
                "verification",
                "verification_digest",
                "replay_receipt_digest",
            },
            "Stage-B replay receipt",
        )
        if data["schema"] != GATED_DEV_TASK_REPLAY_RECEIPT_SCHEMA:
            raise GatedDevValidationError("unsupported Stage-B replay-receipt schema")
        result = cls(
            task_id=data["task_id"],
            task_run_digest=data["task_run_digest"],
            outer_record_digest=data["outer_record_digest"],
            status=data["status"],
            gate_result=data["gate_result"],
            score=_mapping(data["score"], "replay-receipt score"),
            verification=_mapping(
                data["verification"], "visual-semantic replay verification"
            ),
        )
        if (
            data["gate_passed"] is not result.gate_passed
            or data["verification_digest"]
            != canonical_digest(dict(result.verification))
            or result.digest
            != _digest(data["replay_receipt_digest"], "replay receipt digest")
            or result.to_data() != dict(data)
        ):
            raise GatedDevValidationError("Stage-B replay receipt does not reproduce")
        return result

    @classmethod
    def from_verified(
        cls,
        run: GatedDevTaskRun,
        verified: VisualSemanticRunVerification,
    ) -> "GatedDevTaskReplayReceipt":
        if not isinstance(verified, VisualSemanticRunVerification):
            raise TypeError("verified must be VisualSemanticRunVerification")
        return cls(
            task_id=run.selection.task_id,
            task_run_digest=run.digest,
            outer_record_digest=str(run.outer_record["record_digest"]),
            status=run.status,
            gate_result=run.gate_result,
            score=run.score,
            verification=verified.to_data(),
        )


def _strictly_verify_gated_dev_task_run(
    run: GatedDevTaskRun,
    *,
    blob_bytes_by_id: Mapping[str, bytes],
    stage_a_campaign: (
        SemanticCalibrationCampaignArtifact
        | VisualSemanticCalibrationCampaignAnchor
    ),
) -> GatedDevTaskReplayReceipt:
    """Fail closed unless the complete terminal record reconstructs exactly."""

    try:
        campaign_anchor = (
            stage_a_campaign
            if isinstance(
                stage_a_campaign, VisualSemanticCalibrationCampaignAnchor
            )
            else VisualSemanticCalibrationCampaignAnchor.from_verified_campaign(
                stage_a_campaign
            )
        )
        verified = _verify_visual_semantic_run_data_with_verified_anchor(
            run.outer_record,
            blob_bytes_by_id=blob_bytes_by_id,
            campaign_anchor=campaign_anchor,
        )
        receipt = GatedDevTaskReplayReceipt.from_verified(run, verified)
    except (VisualSemanticRunVerificationError, TypeError, ValueError) as exc:
        raise GatedDevValidationError(
            f"Stage-B task {run.selection.task_id} is not a permitted replayable "
            "terminal outcome; generic transport, gate-construction, observer-"
            f"preparation, and query-observer failures are batch-fatal: {exc}"
        ) from exc
    return receipt


@dataclass(frozen=True)
class GatedDevValidationSummary:
    selected_clusters: int
    gate_passed_clusters: int
    status_counts: tuple[tuple[str, int], ...]
    gate_result_counts: tuple[tuple[str, int], ...]
    conditional_image_correct: int
    conditional_image_total: int
    conditional_determinate: int
    conditional_abstentions: int
    conditional_errors: int
    bounds: tuple[RawSimultaneousHoeffdingBound, ...]
    threshold_checks: tuple[tuple[str, bool], ...]
    validation_status: str
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_int(self.selected_clusters, "selected cluster count")
        if not 0 <= self.gate_passed_clusters <= self.selected_clusters:
            raise GatedDevValidationError("invalid gate-passed cluster count")
        if tuple(item.metric for item in self.bounds) != METRIC_NAMES:
            raise GatedDevValidationError("Stage-B bound family/order differs")
        if self.validation_status not in {"pilot", "validated"}:
            raise GatedDevValidationError("unknown Stage-B validation status")
        checks_pass = all(value for _, value in self.threshold_checks)
        if (self.validation_status == "validated") is not checks_pass:
            raise GatedDevValidationError("validation status differs from thresholds")
        expected_failures = tuple(name for name, passed in self.threshold_checks if not passed)
        if self.failure_reasons != expected_failures:
            raise GatedDevValidationError("validation failure reasons differ")

    @classmethod
    def from_replay_receipts(
        cls,
        receipts: Sequence[GatedDevTaskReplayReceipt],
        policy: GatedDevAcceptancePolicy,
    ) -> "GatedDevValidationSummary":
        from collections import Counter

        frozen = tuple(receipts)
        if not frozen:
            raise GatedDevValidationError(
                "Stage-B summary requires strictly replayed task receipts"
            )
        for item in frozen:
            if not isinstance(item, GatedDevTaskReplayReceipt):
                raise TypeError("summary inputs must be replay receipts")
            item.assert_untampered()
        status_counts = Counter(item.status for item in frozen)
        gate_counts = Counter(item.gate_result or "not_reached" for item in frozen)
        gated = tuple(item for item in frozen if item.gate_passed)
        image_correct = 0
        determinate = 0
        abstentions = 0
        errors = 0
        both_correct = 0
        fully_determinate = 0
        any_abstention = 0
        any_error = 0
        for item in gated:
            score = item.score
            current_correct = _score_int(score, "image_correct")
            current_determinate = _score_int(score, "determinate")
            current_abstentions = _score_int(score, "abstentions")
            current_errors = _score_int(score, "errors")
            if current_determinate + current_abstentions != 2:
                raise GatedDevValidationError("conditional query counts do not cover task")
            image_correct += current_correct
            determinate += current_determinate
            abstentions += current_abstentions
            errors += current_errors
            both_correct += int(score.get("puzzle_correct") is True)
            fully_determinate += int(current_determinate == 2)
            any_abstention += int(current_abstentions > 0)
            any_error += int(current_errors > 0)
        selected_n = len(frozen)
        gated_n = len(gated)
        successes = (
            gated_n,
            both_correct,
            fully_determinate,
            any_abstention,
            any_error,
        )
        denominators = (selected_n, gated_n, gated_n, gated_n, gated_n)
        bounds = tuple(
            RawSimultaneousHoeffdingBound(
                metric=name,
                successes=success,
                cluster_count=count,
                confidence_level=policy.confidence_level,
            )
            for name, success, count in zip(
                METRIC_NAMES, successes, denominators, strict=True
            )
        )
        bound_by_name = {item.metric: item for item in bounds}

        def lower(name: str) -> float:
            value = bound_by_name[name].raw_lower
            return float("-inf") if value is None else value

        def upper(name: str) -> float:
            value = bound_by_name[name].raw_upper
            return float("inf") if value is None else value

        checks = (
            (
                "dependence_design_authorized",
                False,
            ),
            (
                "minimum_selected_clusters",
                selected_n >= policy.minimum_selected_clusters,
            ),
            (
                "minimum_gate_passed_clusters",
                gated_n >= policy.minimum_gate_passed_clusters,
            ),
            (
                "minimum_gate_coverage_lower",
                lower("selected_gate_coverage")
                >= policy.minimum_gate_coverage_lower,
            ),
            (
                "minimum_both_query_correct_lower",
                lower("gated_both_query_correct")
                >= policy.minimum_both_query_correct_lower,
            ),
            (
                "minimum_fully_determinate_lower",
                lower("gated_fully_determinate")
                >= policy.minimum_fully_determinate_lower,
            ),
            (
                "maximum_any_abstention_upper",
                upper("gated_any_abstention")
                <= policy.maximum_any_abstention_upper,
            ),
            (
                "maximum_any_error_upper",
                upper("gated_any_error") <= policy.maximum_any_error_upper,
            ),
        )
        return cls(
            selected_clusters=selected_n,
            gate_passed_clusters=gated_n,
            status_counts=tuple(sorted(status_counts.items())),
            gate_result_counts=tuple(sorted(gate_counts.items())),
            conditional_image_correct=image_correct,
            conditional_image_total=2 * gated_n,
            conditional_determinate=determinate,
            conditional_abstentions=abstentions,
            conditional_errors=errors,
            bounds=bounds,
            threshold_checks=checks,
            validation_status="validated" if all(value for _, value in checks) else "pilot",
            failure_reasons=tuple(name for name, passed in checks if not passed),
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GATED_DEV_SUMMARY_SCHEMA,
            "estimand": ESTIMAND,
            "selected_denominator": {
                "clusters": self.selected_clusters,
                "includes_proposal_errors": True,
                "includes_support_rejections": True,
                "status_counts": dict(self.status_counts),
                "gate_result_counts": dict(self.gate_result_counts),
            },
            "gate_passed_conditional_denominator": {
                "clusters": self.gate_passed_clusters,
                "query_images": self.conditional_image_total,
                "image_correct": self.conditional_image_correct,
                "determinate": self.conditional_determinate,
                "abstentions": self.conditional_abstentions,
                "errors": self.conditional_errors,
                "task_both_query_correct_successes": self.bounds[1].successes,
                "task_fully_determinate_successes": self.bounds[2].successes,
                "task_any_abstention_successes": self.bounds[3].successes,
                "task_any_error_successes": self.bounds[4].successes,
            },
            "bounds": [item.to_data() for item in self.bounds],
            "threshold_checks": dict(self.threshold_checks),
            "validation_status": self.validation_status,
            "authorizes_sealed_benchmark": self.validation_status == "validated",
            "failure_reasons": list(self.failure_reasons),
            "interpretation": (
                "conditional operational estimands, not individual semantic proofs"
            ),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        content = self.content_data()
        return {**content, "summary_digest": canonical_digest(content)}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        replay_receipts: Sequence[GatedDevTaskReplayReceipt],
        policy: GatedDevAcceptancePolicy,
    ) -> "GatedDevValidationSummary":
        archived = _mapping(value, "Stage-B summary")
        expected = cls.from_replay_receipts(replay_receipts, policy)
        if expected.to_data() != dict(archived):
            raise GatedDevValidationError("Stage-B summary does not recompute")
        return expected


class _TransportIdentityMonitor:
    """Fail the batch on the first successful receipt identity mismatch."""

    def __init__(self, *, launcher_digest: str, cache_binding: str) -> None:
        self.launcher_digest = launcher_digest
        self.cache_binding = cache_binding
        self._lock = threading.Lock()
        self._fatal: GatedDevTransportIdentityError | None = None

    @property
    def fatal(self) -> GatedDevTransportIdentityError | None:
        with self._lock:
            return self._fatal

    def wrap(self, transport: StructuredTransport) -> StructuredTransport:
        if not callable(transport):
            raise TypeError("transport must be callable")

        def checked(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                fatal = self._fatal
            if fatal is not None:
                raise fatal
            supplied = kwargs.get("expected_launcher_digest")
            if supplied is not None and supplied != self.launcher_digest:
                raise GatedDevTransportIdentityError(
                    "transport call expected another Codex launcher"
                )
            # The default transports authenticate the resolved executable
            # immediately before each actual model call.  Injecting here also
            # makes the requirement visible to test/custom transports.
            kwargs["expected_launcher_digest"] = self.launcher_digest
            result = transport(*args, **kwargs)
            if isinstance(result, CodexStructuredResult):
                receipt = result.receipt
                if not isinstance(receipt, CodexReceipt):
                    error = GatedDevTransportIdentityError(
                        "successful structured transport lacks a Codex receipt"
                    )
                elif (
                    receipt.codex_launcher_digest != self.launcher_digest
                    or receipt.cloud_config_bundle_cache_binding != self.cache_binding
                ):
                    error = GatedDevTransportIdentityError(
                        "successful Codex receipt differs from frozen Stage-A "
                        "launcher/cache identity"
                    )
                else:
                    return result
                with self._lock:
                    if self._fatal is None:
                        self._fatal = error
                raise error
            return result

        return checked


@dataclass(frozen=True)
class GatedDevValidationArtifact:
    plan: GatedDevValidationPlan
    stage_a_campaign: SemanticCalibrationCampaignArtifact
    visual_semantic_policy: VisualSemanticPolicy
    exposure_predecessor: ExposureLedger
    exposure_successor: ExposureLedger
    task_runs: tuple[GatedDevTaskRun, ...]
    task_replay_receipts: tuple[GatedDevTaskReplayReceipt, ...]
    summary: GatedDevValidationSummary
    _stage_a_campaign_payload: bytes = field(
        init=False, repr=False, compare=False
    )
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, GatedDevValidationPlan):
            raise TypeError("plan must be GatedDevValidationPlan")
        self.plan.assert_untampered()
        if not isinstance(
            self.stage_a_campaign, SemanticCalibrationCampaignArtifact
        ):
            raise TypeError(
                "stage_a_campaign must be a semantic calibration campaign"
            )
        self.stage_a_campaign.assert_untampered()
        stage_a_campaign_payload = canonical_json(
            self.stage_a_campaign.to_data()
        )
        stage_a_campaign_data = _mapping(
            json.loads(stage_a_campaign_payload),
            "central Stage-A campaign snapshot",
        )
        if (
            self.stage_a_campaign._sealed_digest
            != self.plan.stage_a_campaign_digest
            or self.stage_a_campaign.calibration._sealed_digest
            != self.plan.stage_a_calibration_digest
            or self.stage_a_campaign.calibration.family.digest()
            != self.plan.stage_a_family_digest
            or self.stage_a_campaign.calibration.protocol.digest()
            != self.plan.stage_a_protocol_digest
            or stage_a_campaign_data.get("campaign_digest")
            != self.plan.stage_a_campaign_digest
        ):
            raise GatedDevValidationError(
                "central Stage-A campaign/calibration/family/protocol differs "
                "from plan"
            )
        object.__setattr__(
            self,
            "_stage_a_campaign_payload",
            stage_a_campaign_payload,
        )
        if not isinstance(self.visual_semantic_policy, VisualSemanticPolicy):
            raise TypeError("visual_semantic_policy must be VisualSemanticPolicy")
        if self.visual_semantic_policy.digest() != self.plan.visual_semantic_policy_digest:
            raise GatedDevValidationError("artifact policy differs from plan")
        if not isinstance(self.exposure_predecessor, ExposureLedger) or not isinstance(
            self.exposure_successor, ExposureLedger
        ):
            raise TypeError("artifact exposure values must be ExposureLedger")
        if self.exposure_predecessor.digest != self.plan.exposure_predecessor_digest:
            raise GatedDevValidationError("artifact exposure predecessor differs")
        if _ledger_exposed_hd_constituent_attributes(
            self.exposure_predecessor
        ) != self.plan.exposed_hd_constituent_attributes:
            raise GatedDevValidationError(
                "artifact predecessor HD constituent inventory differs from plan"
            )
        expected_successor = self.exposure_predecessor.record(
            phase="dev-validation",
            actor=self.plan.exposure_actor,
            purpose=self.plan.exposure_purpose,
            task_ids=(item.task_id for item in self.plan.selections),
            source=self.plan.exposure_source,
            observed_at=self.plan.exposure_observed_at,
            require_unseen=True,
        )
        if expected_successor != self.exposure_successor:
            raise GatedDevValidationError("artifact exposure successor is not exact")
        if (
            not isinstance(self.task_runs, tuple)
            or len(self.task_runs) != len(self.plan.selections)
            or any(not isinstance(item, GatedDevTaskRun) for item in self.task_runs)
        ):
            raise GatedDevValidationError("task runs do not cover exact selection")
        if (
            not isinstance(self.task_replay_receipts, tuple)
            or len(self.task_replay_receipts) != len(self.task_runs)
            or any(
                not isinstance(item, GatedDevTaskReplayReceipt)
                for item in self.task_replay_receipts
            )
        ):
            raise GatedDevValidationError(
                "replay receipts do not cover the exact task-run inventory"
            )
        for selection, run, receipt in zip(
            self.plan.selections,
            self.task_runs,
            self.task_replay_receipts,
            strict=True,
        ):
            run.assert_untampered()
            receipt.assert_untampered()
            if run.selection != selection or run.episode_seed != _episode_seed(
                self.plan, selection
            ):
                raise GatedDevValidationError("task-run order or seed differs")
            if (
                receipt.task_id != selection.task_id
                or receipt.task_run_digest != run.digest
                or receipt.outer_record_digest != run.outer_record["record_digest"]
                or receipt.status != run.status
                or receipt.gate_result != run.gate_result
                or dict(receipt.score) != _canonical_episode_score(run.score)
            ):
                raise GatedDevValidationError(
                    "strict replay receipt differs from its Stage-B task run"
                )
            record = run.outer_record
            if (
                record.get("corpus_manifest_digest")
                != self.plan.source_corpus_manifest_digest
                or record.get("split_source_digest") != self.plan.split_source_digest
                or record.get("calibration_campaign_digest")
                != self.plan.stage_a_campaign_digest
                or record.get("calibration_digest")
                != self.plan.stage_a_calibration_digest
            ):
                raise GatedDevValidationError("outer run trust anchor differs")
            episode_plan = _mapping(record.get("plan"), "outer episode plan")
            if (
                episode_plan.get("predicate_mode")
                != VISUAL_SEMANTIC_PREDICATE_MODE
                or episode_plan.get("predicate_policy_digest")
                != self.plan.visual_semantic_policy_digest
            ):
                raise GatedDevValidationError("outer run predicate policy differs")
            if run.status == EpisodeStatus.COMPLETE.value:
                verify_archive_data(
                    _mapping(record.get("run_archive"), "completed run archive")
                )
        expected_summary = GatedDevValidationSummary.from_replay_receipts(
            self.task_replay_receipts, self.plan.acceptance_policy
        )
        if self.summary.to_data() != expected_summary.to_data():
            raise GatedDevValidationError("artifact summary does not recompute")
        object.__setattr__(self, "_sealed_digest", self.digest)

    def content_data(self) -> dict[str, object]:
        plan = self.plan.to_data()
        task_runs = [item.to_data() for item in self.task_runs]
        replay_receipts = [
            item.to_data() for item in self.task_replay_receipts
        ]
        summary = self.summary.to_data()
        return {
            "schema": GATED_DEV_ARTIFACT_SCHEMA,
            "reference_execution_semantics": REFERENCE_SEMANTICS,
            "estimand": ESTIMAND,
            "stage_a_authority_layout": CENTRAL_CAMPAIGN_LAYOUT,
            "plan": plan,
            "plan_digest": plan["plan_digest"],
            "stage_a_campaign": json.loads(self._stage_a_campaign_payload),
            "stage_a_campaign_digest": self.plan.stage_a_campaign_digest,
            "stage_a_command_receipt_digest": (
                self.plan.stage_a_command_receipt_digest
            ),
            "visual_semantic_policy": self.visual_semantic_policy.to_data(),
            "visual_semantic_policy_digest": self.visual_semantic_policy.digest(),
            "exposure_predecessor": self.exposure_predecessor.to_dict(),
            "exposure_predecessor_digest": self.exposure_predecessor.digest,
            "exposure_successor": self.exposure_successor.to_dict(),
            "exposure_successor_digest": self.exposure_successor.digest,
            "exposure_successor_filename": (
                self.exposure_successor.digest.removeprefix("sha256:")
                + ".exposure.json"
            ),
            "task_runs": task_runs,
            "task_run_digests": [item["task_run_digest"] for item in task_runs],
            "task_replay_receipts": replay_receipts,
            "task_replay_receipt_digests": [
                item["replay_receipt_digest"] for item in replay_receipts
            ],
            "summary": summary,
            "summary_digest": summary["summary_digest"],
            "python_predicate_authoritative": True,
            "python_validation_authoritative": True,
            "python_replay_authoritative": True,
            "stage_a_family_refit_performed": False,
            "optional_checker": None,
            "optional_checker_in_artifact_identity": False,
            "optional_checker_may_affect_result": False,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        content = self.content_data()
        return {
            **content,
            "validation_artifact_digest": canonical_digest(content),
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_data())

    def assert_untampered(self) -> None:
        self.plan.assert_untampered()
        self.stage_a_campaign.assert_untampered()
        for item in self.task_runs:
            item.assert_untampered()
        for item in self.task_replay_receipts:
            item.assert_untampered()
        if self.digest != self._sealed_digest:
            raise GatedDevValidationError("Stage-B artifact changed after sealing")

    def write_once(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_bytes()
        try:
            with destination.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise GatedDevValidationError(
                    f"refusing to overwrite different Stage-B artifact at {destination}"
                )
        return destination

    def write_content_addressed(self, directory: str | Path) -> Path:
        return self.write_once(
            Path(directory) / (self.digest + ".gated-dev-validation.json")
        )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        stage_a_campaign: SemanticCalibrationCampaignArtifact,
        stage_a_command_receipt: StageACommandReceipt,
        corpus: ShapeBongardCorpus,
        source_corpus_manifest: CorpusManifest,
        replay_bytes_by_task: Mapping[str, Mapping[str, bytes]],
    ) -> "GatedDevValidationArtifact":
        data = _fields(
            _mapping(value, "Stage-B validation artifact"),
            {
                "schema",
                "reference_execution_semantics",
                "estimand",
                "stage_a_authority_layout",
                "plan",
                "plan_digest",
                "stage_a_campaign",
                "stage_a_campaign_digest",
                "stage_a_command_receipt_digest",
                "visual_semantic_policy",
                "visual_semantic_policy_digest",
                "exposure_predecessor",
                "exposure_predecessor_digest",
                "exposure_successor",
                "exposure_successor_digest",
                "exposure_successor_filename",
                "task_runs",
                "task_run_digests",
                "task_replay_receipts",
                "task_replay_receipt_digests",
                "summary",
                "summary_digest",
                "python_predicate_authoritative",
                "python_validation_authoritative",
                "python_replay_authoritative",
                "stage_a_family_refit_performed",
                "optional_checker",
                "optional_checker_in_artifact_identity",
                "optional_checker_may_affect_result",
                "validation_artifact_digest",
            },
            "Stage-B validation artifact",
        )
        fixed = {
            "schema": GATED_DEV_ARTIFACT_SCHEMA,
            "reference_execution_semantics": REFERENCE_SEMANTICS,
            "estimand": ESTIMAND,
            "stage_a_authority_layout": CENTRAL_CAMPAIGN_LAYOUT,
            "python_predicate_authoritative": True,
            "python_validation_authoritative": True,
            "python_replay_authoritative": True,
            "stage_a_family_refit_performed": False,
            "optional_checker": None,
            "optional_checker_in_artifact_identity": False,
            "optional_checker_may_affect_result": False,
        }
        for name, expected in fixed.items():
            if data[name] != expected or type(data[name]) is not type(expected):
                raise GatedDevValidationError(f"Stage-B authority changed {name}")
        plan = GatedDevValidationPlan.from_data(
            _mapping(data["plan"], "Stage-B plan")
        )
        if data["plan_digest"] != plan.digest:
            raise GatedDevValidationError("Stage-B plan digest differs")
        if (
            data["stage_a_campaign_digest"]
            != plan.stage_a_campaign_digest
            or data["stage_a_command_receipt_digest"]
            != plan.stage_a_command_receipt_digest
        ):
            raise GatedDevValidationError(
                "central Stage-A campaign/receipt references differ from plan"
            )
        stage_a_campaign.assert_untampered()
        if (
            data["stage_a_campaign_digest"]
            != stage_a_campaign._sealed_digest
        ):
            raise GatedDevValidationError("trusted Stage-A campaign differs")
        _validate_full_source_commitment(
            corpus,
            source_corpus_manifest,
            expected_manifest_digest=plan.source_corpus_manifest_digest,
            expected_split_source_digest=plan.split_source_digest,
        )
        verified_campaign, _ = verify_semantic_campaign_against_corpus(
            _mapping(data["stage_a_campaign"], "central Stage-A campaign"),
            corpus=corpus,
            corpus_manifest=source_corpus_manifest,
        )
        if (
            verified_campaign._sealed_digest
            != stage_a_campaign._sealed_digest
            or verified_campaign._sealed_digest
            != plan.stage_a_campaign_digest
        ):
            raise GatedDevValidationError(
                "central Stage-A campaign does not replay against the trusted "
                "full corpus and external campaign identity"
            )
        if (
            verified_campaign.calibration.family.digest()
            != plan.stage_a_family_digest
            or verified_campaign.calibration.protocol.digest()
            != plan.stage_a_protocol_digest
        ):
            raise GatedDevValidationError(
                "central Stage-A family/protocol differs from Stage-B plan"
            )
        stage_a_campaign = verified_campaign
        campaign_anchor = _campaign_anchor_from_verified_stage_a(
            stage_a_campaign
        )
        receipt = _authenticate_stage_a_command_receipt(
            stage_a_command_receipt,
            campaign=stage_a_campaign,
        )
        if receipt.receipt_digest != plan.stage_a_command_receipt_digest:
            raise GatedDevValidationError(
                "trusted Stage-A command receipt differs from Stage-B plan"
            )
        if data["stage_a_command_receipt_digest"] != receipt.receipt_digest:
            raise GatedDevValidationError(
                "central Stage-A campaign joins another command receipt"
            )
        policy = VisualSemanticPolicy.from_data(
            _mapping(data["visual_semantic_policy"], "visual-semantic policy")
        )
        if data["visual_semantic_policy_digest"] != policy.digest():
            raise GatedDevValidationError("visual-semantic policy digest differs")
        predecessor = ExposureLedger.from_dict(
            _mapping(data["exposure_predecessor"], "exposure predecessor")
        )
        successor = ExposureLedger.from_dict(
            _mapping(data["exposure_successor"], "exposure successor")
        )
        if (
            data["exposure_predecessor_digest"] != predecessor.digest
            or data["exposure_successor_digest"] != successor.digest
            or data["exposure_successor_filename"]
            != successor.digest.removeprefix("sha256:") + ".exposure.json"
        ):
            raise GatedDevValidationError("exposure ledger digest/filename differs")
        proposal_archive = (
            stage_a_campaign.score_batch.commitment_batch.proposal_archive
        )
        if (
            not _ledger_extends(predecessor, proposal_archive.exposure_successor)
            or plan.execution_config != proposal_archive.execution_config
        ):
            raise GatedDevValidationError(
                "Stage-B predecessor/environment does not extend trusted Stage A"
            )
        _audit_gated_dev_selection_against_corpus(
            corpus,
            plan=plan,
            predecessor=predecessor,
        )
        runs = tuple(
            GatedDevTaskRun.from_data(_mapping(item, "Stage-B task run"))
            for item in _list(data["task_runs"], "Stage-B task runs")
        )
        if data["task_run_digests"] != [item.digest for item in runs]:
            raise GatedDevValidationError("task-run digest inventory differs")
        archived_receipts = tuple(
            GatedDevTaskReplayReceipt.from_data(
                _mapping(item, "Stage-B replay receipt")
            )
            for item in _list(
                data["task_replay_receipts"], "Stage-B replay receipts"
            )
        )
        if data["task_replay_receipt_digests"] != [
            item.digest for item in archived_receipts
        ]:
            raise GatedDevValidationError("replay-receipt digest inventory differs")
        if not isinstance(replay_bytes_by_task, Mapping) or any(
            not isinstance(task_id, str) or not isinstance(blobs, Mapping)
            for task_id, blobs in replay_bytes_by_task.items()
        ):
            raise GatedDevValidationError(
                "replay_bytes_by_task must cover every selected task"
            )
        expected_task_ids = {item.selection.task_id for item in runs}
        if set(replay_bytes_by_task) != expected_task_ids:
            raise GatedDevValidationError(
                "strict replay byte inventory differs from selected tasks"
            )
        official_replay_bytes = {
            run.selection.task_id: _capture_gated_dev_task_replay_bytes(
                corpus,
                source_corpus_manifest,
                plan,
                run,
            )
            for run in runs
        }
        supplied_replay_bytes = {
            task_id: dict(blobs)
            for task_id, blobs in replay_bytes_by_task.items()
        }
        if supplied_replay_bytes != official_replay_bytes:
            raise GatedDevValidationError(
                "caller replay preimages differ from exact selected panels in the "
                "trusted full TaskManifests"
            )
        replay_receipts = tuple(
            _strictly_verify_gated_dev_task_run(
                run,
                blob_bytes_by_id=official_replay_bytes[run.selection.task_id],
                stage_a_campaign=campaign_anchor,
            )
            for run in runs
        )
        if [item.to_data() for item in replay_receipts] != [
            item.to_data() for item in archived_receipts
        ]:
            raise GatedDevValidationError(
                "archived Stage-B replay receipts differ from strict reconstruction"
            )
        summary = GatedDevValidationSummary.from_data(
            _mapping(data["summary"], "Stage-B summary"),
            replay_receipts=replay_receipts,
            policy=plan.acceptance_policy,
        )
        if data["summary_digest"] != summary.digest:
            raise GatedDevValidationError("Stage-B summary digest differs")
        result = cls(
            plan,
            stage_a_campaign,
            policy,
            predecessor,
            successor,
            runs,
            replay_receipts,
            summary,
        )
        if result.digest != _digest(
            data["validation_artifact_digest"], "validation artifact digest"
        ) or result.to_data() != dict(data):
            raise GatedDevValidationError("Stage-B artifact does not reproduce")
        return result


def _preflight_execution(
    *,
    plan: GatedDevValidationPlan,
    campaign: SemanticCalibrationCampaignArtifact,
    stage_a_command_receipt: StageACommandReceipt,
    policy: VisualSemanticPolicy,
    corpus: ShapeBongardCorpus,
    manifest: CorpusManifest,
    predecessor: ExposureLedger,
    snapshot: CloudPolicyCacheSnapshot,
    launcher_digest: str,
) -> None:
    plan.assert_untampered()
    receipt = _authenticate_stage_a_command_receipt(
        stage_a_command_receipt,
        campaign=campaign,
        cloud_policy_cache_snapshot=snapshot,
    )
    if receipt.receipt_digest != plan.stage_a_command_receipt_digest:
        raise GatedDevValidationError(
            "execution Stage-A command receipt differs from frozen plan"
        )
    source_digest = _validate_full_source_commitment(
        corpus,
        manifest,
        expected_manifest_digest=plan.source_corpus_manifest_digest,
        expected_split_source_digest=plan.split_source_digest,
    )
    _validate_campaign_policy(campaign, policy, source_manifest_digest=source_digest)
    if (
        campaign.digest != plan.stage_a_campaign_digest
        or campaign.calibration.digest != plan.stage_a_calibration_digest
        or campaign.calibration.family.digest() != plan.stage_a_family_digest
        or campaign.calibration.protocol.digest() != plan.stage_a_protocol_digest
        or policy.digest() != plan.visual_semantic_policy_digest
    ):
        raise GatedDevValidationError("Stage-A/policy preflight differs from plan")
    if (
        snapshot.binding != plan.execution_config.cloud_policy_cache_binding
        or launcher_digest != plan.execution_config.expected_codex_launcher_digest
    ):
        raise GatedDevValidationError("execution launcher/cache differs from plan")
    _audit_gated_dev_selection_against_corpus(
        corpus,
        plan=plan,
        predecessor=predecessor,
    )


def _label_nonce(root: str, task_id: str, plan_digest: str) -> str:
    return hashlib.sha256(
        ("gkm-stage-b-label-nonce/v1\0" + root + "\0" + plan_digest + "\0" + task_id).encode(
            "utf-8"
        )
    ).hexdigest()


def run_gated_dev_validation(
    corpus: ShapeBongardCorpus,
    plan: GatedDevValidationPlan,
    *,
    source_corpus_manifest: CorpusManifest,
    stage_a_campaign: SemanticCalibrationCampaignArtifact,
    stage_a_command_receipt: StageACommandReceipt,
    visual_semantic_policy: VisualSemanticPolicy,
    exposure_predecessor: ExposureLedger,
    exposure_output_directory: str | Path,
    artifact_output_directory: str | Path,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    label_nonce_root: str | None = None,
    verbose: bool = False,
    proposer_transport: StructuredTransport = run_codex_structured,
    scorer_transport: StructuredTransport = run_codex_named_images_structured,
) -> GatedDevValidationArtifact:
    """Execute and persist the preregistered Stage-B batch.

    A performance non-passage still returns and writes the complete ``pilot``
    artifact.  Only trust-boundary failures (corpus, campaign, environment, or
    receipt substitution) abort without claiming a benchmark result.
    """

    if not isinstance(cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("cloud_policy_cache_snapshot must be CloudPolicyCacheSnapshot")
    if not isinstance(stage_a_command_receipt, StageACommandReceipt):
        raise TypeError("stage_a_command_receipt must be StageACommandReceipt")
    plan.assert_untampered()
    # This is a fresh measurement, not a caller assertion.  It happens before
    # preflight can select/open any DEV task and is the identity passed to all
    # transports and successful-receipt checks below.
    launcher = codex_cli_authenticated_fingerprint(
        plan.execution_config.executable,
        expected_launcher_digest=(
            plan.execution_config.expected_codex_launcher_digest
        ),
    )
    if not isinstance(launcher, Mapping) or set(launcher) != {
        "version",
        "launcher_digest",
    }:
        raise GatedDevValidationError("Codex launcher fingerprint is malformed")
    launcher_digest = _digest(
        launcher["launcher_digest"], "measured Codex launcher digest"
    )
    if (
        launcher_digest != plan.execution_config.expected_codex_launcher_digest
        or launcher_digest != stage_a_command_receipt.launcher_digest
        or launcher["version"] != stage_a_command_receipt.launcher_version
    ):
        raise GatedDevValidationError(
            "measured Codex launcher differs from Stage-A command receipt/plan"
        )
    _preflight_execution(
        plan=plan,
        campaign=stage_a_campaign,
        stage_a_command_receipt=stage_a_command_receipt,
        policy=visual_semantic_policy,
        corpus=corpus,
        manifest=source_corpus_manifest,
        predecessor=exposure_predecessor,
        snapshot=cloud_policy_cache_snapshot,
        launcher_digest=launcher_digest,
    )
    # One immutable event records the complete batch, and its content-addressed
    # file is durable before an executor or transport exists.
    exposure_successor = exposure_predecessor.record(
        phase="dev-validation",
        actor=plan.exposure_actor,
        purpose=plan.exposure_purpose,
        task_ids=(item.task_id for item in plan.selections),
        source=plan.exposure_source,
        observed_at=plan.exposure_observed_at,
        known_task_ids=corpus.task_ids,
        require_unseen=True,
    )
    _persist_gated_dev_exposure_precommit(
        exposure_successor,
        exposure_output_directory,
    )

    nonce_root = label_nonce_root or secrets.token_hex(32)
    if not isinstance(nonce_root, str) or not nonce_root:
        raise GatedDevValidationError("label nonce root must be non-empty")
    monitor = _TransportIdentityMonitor(
        launcher_digest=launcher_digest,
        cache_binding=cloud_policy_cache_snapshot.binding,
    )
    checked_proposer = monitor.wrap(proposer_transport)
    checked_scorer = monitor.wrap(scorer_transport)
    calibration = stage_a_campaign.calibration
    campaign_anchor = _campaign_anchor_from_verified_stage_a(
        stage_a_campaign
    )

    def execute(selection: GatedDevSelection) -> GatedDevTaskRun:
        episode_seed = _episode_seed(plan, selection)
        episode_plan = prepare_episode(
            corpus,
            selection.task_id,
            seed=episode_seed,
            corpus_manifest=source_corpus_manifest,
            label_seal_nonce=_label_nonce(
                nonce_root, selection.task_id, plan.digest
            ),
            predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
            predicate_policy_digest=visual_semantic_policy.digest(),
        )
        episode = VisualSemanticEpisode(
            task_id=selection.task_id,
            support_commitment=episode_plan.support,
            policy=visual_semantic_policy,
            family=calibration.family,
            protocol=calibration.protocol,
            proposer_minutes=plan.execution_config.proposer_minutes,
            scorer_minutes=plan.execution_config.scorer_minutes,
            verbose=verbose,
            executable=plan.execution_config.executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_codex_launcher_digest=launcher_digest,
            expected_cloud_policy_cache_binding=(
                cloud_policy_cache_snapshot.binding
            ),
            proposer_transport=checked_proposer,
            scorer_transport=checked_scorer,
        )
        result = run_episode(
            episode_plan,
            episode,
            episode,
            support_gate_policy=SupportGatePolicy.visual_semantic(),
        )
        fatal = monitor.fatal
        if fatal is not None:
            raise fatal
        outer = _build_visual_semantic_run_record_from_verified_anchor(
            corpus_manifest_digest=plan.source_corpus_manifest_digest,
            split_source_digest=plan.split_source_digest,
            official_release=None,
            campaign_anchor=campaign_anchor,
            plan=episode_plan,
            result=result,
            episode=episode,
            exposure=None,
            compact_calibration=True,
        )
        return GatedDevTaskRun(selection, episode_seed, outer)

    # ``executor.map`` preserves preregistered order even when tasks finish out
    # of order.  Each worker owns a distinct VisualSemanticEpisode and all of
    # its observation sessions; only the immutable cache snapshot is shared.
    with ThreadPoolExecutor(
        max_workers=min(plan.task_max_workers, len(plan.selections)),
        thread_name_prefix="bongard-stage-b",
    ) as executor:
        runs = tuple(executor.map(execute, plan.selections))
    fatal = monitor.fatal
    if fatal is not None:
        raise fatal
    # Every terminal outcome is now cold-replayed before it can enter either
    # denominator.  Unsupported infrastructure failures abort the batch; the
    # already-persisted exposure transition remains the honest audit trail.
    replay_bytes = {
        run.selection.task_id: _capture_gated_dev_task_replay_bytes(
            corpus,
            source_corpus_manifest,
            plan,
            run,
        )
        for run in runs
    }
    replay_receipts = tuple(
        _strictly_verify_gated_dev_task_run(
            run,
            blob_bytes_by_id=replay_bytes[run.selection.task_id],
            stage_a_campaign=campaign_anchor,
        )
        for run in runs
    )
    summary = GatedDevValidationSummary.from_replay_receipts(
        replay_receipts, plan.acceptance_policy
    )
    artifact = GatedDevValidationArtifact(
        plan,
        stage_a_campaign,
        visual_semantic_policy,
        exposure_predecessor,
        exposure_successor,
        runs,
        replay_receipts,
        summary,
    )
    cold_verified = GatedDevValidationArtifact.from_data(
        artifact.to_data(),
        stage_a_campaign=stage_a_campaign,
        stage_a_command_receipt=stage_a_command_receipt,
        corpus=corpus,
        source_corpus_manifest=source_corpus_manifest,
        replay_bytes_by_task=replay_bytes,
    )
    if (
        cold_verified.digest != artifact.digest
        or cold_verified.to_data() != artifact.to_data()
    ):
        raise GatedDevValidationError(
            "cold-verified Stage-B reconstruction differs before persistence"
        )
    cold_verified.write_content_addressed(artifact_output_directory)
    return cold_verified


def _capture_gated_dev_task_replay_bytes(
    corpus: ShapeBongardCorpus,
    source_corpus_manifest: CorpusManifest,
    plan: GatedDevValidationPlan,
    run: GatedDevTaskRun,
) -> dict[str, bytes]:
    """Join one run to the trusted TaskManifest, then return exact preimages."""

    _validate_full_source_commitment(
        corpus,
        source_corpus_manifest,
        expected_manifest_digest=plan.source_corpus_manifest_digest,
        expected_split_source_digest=plan.split_source_digest,
    )
    committed_by_id = {
        item.task_id: item for item in source_corpus_manifest.tasks
    }
    try:
        committed_task = committed_by_id[run.selection.task_id]
    except KeyError as exc:
        raise GatedDevValidationError(
            f"selected task {run.selection.task_id} is absent from trusted manifest"
        ) from exc
    expected_task_digest = "sha256:" + canonical_digest(
        committed_task.content_dict()
    )
    fresh_task = corpus.task(run.selection.task_id).build_manifest()
    assignment = corpus.assignment(run.selection.task_id)
    if (
        committed_task.digest != expected_task_digest
        or fresh_task.to_dict() != committed_task.to_dict()
        or committed_task.family != run.selection.family
        or assignment.split != run.selection.split
    ):
        raise GatedDevValidationError(
            f"selected task {run.selection.task_id} differs from its exact trusted "
            "TaskManifest"
        )

    # Recompute the deterministic episode split from the public Stage-B seed.
    # The private label nonce changes only label_commitment_digest, so every
    # pixel-bearing and selection-bearing public field must match exactly.
    expected_episode = prepare_episode(
        corpus,
        run.selection.task_id,
        seed=_episode_seed(plan, run.selection),
        corpus_manifest=source_corpus_manifest,
        label_seal_nonce="0" * 64,
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=plan.visual_semantic_policy_digest,
    )
    archived_plan = _mapping(run.outer_record["plan"], "outer episode plan")
    expected_public_plan = expected_episode.to_data()
    if set(archived_plan) != set(expected_public_plan) or any(
        archived_plan[name] != expected
        for name, expected in expected_public_plan.items()
        if name != "label_commitment_digest"
    ):
        raise GatedDevValidationError(
            f"task {run.selection.task_id} episode commitments do not derive from "
            "the trusted TaskManifest and public Stage-B seed"
        )
    if run.outer_record["support_commitment"] != expected_episode.support.to_data():
        raise GatedDevValidationError(
            f"task {run.selection.task_id} support commitment differs from the "
            "trusted TaskManifest split"
        )

    manifest_panels = {
        (item.polarity == "positive", item.index): item
        for item in committed_task.panels
    }
    sources = list(expected_episode._support_sources)
    archive = run.outer_record.get("run_archive")
    if archive is not None:
        release = _mapping(
            _mapping(archive, "run archive")["query_release"],
            "query release",
        )
        if release["queries"] != [
            item.to_data() for item in expected_episode.queries
        ]:
            raise GatedDevValidationError(
                f"task {run.selection.task_id} query release differs from the "
                "trusted TaskManifest split"
            )
        sources.extend(expected_episode._query_sources)

    task_bytes: dict[str, bytes] = {}
    for source in sources:
        try:
            manifest_panel = manifest_panels[(source.positive, source.source_index)]
        except KeyError as exc:
            raise GatedDevValidationError(
                f"trusted TaskManifest lacks selected panel for "
                f"{run.selection.task_id}/{source.panel.blob_id}"
            ) from exc
        if (
            manifest_panel.sha256.removeprefix("sha256:")
            != source.panel.sha256
            or manifest_panel.size_bytes != source.panel.byte_count
        ):
            raise GatedDevValidationError(
                f"run blob {run.selection.task_id}/{source.panel.blob_id} differs "
                "from its exact trusted PanelManifest"
            )
        task_bytes[source.panel.blob_id] = source.read_verified()
    return task_bytes


def capture_gated_dev_replay_bytes(
    corpus: ShapeBongardCorpus,
    artifact: GatedDevValidationArtifact,
    *,
    source_corpus_manifest: CorpusManifest,
) -> dict[str, dict[str, bytes]]:
    """Resolve exact selected preimages joined to a trusted full manifest."""

    artifact.assert_untampered()
    return {
        run.selection.task_id: _capture_gated_dev_task_replay_bytes(
            corpus,
            source_corpus_manifest,
            artifact.plan,
            run,
        )
        for run in artifact.task_runs
    }


__all__ = (
    "CENTRAL_CAMPAIGN_LAYOUT",
    "GATED_DEV_ARTIFACT_SCHEMA",
    "GATED_DEV_PLAN_SCHEMA",
    "GatedDevAcceptancePolicy",
    "GatedDevSelection",
    "GatedDevTaskReplayReceipt",
    "GatedDevTaskRun",
    "GatedDevTransportIdentityError",
    "GatedDevValidationArtifact",
    "GatedDevValidationError",
    "GatedDevValidationPlan",
    "GatedDevValidationSummary",
    "RawSimultaneousHoeffdingBound",
    "capture_gated_dev_replay_bytes",
    "plan_gated_dev_validation",
    "run_gated_dev_validation",
)
