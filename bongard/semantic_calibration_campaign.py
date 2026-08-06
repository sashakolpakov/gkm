"""End-to-end, pre-family campaign for calibrated visual semantics.

The campaign is deliberately split into four content-addressed phases:

``prospective task/query-0 selection -> support-only typed proposals``
``-> frozen score commitments -> complete blind score batch``
``-> label reveal/join -> fitted family``.

Only train/validation tasks are eligible.  Selection is metadata-only and is
stratified over official generator concept keys, so the twenty HD renderings
of one concept pair cannot masquerade as independent calibration clusters.
An HD exact ordered pair is only the campaign's experimental unit: corpus
metadata does not establish independence between different pairs that share a
constituent attribute.  The interval fit therefore assumes cross-pair
independence; a later policy may tighten selection to disjoint-attribute pairs
or downgrade the inference.
Each selected task first receives the ordinary 6+6 :class:`EpisodePlan`; its
neutral ``query-0`` commitment becomes the one task-weighted development
panel.  Query polarity is retained only in the private episode plan until all
planned scorer attempts have been frozen.

Python is the sole reference implementation.  The final cold decoder accepts
the exact selected panel bytes, reconstructs every proposal/score commitment
and attempt, verifies the post-score label joins, and reproduces the fitted
family without invoking either model transport.

This is Stage-A calibration conditional on a typed proposer emitting a soft
claim.  It does **not** establish calibration conditional on the deployment
pipeline's later exact 12/12 support-gate selection and therefore cannot, by
itself, authorize sealed benchmarking.  That requires an independent clean
DEV end-to-end validation after the family and gate are frozen.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import re
import secrets
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import (
    BlobRef,
    QueryPanel,
    SupportCommitment,
    canonical_digest,
    canonical_json,
    verify_support_commitment_data,
)
from bongard.benchmark import EpisodePlan, PROTOCOL_VERSION, prepare_episode
from bongard.canonical_cache import cached_content_data, cached_content_digest
from bongard.cohorts import (
    ParsedOfficialTaskId,
    build_cohort_report,
    classify_task,
    parse_official_task_id,
)
from bongard.corpus import (
    FAMILIES,
    CorpusManifest,
    ShapeBongardCorpus,
)
from bongard.exposure import semantic_resolver_policy_digest
from bongard.exposure import (
    ExposureLedger,
    ExposureViolation,
    basic_morphology_cluster_id,
    semantic_policy_blocked_keys,
)
from bongard.historical_exposure import (
    DEFAULT_SEED_PATH,
    load_historical_exposure,
)
from bongard.semantic_calibration import (
    CalibrationPanelSelection,
    SemanticCalibrationArtifact,
    SemanticCalibrationMeasurement,
    SemanticCalibrationPlan,
    fit_semantic_calibration,
    join_calibration_label,
)
from bongard.semantic_calibration_scoring import (
    PanelInput,
    SemanticCalibrationScoreAttempt,
    SemanticCalibrationScoreCommitment,
    score_semantic_calibration_panel,
)
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    run_codex_structured,
    snapshot_cloud_policy_cache,
)
from bongard.typed_visual_transport import (
    RejectedTypedVisualProposalAttempt,
    TypedVisualProposalRejected,
    TypedVisualTransportResult,
    propose_typed_visual,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


CAMPAIGN_CANDIDATE_SCHEMA_V1 = (
    "gkm.bongard-semantic-calibration-candidate.v1"
)
CAMPAIGN_CANDIDATE_SCHEMA = "gkm.bongard-semantic-calibration-candidate.v2"
CAMPAIGN_EXECUTION_CONFIG_SCHEMA = (
    "gkm.bongard-semantic-calibration-execution-config.v1"
)
CAMPAIGN_PROPOSAL_RECORD_SCHEMA = (
    "gkm.bongard-semantic-calibration-proposal-record.v1"
)
CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1 = (
    "gkm.bongard-semantic-calibration-proposal-archive.v1"
)
CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA = (
    "gkm.bongard-semantic-calibration-proposal-archive.v2"
)
CAMPAIGN_COMMITMENT_BATCH_SCHEMA = (
    "gkm.bongard-semantic-calibration-commitment-batch.v1"
)
CAMPAIGN_SCORE_BATCH_SCHEMA = "gkm.bongard-semantic-calibration-score-batch.v1"
CAMPAIGN_LABEL_REVEAL_SCHEMA = (
    "gkm.bongard-semantic-calibration-label-reveal.v1"
)
CAMPAIGN_NO_SOFT_FAILURE_SCHEMA = (
    "gkm.bongard-semantic-calibration-no-soft-failure.v1"
)
CAMPAIGN_PROPOSAL_PHASE_FAILURE_SCHEMA = (
    "gkm.bongard-semantic-calibration-proposal-phase-failure.v1"
)
CAMPAIGN_FIT_FAILURE_SCHEMA = (
    "gkm.bongard-semantic-calibration-fit-failure.v1"
)
CAMPAIGN_SCORING_FAILURE_SCHEMA = (
    "gkm.bongard-semantic-calibration-scoring-failure.v1"
)
SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA = (
    "gkm.bongard-semantic-calibration-campaign.v1"
)
CAMPAIGN_SELECTION_ALGORITHM_V1 = (
    "round-robin-family-stratified-unique-generator-concepts-v1"
)
CAMPAIGN_SELECTION_ALGORITHM = (
    "round-robin-bd-morphology-hd-constituent-disjoint-v2"
)
_SELECTION_ALGORITHMS = frozenset(
    {CAMPAIGN_SELECTION_ALGORITHM_V1, CAMPAIGN_SELECTION_ALGORITHM}
)

SOFT_ACCEPTED = "soft_claim_accepted"
DIRECT_ONLY = "direct_only_attrition"
TYPED_REJECTED = "typed_parser_rejected"
TRANSPORT_FAILED = "proposer_transport_failed"
_PROPOSAL_STATUSES = frozenset(
    {SOFT_ACCEPTED, DIRECT_ONLY, TYPED_REJECTED, TRANSPORT_FAILED}
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_OBSERVATION = re.compile(r"development-([0-9]{6})\Z")
_ALLOWED_SPLITS = frozenset({"train", "val"})
_ALLOWED_SEMANTIC_COHORTS = frozenset({"drill", "dev"})

StructuredTransport = Callable[..., Any]


class SemanticCalibrationCampaignError(ValueError):
    """A campaign phase, archive, or causal edge is invalid."""


class SemanticCalibrationCampaignScoringFailed(
    SemanticCalibrationCampaignError
):
    """Every scorer was attempted, but at least one attempt failed.

    ``score_batch`` is a complete, label-free archive of all attempts.  The
    exception is raised before any episode label nonce is opened.
    """

    def __init__(self, score_batch: "SemanticCalibrationScoreBatch") -> None:
        self.score_batch = score_batch
        self.failed_observation_ids = tuple(
            attempt.commitment.selection.observation_id
            for attempt in score_batch.attempts
            if attempt.score_artifact.record.outcome != "present"
        )
        super().__init__(
            "calibration scorer failures cannot be fitted or treated as zero: "
            + ", ".join(self.failed_observation_ids)
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCORING_FAILURE_SCHEMA,
            "terminal_state": "complete-score-batch-has-failures/v1",
            "score_batch": self.score_batch.to_data(),
            "score_batch_digest": self.score_batch.digest,
            "failed_observation_ids": list(self.failed_observation_ids),
            "failure_is_never_score_zero": True,
            "label_state": "withheld",
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "failure_digest": self.digest}


class SemanticCalibrationCampaignNoSoftClaims(
    SemanticCalibrationCampaignError
):
    """Proposal attrition left no soft claims, with disclosures retained."""

    def __init__(
        self, proposal_archive: "SemanticCalibrationProposalArchive"
    ) -> None:
        self.proposal_archive = proposal_archive
        super().__init__(
            "proposal attrition left no soft claims to calibrate; "
            f"archive={proposal_archive.digest}"
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_NO_SOFT_FAILURE_SCHEMA,
            "terminal_state": "proposal-attrition-left-no-soft-claims/v1",
            "proposal_archive": self.proposal_archive.to_data(),
            "proposal_archive_digest": self.proposal_archive.digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "failure_digest": self.digest}


class SemanticCalibrationCampaignProposalPhaseFailed(
    SemanticCalibrationCampaignError
):
    """Unreceipted proposer failure forbids a successful Stage-A fit."""

    def __init__(
        self, proposal_archive: "SemanticCalibrationProposalArchive"
    ) -> None:
        self.proposal_archive = proposal_archive
        self.failed_observation_ids = tuple(
            item.candidate.selection.observation_id
            for item in proposal_archive.records
            if item.status == TRANSPORT_FAILED
        )
        if not self.failed_observation_ids:
            raise ValueError("proposal phase failure requires a failed record")
        super().__init__(
            "unreceipted proposer transport failure forbids Stage-A fitting: "
            + ", ".join(self.failed_observation_ids)
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_PROPOSAL_PHASE_FAILURE_SCHEMA,
            "terminal_state": "unreceipted-proposer-transport-failure/v1",
            "proposal_archive": self.proposal_archive.to_data(),
            "proposal_archive_digest": self.proposal_archive.digest,
            "failed_observation_ids": list(self.failed_observation_ids),
            "family_fit_authorized": False,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "failure_digest": self.digest}


class SemanticCalibrationCampaignFitFailed(
    SemanticCalibrationCampaignError
):
    """Post-reveal join/fit failure retaining every causal parent."""

    def __init__(
        self,
        score_batch: "SemanticCalibrationScoreBatch",
        label_reveals: tuple["SemanticCalibrationLabelReveal", ...],
        measurements: tuple[SemanticCalibrationMeasurement, ...],
        cause: Exception,
    ) -> None:
        self.score_batch = score_batch
        self.label_reveals = label_reveals
        self.measurements = measurements
        self.failure_type, self.failure_reason_digest = _bounded_failure(cause)
        super().__init__(
            "post-reveal calibration join/fit failed; all scored disclosures "
            f"are retained: {self.failure_type}"
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_FIT_FAILURE_SCHEMA,
            "terminal_state": "post-reveal-join-or-fit-failed/v1",
            "score_batch": self.score_batch.to_data(),
            "score_batch_digest": self.score_batch.digest,
            "label_reveals": [item.to_data() for item in self.label_reveals],
            "label_reveal_digests": [
                item.digest for item in self.label_reveals
            ],
            "measurements": [item.to_data() for item in self.measurements],
            "measurement_digests": [
                item.digest for item in self.measurements
            ],
            "failure": {
                "error_type": self.failure_type,
                "reason_digest": self.failure_reason_digest,
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "failure_digest": self.digest}


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticCalibrationCampaignError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise SemanticCalibrationCampaignError(
            f"{label} must be a sha256: content address"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SemanticCalibrationCampaignError(f"invalid {label} {value!r}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SemanticCalibrationCampaignError(f"{label} must be an object")
    return value


def _fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> Mapping[str, Any]:
    actual = set(value)
    if actual != expected:
        raise SemanticCalibrationCampaignError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticCalibrationCampaignError(f"{label} must be a list")
    return value


def _exact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        decoded = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SemanticCalibrationCampaignError(
            "campaign value is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover - Mapping guarantees it.
        raise SemanticCalibrationCampaignError("campaign value must be an object")
    return decoded


def _historical_exposure_cache_identity() -> str:
    """Bind proposal-archive memoization to the exact ambient seed file."""

    try:
        payload = DEFAULT_SEED_PATH.read_bytes()
    except OSError as exc:
        raise SemanticCalibrationCampaignError(
            "cannot read the historical exposure seed for cache validation"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _bounded_failure(error: Exception) -> tuple[str, str]:
    error_type = type(error).__name__
    if _IDENTIFIER.fullmatch(error_type) is None:
        error_type = "ProposerTransportError"
    reason = (str(error) or repr(error)).replace("\x00", "�").strip()
    encoded = reason.encode("utf-8", errors="replace")[:4_000]
    return error_type, hashlib.sha256(encoded).hexdigest()


def semantic_generator_cluster_id(
    family: str, concepts: Sequence[str]
) -> str:
    """Return the dependence cluster for one official generator concept key."""

    if family not in FAMILIES:
        raise SemanticCalibrationCampaignError(
            f"unknown ShapeBongard family {family!r}"
        )
    if isinstance(concepts, (str, bytes)) or not isinstance(concepts, Sequence):
        raise TypeError("concepts must be a sequence")
    frozen = tuple(concepts)
    if not frozen or any(
        not isinstance(item, str) or not item for item in frozen
    ):
        raise SemanticCalibrationCampaignError(
            "generator concepts must be non-empty exact strings"
        )
    normalized = (
        tuple(basic_morphology_cluster_id(item) for item in frozen)
        if family == "bd"
        else frozen
    )
    return "generator-" + canonical_digest(
        {
            "schema": "gkm.bongard-generator-dependence-cluster.v1",
            "family": family,
            "concepts": list(normalized),
        }
    )[:48]


def semantic_campaign_label_reveal_protocol_digest() -> str:
    """Return the sole verifier-owned post-score label-opening procedure."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-semantic-campaign-label-reveal-protocol.v1",
            "episode_protocol": PROTOCOL_VERSION,
            "latent_label_seal": "latent-label-seal/v1",
            "selected_query_rule": "episode-query-0/v1",
            "score_parent": CAMPAIGN_SCORE_BATCH_SCHEMA,
            "causal_order": "complete-score-batch-then-label-opening/v1",
            "join_procedure": (
                "bongard.semantic_calibration.join_calibration_label"
            ),
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


def _disclosure_key_tokens(
    parsed: ParsedOfficialTaskId,
    *,
    selection_algorithm: str = CAMPAIGN_SELECTION_ALGORITHM,
) -> frozenset[str]:
    if selection_algorithm not in _SELECTION_ALGORITHMS:
        raise SemanticCalibrationCampaignError(
            "unknown campaign selection algorithm"
        )
    if parsed.family == "bd":
        return frozenset(
            token
            for concept in parsed.concepts
            for token in (
                "basic_family:" + concept,
                "basic_morphology:" + basic_morphology_cluster_id(concept),
            )
        )
    if parsed.family == "hd":
        pair = "abstract_pair:" + "\0".join(parsed.concepts)
        if selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM_V1:
            return frozenset({pair})
        return frozenset(
            {pair}
            | {
                "abstract_attribute:" + concept
                for concept in parsed.concepts
            }
        )
    return frozenset({"freeform_family:" + "\0".join(parsed.concepts)})


CleanCohortEntry = tuple[str, str, tuple[str, ...], str]


def _clean_cohort_whitelist(
    corpus: ShapeBongardCorpus,
    families: tuple[str, ...],
    semantic_cohort: str,
) -> tuple[
    tuple[CleanCohortEntry, ...],
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Resolve one exact metadata-only clean drill/dev whitelist."""

    if semantic_cohort not in _ALLOWED_SEMANTIC_COHORTS:
        raise SemanticCalibrationCampaignError(
            "semantic_cohort must be drill or dev"
        )
    historical = load_historical_exposure()
    report = build_cohort_report(corpus, historical, cohort=semantic_cohort)
    blocked_clusters = tuple(
        sorted(
            key.concepts[0]
            for key in semantic_policy_blocked_keys(historical)
            if key.kind == "basic_morphology_cluster"
        )
    )
    blocked = set(blocked_clusters)
    raw_entries = tuple(
        sorted(
            (
                record.task_id,
                record.family,
                record.parsed.concepts,
                record.split,
            )
            for record in report.records
            if record.family in families
            and record.split in _ALLOWED_SPLITS
            and record.historically_clean
            and record.semantic_cohort == semantic_cohort
        )
    )
    excluded = tuple(
        task_id
        for task_id, family, concepts, _split in raw_entries
        if family == "bd"
        and any(basic_morphology_cluster_id(item) in blocked for item in concepts)
    )
    excluded_set = set(excluded)
    entries = tuple(item for item in raw_entries if item[0] not in excluded_set)
    if any(split is None for _, _, _, split in entries):  # pragma: no cover.
        raise SemanticCalibrationCampaignError(
            "clean cohort whitelist contains an unassigned split"
        )
    whitelist_data = [
        {
            "task_id": task_id,
            "family": family,
            "concepts": list(concepts),
            "split": split,
        }
        for task_id, family, concepts, split in entries
    ]
    return (
        entries,  # type: ignore[return-value]
        historical.seed_digest,
        semantic_resolver_policy_digest(historical),
        report.digest,
        "sha256:" + canonical_digest(whitelist_data),
        "sha256:"
        + canonical_digest(
            {
                "schema": "gkm.bongard-basic-morphology-block-policy.v1",
                "resolver_policy_digest": semantic_resolver_policy_digest(
                    historical
                ),
                "blocked_clusters": list(blocked_clusters),
            }
        ),
        "sha256:" + canonical_digest(list(excluded)),
        excluded,
        blocked_clusters,
    )


def _select_from_whitelist(
    whitelist: Sequence[CleanCohortEntry],
    *,
    families: tuple[str, ...],
    candidate_count: int,
    seed: str,
    exposure_ledger: ExposureLedger,
    historical_seed_digest: str,
    resolver_policy_digest: str,
    selection_algorithm: str = CAMPAIGN_SELECTION_ALGORITHM,
) -> tuple[ParsedOfficialTaskId, ...]:
    if selection_algorithm not in _SELECTION_ALGORITHMS:
        raise SemanticCalibrationCampaignError(
            "unknown campaign selection algorithm"
        )
    historical = load_historical_exposure()
    grouped: dict[
        str, dict[tuple[str, ...], list[ParsedOfficialTaskId]]
    ] = {family: {} for family in families}
    for task_id, family, concepts, split in whitelist:
        if split not in _ALLOWED_SPLITS:
            raise SemanticCalibrationCampaignError(
                "clean cohort whitelist contains a non-development split"
            )
        parsed = parse_official_task_id(task_id)
        if parsed.family != family or parsed.concepts != concepts:
            raise SemanticCalibrationCampaignError(
                "clean cohort whitelist differs from official task parser"
            )
        try:
            exposure_ledger.assert_unseen(task_ids=(task_id,))
            exposure_ledger.assert_semantically_unseen(
                task_ids=(task_id,),
                historical_seed=historical,
                expected_historical_seed_digest=historical_seed_digest,
                expected_resolver_policy_digest=resolver_policy_digest,
            )
        except ExposureViolation:
            continue
        cluster_concepts = (
            tuple(basic_morphology_cluster_id(item) for item in parsed.concepts)
            if family == "bd"
            else parsed.concepts
        )
        grouped[family].setdefault(cluster_concepts, []).append(parsed)

    queues: dict[str, list[ParsedOfficialTaskId]] = {}
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
        queues[family] = sorted(
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

    selected: list[ParsedOfficialTaskId] = []
    used_disclosure_keys: set[str] = set()
    # Ledger v1 stores exact task IDs rather than semantic tokens.  Recover
    # every parseable prior task and project it through the selection policy,
    # so a new HD pair cannot recycle either constituent attribute.  The
    # ordinary semantic-unseen check above remains the fail-closed authority
    # for malformed/unrecoverable ledger task IDs.
    for exposed_task_id in sorted(exposure_ledger.exposed_task_ids):
        try:
            exposed = parse_official_task_id(exposed_task_id)
        except (TypeError, ValueError):
            continue
        used_disclosure_keys.update(
            _disclosure_key_tokens(
                exposed,
                selection_algorithm=selection_algorithm,
            )
        )
    offsets = {family: 0 for family in families}
    while len(selected) < candidate_count:
        advanced = False
        for family in families:
            queue = queues[family]
            while offsets[family] < len(queue):
                candidate = queue[offsets[family]]
                offsets[family] += 1
                keys = _disclosure_key_tokens(
                    candidate,
                    selection_algorithm=selection_algorithm,
                )
                if keys & used_disclosure_keys:
                    continue
                selected.append(candidate)
                used_disclosure_keys.update(keys)
                advanced = True
                break
            if len(selected) == candidate_count:
                break
        if not advanced:
            raise SemanticCalibrationCampaignError(
                f"requested {candidate_count} candidates but the clean cohort, "
                "morphology policy, live ledger, and within-batch semantic "
                f"independence permit only {len(selected)}"
            )
    return tuple(selected)


def select_semantic_calibration_tasks(
    corpus: ShapeBongardCorpus,
    *,
    candidate_count: int,
    seed: str,
    exposure_ledger: ExposureLedger,
    expected_exposure_ledger_digest: str,
    semantic_cohort: str = "drill",
    families: Sequence[str] = ("bd", "hd"),
) -> tuple[ParsedOfficialTaskId, ...]:
    """Select one task per semantic generator key without opening any PNG.

    Generator keys are first ranked within each requested family, then emitted
    round-robin across families.  HD instance siblings therefore contribute at
    most one task, and BD concepts are collapsed through the frozen morphology
    resolver before selection.
    """

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be ShapeBongardCorpus")
    if not isinstance(exposure_ledger, ExposureLedger):
        raise TypeError(
            "exposure_ledger must be an explicit precommitted ExposureLedger"
        )
    if exposure_ledger.digest != _address(
        expected_exposure_ledger_digest, "expected exposure ledger digest"
    ):
        raise SemanticCalibrationCampaignError(
            "exposure ledger differs from precommitted digest"
        )
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise SemanticCalibrationCampaignError(
            "candidate_count must be a positive integer"
        )
    if not isinstance(seed, str) or not seed.strip():
        raise SemanticCalibrationCampaignError("campaign seed must be non-empty")
    if isinstance(families, (str, bytes)) or not isinstance(families, Sequence):
        raise TypeError("families must be a sequence")
    scope = tuple(families)
    if not scope or len(scope) != len(set(scope)) or any(
        family not in FAMILIES for family in scope
    ):
        raise SemanticCalibrationCampaignError(
            "campaign families must be unique ShapeBongard family IDs"
        )
    if not corpus.split.groups or corpus.split.source_digest is None:
        raise SemanticCalibrationCampaignError(
            "campaign requires an authenticated split index"
        )
    _address(corpus.split.source_digest, "split source digest")

    (
        whitelist,
        historical_seed_digest,
        resolver_policy_digest,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _clean_cohort_whitelist(corpus, scope, semantic_cohort)
    selected = _select_from_whitelist(
        whitelist,
        families=scope,
        candidate_count=candidate_count,
        seed=seed,
        exposure_ledger=exposure_ledger,
        historical_seed_digest=historical_seed_digest,
        resolver_policy_digest=resolver_policy_digest,
    )
    clusters = tuple(
        semantic_generator_cluster_id(item.family, item.concepts)
        for item in selected
    )
    if len(clusters) != len(set(clusters)):
        raise SemanticCalibrationCampaignError(
            "campaign selection repeated a generator dependence cluster"
        )
    return selected


def _development_manifest(
    corpus: ShapeBongardCorpus,
    selected: Sequence[ParsedOfficialTaskId],
) -> CorpusManifest:
    """Hash only selected development tasks, never the official test split."""

    manifests = tuple(corpus.task(item.task_id).build_manifest() for item in selected)
    counts = Counter(item.family for item in selected)
    family_counts = tuple((family, counts.get(family, 0)) for family in FAMILIES)
    provisional = CorpusManifest(
        layout=corpus.layout,
        family_counts=family_counts,
        tasks=manifests,
        split=corpus.split,
        digest="sha256:" + "0" * 64,
    )
    return CorpusManifest(
        layout=provisional.layout,
        family_counts=provisional.family_counts,
        tasks=provisional.tasks,
        split=provisional.split,
        digest="sha256:" + canonical_digest(provisional.content_dict()),
    )


def _blob_from_data(value: object, label: str) -> BlobRef:
    data = _fields(
        _mapping(value, label),
        {"blob_id", "sha256", "byte_count", "media_type"},
        label,
    )
    try:
        result = BlobRef(
            data["blob_id"],
            data["sha256"],
            data["byte_count"],
            data["media_type"],
        )
    except (TypeError, ValueError) as exc:
        raise SemanticCalibrationCampaignError(f"invalid {label}: {exc}") from exc
    if result.to_data() != dict(data):
        raise SemanticCalibrationCampaignError(f"{label} is not canonical")
    return result


def _query_from_data(value: object, label: str) -> QueryPanel:
    data = _fields(_mapping(value, label), {"query_id", "panel"}, label)
    try:
        result = QueryPanel(
            data["query_id"], _blob_from_data(data["panel"], label + " panel")
        )
    except (TypeError, ValueError) as exc:
        raise SemanticCalibrationCampaignError(f"invalid {label}: {exc}") from exc
    if result.to_data() != dict(data):
        raise SemanticCalibrationCampaignError(f"{label} is not canonical")
    return result


def _validate_support_presentation(
    support: SupportCommitment,
    presentation: Sequence[Any],
) -> None:
    by_id = {item.panel.blob_id: item for item in support.support}
    if len(by_id) != 12 or len(presentation) != 12:
        raise SemanticCalibrationCampaignError(
            "campaign proposer presentation must bind twelve supports"
        )
    seen: set[str] = set()
    for item in presentation:
        try:
            stem = item.name.removesuffix(".png")
            side, raw_index = stem.split("_", 1)
            index = int(raw_index)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SemanticCalibrationCampaignError(
                "campaign proposer presentation name is malformed"
            ) from exc
        if side not in {"pos", "neg"} or not 0 <= index < 6:
            raise SemanticCalibrationCampaignError(
                "campaign proposer presentation slot is invalid"
            )
        blob_id = f"support-{'positive' if side == 'pos' else 'negative'}-{index}"
        if blob_id in seen or blob_id not in by_id:
            raise SemanticCalibrationCampaignError(
                "campaign proposer presentation repeats or invents a support slot"
            )
        seen.add(blob_id)
        committed = by_id[blob_id]
        if committed.positive is not (side == "pos") or (
            committed.panel.sha256 != item.content_digest
            or committed.panel.byte_count != item.byte_count
        ):
            raise SemanticCalibrationCampaignError(
                "campaign proposer presentation differs from support commitment"
            )


@dataclass(frozen=True)
class SemanticCalibrationExecutionConfig:
    """Frozen phase barriers, concurrency bounds, and transport environment."""

    proposer_minutes: int
    scorer_minutes: int
    proposer_max_workers: int
    scorer_max_workers: int
    executable: str
    expected_codex_launcher_digest: str
    cloud_policy_cache_binding: str
    scheduling_semantics: str = "phase-barrier-ordered-thread-map/v1"

    def __post_init__(self) -> None:
        for name in ("proposer_minutes", "scorer_minutes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 120
            ):
                raise SemanticCalibrationCampaignError(
                    f"{name} must lie in [1, 120]"
                )
        for name in ("proposer_max_workers", "scorer_max_workers"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 256
            ):
                raise SemanticCalibrationCampaignError(
                    f"{name} must lie in [1, 256]"
                )
        if (
            not isinstance(self.executable, str)
            or not self.executable.strip()
            or "\x00" in self.executable
            or len(self.executable.encode("utf-8")) > 1_024
        ):
            raise SemanticCalibrationCampaignError(
                "executable must be a bounded non-empty string"
            )
        _digest(
            self.expected_codex_launcher_digest,
            "expected Codex launcher digest",
        )
        if self.cloud_policy_cache_binding != "absent":
            _address(
                self.cloud_policy_cache_binding,
                "cloud policy cache binding",
            )
        if self.scheduling_semantics != "phase-barrier-ordered-thread-map/v1":
            raise SemanticCalibrationCampaignError(
                "campaign scheduling semantics changed"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_EXECUTION_CONFIG_SCHEMA,
            "proposer_minutes": self.proposer_minutes,
            "scorer_minutes": self.scorer_minutes,
            "proposer_max_workers": self.proposer_max_workers,
            "scorer_max_workers": self.scorer_max_workers,
            "executable": self.executable,
            "expected_codex_launcher_digest": (
                self.expected_codex_launcher_digest
            ),
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "scheduling_semantics": self.scheduling_semantics,
            "phase_order": [
                "all-candidates-frozen",
                "complete-proposer-phase",
                "all-score-commitments-frozen",
                "complete-scorer-phase",
                "label-reveal-and-fit",
            ],
            "ordered_outputs": True,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "SemanticCalibrationExecutionConfig":
        data = _fields(
            _mapping(value, "campaign execution config"),
            {
                "schema",
                "proposer_minutes",
                "scorer_minutes",
                "proposer_max_workers",
                "scorer_max_workers",
                "executable",
                "expected_codex_launcher_digest",
                "cloud_policy_cache_binding",
                "scheduling_semantics",
                "phase_order",
                "ordered_outputs",
            },
            "campaign execution config",
        )
        if (
            data["schema"] != CAMPAIGN_EXECUTION_CONFIG_SCHEMA
            or data["phase_order"]
            != [
                "all-candidates-frozen",
                "complete-proposer-phase",
                "all-score-commitments-frozen",
                "complete-scorer-phase",
                "label-reveal-and-fit",
            ]
            or data["ordered_outputs"] is not True
        ):
            raise SemanticCalibrationCampaignError(
                "campaign execution phase semantics changed"
            )
        result = cls(
            proposer_minutes=data["proposer_minutes"],
            scorer_minutes=data["scorer_minutes"],
            proposer_max_workers=data["proposer_max_workers"],
            scorer_max_workers=data["scorer_max_workers"],
            executable=data["executable"],
            expected_codex_launcher_digest=data[
                "expected_codex_launcher_digest"
            ],
            cloud_policy_cache_binding=data["cloud_policy_cache_binding"],
            scheduling_semantics=data["scheduling_semantics"],
        )
        if result.to_data() != dict(data):
            raise SemanticCalibrationCampaignError(
                "campaign execution config is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationCandidate:
    """One prospectively chosen, label-free ``query-0`` development panel."""

    selection: CalibrationPanelSelection
    concepts: tuple[str, ...]
    episode_plan_data: Mapping[str, Any]
    support: SupportCommitment
    queries: tuple[QueryPanel, QueryPanel]
    selection_algorithm: str = CAMPAIGN_SELECTION_ALGORITHM
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.selection_algorithm not in _SELECTION_ALGORITHMS:
            raise SemanticCalibrationCampaignError(
                "candidate selection algorithm is unsupported"
            )
        if not isinstance(self.selection, CalibrationPanelSelection):
            raise TypeError("selection must be CalibrationPanelSelection")
        if not isinstance(self.concepts, tuple) or not self.concepts or any(
            not isinstance(item, str) or not item for item in self.concepts
        ):
            raise SemanticCalibrationCampaignError(
                "candidate concepts must be a non-empty immutable tuple"
            )
        parsed = parse_official_task_id(self.selection.task_id)
        if parsed.concepts != self.concepts:
            raise SemanticCalibrationCampaignError(
                "candidate concepts differ from official task identity"
            )
        expected_cluster = semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        )
        if self.selection.dependence_cluster_id != expected_cluster:
            raise SemanticCalibrationCampaignError(
                "candidate dependence cluster differs from generator concept key"
            )
        match = _OBSERVATION.fullmatch(self.selection.observation_id)
        if match is None or self.selection.panel_id != (
            f"calibration-panel-{int(match.group(1)):06d}"
        ):
            raise SemanticCalibrationCampaignError(
                "candidate observation and neutral panel IDs differ"
            )
        if not isinstance(self.support, SupportCommitment):
            raise TypeError("support must be SupportCommitment")
        if verify_support_commitment_data(self.support.to_data()) != self.support:
            raise SemanticCalibrationCampaignError(
                "candidate support is not canonically represented"
            )
        if (
            not isinstance(self.queries, tuple)
            or len(self.queries) != 2
            or any(not isinstance(item, QueryPanel) for item in self.queries)
        ):
            raise TypeError("candidate queries must be two QueryPanel values")
        if tuple(item.query_id for item in self.queries) != (
            "query-0",
            "query-1",
        ) or tuple(item.panel.blob_id for item in self.queries) != (
            "query-panel-0",
            "query-panel-1",
        ):
            raise SemanticCalibrationCampaignError(
                "candidate must select the canonical neutral query-0 panel"
            )
        if len({item.panel.sha256 for item in self.queries}) != 2:
            raise SemanticCalibrationCampaignError(
                "candidate query panels must have distinct bytes"
            )
        if self.selection.panel_digest != self.queries[0].panel.sha256:
            raise SemanticCalibrationCampaignError(
                "candidate selection differs from neutral query-0"
            )
        if self.selection.panel_digest in {
            item.panel.sha256 for item in self.support.support
        }:
            raise SemanticCalibrationCampaignError(
                "candidate panel bytes overlap proposer support"
            )

        plan = _exact_mapping(_mapping(self.episode_plan_data, "episode plan"))
        expected_fields = {
            "version",
            "task_id",
            "family",
            "split",
            "regime",
            "run_id",
            "verifier_id",
            "seed_digest",
            "corpus_digest",
            "task_manifest_digest",
            "support_commitment_digest",
            "latent_query_digest",
            "label_commitment_digest",
        }
        _fields(plan, expected_fields, "campaign episode plan")
        if plan["version"] != PROTOCOL_VERSION:
            raise SemanticCalibrationCampaignError(
                "candidate episode protocol version changed"
            )
        expected = {
            "task_id": self.selection.task_id,
            "family": parsed.family,
            "split": self.selection.split,
            "run_id": self.support.run_id,
            "verifier_id": self.support.issued_by,
            "corpus_digest": self.support.corpus_digest,
            "support_commitment_digest": self.support.digest(),
        }
        for name, wanted in expected.items():
            if plan[name] != wanted:
                raise SemanticCalibrationCampaignError(
                    f"candidate episode {name} differs from frozen inputs"
                )
        for name in (
            "seed_digest",
            "corpus_digest",
            "task_manifest_digest",
            "latent_query_digest",
            "label_commitment_digest",
        ):
            _digest(plan[name], f"candidate episode {name}")
        latent = canonical_digest(
            {
                "version": "latent-two-query-commitment/v1",
                "run_id": self.support.run_id,
                "queries": [item.to_data() for item in self.queries],
            }
        )
        if plan["latent_query_digest"] != latent:
            raise SemanticCalibrationCampaignError(
                "candidate query commitments differ from episode plan"
            )
        object.__setattr__(self, "episode_plan_data", plan)
        object.__setattr__(self, "_sealed_digest", self.digest)

    @classmethod
    def from_episode(
        cls,
        episode: EpisodePlan,
        parsed: ParsedOfficialTaskId,
        *,
        ordinal: int,
    ) -> "SemanticCalibrationCandidate":
        if not isinstance(episode, EpisodePlan):
            raise TypeError("episode must be EpisodePlan")
        if not isinstance(parsed, ParsedOfficialTaskId):
            raise TypeError("parsed must be ParsedOfficialTaskId")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal <= 999_999
        ):
            raise SemanticCalibrationCampaignError(
                "candidate ordinal must lie in [0, 999999]"
            )
        if episode.task_id != parsed.task_id or episode.split not in _ALLOWED_SPLITS:
            raise SemanticCalibrationCampaignError(
                "campaign episode is not a selected development task"
            )
        query = episode.queries[0]
        selection = CalibrationPanelSelection(
            observation_id=f"development-{ordinal:06d}",
            task_id=episode.task_id,
            panel_id=f"calibration-panel-{ordinal:06d}",
            panel_digest=query.panel.sha256,
            split=episode.split,
            dependence_cluster_id=semantic_generator_cluster_id(
                parsed.family, parsed.concepts
            ),
        )
        return cls(
            selection=selection,
            concepts=parsed.concepts,
            episode_plan_data=episode.to_data(),
            support=episode.support,
            queries=episode.queries,
        )

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.selection_algorithm,
            self.selection.digest,
            self.concepts,
            canonical_json(self.episode_plan_data),
            self.support.digest(),
            tuple(canonical_digest(item.to_data()) for item in self.queries),
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": (
                CAMPAIGN_CANDIDATE_SCHEMA
                if self.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM
                else CAMPAIGN_CANDIDATE_SCHEMA_V1
            ),
            "selection_algorithm": self.selection_algorithm,
            "neutral_panel_rule": "episode-query-0/v1",
            "label_state": "withheld",
            "concept_key": {
                "family": self.episode_plan_data["family"],
                "concepts": list(self.concepts),
            },
            "selection": self.selection.to_data(),
            "episode_plan": dict(self.episode_plan_data),
            "episode_plan_digest": canonical_digest(self.episode_plan_data),
            "support_commitment": self.support.to_data(),
            "support_commitment_digest": self.support.digest(),
            "query_panels": [item.to_data() for item in self.queries],
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "candidate_digest": self.digest}

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign candidate changed after sealing"
            )
        type(self).from_data(self.to_data(), expected_digest=self._sealed_digest)

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationCandidate":
        data = _fields(
            _mapping(value, "campaign candidate"),
            {
                "schema",
                "selection_algorithm",
                "neutral_panel_rule",
                "label_state",
                "concept_key",
                "selection",
                "episode_plan",
                "episode_plan_digest",
                "support_commitment",
                "support_commitment_digest",
                "query_panels",
                "candidate_digest",
            },
            "campaign candidate",
        )
        archived_algorithm = data["selection_algorithm"]
        expected_schema = (
            CAMPAIGN_CANDIDATE_SCHEMA
            if archived_algorithm == CAMPAIGN_SELECTION_ALGORITHM
            else (
                CAMPAIGN_CANDIDATE_SCHEMA_V1
                if archived_algorithm == CAMPAIGN_SELECTION_ALGORITHM_V1
                else None
            )
        )
        if (
            expected_schema is None
            or data["schema"] != expected_schema
            or data["neutral_panel_rule"] != "episode-query-0/v1"
            or data["label_state"] != "withheld"
        ):
            raise SemanticCalibrationCampaignError(
                "unsupported campaign candidate or revealed label state"
            )
        concept_key = _fields(
            _mapping(data["concept_key"], "candidate concept key"),
            {"family", "concepts"},
            "candidate concept key",
        )
        raw_concepts = _list(concept_key["concepts"], "candidate concepts")
        selection = CalibrationPanelSelection.from_data(
            _mapping(data["selection"], "candidate selection")
        )
        support = verify_support_commitment_data(data["support_commitment"])
        if support.digest() != _digest(
            data["support_commitment_digest"], "support commitment digest"
        ):
            raise SemanticCalibrationCampaignError(
                "candidate support digest differs"
            )
        queries = tuple(
            _query_from_data(item, f"candidate query {index}")
            for index, item in enumerate(
                _list(data["query_panels"], "candidate query panels")
            )
        )
        result = cls(
            selection=selection,
            concepts=tuple(raw_concepts),
            episode_plan_data=_mapping(data["episode_plan"], "episode plan"),
            support=support,
            queries=queries,  # type: ignore[arg-type]
            selection_algorithm=archived_algorithm,
        )
        if concept_key["family"] != result.episode_plan_data["family"]:
            raise SemanticCalibrationCampaignError(
                "candidate concept family differs"
            )
        if canonical_digest(result.episode_plan_data) != _digest(
            data["episode_plan_digest"], "episode plan digest"
        ):
            raise SemanticCalibrationCampaignError(
                "candidate episode plan digest differs"
            )
        archived = _digest(data["candidate_digest"], "candidate digest")
        if result.digest != archived or (
            expected_digest is not None
            and result.digest != _digest(expected_digest, "expected candidate digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign candidate digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign candidate is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationProposalRecord:
    """One accepted, parser-rejected, direct-only, or failed proposer turn."""

    candidate: SemanticCalibrationCandidate
    status: str
    proposal_transport: TypedVisualTransportResult | None = None
    rejected_attempt: RejectedTypedVisualProposalAttempt | None = None
    failure_type: str | None = None
    failure_reason_digest: str | None = None
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SemanticCalibrationCandidate):
            raise TypeError("candidate must be SemanticCalibrationCandidate")
        self.candidate.assert_untampered()
        if self.status not in _PROPOSAL_STATUSES:
            raise SemanticCalibrationCampaignError(
                f"unknown proposal attrition status {self.status!r}"
            )
        accepted = self.status in {SOFT_ACCEPTED, DIRECT_ONLY}
        rejected = self.status == TYPED_REJECTED
        failed = self.status == TRANSPORT_FAILED
        if (self.proposal_transport is not None) is not accepted:
            raise SemanticCalibrationCampaignError(
                "proposal transport presence differs from attrition status"
            )
        if (self.rejected_attempt is not None) is not rejected:
            raise SemanticCalibrationCampaignError(
                "rejected-attempt presence differs from attrition status"
            )
        if (self.failure_type is not None) is not failed or (
            self.failure_reason_digest is not None
        ) is not failed:
            raise SemanticCalibrationCampaignError(
                "proposal failure fields differ from attrition status"
            )
        if accepted:
            assert self.proposal_transport is not None
            has_soft = self.proposal_transport.proposal.soft_claim is not None
            if has_soft is not (self.status == SOFT_ACCEPTED):
                raise SemanticCalibrationCampaignError(
                    "soft-claim presence differs from attrition status"
                )
            _validate_support_presentation(
                self.candidate.support,
                self.proposal_transport.support_presentation,
            )
        elif rejected:
            assert self.rejected_attempt is not None
            _validate_support_presentation(
                self.candidate.support,
                self.rejected_attempt.support_presentation,
            )
        else:
            _identifier(self.failure_type, "proposal failure type")
            _digest(self.failure_reason_digest, "proposal failure reason digest")
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.candidate.digest,
            self.status,
            None
            if self.proposal_transport is None
            else self.proposal_transport.digest,
            None
            if self.rejected_attempt is None
            else self.rejected_attempt.digest,
            self.failure_type,
            self.failure_reason_digest,
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_PROPOSAL_RECORD_SCHEMA,
            "candidate": self.candidate.to_data(),
            "candidate_digest": self.candidate.digest,
            "status": self.status,
            "proposal_transport": (
                None
                if self.proposal_transport is None
                else self.proposal_transport.to_data()
            ),
            "rejected_attempt": (
                None
                if self.rejected_attempt is None
                else self.rejected_attempt.to_data()
            ),
            "transport_failure": (
                None
                if self.failure_type is None
                else {
                    "error_type": self.failure_type,
                    "reason_digest": self.failure_reason_digest,
                }
            ),
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.digest}

    def assert_untampered(self) -> None:
        self.candidate.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign proposal record changed after sealing"
            )

    def assert_matches_protocol(self, protocol: SoftScorerProtocol) -> None:
        self.candidate.assert_untampered()
        protocol.assert_untampered()
        if self.proposal_transport is not None:
            decoded = TypedVisualTransportResult.from_data(
                self.proposal_transport.to_data(),
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=protocol,
                expected_digest=self.proposal_transport.digest,
            )
            if decoded != self.proposal_transport:
                raise SemanticCalibrationCampaignError(
                    "accepted proposer transport does not cold replay"
                )
        if self.rejected_attempt is not None:
            decoded_rejection = RejectedTypedVisualProposalAttempt.from_data(
                self.rejected_attempt.to_data(),
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=protocol,
                expected_digest=self.rejected_attempt.digest,
            )
            if decoded_rejection != self.rejected_attempt:
                raise SemanticCalibrationCampaignError(
                    "rejected proposer attempt does not cold replay"
                )
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign proposal record changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        protocol: SoftScorerProtocol,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationProposalRecord":
        data = _fields(
            _mapping(value, "campaign proposal record"),
            {
                "schema",
                "candidate",
                "candidate_digest",
                "status",
                "proposal_transport",
                "rejected_attempt",
                "transport_failure",
                "record_digest",
            },
            "campaign proposal record",
        )
        if data["schema"] != CAMPAIGN_PROPOSAL_RECORD_SCHEMA:
            raise SemanticCalibrationCampaignError(
                "unsupported campaign proposal record"
            )
        candidate = SemanticCalibrationCandidate.from_data(
            _mapping(data["candidate"], "proposal candidate"),
            expected_digest=_digest(data["candidate_digest"], "candidate digest"),
        )
        transport = (
            None
            if data["proposal_transport"] is None
            else TypedVisualTransportResult.from_data(
                _mapping(data["proposal_transport"], "accepted proposer transport"),
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=protocol,
            )
        )
        rejected = (
            None
            if data["rejected_attempt"] is None
            else RejectedTypedVisualProposalAttempt.from_data(
                _mapping(data["rejected_attempt"], "rejected proposer attempt"),
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=protocol,
            )
        )
        failure = data["transport_failure"]
        if failure is None:
            failure_type = None
            failure_digest = None
        else:
            failure_data = _fields(
                _mapping(failure, "proposer transport failure"),
                {"error_type", "reason_digest"},
                "proposer transport failure",
            )
            failure_type = failure_data["error_type"]
            failure_digest = failure_data["reason_digest"]
        result = cls(
            candidate=candidate,
            status=data["status"],
            proposal_transport=transport,
            rejected_attempt=rejected,
            failure_type=failure_type,
            failure_reason_digest=failure_digest,
        )
        result.assert_matches_protocol(protocol)
        archived = _digest(data["record_digest"], "proposal record digest")
        if result.digest != archived or (
            expected_digest is not None
            and result.digest
            != _digest(expected_digest, "expected proposal record digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign proposal record digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign proposal record is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationProposalArchive:
    """The complete proposer attrition ledger, still free of query labels."""

    protocol: SoftScorerProtocol
    execution_config: SemanticCalibrationExecutionConfig
    selection_seed: str
    selection_seed_digest: str
    candidate_count: int
    families: tuple[str, ...]
    semantic_cohort: str
    source_corpus_manifest_digest: str
    development_manifest_digest: str
    split_source_digest: str
    split_manifest_digest: str
    historical_seed_digest: str
    resolver_policy_digest: str
    cohort_report_digest: str
    clean_cohort_whitelist_digest: str
    clean_cohort_whitelist: tuple[CleanCohortEntry, ...]
    blocked_policy_digest: str
    blocked_exclusion_digest: str
    blocked_excluded_task_ids: tuple[str, ...]
    blocked_morphology_clusters: tuple[str, ...]
    exposure_predecessor: ExposureLedger
    exposure_successor: ExposureLedger
    records: tuple[SemanticCalibrationProposalRecord, ...]
    selection_algorithm: str = CAMPAIGN_SELECTION_ALGORITHM
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.selection_algorithm not in _SELECTION_ALGORITHMS:
            raise SemanticCalibrationCampaignError(
                "proposal archive selection algorithm is unsupported"
            )
        if not isinstance(self.protocol, SoftScorerProtocol):
            raise TypeError("protocol must be SoftScorerProtocol")
        self.protocol.assert_untampered()
        if not isinstance(
            self.execution_config, SemanticCalibrationExecutionConfig
        ):
            raise TypeError("execution_config must be a campaign execution config")
        _digest(self.selection_seed_digest, "selection seed digest")
        if not isinstance(self.selection_seed, str) or not self.selection_seed.strip():
            raise SemanticCalibrationCampaignError(
                "selection seed must be a non-empty public string"
            )
        if len(self.selection_seed.encode("utf-8")) > 4_096:
            raise SemanticCalibrationCampaignError(
                "selection seed exceeds the archival bound"
            )
        if hashlib.sha256(self.selection_seed.encode("utf-8")).hexdigest() != (
            self.selection_seed_digest
        ):
            raise SemanticCalibrationCampaignError(
                "selection seed digest differs from the archived public seed"
            )
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 1
        ):
            raise SemanticCalibrationCampaignError(
                "candidate_count must be a positive integer"
            )
        if (
            not isinstance(self.families, tuple)
            or not self.families
            or len(self.families) != len(set(self.families))
            or any(item not in FAMILIES for item in self.families)
        ):
            raise SemanticCalibrationCampaignError(
                "proposal archive families must be unique canonical IDs"
            )
        if self.semantic_cohort not in _ALLOWED_SEMANTIC_COHORTS:
            raise SemanticCalibrationCampaignError(
                "proposal archive semantic cohort must be drill or dev"
            )
        _address(
            self.source_corpus_manifest_digest,
            "source corpus manifest digest",
        )
        _address(
            self.development_manifest_digest,
            "development manifest digest",
        )
        _address(self.split_source_digest, "split source digest")
        _digest(self.split_manifest_digest, "split manifest digest")
        _address(self.historical_seed_digest, "historical seed digest")
        _address(self.resolver_policy_digest, "resolver policy digest")
        _address(self.cohort_report_digest, "cohort report digest")
        _address(
            self.clean_cohort_whitelist_digest,
            "clean cohort whitelist digest",
        )
        _address(self.blocked_policy_digest, "blocked morphology policy digest")
        _address(self.blocked_exclusion_digest, "blocked exclusion digest")
        frozen_historical = load_historical_exposure()
        if frozen_historical.seed_digest != self.historical_seed_digest or (
            semantic_resolver_policy_digest(frozen_historical)
            != self.resolver_policy_digest
        ):
            raise SemanticCalibrationCampaignError(
                "clean cohort archive differs from frozen historical resolver"
            )
        expected_blocked = tuple(
            sorted(
                key.concepts[0]
                for key in semantic_policy_blocked_keys(frozen_historical)
                if key.kind == "basic_morphology_cluster"
            )
        )
        if self.blocked_morphology_clusters != expected_blocked:
            raise SemanticCalibrationCampaignError(
                "blocked morphology inventory differs from frozen resolver"
            )
        if tuple(sorted(self.blocked_excluded_task_ids)) != (
            self.blocked_excluded_task_ids
        ) or len(self.blocked_excluded_task_ids) != len(
            set(self.blocked_excluded_task_ids)
        ):
            raise SemanticCalibrationCampaignError(
                "blocked exclusion task IDs must be uniquely sorted"
            )
        for task_id in self.blocked_excluded_task_ids:
            classified = classify_task(
                task_id,
                frozen_historical,
                split=None,
                regime=None,
            )
            if (
                classified.family != "bd"
                or not classified.historically_clean
                or classified.semantic_cohort != self.semantic_cohort
                or not any(
                    basic_morphology_cluster_id(item) in set(expected_blocked)
                    for item in classified.parsed.concepts
                )
            ):
                raise SemanticCalibrationCampaignError(
                    "blocked exclusion task is not a blocked clean-cohort BD task"
                )
        if self.blocked_exclusion_digest != (
            "sha256:" + canonical_digest(list(self.blocked_excluded_task_ids))
        ):
            raise SemanticCalibrationCampaignError(
                "blocked exclusion digest differs from exact task IDs"
            )
        expected_blocked_policy = "sha256:" + canonical_digest(
            {
                "schema": "gkm.bongard-basic-morphology-block-policy.v1",
                "resolver_policy_digest": self.resolver_policy_digest,
                "blocked_clusters": list(expected_blocked),
            }
        )
        if self.blocked_policy_digest != expected_blocked_policy:
            raise SemanticCalibrationCampaignError(
                "blocked morphology policy digest differs"
            )
        if not isinstance(self.clean_cohort_whitelist, tuple) or any(
            not isinstance(item, tuple) or len(item) != 4
            for item in self.clean_cohort_whitelist
        ):
            raise TypeError(
                "clean_cohort_whitelist must be an immutable entry tuple"
            )
        whitelist_data: list[dict[str, object]] = []
        whitelist_task_ids: list[str] = []
        for task_id, family, concepts, split in self.clean_cohort_whitelist:
            _identifier(task_id, "clean cohort task ID")
            if family not in self.families or split not in _ALLOWED_SPLITS:
                raise SemanticCalibrationCampaignError(
                    "clean cohort whitelist is outside family/split scope"
                )
            parsed = parse_official_task_id(task_id)
            if parsed.family != family or parsed.concepts != concepts:
                raise SemanticCalibrationCampaignError(
                    "clean cohort whitelist differs from official parser"
                )
            if family == "bd" and any(
                basic_morphology_cluster_id(item) in set(expected_blocked)
                for item in concepts
            ):
                raise SemanticCalibrationCampaignError(
                    "clean cohort whitelist contains a policy-blocked morphology"
                )
            classified = classify_task(
                task_id,
                frozen_historical,
                split=split,
                regime=None,
            )
            if not classified.historically_clean or (
                classified.semantic_cohort != self.semantic_cohort
            ):
                raise SemanticCalibrationCampaignError(
                    "clean cohort whitelist contains another cohort or exposed semantics"
                )
            whitelist_task_ids.append(task_id)
            whitelist_data.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "concepts": list(concepts),
                    "split": split,
                }
            )
        if whitelist_task_ids != sorted(whitelist_task_ids) or len(
            whitelist_task_ids
        ) != len(set(whitelist_task_ids)):
            raise SemanticCalibrationCampaignError(
                "clean cohort whitelist must be uniquely sorted by task ID"
            )
        reproduced_whitelist_digest = "sha256:" + canonical_digest(
            whitelist_data
        )
        if reproduced_whitelist_digest != self.clean_cohort_whitelist_digest:
            raise SemanticCalibrationCampaignError(
                "clean cohort whitelist digest differs from exact entries"
            )
        if not isinstance(self.records, tuple) or not self.records or any(
            not isinstance(item, SemanticCalibrationProposalRecord)
            for item in self.records
        ):
            raise TypeError("records must be a non-empty immutable tuple")
        observation_ids = tuple(
            item.candidate.selection.observation_id for item in self.records
        )
        if observation_ids != tuple(sorted(observation_ids)) or len(
            observation_ids
        ) != len(set(observation_ids)):
            raise SemanticCalibrationCampaignError(
                "proposal records must be uniquely sorted by observation ID"
            )
        task_ids = tuple(item.candidate.selection.task_id for item in self.records)
        if any(
            item.candidate.selection_algorithm != self.selection_algorithm
            for item in self.records
        ):
            raise SemanticCalibrationCampaignError(
                "proposal candidates use another selection algorithm"
            )
        clusters = tuple(
            item.candidate.selection.dependence_cluster_id
            for item in self.records
        )
        if len(task_ids) != len(set(task_ids)) or len(clusters) != len(
            set(clusters)
        ):
            raise SemanticCalibrationCampaignError(
                "proposal archive repeats a task or generator cluster"
            )
        disclosure_tokens = tuple(
            _disclosure_key_tokens(
                parse_official_task_id(task_id),
                selection_algorithm=self.selection_algorithm,
            )
            for task_id in task_ids
        )
        if self.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM:
            flattened = [token for item in disclosure_tokens for token in item]
            if len(flattened) != len(set(flattened)):
                raise SemanticCalibrationCampaignError(
                    "proposal archive repeats an HD constituent attribute"
                )
        if not set(task_ids) <= set(whitelist_task_ids):
            raise SemanticCalibrationCampaignError(
                "proposal archive contains a task outside clean cohort whitelist"
            )
        if len(task_ids) != self.candidate_count:
            raise SemanticCalibrationCampaignError(
                "proposal archive record count differs from candidate_count"
            )
        if not isinstance(self.exposure_predecessor, ExposureLedger) or not isinstance(
            self.exposure_successor, ExposureLedger
        ):
            raise TypeError("campaign exposure parents must be ExposureLedger values")
        if self.exposure_predecessor.corpus_digest != (
            self.exposure_successor.corpus_digest
        ):
            raise SemanticCalibrationCampaignError(
                "exposure successor belongs to another corpus"
            )
        if self.exposure_predecessor.corpus_digest != (
            self.source_corpus_manifest_digest
        ):
            raise SemanticCalibrationCampaignError(
                "exposure predecessor belongs to another source corpus"
            )
        replayed = _select_from_whitelist(
            self.clean_cohort_whitelist,
            families=self.families,
            candidate_count=self.candidate_count,
            seed=self.selection_seed,
            exposure_ledger=self.exposure_predecessor,
            historical_seed_digest=self.historical_seed_digest,
            resolver_policy_digest=self.resolver_policy_digest,
            selection_algorithm=self.selection_algorithm,
        )
        if tuple(item.task_id for item in replayed) != task_ids:
            raise SemanticCalibrationCampaignError(
                "proposal archive selection differs from cold seed replay"
            )
        resolution = self.exposure_predecessor.assert_semantically_unseen(
            task_ids=task_ids,
            historical_seed=frozen_historical,
            expected_historical_seed_digest=self.historical_seed_digest,
            expected_resolver_policy_digest=self.resolver_policy_digest,
        )
        predecessor_events = self.exposure_predecessor.events
        successor_events = self.exposure_successor.events
        if successor_events[: len(predecessor_events)] != predecessor_events:
            raise SemanticCalibrationCampaignError(
                "exposure successor does not extend exact predecessor"
            )
        additions = successor_events[len(predecessor_events) :]
        if len(additions) != len(task_ids):
            raise SemanticCalibrationCampaignError(
                "exposure successor must record every selected task once"
            )
        for record, task_id, event in zip(
            self.records, task_ids, additions, strict=True
        ):
            if (
                event.task_ids != (task_id,)
                or event.panel_ids
                or event.phase != "semantic-calibration"
                or event.actor != self.protocol.proposer_model_id
                or event.purpose
                != "stage-a-soft-scorer-calibration-candidate"
                or event.source
                != "soft-scorer-protocol:" + self.protocol.digest()
            ):
                raise SemanticCalibrationCampaignError(
                    "exposure successor event differs from campaign selection"
                )
        if resolution.ledger_digest != self.exposure_predecessor.digest:
            raise SemanticCalibrationCampaignError(
                "semantic exposure resolution differs from predecessor"
            )
        for record in self.records:
            record.assert_matches_protocol(self.protocol)
            proposal_receipt = (
                record.proposal_transport.receipt
                if record.proposal_transport is not None
                else (
                    None
                    if record.rejected_attempt is None
                    else record.rejected_attempt.receipt
                )
            )
            if proposal_receipt is not None and (
                proposal_receipt.codex_launcher_digest
                != self.execution_config.expected_codex_launcher_digest
                or proposal_receipt.cloud_config_bundle_cache_binding
                != self.execution_config.cloud_policy_cache_binding
            ):
                raise SemanticCalibrationCampaignError(
                    "proposer receipt differs from frozen launcher/cache identity"
                )
            candidate = record.candidate
            if candidate.selection.split not in _ALLOWED_SPLITS:
                raise SemanticCalibrationCampaignError(
                    "proposal archive contains a non-development task"
                )
            if candidate.episode_plan_data["family"] not in self.families:
                raise SemanticCalibrationCampaignError(
                    "proposal candidate is outside the frozen family scope"
                )
            if candidate.support.corpus_digest != self.development_manifest_digest[7:]:
                raise SemanticCalibrationCampaignError(
                    "proposal candidate belongs to another development manifest"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    @property
    def soft_records(self) -> tuple[SemanticCalibrationProposalRecord, ...]:
        return tuple(item for item in self.records if item.status == SOFT_ACCEPTED)

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.selection_algorithm,
            self.protocol.digest(),
            self.execution_config.digest,
            self.selection_seed,
            self.selection_seed_digest,
            self.candidate_count,
            self.families,
            self.semantic_cohort,
            self.source_corpus_manifest_digest,
            self.development_manifest_digest,
            self.split_source_digest,
            self.split_manifest_digest,
            _historical_exposure_cache_identity(),
            self.historical_seed_digest,
            self.resolver_policy_digest,
            self.cohort_report_digest,
            self.clean_cohort_whitelist_digest,
            self.clean_cohort_whitelist,
            self.blocked_policy_digest,
            self.blocked_exclusion_digest,
            self.blocked_excluded_task_ids,
            self.blocked_morphology_clusters,
            self.exposure_predecessor.digest,
            self.exposure_successor.digest,
            tuple(item.digest for item in self.records),
        )

    def _uncached_content_data(self) -> dict[str, object]:
        counts = Counter(item.status for item in self.records)
        historical = load_historical_exposure()
        resolution = self.exposure_predecessor.assert_semantically_unseen(
            task_ids=tuple(
                item.candidate.selection.task_id for item in self.records
            ),
            historical_seed=historical,
            expected_historical_seed_digest=self.historical_seed_digest,
            expected_resolver_policy_digest=self.resolver_policy_digest,
        )
        return {
            "schema": (
                CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA
                if self.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM
                else CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1
            ),
            "selection_algorithm": self.selection_algorithm,
            "coverage_scope": "stage-a-soft-claim-emission-only/v1",
            "end_to_end_support_gate_coverage_claimed": False,
            "dependence_unit_semantics": (
                "bd-morphology-cluster-and-hd-disjoint-attributes-plus-exact-pair/v2"
                if self.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM
                else "bd-morphology-cluster-and-hd-exact-ordered-pair/v1"
            ),
            "hd_cross_pair_independence_status": (
                "enforced-constituent-disjoint-within-batch-and-live-ledger/v2"
                if self.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM
                else "assumed-for-fit-not-established-by-corpus-metadata/v1"
            ),
            "transport_failure_evidence": (
                "bounded-local-exception-digest-without-codex-receipt/v1"
            ),
            "selection_seed": self.selection_seed,
            "selection_seed_digest": self.selection_seed_digest,
            "candidate_count": self.candidate_count,
            "families": list(self.families),
            "semantic_cohort": self.semantic_cohort,
            "source_corpus_manifest_digest": self.source_corpus_manifest_digest,
            "development_manifest_digest": self.development_manifest_digest,
            "split_source_digest": self.split_source_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "historical_seed_digest": self.historical_seed_digest,
            "resolver_policy_digest": self.resolver_policy_digest,
            "cohort_report_digest": self.cohort_report_digest,
            "clean_cohort_whitelist_digest": self.clean_cohort_whitelist_digest,
            "clean_cohort_whitelist": [
                {
                    "task_id": task_id,
                    "family": family,
                    "concepts": list(concepts),
                    "split": split,
                }
                for task_id, family, concepts, split in self.clean_cohort_whitelist
            ],
            "blocked_policy_digest": self.blocked_policy_digest,
            "blocked_exclusion_digest": self.blocked_exclusion_digest,
            "blocked_excluded_task_ids": list(self.blocked_excluded_task_ids),
            "blocked_morphology_clusters": list(
                self.blocked_morphology_clusters
            ),
            "exposure_predecessor": self.exposure_predecessor.to_dict(),
            "exposure_predecessor_digest": self.exposure_predecessor.digest,
            "semantic_exposure_resolution": resolution.to_dict(),
            "exposure_successor": self.exposure_successor.to_dict(),
            "exposure_successor_digest": self.exposure_successor.digest,
            "prospective_protocol": self.protocol.to_data(),
            "protocol_digest": self.protocol.digest(),
            "records": [item.to_data() for item in self.records],
            "attrition_counts": {
                status: counts.get(status, 0)
                for status in sorted(_PROPOSAL_STATUSES)
            },
            "label_state": "withheld",
            "execution_config": self.execution_config.to_data(),
            "execution_config_digest": self.execution_config.digest,
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "proposal_archive_digest": self.digest}

    def assert_untampered(self) -> None:
        self.protocol.assert_untampered()
        for record in self.records:
            record.assert_matches_protocol(self.protocol)
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign proposal archive changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationProposalArchive":
        data = _fields(
            _mapping(value, "campaign proposal archive"),
            {
                "schema",
                "selection_algorithm",
                "coverage_scope",
                "end_to_end_support_gate_coverage_claimed",
                "dependence_unit_semantics",
                "hd_cross_pair_independence_status",
                "transport_failure_evidence",
                "execution_config",
                "execution_config_digest",
                "selection_seed",
                "selection_seed_digest",
                "candidate_count",
                "families",
                "semantic_cohort",
                "source_corpus_manifest_digest",
                "development_manifest_digest",
                "split_source_digest",
                "split_manifest_digest",
                "historical_seed_digest",
                "resolver_policy_digest",
                "cohort_report_digest",
                "clean_cohort_whitelist_digest",
                "clean_cohort_whitelist",
                "blocked_policy_digest",
                "blocked_exclusion_digest",
                "blocked_excluded_task_ids",
                "blocked_morphology_clusters",
                "exposure_predecessor",
                "exposure_predecessor_digest",
                "semantic_exposure_resolution",
                "exposure_successor",
                "exposure_successor_digest",
                "prospective_protocol",
                "protocol_digest",
                "records",
                "attrition_counts",
                "label_state",
                "proposal_archive_digest",
            },
            "campaign proposal archive",
        )
        archived_algorithm = data["selection_algorithm"]
        if archived_algorithm == CAMPAIGN_SELECTION_ALGORITHM:
            expected_schema = CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA
            expected_dependence = (
                "bd-morphology-cluster-and-hd-disjoint-attributes-plus-exact-pair/v2"
            )
            expected_hd_status = (
                "enforced-constituent-disjoint-within-batch-and-live-ledger/v2"
            )
        elif archived_algorithm == CAMPAIGN_SELECTION_ALGORITHM_V1:
            expected_schema = CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1
            expected_dependence = (
                "bd-morphology-cluster-and-hd-exact-ordered-pair/v1"
            )
            expected_hd_status = (
                "assumed-for-fit-not-established-by-corpus-metadata/v1"
            )
        else:
            expected_schema = None
            expected_dependence = None
            expected_hd_status = None
        if (
            expected_schema is None
            or data["schema"] != expected_schema
            or data["coverage_scope"]
            != "stage-a-soft-claim-emission-only/v1"
            or data["end_to_end_support_gate_coverage_claimed"] is not False
            or data["dependence_unit_semantics"]
            != expected_dependence
            or data["hd_cross_pair_independence_status"]
            != expected_hd_status
            or data["transport_failure_evidence"]
            != "bounded-local-exception-digest-without-codex-receipt/v1"
            or data["label_state"] != "withheld"
        ):
            raise SemanticCalibrationCampaignError(
                "unsupported proposal archive or revealed labels"
            )
        protocol_digest = _digest(data["protocol_digest"], "protocol digest")
        protocol = SoftScorerProtocol.from_data(
            _mapping(data["prospective_protocol"], "prospective protocol"),
            expected_digest=protocol_digest,
        )
        execution_config = SemanticCalibrationExecutionConfig.from_data(
            _mapping(data["execution_config"], "campaign execution config")
        )
        if execution_config.digest != _digest(
            data["execution_config_digest"], "execution config digest"
        ):
            raise SemanticCalibrationCampaignError(
                "campaign execution config digest differs"
            )
        records = tuple(
            SemanticCalibrationProposalRecord.from_data(
                _mapping(item, "proposal record"), protocol=protocol
            )
            for item in _list(data["records"], "proposal records")
        )
        predecessor = ExposureLedger.from_dict(
            _mapping(data["exposure_predecessor"], "exposure predecessor")
        )
        successor = ExposureLedger.from_dict(
            _mapping(data["exposure_successor"], "exposure successor")
        )
        if predecessor.digest != _address(
            data["exposure_predecessor_digest"],
            "exposure predecessor digest",
        ) or successor.digest != _address(
            data["exposure_successor_digest"],
            "exposure successor digest",
        ):
            raise SemanticCalibrationCampaignError(
                "campaign exposure ledger digest differs"
            )
        whitelist: list[CleanCohortEntry] = []
        for raw in _list(
            data["clean_cohort_whitelist"], "clean cohort whitelist"
        ):
            entry = _fields(
                _mapping(raw, "clean cohort whitelist entry"),
                {"task_id", "family", "concepts", "split"},
                "clean cohort whitelist entry",
            )
            whitelist.append(
                (
                    entry["task_id"],
                    entry["family"],
                    tuple(_list(entry["concepts"], "whitelist concepts")),
                    entry["split"],
                )
            )
        result = cls(
            protocol=protocol,
            execution_config=execution_config,
            selection_seed=data["selection_seed"],
            selection_seed_digest=data["selection_seed_digest"],
            candidate_count=data["candidate_count"],
            families=tuple(_list(data["families"], "campaign families")),
            semantic_cohort=data["semantic_cohort"],
            source_corpus_manifest_digest=data[
                "source_corpus_manifest_digest"
            ],
            development_manifest_digest=data["development_manifest_digest"],
            split_source_digest=data["split_source_digest"],
            split_manifest_digest=data["split_manifest_digest"],
            historical_seed_digest=data["historical_seed_digest"],
            resolver_policy_digest=data["resolver_policy_digest"],
            cohort_report_digest=data["cohort_report_digest"],
            clean_cohort_whitelist_digest=data[
                "clean_cohort_whitelist_digest"
            ],
            clean_cohort_whitelist=tuple(whitelist),
            blocked_policy_digest=data["blocked_policy_digest"],
            blocked_exclusion_digest=data["blocked_exclusion_digest"],
            blocked_excluded_task_ids=tuple(
                _list(
                    data["blocked_excluded_task_ids"],
                    "blocked excluded task IDs",
                )
            ),
            blocked_morphology_clusters=tuple(
                _list(
                    data["blocked_morphology_clusters"],
                    "blocked morphology clusters",
                )
            ),
            exposure_predecessor=predecessor,
            exposure_successor=successor,
            records=records,
            selection_algorithm=archived_algorithm,
        )
        if data["semantic_exposure_resolution"] != result.content_data()[
            "semantic_exposure_resolution"
        ]:
            raise SemanticCalibrationCampaignError(
                "semantic exposure resolution receipt differs"
            )
        if data["attrition_counts"] != result.content_data()["attrition_counts"]:
            raise SemanticCalibrationCampaignError(
                "proposal archive attrition counts differ"
            )
        archived = _digest(
            data["proposal_archive_digest"], "proposal archive digest"
        )
        if result.digest != archived or (
            expected_digest is not None
            and result.digest
            != _digest(expected_digest, "expected proposal archive digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign proposal archive digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign proposal archive is not canonical"
            )
        return result


def _panel_map(
    value: Mapping[str, PanelInput],
    observation_ids: Sequence[str],
) -> Mapping[str, PanelInput]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SemanticCalibrationCampaignError(
            "cold panel inputs must be keyed by observation ID"
        )
    expected = set(observation_ids)
    if set(value) != expected:
        raise SemanticCalibrationCampaignError(
            "cold panel inputs differ from the exact planned selection"
        )
    return value


@dataclass(frozen=True)
class SemanticCalibrationCommitmentBatch:
    """All pre-family score inputs, frozen together before any scorer runs."""

    proposal_archive: SemanticCalibrationProposalArchive
    plan: SemanticCalibrationPlan
    commitments: tuple[SemanticCalibrationScoreCommitment, ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.proposal_archive, SemanticCalibrationProposalArchive
        ):
            raise TypeError("proposal_archive must be a proposal archive")
        self.proposal_archive.assert_untampered()
        if not isinstance(self.plan, SemanticCalibrationPlan):
            raise TypeError("plan must be SemanticCalibrationPlan")
        self.plan.assert_untampered()
        protocol = self.proposal_archive.protocol
        expected_plan = {
            "protocol_digest": protocol.digest(),
            "corpus_manifest_digest": (
                self.proposal_archive.source_corpus_manifest_digest
            ),
            "development_manifest_digest": (
                self.proposal_archive.development_manifest_digest
            ),
            "split_source_digest": self.proposal_archive.split_source_digest,
            "split_manifest_digest": self.proposal_archive.split_manifest_digest,
            "label_reveal_protocol_digest": (
                semantic_campaign_label_reveal_protocol_digest()
            ),
        }
        for name, wanted in expected_plan.items():
            if getattr(self.plan, name) != wanted:
                raise SemanticCalibrationCampaignError(
                    f"calibration plan {name} differs from proposal archive"
                )
        soft_records = self.proposal_archive.soft_records
        planned = tuple(item.candidate.selection for item in soft_records)
        if self.plan.selections != planned:
            raise SemanticCalibrationCampaignError(
                "only and all soft-claim candidates must enter calibration plan"
            )
        if not isinstance(self.commitments, tuple) or any(
            not isinstance(item, SemanticCalibrationScoreCommitment)
            for item in self.commitments
        ):
            raise TypeError("commitments must be an immutable typed tuple")
        commitment_ids = tuple(
            item.selection.observation_id for item in self.commitments
        )
        planned_ids = tuple(item.observation_id for item in planned)
        if commitment_ids != planned_ids:
            raise SemanticCalibrationCampaignError(
                "score commitments differ from exact calibration plan"
            )
        for record, commitment in zip(
            soft_records, self.commitments, strict=True
        ):
            commitment.assert_untampered()
            if (
                commitment.plan != self.plan
                or commitment.selection != record.candidate.selection
                or commitment.support != record.candidate.support
                or commitment.proposal_transport != record.proposal_transport
                or commitment.protocol != protocol
            ):
                raise SemanticCalibrationCampaignError(
                    "score commitment differs from its frozen soft proposal"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.proposal_archive.digest,
            self.plan.digest,
            tuple(item.digest for item in self.commitments),
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_COMMITMENT_BATCH_SCHEMA,
            "causal_state": "all_score_commitments_frozen_before_scoring/v1",
            "label_state": "withheld",
            "proposal_archive": self.proposal_archive.to_data(),
            "proposal_archive_digest": self.proposal_archive.digest,
            "calibration_plan": self.plan.to_data(),
            "calibration_plan_digest": self.plan.digest,
            "score_commitments": [item.to_data() for item in self.commitments],
            "score_commitment_digests": [
                item.digest for item in self.commitments
            ],
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "commitment_batch_digest": self.digest}

    def assert_untampered(self) -> None:
        self.proposal_archive.assert_untampered()
        self.plan.assert_untampered()
        for item in self.commitments:
            item.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign commitment batch changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        panels: Mapping[str, PanelInput],
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationCommitmentBatch":
        data = _fields(
            _mapping(value, "campaign commitment batch"),
            {
                "schema",
                "causal_state",
                "label_state",
                "proposal_archive",
                "proposal_archive_digest",
                "calibration_plan",
                "calibration_plan_digest",
                "score_commitments",
                "score_commitment_digests",
                "commitment_batch_digest",
            },
            "campaign commitment batch",
        )
        if (
            data["schema"] != CAMPAIGN_COMMITMENT_BATCH_SCHEMA
            or data["causal_state"]
            != "all_score_commitments_frozen_before_scoring/v1"
            or data["label_state"] != "withheld"
        ):
            raise SemanticCalibrationCampaignError(
                "unsupported campaign commitment batch or label state"
            )
        archive = SemanticCalibrationProposalArchive.from_data(
            _mapping(data["proposal_archive"], "proposal archive"),
            expected_digest=_digest(
                data["proposal_archive_digest"], "proposal archive digest"
            ),
        )
        plan = SemanticCalibrationPlan.from_data(
            _mapping(data["calibration_plan"], "calibration plan"),
            expected_digest=_digest(
                data["calibration_plan_digest"], "calibration plan digest"
            ),
        )
        panel_inputs = _panel_map(
            panels, tuple(item.observation_id for item in plan.selections)
        )
        raw_commitments = _list(
            data["score_commitments"], "score commitments"
        )
        raw_digests = _list(
            data["score_commitment_digests"], "score commitment digests"
        )
        if len(raw_commitments) != len(raw_digests):
            raise SemanticCalibrationCampaignError(
                "score commitment list and digest list differ"
            )
        commitments = tuple(
            SemanticCalibrationScoreCommitment.from_data(
                _mapping(item, "score commitment"),
                expected_digest=_digest(raw_digest, "score commitment digest"),
                panel=panel_inputs[selection.observation_id],
            )
            for item, raw_digest, selection in zip(
                raw_commitments, raw_digests, plan.selections, strict=True
            )
        )
        result = cls(archive, plan, commitments)
        archived = _digest(
            data["commitment_batch_digest"], "commitment batch digest"
        )
        if result.digest != archived or (
            expected_digest is not None
            and result.digest
            != _digest(expected_digest, "expected commitment batch digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign commitment batch digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign commitment batch is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationScoreBatch:
    """All planned one-panel scorer attempts, frozen before label reveal."""

    commitment_batch: SemanticCalibrationCommitmentBatch
    attempts: tuple[SemanticCalibrationScoreAttempt, ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.commitment_batch, SemanticCalibrationCommitmentBatch
        ):
            raise TypeError("commitment_batch must be a commitment batch")
        self.commitment_batch.assert_untampered()
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(item, SemanticCalibrationScoreAttempt)
            for item in self.attempts
        ):
            raise TypeError("attempts must be an immutable typed tuple")
        if len(self.attempts) != len(self.commitment_batch.commitments):
            raise SemanticCalibrationCampaignError(
                "score attempts do not cover every frozen commitment"
            )
        for commitment, attempt in zip(
            self.commitment_batch.commitments, self.attempts, strict=True
        ):
            attempt.assert_untampered()
            if attempt.commitment != commitment:
                raise SemanticCalibrationCampaignError(
                    "score attempt differs from the frozen commitment batch"
                )
            # This explicit edge must survive even though the lower-level
            # attempt constructor checks it too: the final campaign retains
            # full attempts precisely because measurements alone do not.
            if (
                attempt.score_artifact.record.pre_observation_commitment_digest
                != commitment.digest
            ):
                raise SemanticCalibrationCampaignError(
                    "score artifact pre-observation parent differs from commitment"
                )
            scorer_receipt = attempt.score_artifact.receipt
            config = self.commitment_batch.proposal_archive.execution_config
            if isinstance(scorer_receipt, CodexReceipt) and (
                scorer_receipt.codex_launcher_digest
                != config.expected_codex_launcher_digest
                or scorer_receipt.cloud_config_bundle_cache_binding
                != config.cloud_policy_cache_binding
            ):
                raise SemanticCalibrationCampaignError(
                    "scorer receipt differs from frozen launcher/cache identity"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    @property
    def all_present(self) -> bool:
        return all(
            item.score_artifact.record.outcome == "present"
            and item.score_artifact.record.score is not None
            for item in self.attempts
        )

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.commitment_batch.digest,
            tuple(item.digest for item in self.attempts),
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCORE_BATCH_SCHEMA,
            "causal_state": "all_score_attempts_frozen_before_label_reveal/v1",
            "label_state": "withheld",
            "commitment_batch": self.commitment_batch.to_data(),
            "commitment_batch_digest": self.commitment_batch.digest,
            "score_attempts": [item.to_data() for item in self.attempts],
            "score_attempt_digests": [item.digest for item in self.attempts],
            "all_attempts_present": self.all_present,
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "score_batch_digest": self.digest}

    def assert_untampered(self) -> None:
        self.commitment_batch.assert_untampered()
        for item in self.attempts:
            item.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "campaign score batch changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        panels: Mapping[str, PanelInput],
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationScoreBatch":
        data = _fields(
            _mapping(value, "campaign score batch"),
            {
                "schema",
                "causal_state",
                "label_state",
                "commitment_batch",
                "commitment_batch_digest",
                "score_attempts",
                "score_attempt_digests",
                "all_attempts_present",
                "score_batch_digest",
            },
            "campaign score batch",
        )
        if (
            data["schema"] != CAMPAIGN_SCORE_BATCH_SCHEMA
            or data["causal_state"]
            != "all_score_attempts_frozen_before_label_reveal/v1"
            or data["label_state"] != "withheld"
            or type(data["all_attempts_present"]) is not bool
        ):
            raise SemanticCalibrationCampaignError(
                "unsupported campaign score batch or causal state"
            )
        commitment_batch = SemanticCalibrationCommitmentBatch.from_data(
            _mapping(data["commitment_batch"], "commitment batch"),
            panels=panels,
            expected_digest=_digest(
                data["commitment_batch_digest"], "commitment batch digest"
            ),
        )
        raw_attempts = _list(data["score_attempts"], "score attempts")
        raw_digests = _list(
            data["score_attempt_digests"], "score attempt digests"
        )
        if len(raw_attempts) != len(raw_digests) or len(raw_attempts) != len(
            commitment_batch.commitments
        ):
            raise SemanticCalibrationCampaignError(
                "score attempt archive does not cover commitment batch"
            )
        attempts = tuple(
            SemanticCalibrationScoreAttempt.from_data(
                _mapping(item, "score attempt"),
                expected_digest=_digest(raw_digest, "score attempt digest"),
                panel=panels[commitment.selection.observation_id],
            )
            for item, raw_digest, commitment in zip(
                raw_attempts,
                raw_digests,
                commitment_batch.commitments,
                strict=True,
            )
        )
        result = cls(commitment_batch, attempts)
        if data["all_attempts_present"] is not result.all_present:
            raise SemanticCalibrationCampaignError(
                "score batch outcome summary differs from attempts"
            )
        archived = _digest(data["score_batch_digest"], "score batch digest")
        if result.digest != archived or (
            expected_digest is not None
            and result.digest
            != _digest(expected_digest, "expected score batch digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign score batch digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign score batch is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationLabelReveal:
    """Post-score opening of one episode's two-label commitment."""

    candidate_digest: str
    score_batch_digest: str
    run_id: str
    label_commitment_digest: str
    label_nonce: str
    labels: tuple[bool, bool]

    def __post_init__(self) -> None:
        _digest(self.candidate_digest, "candidate digest")
        _digest(self.score_batch_digest, "score batch digest")
        _identifier(self.run_id, "label-reveal run ID")
        _digest(self.label_commitment_digest, "label commitment digest")
        _digest(self.label_nonce, "label nonce")
        if (
            not isinstance(self.labels, tuple)
            or len(self.labels) != 2
            or any(type(item) is not bool for item in self.labels)
            or set(self.labels) != {False, True}
        ):
            raise SemanticCalibrationCampaignError(
                "label reveal must contain one positive and one negative bool"
            )
        expected = canonical_digest(
            {
                "run_id": self.run_id,
                "labels": [
                    {"query_id": f"query-{index}", "positive": positive}
                    for index, positive in enumerate(self.labels)
                ],
                "nonce": self.label_nonce,
                "version": "latent-label-seal/v1",
            }
        )
        if expected != self.label_commitment_digest:
            raise SemanticCalibrationCampaignError(
                "opened episode labels differ from latent commitment"
            )

    @property
    def affirmative_label(self) -> bool:
        return self.labels[0]

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_LABEL_REVEAL_SCHEMA,
            "causal_order": "complete-score-batch-then-label-opening/v1",
            "candidate_digest": self.candidate_digest,
            "score_batch_digest": self.score_batch_digest,
            "run_id": self.run_id,
            "label_commitment_digest": self.label_commitment_digest,
            "label_nonce": self.label_nonce,
            "labels": [
                {"query_id": f"query-{index}", "positive": positive}
                for index, positive in enumerate(self.labels)
            ],
            "selected_query_id": "query-0",
            "affirmative_label": self.affirmative_label,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "label_reveal_digest": self.digest}

    def assert_matches(
        self,
        candidate: SemanticCalibrationCandidate,
        score_batch: SemanticCalibrationScoreBatch,
    ) -> None:
        candidate.assert_untampered()
        score_batch.assert_untampered()
        self._assert_matches_verified_parents(candidate, score_batch)

    def _assert_matches_verified_parents(
        self,
        candidate: SemanticCalibrationCandidate,
        score_batch: SemanticCalibrationScoreBatch,
    ) -> None:
        """Check this reveal after its immutable parents were checked once.

        A complete score batch is shared by every reveal.  Re-running its
        recursive tamper check once per reveal makes campaign construction and
        cold decoding quadratic in the number of soft claims.  Campaign-level
        callers first verify that one shared parent, then use this exact join
        check for each child.
        """

        expected = {
            "candidate_digest": candidate.digest,
            # The campaign boundary has already recursively verified this
            # immutable parent.  Recomputing its digest here serializes the
            # complete M-attempt batch once for every one of M reveals.
            "score_batch_digest": score_batch._sealed_digest,
            "run_id": candidate.support.run_id,
            "label_commitment_digest": candidate.episode_plan_data[
                "label_commitment_digest"
            ],
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise SemanticCalibrationCampaignError(
                    f"label reveal {name} differs from frozen campaign parent"
                )

    @classmethod
    def from_episode(
        cls,
        candidate: SemanticCalibrationCandidate,
        episode: EpisodePlan,
        score_batch: SemanticCalibrationScoreBatch,
    ) -> "SemanticCalibrationLabelReveal":
        candidate.assert_untampered()
        score_batch.assert_untampered()
        return cls._from_episode_with_verified_parents(
            candidate,
            episode,
            score_batch,
        )

    @classmethod
    def _from_episode_with_verified_parents(
        cls,
        candidate: SemanticCalibrationCandidate,
        episode: EpisodePlan,
        score_batch: SemanticCalibrationScoreBatch,
    ) -> "SemanticCalibrationLabelReveal":
        if not score_batch.all_present:
            raise SemanticCalibrationCampaignError(
                "labels cannot be opened while a scorer attempt failed"
            )
        if episode.to_data() != candidate.episode_plan_data:
            raise SemanticCalibrationCampaignError(
                "label episode differs from prospective candidate"
            )
        revealed = episode._revealed_labels()
        if tuple(item.query_id for item in revealed) != ("query-0", "query-1"):
            raise SemanticCalibrationCampaignError(
                "episode revealed non-canonical query IDs"
            )
        result = cls(
            candidate_digest=candidate.digest,
            score_batch_digest=score_batch._sealed_digest,
            run_id=episode.run_id,
            label_commitment_digest=episode.label_commitment_digest,
            label_nonce=episode._label_nonce,
            labels=tuple(item.positive for item in revealed),  # type: ignore[arg-type]
        )
        result._assert_matches_verified_parents(candidate, score_batch)
        return result

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        candidate: SemanticCalibrationCandidate,
        score_batch: SemanticCalibrationScoreBatch,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationLabelReveal":
        candidate.assert_untampered()
        score_batch.assert_untampered()
        return cls._from_data_with_verified_parents(
            value,
            candidate=candidate,
            score_batch=score_batch,
            expected_digest=expected_digest,
        )

    @classmethod
    def _from_data_with_verified_parents(
        cls,
        value: Mapping[str, Any],
        *,
        candidate: SemanticCalibrationCandidate,
        score_batch: SemanticCalibrationScoreBatch,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationLabelReveal":
        data = _fields(
            _mapping(value, "campaign label reveal"),
            {
                "schema",
                "causal_order",
                "candidate_digest",
                "score_batch_digest",
                "run_id",
                "label_commitment_digest",
                "label_nonce",
                "labels",
                "selected_query_id",
                "affirmative_label",
                "label_reveal_digest",
            },
            "campaign label reveal",
        )
        if (
            data["schema"] != CAMPAIGN_LABEL_REVEAL_SCHEMA
            or data["causal_order"]
            != "complete-score-batch-then-label-opening/v1"
            or data["selected_query_id"] != "query-0"
            or type(data["affirmative_label"]) is not bool
        ):
            raise SemanticCalibrationCampaignError(
                "unsupported campaign label reveal"
            )
        raw_labels = _list(data["labels"], "revealed labels")
        if len(raw_labels) != 2:
            raise SemanticCalibrationCampaignError(
                "campaign label reveal must contain two labels"
            )
        labels: list[bool] = []
        for index, raw in enumerate(raw_labels):
            item = _fields(
                _mapping(raw, f"revealed label {index}"),
                {"query_id", "positive"},
                f"revealed label {index}",
            )
            if item["query_id"] != f"query-{index}" or type(
                item["positive"]
            ) is not bool:
                raise SemanticCalibrationCampaignError(
                    "revealed label identity or type differs"
                )
            labels.append(item["positive"])
        result = cls(
            candidate_digest=data["candidate_digest"],
            score_batch_digest=data["score_batch_digest"],
            run_id=data["run_id"],
            label_commitment_digest=data["label_commitment_digest"],
            label_nonce=data["label_nonce"],
            labels=tuple(labels),  # type: ignore[arg-type]
        )
        result._assert_matches_verified_parents(candidate, score_batch)
        if data["affirmative_label"] is not result.affirmative_label:
            raise SemanticCalibrationCampaignError(
                "selected calibration label differs from query-0 reveal"
            )
        archived = _digest(data["label_reveal_digest"], "label reveal digest")
        if result.digest != archived or (
            expected_digest is not None
            and result.digest
            != _digest(expected_digest, "expected label reveal digest")
        ):
            raise SemanticCalibrationCampaignError(
                "campaign label reveal digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "campaign label reveal is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationCampaignArtifact:
    """Complete proposal, scoring, reveal, and fitted-family archive."""

    score_batch: SemanticCalibrationScoreBatch
    label_reveals: tuple[SemanticCalibrationLabelReveal, ...]
    calibration: SemanticCalibrationArtifact
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.score_batch, SemanticCalibrationScoreBatch):
            raise TypeError("score_batch must be SemanticCalibrationScoreBatch")
        self.score_batch.assert_untampered()
        if not self.score_batch.all_present:
            raise SemanticCalibrationCampaignError(
                "failed scorer attempts cannot enter completed campaign"
            )
        if not isinstance(self.label_reveals, tuple) or any(
            not isinstance(item, SemanticCalibrationLabelReveal)
            for item in self.label_reveals
        ):
            raise TypeError("label_reveals must be an immutable typed tuple")
        soft_records = self.score_batch.commitment_batch.proposal_archive.soft_records
        if len(self.label_reveals) != len(soft_records):
            raise SemanticCalibrationCampaignError(
                "label reveals do not cover exact soft-candidate plan"
            )
        for record, reveal in zip(
            soft_records, self.label_reveals, strict=True
        ):
            reveal._assert_matches_verified_parents(
                record.candidate,
                self.score_batch,
            )
        if not isinstance(self.calibration, SemanticCalibrationArtifact):
            raise TypeError("calibration must be SemanticCalibrationArtifact")
        self.calibration.assert_untampered()
        batch = self.score_batch.commitment_batch
        if (
            self.calibration.plan != batch.plan
            or self.calibration.protocol != batch.proposal_archive.protocol
            or len(self.calibration.measurements) != len(self.score_batch.attempts)
        ):
            raise SemanticCalibrationCampaignError(
                "fitted calibration differs from complete campaign parents"
            )
        for attempt, reveal, measurement in zip(
            self.score_batch.attempts,
            self.label_reveals,
            self.calibration.measurements,
            strict=True,
        ):
            commitment = attempt.commitment
            record = attempt.score_artifact.record
            if record.pre_observation_commitment_digest != commitment.digest:
                raise SemanticCalibrationCampaignError(
                    "measurement score artifact lost its commitment parent"
                )
            expected = {
                "selection": commitment.selection,
                "score_artifact_digest": attempt.score_artifact.digest,
                "label_reveal_receipt_digest": reveal.digest,
            }
            for name, wanted in expected.items():
                if getattr(measurement, name) != wanted:
                    raise SemanticCalibrationCampaignError(
                        f"campaign measurement {name} differs from full parents"
                    )
            if measurement.join_receipt.score_record_digest != record.digest():
                raise SemanticCalibrationCampaignError(
                    "measurement join differs from retained score record"
                )
            if (
                measurement.development_unit.affirmative_label
                is not reveal.affirmative_label
            ):
                raise SemanticCalibrationCampaignError(
                    "measurement label differs from post-score query-0 reveal"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            self.score_batch.digest,
            tuple(item.digest for item in self.label_reveals),
            self.calibration.digest,
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA,
            "reference_execution_semantics": "pure-python-semantic-calibration/v1",
            "calibration_scope": (
                "stage-a-conditional-on-soft-claim-emission-not-support-gate-pass/v1"
            ),
            "python_predicate_authoritative": True,
            "optional_checker_may_affect_result": False,
            "score_batch": self.score_batch.to_data(),
            "score_batch_digest": self.score_batch.digest,
            "label_reveals": [item.to_data() for item in self.label_reveals],
            "label_reveal_digests": [item.digest for item in self.label_reveals],
            "semantic_calibration": self.calibration.to_data(),
            "semantic_calibration_digest": self.calibration.digest,
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "campaign_digest": self.digest}

    def assert_untampered(self) -> None:
        self.score_batch.assert_untampered()
        self.calibration.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationCampaignError(
                "semantic calibration campaign changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        panels: Mapping[str, PanelInput],
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationCampaignArtifact":
        """Cold-decode all parents and replay against exact selected PNGs."""

        data = _fields(
            _mapping(value, "semantic calibration campaign"),
            {
                "schema",
                "reference_execution_semantics",
                "calibration_scope",
                "python_predicate_authoritative",
                "optional_checker_may_affect_result",
                "score_batch",
                "score_batch_digest",
                "label_reveals",
                "label_reveal_digests",
                "semantic_calibration",
                "semantic_calibration_digest",
                "campaign_digest",
            },
            "semantic calibration campaign",
        )
        if (
            data["schema"] != SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA
            or data["reference_execution_semantics"]
            != "pure-python-semantic-calibration/v1"
            or data["calibration_scope"]
            != "stage-a-conditional-on-soft-claim-emission-not-support-gate-pass/v1"
            or data["python_predicate_authoritative"] is not True
            or data["optional_checker_may_affect_result"] is not False
        ):
            raise SemanticCalibrationCampaignError(
                "campaign reference semantics or authority changed"
            )
        score_batch = SemanticCalibrationScoreBatch.from_data(
            _mapping(data["score_batch"], "score batch"),
            panels=panels,
            expected_digest=_digest(data["score_batch_digest"], "score batch digest"),
        )
        raw_reveals = _list(data["label_reveals"], "label reveals")
        raw_digests = _list(
            data["label_reveal_digests"], "label reveal digests"
        )
        soft_records = score_batch.commitment_batch.proposal_archive.soft_records
        if len(raw_reveals) != len(raw_digests) or len(raw_reveals) != len(
            soft_records
        ):
            raise SemanticCalibrationCampaignError(
                "label reveal archive differs from soft candidate plan"
            )
        reveals = tuple(
            SemanticCalibrationLabelReveal._from_data_with_verified_parents(
                _mapping(raw, "label reveal"),
                candidate=record.candidate,
                score_batch=score_batch,
                expected_digest=_digest(raw_digest, "label reveal digest"),
            )
            for raw, raw_digest, record in zip(
                raw_reveals, raw_digests, soft_records, strict=True
            )
        )
        calibration = SemanticCalibrationArtifact.from_data(
            _mapping(data["semantic_calibration"], "semantic calibration"),
            expected_digest=_digest(
                data["semantic_calibration_digest"],
                "semantic calibration digest",
            ),
        )
        result = cls(score_batch, reveals, calibration)
        archived = _digest(data["campaign_digest"], "campaign digest")
        if result.digest != archived or (
            expected_digest is not None
            and result.digest != _digest(expected_digest, "expected campaign digest")
        ):
            raise SemanticCalibrationCampaignError(
                "semantic calibration campaign digest differs"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationCampaignError(
                "semantic calibration campaign is not canonical"
            )
        return result


def _proposal_archive_from_full_campaign(
    campaign: SemanticCalibrationCampaignArtifact | Mapping[str, Any],
) -> SemanticCalibrationProposalArchive:
    if isinstance(campaign, SemanticCalibrationCampaignArtifact):
        campaign.assert_untampered()
        return campaign.score_batch.commitment_batch.proposal_archive
    data = _fields(
        _mapping(campaign, "semantic calibration campaign"),
        {
            "schema",
            "reference_execution_semantics",
            "calibration_scope",
            "python_predicate_authoritative",
            "optional_checker_may_affect_result",
            "score_batch",
            "score_batch_digest",
            "label_reveals",
            "label_reveal_digests",
            "semantic_calibration",
            "semantic_calibration_digest",
            "campaign_digest",
        },
        "semantic calibration campaign",
    )
    if (
        data["schema"] != SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA
        or data["reference_execution_semantics"]
        != "pure-python-semantic-calibration/v1"
        or data["python_predicate_authoritative"] is not True
        or data["optional_checker_may_affect_result"] is not False
    ):
        raise SemanticCalibrationCampaignError(
            "raw campaign is not the canonical Python-authoritative schema"
        )
    score_batch = _fields(
        _mapping(data["score_batch"], "campaign score batch"),
        {
            "schema",
            "causal_state",
            "label_state",
            "commitment_batch",
            "commitment_batch_digest",
            "score_attempts",
            "score_attempt_digests",
            "all_attempts_present",
            "score_batch_digest",
        },
        "campaign score batch",
    )
    if score_batch["schema"] != CAMPAIGN_SCORE_BATCH_SCHEMA:
        raise SemanticCalibrationCampaignError(
            "raw campaign contains another score-batch schema"
        )
    commitment_batch = _fields(
        _mapping(score_batch["commitment_batch"], "campaign commitment batch"),
        {
            "schema",
            "causal_state",
            "label_state",
            "proposal_archive",
            "proposal_archive_digest",
            "calibration_plan",
            "calibration_plan_digest",
            "score_commitments",
            "score_commitment_digests",
            "commitment_batch_digest",
        },
        "campaign commitment batch",
    )
    if commitment_batch["schema"] != CAMPAIGN_COMMITMENT_BATCH_SCHEMA:
        raise SemanticCalibrationCampaignError(
            "raw campaign contains another commitment-batch schema"
        )
    return SemanticCalibrationProposalArchive.from_data(
        _mapping(commitment_batch["proposal_archive"], "proposal archive"),
        expected_digest=_digest(
            commitment_batch["proposal_archive_digest"],
            "proposal archive digest",
        ),
    )


def _resolve_semantic_campaign_bindings(
    campaign: SemanticCalibrationCampaignArtifact | Mapping[str, Any],
    *,
    corpus: ShapeBongardCorpus,
    corpus_manifest: CorpusManifest,
) -> tuple[
    SemanticCalibrationProposalArchive,
    Mapping[str, PanelInput],
    Mapping[str, tuple[bool, bool]],
]:
    """Resolve exact query-0 paths for cold replay without opening test PNGs.

    ``corpus_manifest`` must be an externally authenticated full-release
    manifest supplied by the verifier.  This function checks its content
    address and campaign binding; it does not establish official provenance
    from a self-supplied digest.  The resolver validates its split and metadata
    inventory, but re-hashes pixels only for campaign-selected development
    tasks.  Every selected task's 12 support and two query commitments must
    form an exact, polarity-consistent partition of its official 14 panels.
    """

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be ShapeBongardCorpus")
    if not isinstance(corpus_manifest, CorpusManifest):
        raise TypeError("corpus_manifest must be CorpusManifest")
    expected_full_digest = "sha256:" + canonical_digest(
        corpus_manifest.content_dict()
    )
    if corpus_manifest.digest != expected_full_digest:
        raise SemanticCalibrationCampaignError(
            "full corpus manifest content differs from its digest"
        )
    archive = _proposal_archive_from_full_campaign(campaign)
    if archive.source_corpus_manifest_digest != corpus_manifest.digest:
        raise SemanticCalibrationCampaignError(
            "campaign source digest differs from the externally authenticated "
            "full manifest"
        )
    if corpus_manifest.layout != corpus.layout or dict(
        corpus_manifest.family_counts
    ) != dict(corpus.family_counts):
        raise SemanticCalibrationCampaignError(
            "full manifest layout or family inventory differs from corpus"
        )
    manifest_inventory = tuple(
        (item.task_id, item.family) for item in corpus_manifest.tasks
    )
    corpus_inventory = tuple((item.task_id, item.family) for item in corpus.tasks)
    if (
        manifest_inventory != corpus_inventory
        or len(manifest_inventory) != len(set(manifest_inventory))
    ):
        raise SemanticCalibrationCampaignError(
            "full manifest task inventory differs from corpus metadata"
        )
    if (
        corpus.split.source_digest is None
        or archive.split_source_digest != corpus.split.source_digest
        or corpus_manifest.split.to_manifest_dict()
        != corpus.split.to_manifest_dict()
        or archive.split_manifest_digest
        != canonical_digest(corpus.split.to_manifest_dict())
    ):
        raise SemanticCalibrationCampaignError(
            "campaign, corpus, and full-manifest split identities differ"
        )
    reproduced_cohort = _clean_cohort_whitelist(
        corpus,
        archive.families,
        archive.semantic_cohort,
    )
    archived_cohort = (
        archive.clean_cohort_whitelist,
        archive.historical_seed_digest,
        archive.resolver_policy_digest,
        archive.cohort_report_digest,
        archive.clean_cohort_whitelist_digest,
        archive.blocked_policy_digest,
        archive.blocked_exclusion_digest,
        archive.blocked_excluded_task_ids,
        archive.blocked_morphology_clusters,
    )
    if reproduced_cohort != archived_cohort:
        raise SemanticCalibrationCampaignError(
            "campaign clean-cohort whitelist is not exhaustive for the "
            "authenticated corpus inventory"
        )

    selected = tuple(
        parse_official_task_id(record.candidate.selection.task_id)
        for record in archive.records
    )
    if any(
        corpus.split.assignment(item.task_id).split not in _ALLOWED_SPLITS
        for item in selected
    ):
        raise SemanticCalibrationCampaignError(
            "campaign panel resolution encountered an official test task"
        )
    development_manifest = _development_manifest(corpus, selected)
    if development_manifest.digest != archive.development_manifest_digest:
        raise SemanticCalibrationCampaignError(
            "selected task pixels differ from the development submanifest"
        )
    full_tasks = {item.task_id: item for item in corpus_manifest.tasks}
    fresh_tasks = {
        item.task_id: item for item in development_manifest.tasks
    }
    panel_inputs: dict[str, PanelInput] = {}
    query_polarities: dict[str, tuple[bool, bool]] = {}
    for ordinal, record in enumerate(archive.records):
        candidate = record.candidate
        task_id = candidate.selection.task_id
        assignment = corpus.split.assignment(task_id)
        if (
            assignment.split not in _ALLOWED_SPLITS
            or candidate.selection.split != assignment.split
            or candidate.episode_plan_data["split"] != assignment.split
            or candidate.episode_plan_data["regime"] != assignment.regime
        ):
            raise SemanticCalibrationCampaignError(
                "candidate plan differs from official development assignment"
            )
        fresh = fresh_tasks[task_id]
        trusted = full_tasks[task_id]
        if fresh.to_dict() != trusted.to_dict():
            raise SemanticCalibrationCampaignError(
                "selected task differs from the trusted full manifest"
            )
        if candidate.episode_plan_data["task_manifest_digest"] != (
            fresh.digest.removeprefix("sha256:")
        ) or candidate.episode_plan_data["corpus_digest"] != (
            development_manifest.digest.removeprefix("sha256:")
        ):
            raise SemanticCalibrationCampaignError(
                "candidate episode plan differs from official selected manifests"
            )
        reference_episode = prepare_episode(
            corpus,
            task_id,
            seed=(
                f"semantic-calibration:{archive.selection_seed}:"
                f"{ordinal:06d}"
            ),
            corpus_manifest=development_manifest,
            verifier_id=candidate.support.issued_by,
            label_seal_nonce="0" * 64,
        )
        reference_plan = reference_episode.to_data()
        candidate_plan_without_label = {
            key: value
            for key, value in candidate.episode_plan_data.items()
            if key != "label_commitment_digest"
        }
        reference_plan_without_label = {
            key: value
            for key, value in reference_plan.items()
            if key != "label_commitment_digest"
        }
        if (
            candidate_plan_without_label != reference_plan_without_label
            or candidate.support != reference_episode.support
            or candidate.queries != reference_episode.queries
        ):
            raise SemanticCalibrationCampaignError(
                "candidate support/query episode differs from deterministic "
                "official selection"
            )

        identities: dict[tuple[str, int], Any] = {}
        for panel in fresh.panels:
            key = (panel.sha256.removeprefix("sha256:"), panel.size_bytes)
            if key in identities:
                raise SemanticCalibrationCampaignError(
                    "official selected task has ambiguous duplicate panel bytes"
                )
            identities[key] = panel
        consumed: set[tuple[str, int]] = set()
        for support in candidate.support.support:
            key = (support.panel.sha256, support.panel.byte_count)
            panel = identities.get(key)
            if (
                panel is None
                or key in consumed
                or (panel.polarity == "positive") is not support.positive
            ):
                raise SemanticCalibrationCampaignError(
                    "candidate support does not match official selected pixels"
                )
            consumed.add(key)
        query_paths: list[Path] = []
        official_query_polarities: list[bool] = []
        for query in candidate.queries:
            key = (query.panel.sha256, query.panel.byte_count)
            panel = identities.get(key)
            if panel is None or key in consumed:
                raise SemanticCalibrationCampaignError(
                    "candidate query does not match official selected pixels"
                )
            consumed.add(key)
            query_paths.append(panel.path)
            official_query_polarities.append(panel.polarity == "positive")
        if len(consumed) != 14 or len(query_paths) != 2:
            raise SemanticCalibrationCampaignError(
                "candidate support/query commitments do not exhaust 14 panels"
            )
        if record.status == SOFT_ACCEPTED:
            panel_inputs[candidate.selection.observation_id] = query_paths[0]
            query_polarities[candidate.selection.observation_id] = tuple(
                official_query_polarities
            )  # type: ignore[assignment]

    expected_observations = {
        item.candidate.selection.observation_id for item in archive.soft_records
    }
    if set(panel_inputs) != expected_observations:
        raise SemanticCalibrationCampaignError(
            "resolved query-0 paths differ from exact soft calibration plan"
        )
    return archive, panel_inputs, query_polarities


def resolve_semantic_campaign_panels(
    campaign: SemanticCalibrationCampaignArtifact | Mapping[str, Any],
    *,
    corpus: ShapeBongardCorpus,
    corpus_manifest: CorpusManifest,
) -> Mapping[str, PanelInput]:
    """Return query-0 inputs bound to a verifier-authenticated full manifest.

    The caller, not this self-consistency checker, is responsible for
    authenticating the provenance of ``corpus_manifest``.
    """

    _, panels, _ = _resolve_semantic_campaign_bindings(
        campaign,
        corpus=corpus,
        corpus_manifest=corpus_manifest,
    )
    return panels


def verify_semantic_campaign_against_corpus(
    campaign: SemanticCalibrationCampaignArtifact | Mapping[str, Any],
    *,
    corpus: ShapeBongardCorpus,
    corpus_manifest: CorpusManifest,
) -> tuple[SemanticCalibrationCampaignArtifact, Mapping[str, PanelInput]]:
    """Cold-verify against a caller-authenticated full manifest and corpus."""

    _, panels, query_polarities = _resolve_semantic_campaign_bindings(
        campaign,
        corpus=corpus,
        corpus_manifest=corpus_manifest,
    )
    raw = campaign.to_data() if isinstance(
        campaign, SemanticCalibrationCampaignArtifact
    ) else campaign
    verified = SemanticCalibrationCampaignArtifact.from_data(
        raw,
        panels=panels,
    )
    soft_records = (
        verified.score_batch.commitment_batch.proposal_archive.soft_records
    )
    for record, reveal in zip(
        soft_records, verified.label_reveals, strict=True
    ):
        observation_id = record.candidate.selection.observation_id
        if reveal.labels != query_polarities[observation_id]:
            raise SemanticCalibrationCampaignError(
                "campaign label reveal differs from official query polarities"
            )
    return verified, panels


def _label_nonce(root_nonce: str, task_id: str, ordinal: int) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-semantic-campaign-label-nonce.v1",
            "root_nonce": root_nonce,
            "task_id": task_id,
            "ordinal": ordinal,
        }
    )


def run_semantic_calibration_campaign(
    corpus: ShapeBongardCorpus,
    protocol: SoftScorerProtocol,
    *,
    candidate_count: int,
    seed: str,
    source_corpus_manifest_digest: str,
    expected_codex_launcher_digest: str,
    exposure_ledger: ExposureLedger,
    expected_exposure_ledger_digest: str,
    label_reveal_protocol_digest: str | None = None,
    semantic_cohort: str = "drill",
    families: Sequence[str] = ("bd", "hd"),
    verifier_id: str = "canonical-bongard-verifier",
    label_nonce_root: str | None = None,
    proposer_minutes: int = 15,
    scorer_minutes: int = 10,
    proposer_max_workers: int = 1,
    scorer_max_workers: int = 1,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    proposer_transport: StructuredTransport = run_codex_structured,
    scorer_transport: StructuredTransport = run_codex_named_images_structured,
    on_exposure_precommit: Callable[[ExposureLedger], None] | None = None,
) -> SemanticCalibrationCampaignArtifact:
    """Run the complete causal campaign and return its cold-replayable archive.

    The only pixel reads before proposal attrition are the selected development
    tasks needed to form their 6+6 support and two unlabeled query commitments.
    Official test membership is rejected by metadata before that point.  Every
    accepted typed proposal is archived; direct-only and parser-rejected turns
    remain explicit attrition records and never enter the soft calibration
    plan.
    """

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be ShapeBongardCorpus")
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be SoftScorerProtocol")
    protocol.assert_untampered()
    source_manifest = _address(
        source_corpus_manifest_digest,
        "source corpus manifest digest",
    )
    if not isinstance(exposure_ledger, ExposureLedger):
        raise TypeError("exposure_ledger must be ExposureLedger")
    if exposure_ledger.corpus_digest != source_manifest:
        raise SemanticCalibrationCampaignError(
            "exposure ledger corpus differs from trusted source manifest"
        )
    reveal_protocol = semantic_campaign_label_reveal_protocol_digest()
    if label_reveal_protocol_digest is not None and _digest(
        label_reveal_protocol_digest, "label reveal protocol digest"
    ) != reveal_protocol:
        raise SemanticCalibrationCampaignError(
            "label reveal protocol differs from verifier-owned procedure"
        )
    if not isinstance(verifier_id, str) or not verifier_id.strip():
        raise SemanticCalibrationCampaignError("verifier_id must be non-empty")
    if cloud_policy_cache_snapshot is None:
        frozen_cache_snapshot = snapshot_cloud_policy_cache()
    elif isinstance(cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
        frozen_cache_snapshot = cloud_policy_cache_snapshot
    else:
        raise TypeError(
            "cloud_policy_cache_snapshot must be CloudPolicyCacheSnapshot or None"
        )
    execution_config = SemanticCalibrationExecutionConfig(
        proposer_minutes=proposer_minutes,
        scorer_minutes=scorer_minutes,
        proposer_max_workers=proposer_max_workers,
        scorer_max_workers=scorer_max_workers,
        executable=executable,
        expected_codex_launcher_digest=expected_codex_launcher_digest,
        cloud_policy_cache_binding=frozen_cache_snapshot.binding,
    )
    if not callable(proposer_transport) or not callable(scorer_transport):
        raise TypeError("campaign transports must be callable")
    if on_exposure_precommit is not None and not callable(
        on_exposure_precommit
    ):
        raise TypeError("on_exposure_precommit must be callable or None")
    scope = tuple(families)
    selected = select_semantic_calibration_tasks(
        corpus,
        candidate_count=candidate_count,
        seed=seed,
        exposure_ledger=exposure_ledger,
        expected_exposure_ledger_digest=expected_exposure_ledger_digest,
        semantic_cohort=semantic_cohort,
        families=scope,
    )
    (
        clean_cohort_whitelist,
        historical_seed_digest,
        resolver_policy_digest,
        cohort_report_digest,
        clean_cohort_whitelist_digest,
        blocked_policy_digest,
        blocked_exclusion_digest,
        blocked_excluded_task_ids,
        blocked_morphology_clusters,
    ) = _clean_cohort_whitelist(corpus, scope, semantic_cohort)

    # Record the disclosure against the exact precommitted ledger before the
    # first selected task is materialized for semantic extraction or a model.
    exposure_successor = exposure_ledger
    historical = load_historical_exposure()
    for parsed in selected:
        exposure_successor.assert_semantically_unseen(
            task_ids=(parsed.task_id,),
            historical_seed=historical,
            expected_historical_seed_digest=historical_seed_digest,
            expected_resolver_policy_digest=resolver_policy_digest,
        )
        exposure_successor = exposure_successor.record(
            phase="semantic-calibration",
            actor=protocol.proposer_model_id,
            purpose="stage-a-soft-scorer-calibration-candidate",
            task_ids=(parsed.task_id,),
            source="soft-scorer-protocol:" + protocol.digest(),
            known_task_ids=corpus.task_ids,
            require_unseen=True,
        )
    # A canonical operational caller writes and fsyncs this exact full-batch
    # successor before returning from the hook.  The hook is deliberately
    # before `_development_manifest`: no selected PNG is opened until the
    # complete disclosure transition is durable.  Low-level in-memory tests
    # may omit the hook, but such a call is not crash-safe for live execution.
    if on_exposure_precommit is not None:
        try:
            on_exposure_precommit(exposure_successor)
        except Exception as exc:
            raise SemanticCalibrationCampaignError(
                "exposure precommit failed before selected semantic access"
            ) from exc
    manifest = _development_manifest(corpus, selected)
    if label_nonce_root is None:
        label_nonce_root = secrets.token_hex(32)
    root_nonce = _digest(label_nonce_root, "label nonce root")

    episodes: list[EpisodePlan] = []
    candidates: list[SemanticCalibrationCandidate] = []
    for ordinal, parsed in enumerate(selected):
        assignment = corpus.split.assignment(parsed.task_id)
        if assignment.split not in _ALLOWED_SPLITS:
            # This is repeated immediately before prepare_episode so a changed
            # split cannot turn metadata selection into a test-pixel read.
            raise SemanticCalibrationCampaignError(
                "selected calibration task is no longer train/val"
            )
        episode = prepare_episode(
            corpus,
            parsed.task_id,
            seed=f"semantic-calibration:{seed}:{ordinal:06d}",
            corpus_manifest=manifest,
            verifier_id=verifier_id,
            label_seal_nonce=_label_nonce(root_nonce, parsed.task_id, ordinal),
        )
        episodes.append(episode)
        candidates.append(
            SemanticCalibrationCandidate.from_episode(
                episode, parsed, ordinal=ordinal
            )
        )

    # Phase 1: run every support-only proposer.  No selected query path is
    # passed to this transport, and no scorer starts until the full ledger is
    # sealed below.
    def run_proposer_phase_item(
        item: tuple[SemanticCalibrationCandidate, EpisodePlan],
    ) -> SemanticCalibrationProposalRecord:
        candidate, episode = item
        positive_paths = tuple(
            source.path for source in episode._support_sources if source.positive
        )
        negative_paths = tuple(
            source.path for source in episode._support_sources if not source.positive
        )
        try:
            transport_result = propose_typed_visual(
                positive_paths,
                negative_paths,
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=protocol,
                minutes=proposer_minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=frozen_cache_snapshot,
                transport=proposer_transport,
            )
        except TypedVisualProposalRejected as exc:
            return SemanticCalibrationProposalRecord(
                candidate,
                TYPED_REJECTED,
                rejected_attempt=exc.attempt,
            )
        except Exception as exc:  # noqa: BLE001 - attrition is archived.
            failure_type, reason_digest = _bounded_failure(exc)
            return SemanticCalibrationProposalRecord(
                candidate,
                TRANSPORT_FAILED,
                failure_type=failure_type,
                failure_reason_digest=reason_digest,
            )
        status = (
            SOFT_ACCEPTED
            if transport_result.proposal.soft_claim is not None
            else DIRECT_ONLY
        )
        return SemanticCalibrationProposalRecord(
            candidate,
            status,
            proposal_transport=transport_result,
        )

    proposer_inputs = tuple(zip(candidates, episodes, strict=True))
    if execution_config.proposer_max_workers == 1:
        proposal_records = tuple(map(run_proposer_phase_item, proposer_inputs))
    else:
        with ThreadPoolExecutor(
            max_workers=execution_config.proposer_max_workers,
            thread_name_prefix="semantic-calibration-proposer",
        ) as executor:
            proposal_records = tuple(
                executor.map(run_proposer_phase_item, proposer_inputs)
            )

    split_source = corpus.split.source_digest
    assert split_source is not None
    proposal_archive = SemanticCalibrationProposalArchive(
        protocol=protocol,
        execution_config=execution_config,
        selection_seed=seed,
        selection_seed_digest=hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        candidate_count=candidate_count,
        families=scope,
        semantic_cohort=semantic_cohort,
        source_corpus_manifest_digest=source_manifest,
        development_manifest_digest=manifest.digest,
        split_source_digest=split_source,
        split_manifest_digest=canonical_digest(corpus.split.to_manifest_dict()),
        historical_seed_digest=historical_seed_digest,
        resolver_policy_digest=resolver_policy_digest,
        cohort_report_digest=cohort_report_digest,
        clean_cohort_whitelist_digest=clean_cohort_whitelist_digest,
        clean_cohort_whitelist=clean_cohort_whitelist,
        blocked_policy_digest=blocked_policy_digest,
        blocked_exclusion_digest=blocked_exclusion_digest,
        blocked_excluded_task_ids=blocked_excluded_task_ids,
        blocked_morphology_clusters=blocked_morphology_clusters,
        exposure_predecessor=exposure_ledger,
        exposure_successor=exposure_successor,
        records=proposal_records,
    )
    if any(item.status == TRANSPORT_FAILED for item in proposal_archive.records):
        raise SemanticCalibrationCampaignProposalPhaseFailed(proposal_archive)
    soft_records = proposal_archive.soft_records
    if not soft_records:
        raise SemanticCalibrationCampaignNoSoftClaims(proposal_archive)

    # Phase 2: the one exact label-free plan and *all* score commitments are
    # created before the first blind scorer invocation.
    calibration_plan = SemanticCalibrationPlan.create(
        protocol,
        corpus.split,
        tuple(item.candidate.selection for item in soft_records),
        corpus_manifest_digest=source_manifest,
        development_manifest_digest=manifest.digest,
        label_reveal_protocol_digest=reveal_protocol,
    )
    episode_by_task = {episode.task_id: episode for episode in episodes}
    commitments: list[SemanticCalibrationScoreCommitment] = []
    panel_by_observation: dict[str, PanelInput] = {}
    for record in soft_records:
        candidate = record.candidate
        episode = episode_by_task[candidate.selection.task_id]
        source = episode._query_sources[0]
        if source.panel.sha256 != candidate.selection.panel_digest:
            raise SemanticCalibrationCampaignError(
                "private query-0 source differs from prospective selection"
            )
        assert record.proposal_transport is not None
        panel_by_observation[candidate.selection.observation_id] = source.path
        commitments.append(
            SemanticCalibrationScoreCommitment.from_panel(
                plan=calibration_plan,
                selection=candidate.selection,
                support=candidate.support,
                proposal_transport=record.proposal_transport,
                protocol=protocol,
                panel=source.path,
            )
        )
    commitment_batch = SemanticCalibrationCommitmentBatch(
        proposal_archive,
        calibration_plan,
        tuple(commitments),
    )

    # Phase 3: execute exactly one descendant scorer call per commitment.  The
    # lower transport records failures explicitly, allowing the loop to finish
    # and the complete failure batch to be returned on the exception.
    def run_scorer_phase_item(
        commitment: SemanticCalibrationScoreCommitment,
    ) -> SemanticCalibrationScoreAttempt:
        return score_semantic_calibration_panel(
            panel_by_observation[commitment.selection.observation_id],
            commitment,
            minutes=scorer_minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=frozen_cache_snapshot,
            transport=scorer_transport,
        )

    if execution_config.scorer_max_workers == 1:
        attempts = tuple(map(run_scorer_phase_item, commitment_batch.commitments))
    else:
        with ThreadPoolExecutor(
            max_workers=execution_config.scorer_max_workers,
            thread_name_prefix="semantic-calibration-scorer",
        ) as executor:
            attempts = tuple(
                executor.map(
                    run_scorer_phase_item,
                    commitment_batch.commitments,
                )
            )
    score_batch = SemanticCalibrationScoreBatch(commitment_batch, attempts)
    if not score_batch.all_present:
        raise SemanticCalibrationCampaignScoringFailed(score_batch)

    # Phase 4: only now open the episode label seals, bind each opening to the
    # complete score-batch digest, join query-0 labels, and fit the family.
    reveal_list: list[SemanticCalibrationLabelReveal] = []
    measurements: list[SemanticCalibrationMeasurement] = []
    try:
        for record in soft_records:
            reveal_list.append(
                SemanticCalibrationLabelReveal._from_episode_with_verified_parents(
                    record.candidate,
                    episode_by_task[record.candidate.selection.task_id],
                    score_batch,
                )
            )
        reveals = tuple(reveal_list)
        for record, attempt, reveal in zip(
            soft_records, attempts, reveals, strict=True
        ):
            measurements.append(
                join_calibration_label(
                    calibration_plan,
                    protocol,
                    record.candidate.selection.observation_id,
                    attempt.score_artifact,
                    reveal.affirmative_label,
                    label_reveal_receipt_digest=reveal.digest,
                )
            )
        calibration = fit_semantic_calibration(
            calibration_plan, protocol, tuple(measurements)
        )
    except Exception as exc:  # noqa: BLE001 - retain the complete causal record.
        raise SemanticCalibrationCampaignFitFailed(
            score_batch,
            tuple(reveal_list),
            tuple(measurements),
            exc,
        ) from exc
    return SemanticCalibrationCampaignArtifact(
        score_batch,
        reveals,
        calibration,
    )


__all__ = [
    "CAMPAIGN_CANDIDATE_SCHEMA",
    "CAMPAIGN_CANDIDATE_SCHEMA_V1",
    "CAMPAIGN_COMMITMENT_BATCH_SCHEMA",
    "CAMPAIGN_LABEL_REVEAL_SCHEMA",
    "CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA",
    "CAMPAIGN_PROPOSAL_ARCHIVE_SCHEMA_V1",
    "CAMPAIGN_PROPOSAL_RECORD_SCHEMA",
    "CAMPAIGN_SCORE_BATCH_SCHEMA",
    "CAMPAIGN_SELECTION_ALGORITHM",
    "CAMPAIGN_SELECTION_ALGORITHM_V1",
    "DIRECT_ONLY",
    "SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA",
    "SOFT_ACCEPTED",
    "TRANSPORT_FAILED",
    "TYPED_REJECTED",
    "SemanticCalibrationCampaignArtifact",
    "SemanticCalibrationCampaignError",
    "SemanticCalibrationCampaignFitFailed",
    "SemanticCalibrationCampaignNoSoftClaims",
    "SemanticCalibrationCampaignProposalPhaseFailed",
    "SemanticCalibrationCampaignScoringFailed",
    "SemanticCalibrationCandidate",
    "SemanticCalibrationCommitmentBatch",
    "SemanticCalibrationExecutionConfig",
    "SemanticCalibrationLabelReveal",
    "SemanticCalibrationProposalArchive",
    "SemanticCalibrationProposalRecord",
    "SemanticCalibrationScoreBatch",
    "run_semantic_calibration_campaign",
    "resolve_semantic_campaign_panels",
    "verify_semantic_campaign_against_corpus",
    "select_semantic_calibration_tasks",
    "semantic_generator_cluster_id",
    "semantic_campaign_label_reveal_protocol_digest",
]
