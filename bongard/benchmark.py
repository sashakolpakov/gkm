"""Canonical sealed evaluation for one official 7+7 Bongard task.

The runner enforces the information flow rather than merely documenting it::

    6+6 labelled support -> one proposal -> frozen formula
        -> two isolated, unlabeled queries -> joint prediction commitment
        -> label reveal -> model-free replay

Source paths, task identifiers, split names, regimes, and query labels never
cross either callback boundary.  Callbacks receive short-lived neutral paths;
the query paths do not exist until after :class:`ProposalFreeze` has been
created.  An indeterminate or error observation remains an abstention and is
scored as wrong, including when the hidden label is negative.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bongard.admission import TypedAttachmentContract
from bongard.artifacts import (
    ArtifactTamperError,
    AtomPath,
    BlobRef,
    ColdReplayInputs,
    LabelReveal,
    PredictionCommitment,
    ProposalFreeze,
    QueryPanel,
    QueryRelease,
    RevealedLabel,
    RunArtifactBundle,
    SupportCommitment,
    SupportExample,
    TruthEvidenceRecord,
    atom_paths,
    canonical_digest,
)
from bongard.corpus import (
    BongardTask,
    CorpusManifest,
    ShapeBongardCorpus,
    SplitIndex,
    TaskManifest,
)
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import AllOf, AnyOf, Atom, Formula
from bongard.legs import PANEL, LegRegistry, TypedValue
from bongard.predicate_backend import (
    PYTHON_PREDICATE_BACKEND,
    PredicateBackend,
)


PROTOCOL_VERSION = "official-two-query-benchmark/v3"
DEFAULT_VERIFIER = "canonical-bongard-verifier"
SUPPORT_PROTOTYPE_PREDICATE_MODE = "support_prototype"
_SHA256 = re.compile(r"(?:sha256:)?([0-9a-f]{64})\Z")
_NEUTRAL_QUERY_ID = "query"
_NEUTRAL_QUERY_BLOB_ID = "query-panel"


class BenchmarkProtocolError(RuntimeError):
    """The episode could not satisfy the sealed evaluation protocol."""


class SealedMutationError(BenchmarkProtocolError):
    """Committed corpus or split bytes changed during a sealed run."""


class EpisodeStatus(str, Enum):
    COMPLETE = "complete"
    PROPOSAL_ERROR = "proposal_error"
    SUPPORT_REJECTED = "support_rejected"
    OBSERVATION_ERROR = "observation_error"


class SupportGateMode(str, Enum):
    """Verifier-owned choice made before the proposer receives support."""

    EMPIRICAL_REPLAY = "empirical_replay"
    SUPPORT_PROTOTYPE_REPLAY = "support_prototype_replay"
    TEST_BYPASS = "verifier_test_bypass"


class SupportGateResult(str, Enum):
    ALIGNED = "aligned"
    MISORIENTED = "misoriented"
    UNSUPPORTED = "unsupported"
    OBSERVER_FAILURE = "observer_failure"
    TEST_BYPASSED = "test_bypassed"


SUPPORT_GATE_POLICY_VERSION = "headless-hybrid-support-replay/v2"
SUPPORT_PROTOTYPE_GATE_POLICY_VERSION = "support-prototype-replay/v1"
_PENDING_SUPPORT_GATE_DIGEST = canonical_digest(
    {"schema": SUPPORT_GATE_POLICY_VERSION, "state": "pending"}
)


@dataclass(frozen=True, slots=True)
class SupportGatePolicy:
    """Policy selected by the verifier, never by the proposer."""

    mode: SupportGateMode
    reason: str | None = None
    version: str = SUPPORT_GATE_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.mode is SupportGateMode.EMPIRICAL_REPLAY:
            if self.version != SUPPORT_GATE_POLICY_VERSION:
                raise ValueError("unsupported empirical support replay policy")
            if self.reason is not None:
                raise ValueError("empirical support replay has no bypass reason")
        elif self.mode is SupportGateMode.SUPPORT_PROTOTYPE_REPLAY:
            if self.version != SUPPORT_PROTOTYPE_GATE_POLICY_VERSION:
                raise ValueError("unsupported support-prototype replay policy")
            if self.reason is not None:
                raise ValueError("support-prototype replay has no bypass reason")
        elif self.mode is SupportGateMode.TEST_BYPASS:
            if self.version != SUPPORT_GATE_POLICY_VERSION:
                raise ValueError("unsupported test support-gate bypass policy")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError(
                    "test support-gate bypass requires an explicit reason"
                )
        else:  # pragma: no cover - the enum is closed, but callers are runtime Python.
            raise ValueError("unsupported support replay gate mode")

    @classmethod
    def empirical(cls) -> "SupportGatePolicy":
        return cls(SupportGateMode.EMPIRICAL_REPLAY)

    @classmethod
    def prototype(cls) -> "SupportGatePolicy":
        """Require deterministic support-prototype extraction and evaluation."""

        return cls(
            SupportGateMode.SUPPORT_PROTOTYPE_REPLAY,
            version=SUPPORT_PROTOTYPE_GATE_POLICY_VERSION,
        )

    @classmethod
    def verifier_test_bypass(cls, reason: str) -> "SupportGatePolicy":
        return cls(SupportGateMode.TEST_BYPASS, reason)

    def to_data(self) -> dict[str, object]:
        if self.mode is SupportGateMode.SUPPORT_PROTOTYPE_REPLAY:
            return {
                "version": self.version,
                "mode": self.mode.value,
                "reason": self.reason,
                "call_count": 12,
                "positive_count": 6,
                "negative_count": 6,
                "extractor_input_contract": (
                    "panel_bytes_only_no_task_candidate_side_or_role_context_v1"
                ),
                "positive_outcome": Disposition.PRESENT.value,
                "negative_outcome": Disposition.CERTIFIED_ABSENT.value,
                "certified_absence_semantics": (
                    "operational_contrastive_nonmatch_for_frozen_support_prototype"
                ),
                "dispositions": [item.value for item in Disposition],
                "fresh_candidate_independent_extraction_per_panel": True,
                "fresh_frozen_predicate_evaluation_per_panel": True,
                "polarity_flip_allowed": False,
            }
        return {
            "version": self.version,
            "mode": self.mode.value,
            "reason": self.reason,
            "call_count": 12 if self.mode is SupportGateMode.EMPIRICAL_REPLAY else 0,
            "positive_count": 6,
            "negative_count": 6,
            "image_name": "query.png",
            "positive_outcome": "present",
            "negative_outcome": "nonmatch",
            "nonmatch_certificate_semantics": (
                "archived_model_nonmatch_for_frozen_operational_claim"
            ),
            "nonmatch_reason_semantics": (
                "optional_overall_model_summary_bound_inside_certificate"
            ),
            "nonmatch_cue_keyed_findings_required": True,
            "nonmatch_visibility_statement_required": True,
            "fresh_isolated_transport_per_panel": True,
            "polarity_flip_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportGateMeasurement:
    """One callback result; the adapter never receives its support label."""

    evidence: Evidence[bool]
    observer_artifact: Mapping[str, Any]
    transport_attempted: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Evidence):
            raise ValueError("support gate measurement requires Evidence")
        if not isinstance(self.transport_attempted, bool):
            raise ValueError("support gate transport_attempted must be Boolean")
        if (
            self.evidence.disposition is Disposition.PRESENT
            and self.evidence.value is not True
        ):
            raise ValueError("present support gate evidence must contain True")
        # Copy through canonical JSON so a callback cannot mutate an artifact
        # after the verifier has classified it.
        try:
            frozen = copy.deepcopy(dict(self.observer_artifact))
            canonical_digest(frozen)
        except (TypeError, ValueError) as exc:
            raise ValueError("support observer artifact is not canonical JSON") from exc
        object.__setattr__(self, "observer_artifact", frozen)


@dataclass(frozen=True, slots=True)
class SupportGateEntry:
    slot_id: str
    panel: BlobRef
    positive: bool
    evidence: TruthEvidenceRecord
    observer_artifact: Mapping[str, Any]
    transport_attempted: bool

    def to_data(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "panel": self.panel.to_data(),
            "positive": self.positive,
            "evidence": self.evidence.to_data(),
            "observer_artifact": dict(self.observer_artifact),
            "transport_attempted": self.transport_attempted,
        }


@dataclass(frozen=True, slots=True)
class SupportGateArtifact:
    """Verifier-side join of hidden support labels with isolated judgments."""

    run_id: str
    proposal_digest: str
    support_commitment_digest: str
    policy: SupportGatePolicy
    entries: tuple[SupportGateEntry, ...]
    forward_matches: int
    reverse_matches: int
    present_count: int
    nonmatch_count: int
    indeterminate_count: int
    error_count: int
    transport_attempt_count: int
    result: SupportGateResult
    version: str = "support-replay-gate-artifact/v1"

    def __post_init__(self) -> None:
        _hex_digest(self.proposal_digest, "support gate proposal digest")
        _hex_digest(self.support_commitment_digest, "support commitment digest")
        counts = (
            self.forward_matches,
            self.reverse_matches,
            self.present_count,
            self.nonmatch_count,
            self.indeterminate_count,
            self.error_count,
            self.transport_attempt_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("support gate counts must be non-negative integers")
        if self.policy.mode in {
            SupportGateMode.EMPIRICAL_REPLAY,
            SupportGateMode.SUPPORT_PROTOTYPE_REPLAY,
        }:
            if len(self.entries) != 12:
                raise ValueError("replay support gate requires twelve entries")
            if self.present_count + self.nonmatch_count + self.indeterminate_count + self.error_count != 12:
                raise ValueError("support gate dispositions do not cover twelve panels")
            if self.result is SupportGateResult.TEST_BYPASSED:
                raise ValueError("replay support gate cannot be marked bypassed")
            if (
                self.result is SupportGateResult.ALIGNED
                and self.transport_attempt_count != 12
            ):
                raise ValueError("aligned support gate requires twelve transport attempts")
        else:
            if self.entries or any(counts):
                raise ValueError("test bypass cannot contain empirical observations")
            if self.result is not SupportGateResult.TEST_BYPASSED:
                raise ValueError("test bypass must have test_bypassed result")

    @property
    def accepted(self) -> bool:
        return self.result in {
            SupportGateResult.ALIGNED,
            SupportGateResult.TEST_BYPASSED,
        }

    def content_data(self) -> dict[str, object]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "proposal_digest": self.proposal_digest,
            "support_commitment_digest": self.support_commitment_digest,
            "policy": self.policy.to_data(),
            "ordered_entries": [entry.to_data() for entry in self.entries],
            "counts": {
                "forward_matches": self.forward_matches,
                "reverse_matches": self.reverse_matches,
                "present": self.present_count,
                "nonmatch": self.nonmatch_count,
                "indeterminate": self.indeterminate_count,
                "error": self.error_count,
                "transport_attempts": self.transport_attempt_count,
            },
            "result": self.result.value,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "gate_digest": self.digest}


def _hex_digest(value: str, label: str) -> str:
    match = _SHA256.fullmatch(value)
    if match is None:
        raise BenchmarkProtocolError(f"{label} is not a SHA-256 digest")
    return match.group(1)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _derive_hex(domain: str, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update((PROTOCOL_VERSION + "\0" + domain).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def _draw(task_id: str, seed: str, purpose: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("draw modulus must be positive")
    return int(_derive_hex("selection:" + purpose, task_id, seed), 16) % modulus


def _manifest_digest(manifest: CorpusManifest) -> str:
    expected = canonical_digest(manifest.content_dict())
    actual = _hex_digest(manifest.digest, "corpus manifest digest")
    if actual != expected:
        raise BenchmarkProtocolError("corpus manifest content does not match its digest")
    return actual


def _task_manifest(
    manifest: CorpusManifest, task_id: str
) -> TaskManifest:
    matches = tuple(item for item in manifest.tasks if item.task_id == task_id)
    if len(matches) != 1:
        raise BenchmarkProtocolError(
            f"corpus manifest contains {len(matches)} entries for task {task_id!r}"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class SupportInput:
    """The entire and only value sent to a proposer.

    Every path has a neutral basename and lives in an otherwise empty private
    directory.  There is intentionally no run, corpus, task, family, split,
    regime, source-index, concept, query, or source-path field.
    """

    positive_paths: tuple[Path, Path, Path, Path, Path, Path]
    negative_paths: tuple[Path, Path, Path, Path, Path, Path]

    def __post_init__(self) -> None:
        if len(self.positive_paths) != 6 or len(self.negative_paths) != 6:
            raise ValueError("canonical proposer input is exactly 6+6 support panels")
        paths = (*self.positive_paths, *self.negative_paths)
        if len(set(paths)) != 12:
            raise ValueError("support paths must be distinct")


@dataclass(frozen=True, slots=True)
class ProposedRule:
    """Typed output expected from the injected support-only proposer adapter."""

    proposal_id: str
    proposer_digest: str
    formula: Formula
    registry: LegRegistry = field(repr=False, compare=False)
    attachment_contract: TypedAttachmentContract

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.proposal_id) is None:
            raise ValueError(f"invalid proposal id {self.proposal_id!r}")
        _hex_digest(self.proposer_digest, "proposer digest")
        if not self.registry.frozen:
            raise ValueError("proposed rule registry must already be frozen")
        self.attachment_contract.validate(self.formula, self.registry)


@dataclass(frozen=True, slots=True)
class ObservationInput:
    """One isolated query and the already-frozen rule sent to an observer.

    ``query_id`` and the blob identifier are callback-local neutral sentinels,
    not the identifiers used by the artifact chain.  Consequently the
    empirical observer cannot learn which of the two query slots it received
    from this value.
    """

    query_id: str
    panel_path: Path
    panel: BlobRef
    freeze: ProposalFreeze
    registry: LegRegistry = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.query_id != _NEUTRAL_QUERY_ID:
            raise ValueError("observer input query id must be the neutral sentinel")
        if self.panel.blob_id != _NEUTRAL_QUERY_BLOB_ID:
            raise ValueError("observer input panel id must be the neutral sentinel")
        if not self.registry.frozen:
            raise ValueError("observer input registry must be frozen")


@runtime_checkable
class Proposer(Protocol):
    """One call per task, with support only."""

    def propose(self, support: SupportInput) -> ProposedRule:
        ...


@runtime_checkable
class Observer(Protocol):
    """One independent call per query, returning evidence for every IR atom."""

    def observe(
        self, query: ObservationInput
    ) -> Mapping[AtomPath, Evidence[bool]]:
        ...


@runtime_checkable
class ObserverFactory(Protocol):
    """Create a fresh observer object for one and only one query.

    Implementations should not share mutable model/session state between the
    objects they return.  The runner checks object identity, but Python object
    identity cannot detect covert shared globals or external services.
    """

    def create_observer(self) -> Observer:
        ...


@runtime_checkable
class SupportReplayObserver(Protocol):
    """One fresh support-only observer transport with no label field."""

    def observe_support(self, panel: ObservationInput) -> SupportGateMeasurement:
        ...


@runtime_checkable
class SupportReplayFactory(Protocol):
    """Issue a never-reused isolated support observer for each panel."""

    def create_support_observer(self) -> SupportReplayObserver:
        ...


def _atoms_by_path(
    formula: Formula, prefix: AtomPath = ()
) -> dict[AtomPath, Atom]:
    if isinstance(formula, Atom):
        return {prefix: formula}
    assert isinstance(formula, (AllOf, AnyOf))
    result: dict[AtomPath, Atom] = {}
    for index, term in enumerate(formula.terms):
        result.update(_atoms_by_path(term, (*prefix, index)))
    return result


class ClosedIRObserver:
    """Default adapter for registered one-panel predicates.

    The reference backend is pure Python. Injection exists so an independent
    checker can be tested without changing the frozen IR or evidence formats.
    """

    def __init__(
        self,
        *,
        backend: PredicateBackend = PYTHON_PREDICATE_BACKEND,
    ) -> None:
        self._backend = backend

    def observe(self, query: ObservationInput) -> Mapping[AtomPath, Evidence[bool]]:
        bindings = {"panel": TypedValue(PANEL, query.panel_path)}
        return {
            path: self._backend.evaluate(atom, query.registry, bindings)
            for path, atom in _atoms_by_path(query.freeze.formula).items()
        }


class _ObserverSessions:
    """Prepare two distinct callback objects before either query is released.

    An explicit factory is preferred.  Legacy observer objects are snapshotted
    once after proposal freeze and deep-copied twice.  This prevents ordinary
    per-instance state from flowing from query one to query two.  It is not a
    cryptographic sandbox: malicious in-process code can communicate through
    globals, custom copy hooks, threads, files, or other external state.
    """

    def __init__(self, observer: Observer | ObserverFactory) -> None:
        self._factory: ObserverFactory | None = None
        self._legacy: Observer | None = None
        self._prototype: Observer | None = None
        self._issued: list[tuple[str, Observer]] = []
        if isinstance(observer, ObserverFactory):
            self._factory = observer
            return
        if not isinstance(observer, Observer):
            raise BenchmarkProtocolError(
                "observer must implement observe() or create_observer()"
            )
        try:
            prototype = copy.deepcopy(observer)
        except Exception as exc:  # noqa: BLE001 - adapter isolation boundary.
            raise BenchmarkProtocolError(
                "legacy observer cannot be isolated; provide an ObserverFactory"
            ) from exc
        if prototype is observer or not isinstance(prototype, Observer):
            raise BenchmarkProtocolError(
                "legacy observer did not produce an independent snapshot; "
                "provide an ObserverFactory"
            )
        self._legacy = observer
        self._prototype = prototype

    def prepare(self, count: int) -> tuple[Observer, ...]:
        sessions: list[Observer] = []
        for _index in range(count):
            try:
                session = (
                    self._factory.create_observer()
                    if self._factory is not None
                    else copy.deepcopy(self._prototype)
                )
            except Exception as exc:  # noqa: BLE001 - factory boundary.
                raise BenchmarkProtocolError(
                    "observer factory failed before query release"
                ) from exc
            if not isinstance(session, Observer):
                raise BenchmarkProtocolError(
                    "observer factory returned an object without observe()"
                )
            if session is self._factory or session is self._legacy:
                raise BenchmarkProtocolError(
                    "observer factory reused its own stateful callback object"
                )
            if any(session is existing for existing in sessions):
                raise BenchmarkProtocolError(
                    "observer factory reused one callback object for both queries"
                )
            sessions.append(session)
        return tuple(sessions)

    def record(self, public_query_id: str, session: Observer) -> None:
        self._issued.append((public_query_id, session))

    def finalize(self, expected_count: int = 2) -> None:
        """Collect receipts only after both isolated calls have completed.

        Factories may expose a post-hoc collector.  The legacy private-map
        adapter preserves ``HeadlessCodexEpisode.artifact_data()`` without
        giving the second session access to the first session's observation.
        """

        if len(self._issued) != expected_count:
            raise BenchmarkProtocolError(
                f"exactly {expected_count} observer sessions must complete"
            )
        if self._factory is not None:
            collector = getattr(self._factory, "collect_observers", None)
            if collector is not None:
                if not callable(collector):
                    raise BenchmarkProtocolError(
                        "observer factory collect_observers attribute is not callable"
                    )
                collector(tuple(self._issued))
            return
        if self._legacy is None or self._prototype is None:
            raise AssertionError("legacy observer adapter is incomplete")
        original = getattr(self._legacy, "_observations", None)
        prototype = getattr(self._prototype, "_observations", None)
        if original is None and prototype is None:
            return
        if not isinstance(original, dict) or not isinstance(prototype, dict):
            raise BenchmarkProtocolError(
                "legacy observer receipt state is not an observation dictionary"
            )
        if original != prototype:
            raise BenchmarkProtocolError(
                "legacy observer state changed while isolated sessions ran"
            )
        additions: dict[str, object] = {}
        for public_query_id, session in self._issued:
            isolated = getattr(session, "_observations", None)
            if not isinstance(isolated, dict):
                raise BenchmarkProtocolError(
                    "isolated observer lost its observation receipt dictionary"
                )
            changed = {
                key: value
                for key, value in isolated.items()
                if key not in prototype or prototype[key] != value
            }
            if not changed:
                continue
            if set(changed) != {_NEUTRAL_QUERY_ID}:
                raise BenchmarkProtocolError(
                    "isolated observer recorded unexpected query identifiers"
                )
            additions[public_query_id] = changed[_NEUTRAL_QUERY_ID]
        if set(additions) & set(original):
            raise BenchmarkProtocolError("legacy observer receipt ids already exist")
        original.update(additions)


@dataclass(frozen=True, slots=True)
class _PanelSource:
    panel: BlobRef
    path: Path = field(repr=False, compare=False)
    positive: bool = field(repr=False)
    source_index: int = field(repr=False)

    def read_verified(self) -> bytes:
        try:
            payload = self.path.read_bytes()
            self.panel.verify_bytes(payload)
        except (OSError, ArtifactTamperError) as exc:
            raise SealedMutationError(
                f"committed panel bytes changed for {self.panel.blob_id}"
            ) from exc
        return payload


@dataclass(frozen=True, slots=True)
class EpisodePlan:
    """Sealed plan; callback-specific private views are constructed by the runner.

    ``to_data`` is deliberately safe to publish after planning.  It contains
    no source paths, plaintext seed, or secret label salt and exposes the
    already sealed support/query/label commitments only by digest.  The exact
    same canonical object defines :attr:`digest`, so an outer run record can
    carry the plan without introducing a second, weaker description of it.
    """

    task_id: str
    family: str
    split: str | None
    regime: str | None
    run_id: str
    verifier_id: str
    seed_digest: str
    corpus_digest: str
    task_manifest_digest: str
    support: SupportCommitment
    queries: tuple[QueryPanel, QueryPanel]
    latent_query_digest: str
    label_commitment_digest: str
    _support_sources: tuple[_PanelSource, ...] = field(repr=False, compare=False)
    _query_sources: tuple[_PanelSource, _PanelSource] = field(
        repr=False, compare=False
    )
    _label_nonce: str = field(repr=False, compare=False)
    predicate_mode: str | None = None
    predicate_policy_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.family or not self.verifier_id:
            raise ValueError("episode plan identity fields must be non-empty")
        if not self.task_id.startswith(f"{self.family}_"):
            raise ValueError("episode task id and family differ")
        for label, value in (
            ("seed digest", self.seed_digest),
            ("corpus digest", self.corpus_digest),
            ("task manifest digest", self.task_manifest_digest),
            ("latent query digest", self.latent_query_digest),
            ("label commitment digest", self.label_commitment_digest),
        ):
            _hex_digest(value, label)
        _hex_digest(self._label_nonce, "private label seal nonce")
        if self.support.run_id != self.run_id:
            raise ValueError("support commitment belongs to another run")
        if self.support.issued_by != self.verifier_id:
            raise ValueError("support issuer differs from plan verifier")
        if self.support.corpus_digest != self.corpus_digest:
            raise ValueError("support corpus differs from plan corpus")
        if len(self._support_sources) != 12:
            raise ValueError("episode plan requires twelve support sources")
        if len(self._query_sources) != 2 or len(self.queries) != 2:
            raise ValueError("episode plan requires two query sources")
        if sum(source.positive for source in self._support_sources) != 6:
            raise ValueError("episode support is not 6 positive and 6 negative")
        if {source.positive for source in self._query_sources} != {False, True}:
            raise ValueError("episode queries are not one positive and one negative")
        if tuple(item.panel for item in self.queries) != tuple(
            source.panel for source in self._query_sources
        ):
            raise ValueError("query public commitments differ from private sources")
        support_hashes = {item.panel.sha256 for item in self.support.support}
        query_hashes = {item.panel.sha256 for item in self.queries}
        if len(query_hashes) != 2:
            raise BenchmarkProtocolError("the two selected queries have identical bytes")
        if support_hashes & query_hashes:
            raise BenchmarkProtocolError("selected query bytes overlap support bytes")
        expected_latent = canonical_digest(
            {
                "version": "latent-two-query-commitment/v1",
                "run_id": self.run_id,
                "queries": [item.to_data() for item in self.queries],
            }
        )
        if self.latent_query_digest != expected_latent:
            raise ValueError("latent query commitment digest differs")
        if (self.predicate_mode is None) != (
            self.predicate_policy_digest is None
        ):
            raise ValueError(
                "predicate mode and policy digest must be committed together"
            )
        if self.predicate_mode is not None:
            if self.predicate_mode != SUPPORT_PROTOTYPE_PREDICATE_MODE:
                raise ValueError(
                    f"unsupported predicate mode {self.predicate_mode!r}"
                )
            if not isinstance(self.predicate_policy_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", self.predicate_policy_digest
            ) is None:
                raise ValueError(
                    "predicate policy digest must be an unprefixed lowercase SHA-256"
                )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        """Return the complete, path-free public plan commitment."""

        data: dict[str, object] = {
            "version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "regime": self.regime,
            "run_id": self.run_id,
            "verifier_id": self.verifier_id,
            "seed_digest": self.seed_digest,
            "corpus_digest": self.corpus_digest,
            "task_manifest_digest": self.task_manifest_digest,
            "support_commitment_digest": self.support.digest(),
            "latent_query_digest": self.latent_query_digest,
            "label_commitment_digest": self.label_commitment_digest,
        }
        if self.predicate_mode is not None:
            assert self.predicate_policy_digest is not None
            data["predicate_mode"] = self.predicate_mode
            data["predicate_policy_digest"] = self.predicate_policy_digest
        return data

    def verify_sources(self) -> None:
        for source in (*self._support_sources, *self._query_sources):
            source.read_verified()

    def _revealed_labels(self) -> tuple[RevealedLabel, RevealedLabel]:
        labels = tuple(
            RevealedLabel(query.query_id, source.positive)
            for query, source in zip(self.queries, self._query_sources, strict=True)
        )
        actual_commitment = canonical_digest(
            {
                "run_id": self.run_id,
                "labels": [label.to_data() for label in labels],
                "nonce": self._label_nonce,
                "version": "latent-label-seal/v1",
            }
        )
        if actual_commitment != self.label_commitment_digest:
            raise ArtifactTamperError("revealed labels differ from latent label seal")
        return labels  # type: ignore[return-value]


def prepare_episode(
    corpus: ShapeBongardCorpus,
    task_id: str,
    *,
    seed: str,
    corpus_manifest: CorpusManifest | None = None,
    verifier_id: str = DEFAULT_VERIFIER,
    label_seal_nonce: str | None = None,
    predicate_mode: str | None = None,
    predicate_policy_digest: str | None = None,
) -> EpisodePlan:
    """Commit a deterministic 6+6 support / one-per-side query split.

    Pass one pre-built ``corpus_manifest`` to every episode in a large run so
    the complete corpus is hashed once rather than once per task.  Selection
    is seed-deterministic; the label commitment is randomized unless a
    verifier supplies a precommitted 256-bit ``label_seal_nonce``.
    """

    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("episode seed must be a non-empty string")
    if not verifier_id.strip():
        raise ValueError("verifier id must be non-empty")
    if label_seal_nonce is None:
        label_seal_nonce = secrets.token_hex(32)
    _hex_digest(label_seal_nonce, "label seal nonce")
    task = corpus.task(task_id)
    manifest = corpus_manifest or corpus.build_manifest()
    corpus_digest = _manifest_digest(manifest)
    committed_task = _task_manifest(manifest, task_id)
    fresh_task = task.build_manifest()
    if fresh_task.digest != committed_task.digest:
        raise SealedMutationError("task bytes differ from the supplied corpus manifest")

    assignment = corpus.assignment(task_id)
    panels_by_side: dict[str, list[object]] = {"positive": [], "negative": []}
    for panel in committed_task.panels:
        panels_by_side[panel.polarity].append(panel)
    positive = tuple(sorted(panels_by_side["positive"], key=lambda item: item.index))
    negative = tuple(sorted(panels_by_side["negative"], key=lambda item: item.index))
    if len(positive) != 7 or len(negative) != 7:
        raise BenchmarkProtocolError("official episodes require exactly seven panels per side")

    positive_query = _draw(task_id, seed, "positive-query", 7)
    negative_query = _draw(task_id, seed, "negative-query", 7)
    positive_support = tuple(
        panel for panel in positive if panel.index != positive_query
    )
    negative_support = tuple(
        panel for panel in negative if panel.index != negative_query
    )

    seed_digest = _derive_hex("seed", seed)
    run_hash = _derive_hex("run", corpus_digest, task_id, seed)
    run_id = "run-" + run_hash[:32]

    def source_for(panel: object, blob_id: str, side: bool) -> _PanelSource:
        return _PanelSource(
            BlobRef(
                blob_id=blob_id,
                sha256=_hex_digest(panel.sha256, "panel digest"),
                byte_count=panel.size_bytes,
                media_type="image/png",
            ),
            panel.path,
            side,
            panel.index,
        )

    support_sources = tuple(
        [
            source_for(panel, f"support-negative-{slot}", False)
            for slot, panel in enumerate(negative_support)
        ]
        + [
            source_for(panel, f"support-positive-{slot}", True)
            for slot, panel in enumerate(positive_support)
        ]
    )
    support_examples = tuple(
        SupportExample(source.panel, source.positive) for source in support_sources
    )
    support = SupportCommitment(
        run_id=run_id,
        issued_by=verifier_id,
        corpus_digest=corpus_digest,
        support=support_examples,
        verifier_nonce=_derive_hex("support-nonce", run_hash),
    )

    selected = ((positive[positive_query], True), (negative[negative_query], False))
    if _draw(task_id, seed, "query-order", 2):
        selected = tuple(reversed(selected))
    query_sources = tuple(
        source_for(panel, f"query-panel-{slot}", side)
        for slot, (panel, side) in enumerate(selected)
    )
    queries = tuple(
        QueryPanel(f"query-{slot}", source.panel)
        for slot, source in enumerate(query_sources)
    )
    latent_query_digest = canonical_digest(
        {
            "version": "latent-two-query-commitment/v1",
            "run_id": run_id,
            "queries": [item.to_data() for item in queries],
        }
    )
    label_commitment_digest = canonical_digest(
        {
            "run_id": run_id,
            "labels": [
                {"query_id": f"query-{slot}", "positive": source.positive}
                for slot, source in enumerate(query_sources)
            ],
            "nonce": label_seal_nonce,
            "version": "latent-label-seal/v1",
        }
    )
    return EpisodePlan(
        task_id=task.task_id,
        family=task.family,
        split=assignment.split,
        regime=assignment.regime,
        run_id=run_id,
        verifier_id=verifier_id,
        seed_digest=seed_digest,
        corpus_digest=corpus_digest,
        task_manifest_digest=_hex_digest(committed_task.digest, "task manifest digest"),
        support=support,
        queries=queries,  # type: ignore[arg-type]
        latent_query_digest=latent_query_digest,
        label_commitment_digest=label_commitment_digest,
        _support_sources=support_sources,
        _query_sources=query_sources,  # type: ignore[arg-type]
        _label_nonce=label_seal_nonce,
        predicate_mode=predicate_mode,
        predicate_policy_digest=predicate_policy_digest,
    )


@dataclass(frozen=True, slots=True)
class SealedTestGuard:
    """Reusable content guard for all tasks in one official test snapshot."""

    corpus_digest: str
    split_digest: str
    task_digests: tuple[tuple[str, str], ...]
    _split_path: Path = field(repr=False, compare=False)
    _tasks: tuple[BongardTask, ...] = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        corpus: ShapeBongardCorpus,
        *,
        corpus_manifest: CorpusManifest | None = None,
        require_complete: bool = False,
    ) -> "SealedTestGuard":
        if require_complete:
            corpus.validate_complete(require_split=True)
        if corpus.split.source_path is None or corpus.split.source_digest is None:
            raise BenchmarkProtocolError("sealed test guard requires a split source file")
        tasks = corpus.tasks_in_split("test")
        if not tasks:
            raise BenchmarkProtocolError("sealed test guard found no test tasks")
        manifest = corpus_manifest or corpus.build_manifest()
        corpus_digest = _manifest_digest(manifest)
        by_id = {item.task_id: item for item in manifest.tasks}
        task_digests: list[tuple[str, str]] = []
        for task in tasks:
            try:
                committed = by_id[task.task_id]
            except KeyError as exc:
                raise BenchmarkProtocolError(
                    f"test task {task.task_id!r} is absent from corpus manifest"
                ) from exc
            task_digests.append(
                (task.task_id, _hex_digest(committed.digest, "task manifest digest"))
            )
        return cls(
            corpus_digest=corpus_digest,
            split_digest=_hex_digest(corpus.split.source_digest, "split digest"),
            task_digests=tuple(sorted(task_digests)),
            _split_path=corpus.split.source_path,
            _tasks=tuple(sorted(tasks, key=lambda item: item.task_id)),
        )

    def _verify_split(self) -> None:
        try:
            payload = self._split_path.read_bytes()
        except OSError as exc:
            raise SealedMutationError("sealed split source is unavailable") from exc
        if _sha256_bytes(payload) != self.split_digest:
            raise SealedMutationError("sealed split bytes changed")
        # Parsing rejects malformed or duplicate membership even if a caller
        # somehow constructed the guard without going through ``capture``.
        SplitIndex.load(self._split_path)

    def verify_episode(self, plan: EpisodePlan) -> None:
        if plan.split != "test":
            raise BenchmarkProtocolError("sealed test guard used for a non-test episode")
        if plan.corpus_digest != self.corpus_digest:
            raise SealedMutationError("episode uses a different corpus snapshot")
        expected = dict(self.task_digests).get(plan.task_id)
        if expected is None:
            raise BenchmarkProtocolError("episode task is not in the sealed test set")
        if expected != plan.task_manifest_digest:
            raise SealedMutationError("episode task manifest differs from sealed test")
        self._verify_split()
        plan.verify_sources()

    def verify_all(self) -> None:
        """Expensive full-test audit, intended before and after a suite run."""

        self._verify_split()
        expected = dict(self.task_digests)
        if {task.task_id for task in self._tasks} != set(expected):
            raise SealedMutationError("sealed test task inventory changed")
        for task in self._tasks:
            actual = _hex_digest(task.build_manifest().digest, "task manifest digest")
            if actual != expected[task.task_id]:
                raise SealedMutationError(
                    f"sealed test task bytes changed for {task.task_id}"
                )


@dataclass(frozen=True, slots=True)
class EpisodeFailure:
    stage: str
    error_type: str
    reason: str

    def to_data(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    image_correct: int
    image_total: int
    puzzle_correct: bool
    determinate: int
    abstentions: int
    errors: int

    def __post_init__(self) -> None:
        if self.image_total != 2:
            raise ValueError("one canonical episode always has two query images")
        if not 0 <= self.image_correct <= self.image_total:
            raise ValueError("invalid image score")
        if self.puzzle_correct != (self.image_correct == self.image_total):
            raise ValueError("puzzle correctness must mean both queries are correct")
        if self.determinate + self.abstentions != self.image_total:
            raise ValueError("determinate and abstention counts do not cover queries")
        if not 0 <= self.errors <= self.abstentions:
            raise ValueError("errors must be a subset of abstentions")

    @property
    def image_accuracy(self) -> float:
        return self.image_correct / self.image_total

    @property
    def puzzle_accuracy(self) -> float:
        return float(self.puzzle_correct)

    def to_data(self) -> dict[str, object]:
        return {
            "image_correct": self.image_correct,
            "image_total": self.image_total,
            "image_accuracy": self.image_accuracy,
            "puzzle_correct": self.puzzle_correct,
            "puzzle_accuracy": self.puzzle_accuracy,
            "determinate": self.determinate,
            "abstentions": self.abstentions,
            "errors": self.errors,
        }


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    task_id: str
    family: str
    split: str | None
    regime: str | None
    run_id: str
    plan_digest: str
    status: EpisodeStatus
    score: EpisodeScore
    phases: tuple[str, ...]
    bundle: RunArtifactBundle | None = field(repr=False)
    failure: EpisodeFailure | None = None
    support_gate: SupportGateArtifact | None = field(
        default=None, repr=False, compare=False
    )
    proposal_freeze: ProposalFreeze | None = field(
        default=None, repr=False, compare=False
    )

    def to_data(self) -> dict[str, object]:
        return {
            "version": PROTOCOL_VERSION,
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "regime": self.regime,
            "run_id": self.run_id,
            "plan_digest": self.plan_digest,
            "status": self.status.value,
            "score": self.score.to_data(),
            "phases": list(self.phases),
            "artifact_chain": self.bundle.chain_data() if self.bundle else None,
            "failure": self.failure.to_data() if self.failure else None,
        }


def _verify_integrity(
    plan: EpisodePlan, guard: SealedTestGuard | None
) -> None:
    if plan.split == "test":
        if guard is None:
            raise BenchmarkProtocolError(
                "official test episodes require a captured SealedTestGuard"
            )
        guard.verify_episode(plan)
    else:
        if guard is not None:
            raise BenchmarkProtocolError("a sealed test guard cannot be used outside test")
        plan.verify_sources()


def _verify_predicate_policy_binding(plan: EpisodePlan, proposer: Proposer) -> None:
    """Match a verifier-owned proposer adapter to the pre-support plan policy."""

    expected = (plan.predicate_mode, plan.predicate_policy_digest)
    actual = (
        getattr(proposer, "predicate_mode", None),
        getattr(proposer, "predicate_policy_digest", None),
    )
    if expected != actual:
        raise BenchmarkProtocolError(
            "proposer predicate policy differs from the committed episode plan"
        )


def _write_neutral(path: Path, source: _PanelSource) -> None:
    payload = source.read_verified()
    path.write_bytes(payload)
    source.panel.verify_bytes(path.read_bytes())


def _support_input(plan: EpisodePlan, root: Path) -> SupportInput:
    positive = tuple(source for source in plan._support_sources if source.positive)
    negative = tuple(source for source in plan._support_sources if not source.positive)
    positive_paths: list[Path] = []
    negative_paths: list[Path] = []
    for slot, source in enumerate(positive):
        path = root / f"pos_{slot}.png"
        _write_neutral(path, source)
        positive_paths.append(path)
    for slot, source in enumerate(negative):
        path = root / f"neg_{slot}.png"
        _write_neutral(path, source)
        negative_paths.append(path)
    return SupportInput(
        tuple(positive_paths),  # type: ignore[arg-type]
        tuple(negative_paths),  # type: ignore[arg-type]
    )


def _protocol_error_evidence(
    plan: EpisodePlan,
    freeze: ProposalFreeze,
    panel: BlobRef,
    error_type: str,
    reason: str,
) -> Evidence[bool]:
    provenance = Provenance(
        producer="bongard.benchmark",
        version="1",
        method="observer-protocol-boundary",
        input_digests=(freeze.digest(), panel.sha256),
        run_id=plan.run_id,
    )
    return Evidence.error(provenance, error_type, reason)


def _validate_registry_against_freeze(
    proposed: ProposedRule,
    freeze: ProposalFreeze,
    registry: LegRegistry,
) -> None:
    """Recompute contract identity against digests committed before queries."""

    if proposed.attachment_contract.digest() != freeze.attachment_contract_digest:
        raise ValueError("attachment contract differs from proposal freeze")
    if registry.digest() != freeze.registry_digest:
        raise ValueError("registry differs from proposal freeze")
    proposed.attachment_contract.validate(freeze.formula, registry)


def _isolated_registry(
    proposed: ProposedRule, freeze: ProposalFreeze
) -> LegRegistry:
    """Revalidate live implementation identity and return a query-local copy."""

    try:
        _validate_registry_against_freeze(proposed, freeze, proposed.registry)
        registry = copy.deepcopy(proposed.registry)
        if registry is proposed.registry:
            raise ValueError("registry copy retained the original object identity")
        _validate_registry_against_freeze(proposed, freeze, registry)
    except Exception as exc:  # noqa: BLE001 - registry trust boundary.
        raise BenchmarkProtocolError(
            "frozen registry implementation/snapshot changed before query"
        ) from exc
    return registry


def _revalidate_registries(
    proposed: ProposedRule,
    freeze: ProposalFreeze,
    query_registry: LegRegistry,
) -> None:
    """Reject callback-time mutation of either registry before continuing."""

    try:
        _validate_registry_against_freeze(proposed, freeze, proposed.registry)
        _validate_registry_against_freeze(proposed, freeze, query_registry)
    except Exception as exc:  # noqa: BLE001 - registry trust boundary.
        raise BenchmarkProtocolError(
            "observer changed a frozen registry implementation or contract"
        ) from exc


def _observe_query(
    plan: EpisodePlan,
    freeze: ProposalFreeze,
    query: QueryPanel,
    source: _PanelSource,
    observer: Observer,
    registry: LegRegistry,
) -> Mapping[AtomPath, Evidence[bool]]:
    expected = set(atom_paths(freeze.formula))
    neutral_panel = BlobRef(
        blob_id=_NEUTRAL_QUERY_BLOB_ID,
        sha256=query.panel.sha256,
        byte_count=query.panel.byte_count,
        media_type=query.panel.media_type,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="bongard-query-") as directory:
            path = Path(directory) / "query.png"
            _write_neutral(path, source)
            observed = observer.observe(
                ObservationInput(
                    query_id=_NEUTRAL_QUERY_ID,
                    panel_path=path,
                    panel=neutral_panel,
                    freeze=copy.deepcopy(freeze),
                    registry=registry,
                )
            )
    except Exception as exc:  # noqa: BLE001 - fail-closed observer boundary.
        # A launcher/model/policy substitution is a run-level protocol
        # violation, not ordinary visual uncertainty.  Preserve the raw
        # response in the isolated Headless session and let the runner reject
        # the episode explicitly.
        from bongard.proposer import TransportIdentityError

        if isinstance(exc, TransportIdentityError):
            raise
        error = _protocol_error_evidence(
            plan,
            freeze,
            query.panel,
            type(exc).__name__,
            "observer raised before producing a complete atom record",
        )
        return {path: error for path in expected}

    if not isinstance(observed, Mapping) or set(observed) != expected:
        error = _protocol_error_evidence(
            plan,
            freeze,
            query.panel,
            "ObserverProtocolError",
            "observer did not return exactly one evidence value per frozen atom",
        )
        return {path: error for path in expected}
    normalised: dict[AtomPath, Evidence[bool]] = {}
    for path in expected:
        evidence = observed[path]
        if not isinstance(evidence, Evidence):
            normalised[path] = _protocol_error_evidence(
                plan,
                freeze,
                query.panel,
                "ObserverProtocolError",
                "observer returned a non-Evidence atom value",
            )
        elif evidence.disposition is Disposition.PRESENT and evidence.value is not True:
            normalised[path] = _protocol_error_evidence(
                plan,
                freeze,
                query.panel,
                "ObserverProtocolError",
                "present truth evidence must contain True",
            )
        else:
            normalised[path] = evidence
    return normalised


def _score_bundle(bundle: RunArtifactBundle) -> EpisodeScore:
    prediction_by_id = {
        item.query_id: item for item in bundle.predictions.predictions
    }
    labels = {item.query_id: item.positive for item in bundle.labels.labels}
    correct = sum(
        prediction.positive is not None
        and prediction.positive == labels[prediction.query_id]
        for prediction in prediction_by_id.values()
    )
    dispositions = Counter(
        prediction.disposition for prediction in prediction_by_id.values()
    )
    abstentions = (
        dispositions[Disposition.INDETERMINATE] + dispositions[Disposition.ERROR]
    )
    return EpisodeScore(
        image_correct=correct,
        image_total=2,
        puzzle_correct=correct == 2,
        determinate=2 - abstentions,
        abstentions=abstentions,
        errors=dispositions[Disposition.ERROR],
    )


def _support_protocol_measurement(
    plan: EpisodePlan,
    provisional_freeze: ProposalFreeze,
    panel: BlobRef,
    error_type: str,
    reason: str,
    *,
    transport_attempted: bool,
) -> SupportGateMeasurement:
    evidence = _protocol_error_evidence(
        plan, provisional_freeze, panel, error_type, reason
    )
    return SupportGateMeasurement(
        evidence=evidence,
        observer_artifact={
            "schema": "support-replay-protocol-error/v1",
            "error_type": error_type,
            "reason": reason,
        },
        transport_attempted=transport_attempted,
    )


def _classify_support_gate(
    *,
    plan: EpisodePlan,
    proposed: ProposedRule,
    policy: SupportGatePolicy,
    measurements: Sequence[SupportGateMeasurement],
) -> SupportGateArtifact:
    if len(measurements) != 12:
        raise BenchmarkProtocolError("support gate must classify exactly twelve panels")
    entries = tuple(
        SupportGateEntry(
            slot_id=source.panel.blob_id,
            panel=source.panel,
            positive=source.positive,
            evidence=TruthEvidenceRecord.from_evidence(measurement.evidence),
            observer_artifact=measurement.observer_artifact,
            transport_attempted=measurement.transport_attempted,
        )
        for source, measurement in zip(
            plan._support_sources, measurements, strict=True
        )
    )
    dispositions = Counter(
        measurement.evidence.disposition for measurement in measurements
    )
    forward = sum(
        (
            source.positive
            and measurement.evidence.disposition is Disposition.PRESENT
        )
        or (
            not source.positive
            and measurement.evidence.disposition is Disposition.CERTIFIED_ABSENT
        )
        for source, measurement in zip(
            plan._support_sources, measurements, strict=True
        )
    )
    reverse = sum(
        (
            source.positive
            and measurement.evidence.disposition is Disposition.CERTIFIED_ABSENT
        )
        or (
            not source.positive
            and measurement.evidence.disposition is Disposition.PRESENT
        )
        for source, measurement in zip(
            plan._support_sources, measurements, strict=True
        )
    )
    if dispositions[Disposition.INDETERMINATE] or dispositions[Disposition.ERROR]:
        result = SupportGateResult.OBSERVER_FAILURE
    elif forward == 12:
        result = SupportGateResult.ALIGNED
    elif reverse > forward:
        result = SupportGateResult.MISORIENTED
    else:
        result = SupportGateResult.UNSUPPORTED
    return SupportGateArtifact(
        run_id=plan.run_id,
        proposal_digest=proposed.proposer_digest,
        support_commitment_digest=plan.support.digest(),
        policy=policy,
        entries=entries,
        forward_matches=forward,
        reverse_matches=reverse,
        present_count=dispositions[Disposition.PRESENT],
        nonmatch_count=dispositions[Disposition.CERTIFIED_ABSENT],
        indeterminate_count=dispositions[Disposition.INDETERMINATE],
        error_count=dispositions[Disposition.ERROR],
        transport_attempt_count=sum(
            measurement.transport_attempted for measurement in measurements
        ),
        result=result,
    )


def _test_bypass_gate(
    plan: EpisodePlan,
    proposed: ProposedRule,
    policy: SupportGatePolicy,
) -> SupportGateArtifact:
    return SupportGateArtifact(
        run_id=plan.run_id,
        proposal_digest=proposed.proposer_digest,
        support_commitment_digest=plan.support.digest(),
        policy=policy,
        entries=(),
        forward_matches=0,
        reverse_matches=0,
        present_count=0,
        nonmatch_count=0,
        indeterminate_count=0,
        error_count=0,
        transport_attempt_count=0,
        result=SupportGateResult.TEST_BYPASSED,
    )


def _run_support_replay_gate(
    plan: EpisodePlan,
    proposed: ProposedRule,
    proposer: Proposer,
    policy: SupportGatePolicy,
    *,
    sealed_guard: SealedTestGuard | None,
) -> SupportGateArtifact:
    if policy.mode is SupportGateMode.TEST_BYPASS:
        if bool(getattr(proposer, "requires_empirical_support_gate", False)):
            raise BenchmarkProtocolError(
                "HeadlessCodexEpisode cannot bypass empirical support replay"
            )
        return _test_bypass_gate(plan, proposed, policy)
    if not isinstance(proposer, SupportReplayFactory):
        raise BenchmarkProtocolError(
            "empirical support replay requires create_support_observer()"
        )

    # The proposal/registry is already immutable.  This provisional object is
    # used only to give callback adapters the same closed formula boundary;
    # query release can depend only on the later freeze that binds the actual
    # gate digest.
    provisional_freeze = ProposalFreeze.create(
        support=plan.support,
        proposal_id=proposed.proposal_id,
        formula=proposed.formula,
        proposer_digest=proposed.proposer_digest,
        attachment_contract=proposed.attachment_contract,
        registry=proposed.registry,
        support_gate_digest=_PENDING_SUPPORT_GATE_DIGEST,
        verifier_nonce=_derive_hex("proposal-freeze-nonce", plan.digest),
    )
    sessions: list[SupportReplayObserver | None] = []
    creation_errors: list[Exception | None] = []
    for _slot in range(12):
        try:
            session = proposer.create_support_observer()
            if not isinstance(session, SupportReplayObserver):
                raise TypeError("support observer lacks observe_support()")
            if session is proposer or any(session is existing for existing in sessions):
                raise ValueError("support observer factory reused callback state")
            sessions.append(session)
            creation_errors.append(None)
        except Exception as exc:  # noqa: BLE001 - isolated callback boundary.
            sessions.append(None)
            creation_errors.append(exc)

    measurements: list[SupportGateMeasurement] = []
    for source, session, creation_error in zip(
        plan._support_sources, sessions, creation_errors, strict=True
    ):
        if creation_error is not None:
            measurements.append(
                _support_protocol_measurement(
                    plan,
                    provisional_freeze,
                    source.panel,
                    type(creation_error).__name__,
                    "support observer factory failed before isolated transport",
                    transport_attempted=False,
                )
            )
            _verify_integrity(plan, sealed_guard)
            continue
        assert session is not None
        neutral_panel = BlobRef(
            blob_id=_NEUTRAL_QUERY_BLOB_ID,
            sha256=source.panel.sha256,
            byte_count=source.panel.byte_count,
            media_type=source.panel.media_type,
        )
        registry = _isolated_registry(proposed, provisional_freeze)
        try:
            with tempfile.TemporaryDirectory(prefix="bongard-support-replay-") as directory:
                path = Path(directory) / "query.png"
                _write_neutral(path, source)
                measurement = session.observe_support(
                    ObservationInput(
                        query_id=_NEUTRAL_QUERY_ID,
                        panel_path=path,
                        panel=neutral_panel,
                        freeze=copy.deepcopy(provisional_freeze),
                        registry=registry,
                    )
                )
            if not isinstance(measurement, SupportGateMeasurement):
                raise TypeError("support observer returned the wrong measurement type")
            _revalidate_registries(proposed, provisional_freeze, registry)
        except Exception as exc:  # noqa: BLE001 - record and continue all 12 calls.
            measurement = _support_protocol_measurement(
                plan,
                provisional_freeze,
                source.panel,
                type(exc).__name__,
                str(exc) or "support observer failed",
                transport_attempted=True,
            )
        measurements.append(measurement)
        _verify_integrity(plan, sealed_guard)
    return _classify_support_gate(
        plan=plan,
        proposed=proposed,
        policy=policy,
        measurements=measurements,
    )


def run_episode(
    plan: EpisodePlan,
    proposer: Proposer,
    observer: Observer | ObserverFactory,
    *,
    support_gate_policy: SupportGatePolicy,
    sealed_guard: SealedTestGuard | None = None,
) -> EpisodeResult:
    """Run one proposal, a query-blind 12-panel gate, then two queries.

    The two predictions are jointly committed before ``_revealed_labels`` is
    called.  Each observer call gets a distinct callback object, a query-local
    registry, and a separate directory containing only one neutral
    ``query.png``.  An explicit :class:`ObserverFactory` is preferred; legacy
    observer objects are independently deep-copied.
    """

    phases: list[str] = ["plan_committed"]
    _verify_integrity(plan, sealed_guard)
    _verify_predicate_policy_binding(plan, proposer)
    try:
        with tempfile.TemporaryDirectory(prefix="bongard-support-") as directory:
            support_input = _support_input(plan, Path(directory))
            phases.append("support_released")
            proposed = proposer.propose(support_input)
        if not isinstance(proposed, ProposedRule):
            raise TypeError("proposer adapter did not return ProposedRule")
        if proposed.attachment_contract.issued_by != plan.verifier_id:
            raise ValueError("proposal attachment issuer differs from episode verifier")
        _hex_digest(proposed.proposer_digest, "proposer digest")
        phases.append("proposal_fixed")
    except Exception as exc:  # noqa: BLE001 - no query release on proposal failure.
        _verify_integrity(plan, sealed_guard)
        phases.append("proposal_failed")
        return EpisodeResult(
            task_id=plan.task_id,
            family=plan.family,
            split=plan.split,
            regime=plan.regime,
            run_id=plan.run_id,
            plan_digest=plan.digest,
            status=EpisodeStatus.PROPOSAL_ERROR,
            score=EpisodeScore(0, 2, False, 0, 2, 2),
            phases=tuple(phases),
            bundle=None,
            failure=EpisodeFailure(
                "proposal", type(exc).__name__, str(exc) or "proposal failed"
            ),
        )

    # Re-observe every support image in a new single-image transport.  The
    # callback sees only query.png and the already-fixed operational claim;
    # labels are joined below, inside this verifier-owned runner.
    try:
        gate = _run_support_replay_gate(
            plan,
            proposed,
            proposer,
            support_gate_policy,
            sealed_guard=sealed_guard,
        )
        freeze = ProposalFreeze.create(
            support=plan.support,
            proposal_id=proposed.proposal_id,
            formula=proposed.formula,
            proposer_digest=_hex_digest(proposed.proposer_digest, "proposer digest"),
            attachment_contract=proposed.attachment_contract,
            registry=proposed.registry,
            support_gate_digest=gate.digest,
            verifier_nonce=_derive_hex("proposal-freeze-nonce", plan.digest),
        )
        _validate_registry_against_freeze(proposed, freeze, proposed.registry)
        phases.append("support_gate_replayed")
        phases.append("proposal_frozen")
    except Exception as exc:  # noqa: BLE001 - no query release on gate failure.
        _verify_integrity(plan, sealed_guard)
        phases.append("support_gate_failed")
        return EpisodeResult(
            task_id=plan.task_id,
            family=plan.family,
            split=plan.split,
            regime=plan.regime,
            run_id=plan.run_id,
            plan_digest=plan.digest,
            status=EpisodeStatus.SUPPORT_REJECTED,
            score=EpisodeScore(0, 2, False, 0, 2, 2),
            phases=tuple(phases),
            bundle=None,
            failure=EpisodeFailure(
                "support_gate", type(exc).__name__, str(exc) or "support gate failed"
            ),
        )

    if not gate.accepted:
        _verify_integrity(plan, sealed_guard)
        phases.append("support_gate_rejected")
        observer_failed = gate.result is SupportGateResult.OBSERVER_FAILURE
        return EpisodeResult(
            task_id=plan.task_id,
            family=plan.family,
            split=plan.split,
            regime=plan.regime,
            run_id=plan.run_id,
            plan_digest=plan.digest,
            status=EpisodeStatus.SUPPORT_REJECTED,
            score=EpisodeScore(0, 2, False, 0, 2, 2 if observer_failed else 0),
            phases=tuple(phases),
            bundle=None,
            failure=EpisodeFailure(
                "support_gate",
                "SupportGateRejected",
                gate.result.value,
            ),
            support_gate=gate,
            proposal_freeze=freeze,
        )

    try:
        observer_sessions = _ObserverSessions(observer)
        isolated_observers = observer_sessions.prepare(2)
        # Factory/copy hooks are arbitrary in-process Python.  Check that they
        # did not alter the signed implementation boundary before releasing a
        # single query pixel.
        _validate_registry_against_freeze(proposed, freeze, proposed.registry)
    except Exception as exc:  # noqa: BLE001 - no query release on isolation failure.
        _verify_integrity(plan, sealed_guard)
        phases.append("query_observer_prepare_failed")
        return EpisodeResult(
            task_id=plan.task_id,
            family=plan.family,
            split=plan.split,
            regime=plan.regime,
            run_id=plan.run_id,
            plan_digest=plan.digest,
            status=EpisodeStatus.PROPOSAL_ERROR,
            score=EpisodeScore(0, 2, False, 0, 2, 2),
            phases=tuple(phases),
            bundle=None,
            failure=EpisodeFailure(
                "observer_prepare", type(exc).__name__, str(exc) or "observer preparation failed"
            ),
            support_gate=gate,
            proposal_freeze=freeze,
        )

    # This check occurs after the callback and before query release.  A
    # proposer with an out-of-band source reference still cannot mutate a
    # sealed task and continue.
    _verify_integrity(plan, sealed_guard)
    release = QueryRelease.create(
        freeze,
        plan.queries,
        verifier_nonce=_derive_hex("query-release-nonce", plan.digest),
    )
    released_latent_digest = canonical_digest(
        {
            "version": "latent-two-query-commitment/v1",
            "run_id": release.run_id,
            "queries": [item.to_data() for item in release.queries],
        }
    )
    if released_latent_digest != plan.latent_query_digest:
        raise ArtifactTamperError("released queries differ from latent commitment")
    phases.append("query_released")

    atom_evidence: dict[str, Mapping[AtomPath, Evidence[bool]]] = {}
    for query, source, isolated_observer in zip(
        plan.queries,
        plan._query_sources,
        isolated_observers,
        strict=True,
    ):
        query_registry = _isolated_registry(proposed, freeze)
        try:
            atom_evidence[query.query_id] = _observe_query(
                plan,
                freeze,
                query,
                source,
                isolated_observer,
                query_registry,
            )
        except Exception as exc:  # noqa: BLE001 - identify transport substitution.
            from bongard.proposer import TransportIdentityError

            if not isinstance(exc, TransportIdentityError):
                raise
            observer_sessions.record(query.query_id, isolated_observer)
            observer_sessions.finalize(expected_count=len(observer_sessions._issued))
            _verify_integrity(plan, sealed_guard)
            phases.append("query_observer_failed")
            return EpisodeResult(
                task_id=plan.task_id,
                family=plan.family,
                split=plan.split,
                regime=plan.regime,
                run_id=plan.run_id,
                plan_digest=plan.digest,
                status=EpisodeStatus.OBSERVATION_ERROR,
                score=EpisodeScore(0, 2, False, 0, 2, 2),
                phases=tuple(phases),
                bundle=None,
                failure=EpisodeFailure(
                    "query_observer",
                    type(exc).__name__,
                    str(exc) or "query observer transport identity differed",
                ),
                support_gate=gate,
                proposal_freeze=freeze,
            )
        _revalidate_registries(proposed, freeze, query_registry)
        observer_sessions.record(query.query_id, isolated_observer)
        _verify_integrity(plan, sealed_guard)
    observer_sessions.finalize()
    _revalidate_registries(proposed, freeze, proposed.registry)

    cold_inputs = ColdReplayInputs.capture(
        freeze=freeze,
        release=release,
        atom_evidence=atom_evidence,
    )
    predictions = PredictionCommitment.create(
        freeze=freeze,
        release=release,
        cold_inputs=cold_inputs,
        verifier_nonce=_derive_hex("prediction-nonce", plan.digest),
    )
    phases.append("predictions_committed")

    # Ground-truth labels are materialised only after the joint commitment.
    _verify_integrity(plan, sealed_guard)
    labels = LabelReveal.create(
        predictions,
        plan._revealed_labels(),
        # The random label-seal salt is disclosed only after both predictions
        # have been committed.  Publishing EpisodePlan.to_data() before then
        # no longer permits enumeration of the two possible label orders.
        verifier_nonce=plan._label_nonce,
    )
    phases.append("labels_revealed")
    bundle = RunArtifactBundle(
        support=plan.support,
        attachment_contract=proposed.attachment_contract,
        freeze=freeze,
        release=release,
        cold_inputs=cold_inputs,
        predictions=predictions,
        labels=labels,
    )
    bundle.verify()
    _verify_integrity(plan, sealed_guard)
    phases.append("cold_replay_verified")
    return EpisodeResult(
        task_id=plan.task_id,
        family=plan.family,
        split=plan.split,
        regime=plan.regime,
        run_id=plan.run_id,
        plan_digest=plan.digest,
        status=EpisodeStatus.COMPLETE,
        score=_score_bundle(bundle),
        phases=tuple(phases),
        bundle=bundle,
        support_gate=gate,
        proposal_freeze=freeze,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    episode_total: int
    episode_complete: int
    image_correct: int
    image_total: int
    puzzle_correct: int
    puzzle_total: int
    determinate: int
    abstentions: int
    errors: int
    dispositions: tuple[tuple[str, int], ...]

    @property
    def image_accuracy(self) -> float:
        return self.image_correct / self.image_total if self.image_total else 0.0

    @property
    def puzzle_accuracy(self) -> float:
        return self.puzzle_correct / self.puzzle_total if self.puzzle_total else 0.0

    def to_data(self) -> dict[str, object]:
        return {
            "episode_total": self.episode_total,
            "episode_complete": self.episode_complete,
            "image_correct": self.image_correct,
            "image_total": self.image_total,
            "image_accuracy": self.image_accuracy,
            "puzzle_correct": self.puzzle_correct,
            "puzzle_total": self.puzzle_total,
            "puzzle_accuracy": self.puzzle_accuracy,
            "determinate": self.determinate,
            "abstentions": self.abstentions,
            "errors": self.errors,
            "dispositions": dict(self.dispositions),
        }


def score_results(results: Sequence[EpisodeResult]) -> BenchmarkScore:
    """Aggregate with abstentions/errors in both accuracy denominators."""

    disposition_counts: Counter[str] = Counter()
    for result in results:
        if result.bundle is None:
            disposition_counts[Disposition.ERROR.value] += 2
        else:
            disposition_counts.update(
                item.disposition.value for item in result.bundle.predictions.predictions
            )
    return BenchmarkScore(
        episode_total=len(results),
        episode_complete=sum(
            result.status is EpisodeStatus.COMPLETE for result in results
        ),
        image_correct=sum(result.score.image_correct for result in results),
        image_total=sum(result.score.image_total for result in results),
        puzzle_correct=sum(result.score.puzzle_correct for result in results),
        puzzle_total=len(results),
        determinate=sum(result.score.determinate for result in results),
        abstentions=sum(result.score.abstentions for result in results),
        errors=sum(result.score.errors for result in results),
        dispositions=tuple(sorted(disposition_counts.items())),
    )


__all__ = [
    "BenchmarkProtocolError",
    "BenchmarkScore",
    "ClosedIRObserver",
    "DEFAULT_VERIFIER",
    "EpisodeFailure",
    "EpisodePlan",
    "EpisodeResult",
    "EpisodeScore",
    "EpisodeStatus",
    "ObservationInput",
    "Observer",
    "ObserverFactory",
    "PROTOCOL_VERSION",
    "SUPPORT_PROTOTYPE_PREDICATE_MODE",
    "ProposedRule",
    "Proposer",
    "SealedMutationError",
    "SealedTestGuard",
    "SupportInput",
    "SupportGateArtifact",
    "SupportGateEntry",
    "SupportGateMeasurement",
    "SupportGateMode",
    "SupportGatePolicy",
    "SupportGateResult",
    "SUPPORT_PROTOTYPE_GATE_POLICY_VERSION",
    "SupportReplayFactory",
    "SupportReplayObserver",
    "prepare_episode",
    "run_episode",
    "score_results",
]
