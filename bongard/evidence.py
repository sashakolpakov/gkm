"""Epistemically honest evidence values for the canonical Bongard track.

An extractor has four, and only four, possible outcomes.  In particular,
failure to construct a witness is not silently reinterpreted as evidence that
the witness is absent.  Soft semantic descriptions are empirical observations
with provenance and a score interval; they deliberately have no truth-value.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Generic, NoReturn, TypeVar


T = TypeVar("T")
U = TypeVar("U")


class Disposition(str, Enum):
    """The exhaustive runtime state of an observation."""

    PRESENT = "present"
    CERTIFIED_ABSENT = "certified_absent"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


@dataclass(frozen=True)
class Provenance:
    """Auditable origin of an observation.

    ``input_digests`` bind the observation to exact input bytes (or exact
    parent observations).  ``details`` is a canonical tuple rather than a
    mutable mapping so provenance itself can be hashed and archived.
    """

    producer: str
    version: str
    method: str
    input_digests: tuple[str, ...] = ()
    artifact_digest: str | None = None
    run_id: str | None = None
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("producer", self.producer),
            ("version", self.version),
            ("method", self.method),
        ):
            if not value.strip():
                raise ValueError(f"provenance {label} must be non-empty")
        if any(not value.strip() for value in self.input_digests):
            raise ValueError("provenance input digests must be non-empty")
        keys = [key for key, _ in self.details]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("provenance detail keys must be unique and sorted")
        if any(not key.strip() for key in keys):
            raise ValueError("provenance detail keys must be non-empty")

    def canonical_data(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def composed(
        cls,
        producer: str,
        version: str,
        method: str,
        parents: tuple["Provenance", ...],
        *,
        details: tuple[tuple[str, str], ...] = (),
    ) -> "Provenance":
        return cls(
            producer=producer,
            version=version,
            method=method,
            input_digests=tuple(parent.digest() for parent in parents),
            details=tuple(sorted(details)),
        )


@dataclass(frozen=True)
class Uncertainty:
    """A closed interval known to contain a scalar observation.

    The interval is unit-agnostic here; units belong to the typed leg and IR
    layers.  ``confidence_level`` records calibration when one exists, but an
    interval may also be a deterministic numerical enclosure.
    """

    lower: float
    upper: float
    confidence_level: float | None = None
    causes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Canonical evidence archives use JSON floating-point numbers for
        # intervals.  Python's ``bool`` is an ``int`` and permissive numeric
        # coercion used to let ``False``/``True`` enter an in-memory archive
        # which changed to ``0.0``/``1.0`` during cold decoding.  Require the
        # one in-memory representation that round-trips byte-for-byte.
        if type(self.lower) is not float or type(self.upper) is not float:
            raise ValueError(
                "uncertainty bounds must be literal canonical floats"
            )
        if self.confidence_level is not None and type(
            self.confidence_level
        ) is not float:
            raise ValueError(
                "uncertainty confidence_level must be a literal canonical float"
            )
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("uncertainty bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("uncertainty lower bound exceeds upper bound")
        if self.confidence_level is not None and not (
            0.0 < self.confidence_level <= 1.0
        ):
            raise ValueError("confidence_level must lie in (0, 1]")
        if len(self.causes) != len(set(self.causes)):
            raise ValueError("uncertainty causes must be unique")
        if any(not cause.strip() for cause in self.causes):
            raise ValueError("uncertainty causes must be non-empty")

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class SoftSemanticObservation:
    """A vision model's graded, empirical description of a panel.

    This can ground an IR atom after calibration, but is not itself the fact
    that a panel *is* bird-like, oblique, and so on.  Calling ``bool`` is an
    error so downstream code must compare the support interval explicitly.
    """

    phrase: str
    support: Uncertainty
    provenance: Provenance
    description: str = ""
    witness_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.phrase.strip():
            raise ValueError("soft semantic phrase must be non-empty")
        if self.support.lower < 0.0 or self.support.upper > 1.0:
            raise ValueError("soft semantic support must lie in [0, 1]")
        if len(self.witness_ids) != len(set(self.witness_ids)):
            raise ValueError("soft semantic witness ids must be unique")
        if any(not witness_id.strip() for witness_id in self.witness_ids):
            raise ValueError("soft semantic witness ids must be non-empty")

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "a soft semantic observation is empirical evidence, not truth; "
            "compare its support interval through the closed IR"
        )


@dataclass(frozen=True)
class Evidence(Generic[T]):
    """One provenance-bearing observation with exactly one disposition."""

    disposition: Disposition
    provenance: Provenance
    value: T | None = None
    uncertainty: Uncertainty | None = None
    certificate: str | None = None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.disposition is Disposition.PRESENT:
            if self.value is None:
                raise ValueError("present evidence requires a value")
            if self.certificate is not None or self.reason is not None:
                raise ValueError("present evidence cannot carry failure fields")
            if self.error_type is not None:
                raise ValueError("present evidence cannot carry an error type")
            if isinstance(self.value, SoftSemanticObservation):
                if self.value.provenance != self.provenance:
                    raise ValueError(
                        "soft semantic observation and evidence provenance differ"
                    )
                if self.uncertainty not in (None, self.value.support):
                    raise ValueError(
                        "soft semantic evidence uncertainty must match support"
                    )
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if self.value is not None:
                raise ValueError("certified absence cannot carry a value")
            if self.certificate is None or not self.certificate.strip():
                raise ValueError("certified absence requires a certificate")
            if self.reason is not None or self.error_type is not None:
                raise ValueError("certified absence cannot carry failure fields")
        elif self.disposition is Disposition.INDETERMINATE:
            if self.value is not None or self.certificate is not None:
                raise ValueError("indeterminate evidence cannot carry a value/certificate")
            if self.reason is None or not self.reason.strip():
                raise ValueError("indeterminate evidence requires a reason")
            if self.error_type is not None:
                raise ValueError("indeterminate evidence cannot carry an error type")
        elif self.disposition is Disposition.ERROR:
            if self.value is not None or self.certificate is not None:
                raise ValueError("error evidence cannot carry a value/certificate")
            if self.reason is None or not self.reason.strip():
                raise ValueError("error evidence requires a reason")
            if self.error_type is None or not self.error_type.strip():
                raise ValueError("error evidence requires an error type")
        else:  # pragma: no cover - protects callers bypassing the Enum type.
            raise ValueError(f"unknown evidence disposition {self.disposition!r}")

    @classmethod
    def present(
        cls,
        value: T,
        provenance: Provenance,
        uncertainty: Uncertainty | None = None,
    ) -> "Evidence[T]":
        if isinstance(value, SoftSemanticObservation) and uncertainty is None:
            uncertainty = value.support
        return cls(
            disposition=Disposition.PRESENT,
            value=value,
            provenance=provenance,
            uncertainty=uncertainty,
        )

    @classmethod
    def certified_absent(
        cls,
        provenance: Provenance,
        certificate: str,
        uncertainty: Uncertainty | None = None,
    ) -> "Evidence[T]":
        return cls(
            disposition=Disposition.CERTIFIED_ABSENT,
            provenance=provenance,
            certificate=certificate,
            uncertainty=uncertainty,
        )

    @classmethod
    def indeterminate(
        cls,
        provenance: Provenance,
        reason: str,
        uncertainty: Uncertainty | None = None,
    ) -> "Evidence[T]":
        return cls(
            disposition=Disposition.INDETERMINATE,
            provenance=provenance,
            reason=reason,
            uncertainty=uncertainty,
        )

    @classmethod
    def operational_nonmatch(
        cls,
        provenance: Provenance,
        reason: str,
        uncertainty: Uncertainty | None = None,
    ) -> "Evidence[T]":
        """Record an uncalibrated observer nonmatch as an abstention.

        Operational nonmatch is deliberately represented by the existing
        ``INDETERMINATE`` truth disposition.  It is useful exploratory data,
        but it is not a counter-witness and therefore can never be consumed as
        ``CERTIFIED_ABSENT`` by Boolean synthesis.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("operational nonmatch requires a non-empty reason")
        return cls.indeterminate(
            provenance,
            "operational nonmatch (uncalibrated): " + reason,
            uncertainty,
        )

    @property
    def is_operational_nonmatch(self) -> bool:
        return (
            self.disposition is Disposition.INDETERMINATE
            and self.reason is not None
            and self.reason.startswith("operational nonmatch (uncalibrated): ")
        )

    @classmethod
    def error(
        cls,
        provenance: Provenance,
        error_type: str,
        reason: str,
    ) -> "Evidence[T]":
        return cls(
            disposition=Disposition.ERROR,
            provenance=provenance,
            error_type=error_type,
            reason=reason,
        )

    @property
    def is_present(self) -> bool:
        return self.disposition is Disposition.PRESENT

    def unwrap(self) -> T:
        if not self.is_present:
            raise ValueError(f"cannot unwrap {self.disposition.value} evidence")
        # ``None`` is forbidden for PRESENT in __post_init__.
        return self.value  # type: ignore[return-value]

    def map(self, function: Callable[[T], U]) -> "Evidence[U]":
        """Map a present value while preserving every non-present state.

        An exception is an implementation error, never certified absence.
        """

        if self.disposition is Disposition.PRESENT:
            try:
                return Evidence.present(
                    function(self.unwrap()), self.provenance, self.uncertainty
                )
            except Exception as exc:  # noqa: BLE001 - disposition boundary.
                return Evidence.error(
                    self.provenance, type(exc).__name__, str(exc) or repr(exc)
                )
        return Evidence(
            disposition=self.disposition,
            provenance=self.provenance,
            uncertainty=self.uncertainty,
            certificate=self.certificate,
            reason=self.reason,
            error_type=self.error_type,
        )

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "Evidence has four dispositions and cannot be coerced to bool"
        )
