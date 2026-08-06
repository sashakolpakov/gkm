"""Typed, version-pinned contracts for registered visual and semantic legs."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from bongard.evidence import Evidence, Provenance


Literal = str | int | float | bool | None


class Unit(str, Enum):
    """Closed unit vocabulary admitted by the primary IR."""

    NONE = "none"
    DIMENSIONLESS = "dimensionless"
    COUNT = "count"
    PIXEL = "pixel"
    RADIANS = "radians"
    DEGREES = "degrees"
    FRACTION = "fraction"
    PROBABILITY = "probability"


@dataclass(frozen=True, order=True)
class ValueType:
    """A semantic carrier and, when scalar, its physical unit."""

    name: str
    unit: Unit = Unit.NONE

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid value type name {self.name!r}")
        if self.unit is not Unit.NONE and self.name not in {
            "measurement",
            "soft_semantic",
            "scalar",
        }:
            raise ValueError(
                f"non-scalar type {self.name!r} cannot carry unit {self.unit.value}"
            )

    def to_data(self) -> dict[str, str]:
        return {"name": self.name, "unit": self.unit.value}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ValueType":
        """Decode the exact static representation used in run archives."""

        if not isinstance(data, Mapping) or set(data) != {"name", "unit"}:
            raise ValueError("value type fields must be exactly name/unit")
        name = data["name"]
        unit = data["unit"]
        if not isinstance(name, str) or not isinstance(unit, str):
            raise TypeError("value type name/unit must be strings")
        return cls(name, Unit(unit))


PANEL = ValueType("panel")
FROZEN_VISUAL_SCORE = ValueType("frozen_visual_score")
OBJECT = ValueType("object")
WITNESS = ValueType("witness")
BOOLEAN_WITNESS = ValueType("boolean_witness")
SOFT_SEMANTIC = ValueType("soft_semantic", Unit.PROBABILITY)


@dataclass(frozen=True)
class TypedValue:
    """A runtime value carrying the exact type declared at the boundary."""

    type: ValueType
    value: Any

    def __post_init__(self) -> None:
        if self.value is None:
            raise ValueError("typed values cannot contain None")
        if self.type == BOOLEAN_WITNESS and self.value is not True:
            raise TypeError(
                "boolean_witness carries only the affirmative witness True; "
                "certified absence, indeterminacy, and errors belong in the "
                "Evidence disposition"
            )
        if self.type.unit is not Unit.NONE:
            if isinstance(self.value, bool):
                raise TypeError("boolean is not a scalar measurement")
            if isinstance(self.value, float) and not math.isfinite(self.value):
                raise ValueError("scalar measurement must be finite")
            # Interval-like values are accepted by protocol to avoid a module
            # cycle: the IR checks their bounds and exact unit before use.
            interval_like = hasattr(self.value, "lower") and hasattr(
                self.value, "upper"
            )
            if not isinstance(self.value, (int, float)) and not interval_like:
                # Soft observations expose a support interval rather than
                # pretending that a vision score is a direct truth value.
                if self.type != SOFT_SEMANTIC or not hasattr(
                    self.value, "support"
                ):
                    raise TypeError(
                        f"{self.type.name}/{self.type.unit.value} requires a "
                        "numeric, interval, or soft semantic value"
                    )


class Transform(str, Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    UNIFORM_SCALE = "uniform_scale"
    STROKE_WIDTH = "stroke_width"
    RASTER_RESOLUTION = "raster_resolution"
    STYLE = "style"


@dataclass(frozen=True)
class InvarianceContract:
    """Declared behavior under named nuisance transformations."""

    invariant_under: frozenset[Transform] = frozenset()
    equivariant_under: frozenset[Transform] = frozenset()
    sensitive_to: frozenset[Transform] = frozenset()

    def __post_init__(self) -> None:
        groups = (
            self.invariant_under,
            self.equivariant_under,
            self.sensitive_to,
        )
        if any(not isinstance(group, frozenset) for group in groups):
            raise TypeError("invariance transform groups must be frozensets")
        if any(
            not isinstance(transform, Transform)
            for group in groups
            for transform in group
        ):
            raise TypeError("invariance groups contain an unknown transform")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(
                "a transform cannot be invariant, equivariant, and/or "
                "sensitive in the same contract"
            )

    def to_data(self) -> dict[str, list[str]]:
        return {
            "invariant_under": sorted(item.value for item in self.invariant_under),
            "equivariant_under": sorted(item.value for item in self.equivariant_under),
            "sensitive_to": sorted(item.value for item in self.sensitive_to),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "InvarianceContract":
        required = {"invariant_under", "equivariant_under", "sensitive_to"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise ValueError("invariance fields differ from the static schema")

        def transforms(name: str) -> frozenset[Transform]:
            raw = data[name]
            if not isinstance(raw, list) or any(
                not isinstance(item, str) for item in raw
            ):
                raise TypeError(f"invariance {name} must be a list of strings")
            if raw != sorted(raw) or len(raw) != len(set(raw)):
                raise ValueError(f"invariance {name} must be unique and sorted")
            return frozenset(Transform(item) for item in raw)

        return cls(
            invariant_under=transforms("invariant_under"),
            equivariant_under=transforms("equivariant_under"),
            sensitive_to=transforms("sensitive_to"),
        )


class LegSemantics(str, Enum):
    EMPIRICAL_WITNESS = "empirical_witness"
    DETERMINISTIC_MEASUREMENT = "deterministic_measurement"
    DERIVED = "derived"


class AffirmativeRelation(str, Enum):
    """Comparison directions a leg declares to mean *more of its claim*.

    This is the signed-observable contract.  A synthesizer cannot rescue a
    badly oriented scalar by silently trying the opposite inequality: that
    inequality must have been explicitly admitted with the leg itself.
    """

    PRESENT = "present"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    BETWEEN = "between"


def implementation_sha256(implementation: Callable[..., object]) -> str:
    """Hash normalized source when available, with bytecode as a fallback."""

    try:
        source = inspect.getsource(implementation).strip()
    except (OSError, TypeError):
        code = getattr(implementation, "__code__", None)
        if code is None:
            raise TypeError("leg implementation must expose source or bytecode")
        source = "|".join(
            (
                code.co_code.hex(),
                repr(code.co_consts),
                repr(code.co_names),
                repr(code.co_varnames),
            )
        )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _is_literal(value: object) -> bool:
    return value is None or (
        isinstance(value, (str, int, float, bool))
        and not (isinstance(value, float) and not math.isfinite(value))
    )


@dataclass(frozen=True)
class LegContract:
    """The complete verifier-visible boundary of one callable leg."""

    name: str
    version: str
    domain: tuple[ValueType, ...]
    codomain: ValueType
    implementation: Callable[..., Evidence[Any]] = field(
        repr=False, compare=False, hash=False
    )
    affirmative_relations: frozenset[AffirmativeRelation] = frozenset(
        {AffirmativeRelation.PRESENT}
    )
    invariance: InvarianceContract = InvarianceContract()
    semantics: LegSemantics = LegSemantics.EMPIRICAL_WITNESS
    parameter_names: frozenset[str] = frozenset()
    cost: int = 1
    operational_digest: str | None = None
    source_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid leg name {self.name!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.version):
            raise ValueError(f"invalid leg version {self.version!r}")
        if not self.domain:
            raise ValueError("leg domain must contain at least one input")
        if not callable(self.implementation):
            raise TypeError("leg implementation must be callable")
        if not self.affirmative_relations:
            raise ValueError("leg must declare at least one affirmative relation")
        if self.codomain.unit is Unit.NONE:
            if self.affirmative_relations != frozenset(
                {AffirmativeRelation.PRESENT}
            ):
                raise ValueError(
                    "non-scalar legs admit only the affirmative present relation"
                )
        elif AffirmativeRelation.PRESENT in self.affirmative_relations:
            raise ValueError(
                "scalar legs must declare signed interval comparisons, not present"
            )
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, int)
            or self.cost <= 0
        ):
            raise ValueError("leg cost must be a positive integer")
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in self.parameter_names
        ):
            raise ValueError("leg parameter names must be lower snake case")
        try:
            inspect.signature(self.implementation).bind(
                *([None] * len(self.domain)),
                **{name: None for name in self.parameter_names},
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"implementation does not accept declared leg signature: {exc}"
            ) from exc
        if self.operational_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.operational_digest
        ):
            raise ValueError("leg operational_digest must be a lowercase sha256")
        # This identity is never caller-overridable.  Configuration or closure
        # state needed to define the operational procedure is committed
        # separately through ``operational_digest``.
        object.__setattr__(
            self, "source_digest", implementation_sha256(self.implementation)
        )

    def contract_data(self) -> dict[str, object]:
        recomputed_source_digest = implementation_sha256(self.implementation)
        if recomputed_source_digest != self.source_digest:
            raise ValueError(
                f"leg implementation changed after contract creation: "
                f"{self.name}@{self.version}"
            )
        return {
            "name": self.name,
            "version": self.version,
            "domain": [value_type.to_data() for value_type in self.domain],
            "codomain": self.codomain.to_data(),
            "invariance": self.invariance.to_data(),
            "semantics": self.semantics.value,
            "affirmative_relations": sorted(
                relation.value for relation in self.affirmative_relations
            ),
            "parameter_names": sorted(self.parameter_names),
            "cost": self.cost,
            "source_digest": recomputed_source_digest,
            "operational_digest": self.operational_digest,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.contract_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def snapshot(self) -> "LegContractSnapshot":
        """Return the immutable, non-executable verifier representation."""

        return LegContractSnapshot.from_data(self.contract_data())


@dataclass(frozen=True, order=True)
class LegReference:
    """An immutable, static reference embedded in candidate IR."""

    name: str
    version: str
    contract_digest: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid leg reference name {self.name!r}")
        if not self.version:
            raise ValueError("leg reference version must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.contract_digest):
            raise ValueError("leg reference requires a lowercase sha256 digest")

    def to_data(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "contract_digest": self.contract_digest,
        }


@dataclass(frozen=True)
class LegContractSnapshot:
    """Complete static contract data needed for model-free verification.

    The callable is deliberately absent.  Its always-recomputed source digest
    and any separately declared operational/configuration digest remain part of
    the contract identity.
    """

    name: str
    version: str
    domain: tuple[ValueType, ...]
    codomain: ValueType
    affirmative_relations: frozenset[AffirmativeRelation]
    invariance: InvarianceContract
    semantics: LegSemantics
    parameter_names: frozenset[str]
    cost: int
    source_digest: str
    operational_digest: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid leg name {self.name!r}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.version):
            raise ValueError(f"invalid leg version {self.version!r}")
        if not isinstance(self.domain, tuple) or not self.domain or any(
            not isinstance(value_type, ValueType) for value_type in self.domain
        ):
            raise ValueError("leg snapshot domain must be an immutable typed tuple")
        if not isinstance(self.codomain, ValueType):
            raise TypeError("leg snapshot codomain must be a ValueType")
        if not isinstance(self.affirmative_relations, frozenset):
            raise TypeError("leg snapshot affirmative_relations must be a frozenset")
        if not self.affirmative_relations or any(
            not isinstance(relation, AffirmativeRelation)
            for relation in self.affirmative_relations
        ):
            raise ValueError("leg snapshot must declare affirmative relations")
        if self.codomain.unit is Unit.NONE:
            if self.affirmative_relations != frozenset(
                {AffirmativeRelation.PRESENT}
            ):
                raise ValueError(
                    "non-scalar legs admit only the affirmative present relation"
                )
        elif AffirmativeRelation.PRESENT in self.affirmative_relations:
            raise ValueError(
                "scalar legs must declare signed interval comparisons, not present"
            )
        if not isinstance(self.invariance, InvarianceContract):
            raise TypeError("leg snapshot invariance contract is malformed")
        if not isinstance(self.semantics, LegSemantics):
            raise TypeError("leg snapshot semantics are malformed")
        if not isinstance(self.parameter_names, frozenset):
            raise TypeError("leg snapshot parameter_names must be a frozenset")
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in self.parameter_names
        ):
            raise ValueError("leg parameter names must be lower snake case")
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, int)
            or self.cost <= 0
        ):
            raise ValueError("leg cost must be a positive integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError("leg source_digest must be a lowercase sha256")
        if self.operational_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.operational_digest
        ):
            raise ValueError("leg operational_digest must be a lowercase sha256")

    def contract_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "domain": [value_type.to_data() for value_type in self.domain],
            "codomain": self.codomain.to_data(),
            "invariance": self.invariance.to_data(),
            "semantics": self.semantics.value,
            "affirmative_relations": sorted(
                relation.value for relation in self.affirmative_relations
            ),
            "parameter_names": sorted(self.parameter_names),
            "cost": self.cost,
            "source_digest": self.source_digest,
            "operational_digest": self.operational_digest,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.contract_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LegContractSnapshot":
        required = {
            "name",
            "version",
            "domain",
            "codomain",
            "invariance",
            "semantics",
            "affirmative_relations",
            "parameter_names",
            "cost",
            "source_digest",
            "operational_digest",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise ValueError("leg contract snapshot fields differ from schema")
        raw_domain = data["domain"]
        raw_codomain = data["codomain"]
        raw_invariance = data["invariance"]
        if not isinstance(raw_domain, list):
            raise TypeError("leg snapshot domain must be a list")
        if not isinstance(raw_codomain, Mapping):
            raise TypeError("leg snapshot codomain must be an object")
        if not isinstance(raw_invariance, Mapping):
            raise TypeError("leg snapshot invariance must be an object")

        raw_relations = data["affirmative_relations"]
        if not isinstance(raw_relations, list) or any(
            not isinstance(item, str) for item in raw_relations
        ):
            raise TypeError("affirmative_relations must be a list of strings")
        if raw_relations != sorted(raw_relations) or len(raw_relations) != len(
            set(raw_relations)
        ):
            raise ValueError("affirmative_relations must be unique and sorted")

        raw_parameters = data["parameter_names"]
        if not isinstance(raw_parameters, list) or any(
            not isinstance(item, str) for item in raw_parameters
        ):
            raise TypeError("parameter_names must be a list of strings")
        if raw_parameters != sorted(raw_parameters) or len(raw_parameters) != len(
            set(raw_parameters)
        ):
            raise ValueError("parameter_names must be unique and sorted")

        name = data["name"]
        version = data["version"]
        semantics = data["semantics"]
        source_digest = data["source_digest"]
        operational_digest = data["operational_digest"]
        if not isinstance(name, str) or not isinstance(version, str):
            raise TypeError("leg snapshot name/version must be strings")
        if not isinstance(semantics, str) or not isinstance(source_digest, str):
            raise TypeError("leg snapshot semantics/source_digest must be strings")
        if operational_digest is not None and not isinstance(
            operational_digest, str
        ):
            raise TypeError("operational_digest must be a string or null")
        return cls(
            name=name,
            version=version,
            domain=tuple(ValueType.from_data(item) for item in raw_domain),
            codomain=ValueType.from_data(raw_codomain),
            affirmative_relations=frozenset(
                AffirmativeRelation(item) for item in raw_relations
            ),
            invariance=InvarianceContract.from_data(raw_invariance),
            semantics=LegSemantics(semantics),
            parameter_names=frozenset(raw_parameters),
            cost=data["cost"],
            source_digest=source_digest,
            operational_digest=operational_digest,
        )


@dataclass(frozen=True)
class RegistrySnapshot:
    """Sorted immutable registry sufficient for offline static validation."""

    contracts: tuple[LegContractSnapshot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contracts, tuple):
            raise TypeError("registry snapshot contracts must be an immutable tuple")
        if any(
            not isinstance(contract, LegContractSnapshot)
            for contract in self.contracts
        ):
            raise TypeError("registry snapshot contains a non-contract value")
        keys = [(contract.name, contract.version) for contract in self.contracts]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("registry snapshot contracts must be unique and sorted")

    @classmethod
    def from_data(cls, data: object) -> "RegistrySnapshot":
        if not isinstance(data, list) or any(
            not isinstance(item, Mapping) for item in data
        ):
            raise TypeError("registry snapshot must be a list of contract objects")
        return cls(tuple(LegContractSnapshot.from_data(item) for item in data))

    def to_data(self) -> list[dict[str, object]]:
        return [contract.contract_data() for contract in self.contracts]

    def digest(self) -> str:
        payload = json.dumps(
            self.to_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolve(self, reference: LegReference) -> LegContractSnapshot:
        matches = tuple(
            contract
            for contract in self.contracts
            if (contract.name, contract.version)
            == (reference.name, reference.version)
        )
        if len(matches) != 1:
            raise ContractViolation(
                f"unregistered leg {reference.name}@{reference.version}"
            )
        contract = matches[0]
        if contract.digest() != reference.contract_digest:
            raise ContractViolation(
                f"contract digest mismatch for {reference.name}@{reference.version}"
            )
        return contract


class ContractViolation(ValueError):
    """A candidate call does not match the verifier's frozen registry."""


class LegRegistry:
    """Verifier-owned registry; formulas can only call exact registered refs."""

    def __init__(self) -> None:
        self._contracts: dict[tuple[str, str], LegContract] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register(self, contract: LegContract) -> LegReference:
        if self._frozen:
            raise ContractViolation("cannot register a leg after registry freeze")
        key = (contract.name, contract.version)
        if key in self._contracts:
            raise ContractViolation(
                f"duplicate leg contract {contract.name}@{contract.version}"
            )
        self._contracts[key] = contract
        return self.reference(contract.name, contract.version)

    def freeze(self) -> "LegRegistry":
        self._frozen = True
        return self

    def reference(self, name: str, version: str) -> LegReference:
        try:
            contract = self._contracts[(name, version)]
        except KeyError as exc:
            raise ContractViolation(f"unregistered leg {name}@{version}") from exc
        return LegReference(name, version, contract.digest())

    def resolve(self, reference: LegReference) -> LegContract:
        try:
            contract = self._contracts[(reference.name, reference.version)]
        except KeyError as exc:
            raise ContractViolation(
                f"unregistered leg {reference.name}@{reference.version}"
            ) from exc
        if contract.digest() != reference.contract_digest:
            raise ContractViolation(
                f"contract digest mismatch for {reference.name}@{reference.version}"
            )
        return contract

    def contracts(self) -> tuple[LegContract, ...]:
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(
            tuple(contract.snapshot() for contract in self.contracts())
        )

    def digest(self) -> str:
        return self.snapshot().digest()

    def view(self) -> Mapping[tuple[str, str], LegContract]:
        return MappingProxyType(self._contracts)

    def invoke(
        self,
        reference: LegReference,
        arguments: tuple[TypedValue, ...],
        parameters: tuple[tuple[str, Literal], ...] = (),
    ) -> Evidence[TypedValue]:
        """Invoke one exact registered leg and enforce its typed boundary."""

        contract = self.resolve(reference)
        if len(arguments) != len(contract.domain):
            raise ContractViolation(
                f"{contract.name} expects {len(contract.domain)} arguments, "
                f"got {len(arguments)}"
            )
        for index, (argument, expected) in enumerate(
            zip(arguments, contract.domain, strict=True)
        ):
            if argument.type != expected:
                raise ContractViolation(
                    f"{contract.name} argument {index} has {argument.type}, "
                    f"expected {expected}"
                )
        names = [name for name, _ in parameters]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ContractViolation("leg parameters must be unique and sorted")
        if set(names) - contract.parameter_names:
            raise ContractViolation(
                "undeclared leg parameters: "
                + ", ".join(sorted(set(names) - contract.parameter_names))
            )
        if any(not _is_literal(value) for _, value in parameters):
            raise ContractViolation("leg parameters must be finite JSON literals")

        fallback_provenance = Provenance(
            producer=f"leg:{contract.name}",
            version=contract.version,
            method=contract.semantics.value,
            details=(("contract_digest", contract.digest()),),
        )
        try:
            result = contract.implementation(
                *(argument.value for argument in arguments), **dict(parameters)
            )
        except Exception as exc:  # noqa: BLE001 - hard disposition boundary.
            return Evidence.error(
                fallback_provenance,
                type(exc).__name__,
                str(exc) or repr(exc),
            )
        if not isinstance(result, Evidence):
            return Evidence.error(
                fallback_provenance,
                "ContractViolation",
                f"{contract.name} returned {type(result).__name__}, not Evidence",
            )

        if not result.is_present:
            return Evidence(
                disposition=result.disposition,
                provenance=result.provenance,
                uncertainty=result.uncertainty,
                certificate=result.certificate,
                reason=result.reason,
                error_type=result.error_type,
            )
        value = result.unwrap()
        if isinstance(value, TypedValue):
            if value.type != contract.codomain:
                return Evidence.error(
                    result.provenance,
                    "ContractViolation",
                    f"{contract.name} returned {value.type}, expected "
                    f"{contract.codomain}",
                )
            typed = value
        else:
            try:
                typed = TypedValue(contract.codomain, value)
            except (TypeError, ValueError) as exc:
                return Evidence.error(
                    result.provenance,
                    "ContractViolation",
                    str(exc),
                )
        if contract.codomain.unit is not Unit.NONE and result.uncertainty is not None:
            point = typed.value
            if isinstance(point, (int, float)) and not isinstance(point, bool):
                if not (
                    result.uncertainty.lower
                    <= point
                    <= result.uncertainty.upper
                ):
                    return Evidence.error(
                        result.provenance,
                        "ContractViolation",
                        f"{contract.name} scalar point {point} lies outside its "
                        f"uncertainty interval [{result.uncertainty.lower}, "
                        f"{result.uncertainty.upper}]",
                    )
        return Evidence.present(typed, result.provenance, result.uncertainty)
