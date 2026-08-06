"""Adversarial tests for the frozen Codex output-schema boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from typing import Any

import pytest

import bongard.transport as T


def _object_schema(
        property_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": (
                {"type": "string"}
                if property_schema is None else property_schema
            ),
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({}, "non-empty object"),
        ({"type": "string"}, "root must be a strict object"),
        (
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
                "minProperties": 1,
            },
            "unsupported keywords: minProperties",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer", "answer"],
                "additionalProperties": False,
            },
            "unique strings",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            "must equal its properties",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer", "undeclared"],
                "additionalProperties": False,
            },
            "must equal its properties",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": [7],
                "additionalProperties": False,
            },
            "unique strings",
        ),
        (_object_schema({"type": ["string", "null"]}), "not allowlisted"),
        (_object_schema({"type": "string", "pattern": ".*"}),
         "unsupported keywords: pattern"),
        (_object_schema({"type": "array"}), "invalid keywords"),
        (_object_schema({"anyOf": []}), "must have alternatives"),
        (_object_schema({}), "non-empty object"),
    ],
)
def test_strict_schema_rejects_non_dialect_shapes(
        schema: Mapping[str, Any], message: str) -> None:
    with pytest.raises(T.CodexProposerFailure, match=message):
        T.validate_codex_strict_output_schema(schema)


def test_strict_schema_rejects_non_mapping_root_and_non_string_keys() -> None:
    with pytest.raises(T.CodexProposerFailure, match="must be a mapping"):
        T.validate_codex_strict_output_schema("string")  # type: ignore[arg-type]

    schema = {
        "type": "object",
        "properties": {7: {"type": "string"}},
        "required": ["7"],
        "additionalProperties": False,
    }
    with pytest.raises(T.CodexProposerFailure, match="non-string key"):
        T.validate_codex_strict_output_schema(schema)


def test_strict_schema_keyword_allowlist_is_recursive_not_property_named() -> None:
    schema = {
        "type": "object",
        "properties": {
            "minProperties": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "array", "items": {"type": "integer"}},
                ]
            },
        },
        "required": ["minProperties"],
        "additionalProperties": False,
    }
    T.validate_codex_strict_output_schema(schema)


@pytest.mark.parametrize(
    "poison",
    [float("nan"), float("inf"), float("-inf")],
)
def test_strict_schema_rejects_non_finite_json(poison: float) -> None:
    with pytest.raises(T.CodexProposerFailure, match="finite"):
        T.validate_codex_strict_output_schema(
            _object_schema({"type": "number", "enum": [poison]}))


def test_strict_schema_rejects_oversized_canonical_bytes() -> None:
    schema = _object_schema()
    schema["description"] = "x" * (T.MAX_SCHEMA_UTF8_BYTES + 1)
    with pytest.raises(T.CodexProposerFailure, match="oversized"):
        T.validate_codex_strict_output_schema(schema)


class _OneReadMapping(Mapping[str, Any]):
    """Expose a different mapping after the first materialization."""

    def __init__(
            self, first: Mapping[str, Any], later: Mapping[str, Any]) -> None:
        self.first = dict(first)
        self.later = dict(later)
        self.materializations = 0

    def __getitem__(self, key: str) -> Any:
        return self.first[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.first)

    def __len__(self) -> int:
        return len(self.first)

    def items(self):
        self.materializations += 1
        source = self.first if self.materializations == 1 else self.later
        return source.items()


def test_schema_is_deep_frozen_once_before_validation_and_digest() -> None:
    flipping = _OneReadMapping(
        {"type": "string"},
        {"type": "string", "minProperties": 1},
    )
    caller = _object_schema(flipping)

    frozen, schema_bytes, schema_digest = (
        T._freeze_codex_strict_output_schema(caller)
    )
    assert flipping.materializations == 1
    assert frozen == _object_schema()
    assert schema_bytes == json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert schema_digest == hashlib.sha256(schema_bytes).hexdigest()

    # Neither a later custom-mapping view nor mutation of an ordinary nested
    # container can change what validation accepted or what Codex will see.
    assert dict(flipping.items())["minProperties"] == 1
    caller["required"].append("later")
    assert flipping.materializations == 2
    assert frozen == _object_schema()
    assert hashlib.sha256(schema_bytes).hexdigest() == schema_digest


def test_invalid_first_custom_mapping_view_fails_closed() -> None:
    flipping = _OneReadMapping(
        {"type": "string", "minProperties": 1},
        {"type": "string"},
    )
    with pytest.raises(
        T.CodexProposerFailure,
        match="unsupported keywords: minProperties",
    ):
        T.validate_codex_strict_output_schema(_object_schema(flipping))
    assert flipping.materializations == 1


def _nested_object_schema(schema_depth: int) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "string"}
    for _ in range(schema_depth - 1):
        node = {
            "type": "object",
            "properties": {"child": node},
            "required": ["child"],
            "additionalProperties": False,
        }
    return node


def test_strict_schema_enforces_ten_schema_levels() -> None:
    T.validate_codex_strict_output_schema(_nested_object_schema(10))
    with pytest.raises(T.CodexProposerFailure, match="exceeds depth 10"):
        T.validate_codex_strict_output_schema(_nested_object_schema(11))


def test_raw_depth_1500_fails_as_codex_failure_not_recursion_error() -> None:
    raw: dict[str, Any] = {"type": "string"}
    for _ in range(1_500):
        raw = {"child": raw}
    with pytest.raises(T.CodexProposerFailure, match="deeply nested"):
        T.validate_codex_strict_output_schema(raw)


class _HugeMapping(Mapping[str, Any]):
    def __init__(self) -> None:
        self.iterated = False

    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"huge mapping was indexed at {key!r}")

    def __iter__(self) -> Iterator[str]:
        self.iterated = True
        raise AssertionError("huge mapping was iterated")

    def __len__(self) -> int:
        return T.MAX_STRICT_SCHEMA_PROPERTIES + 1

    def items(self):
        self.iterated = True
        raise AssertionError("huge mapping items were materialized")


class _HugeSequence(Sequence[str]):
    def __init__(self) -> None:
        self.iterated = False

    def __getitem__(self, index: int) -> str:
        self.iterated = True
        raise AssertionError(f"huge sequence was indexed at {index}")

    def __len__(self) -> int:
        return T.MAX_STRICT_SCHEMA_PROPERTIES + 1


def test_huge_custom_mapping_and_sequence_reject_before_iteration() -> None:
    huge_mapping = _HugeMapping()
    schema = _object_schema()
    schema["properties"] = huge_mapping
    with pytest.raises(T.CodexProposerFailure, match="mapping.*oversized"):
        T.validate_codex_strict_output_schema(schema)
    assert not huge_mapping.iterated

    huge_sequence = _HugeSequence()
    schema = _object_schema()
    schema["required"] = huge_sequence
    with pytest.raises(T.CodexProposerFailure, match="sequence.*oversized"):
        T.validate_codex_strict_output_schema(schema)
    assert not huge_sequence.iterated


def test_strict_schema_enforces_property_enum_and_string_budgets() -> None:
    properties = {
        f"field_{index}": {"type": "string"}
        for index in range(T.MAX_STRICT_SCHEMA_PROPERTIES + 1)
    }
    with pytest.raises(T.CodexProposerFailure, match="oversized|property budget"):
        T.validate_codex_strict_output_schema(
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        )

    with pytest.raises(T.CodexProposerFailure, match="enum value budget"):
        T.validate_codex_strict_output_schema(
            _object_schema({
                "type": "integer",
                "enum": list(range(T.MAX_STRICT_SCHEMA_ENUM_VALUES + 1)),
            })
        )

    long_name = "x" * (T.MAX_STRICT_SCHEMA_STRING_CHARS + 1)
    with pytest.raises(T.CodexProposerFailure, match="string budget"):
        T.validate_codex_strict_output_schema(
            {
                "type": "object",
                "properties": {long_name: {"type": "string"}},
                "required": [long_name],
                "additionalProperties": False,
            }
        )

    enum_values = ["x" * 60 + str(index) for index in range(251)]
    with pytest.raises(T.CodexProposerFailure, match="large enum string budget"):
        T.validate_codex_strict_output_schema(
            _object_schema({"type": "string", "enum": enum_values})
        )
