"""Role-free raw inference custody for the passed skeleton-graph observer.

This module archives only content-addressed PNG observations and raw model
outputs.  It deliberately has no calibration, typed projection, support/query
assignment, task identity, polarity, formula, or truth interface.  Equal PNG
bytes share one row.  Fresh verification decodes and infers again; cold replay
parses and joins the archived canonical records with zero pixel or model calls.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_skeleton_graph_dev_command as core
from bongard import panel_action_count_skeleton_graph_passed_fit_protocol as passed_fit_module
from bongard.panel_action_count_skeleton_graph_passed_fit_protocol import (
    SkeletonGraphPassedFitOutcome,
    SkeletonGraphPassedFitProtocol,
    verify_skeleton_graph_passed_fit_protocol,
)


ROW_SCHEMA: Final = "gkm.bongard-skeleton-graph-raw-inference-row.v1"
BATCH_SCHEMA: Final = "gkm.bongard-skeleton-graph-raw-inference-batch.v1"
RECOMPUTE_RECEIPT_SCHEMA: Final = (
    "gkm.bongard-skeleton-graph-inference-recompute-receipt.v1"
)
FEATURE_DTYPE: Final = "<f4"
PROBABILITY_DTYPE: Final = "<f8"
FEATURE_SHAPE: Final = (112,)
DIRECT_PAIR_CLASS_ORDER: Final = core.OBSERVED_TRAIN_PAIR_CLASS_ORDER
CATALOG_CLASS_ORDER: Final = core.CATALOG_CLASS_ORDER
MAX_INPUT_OCCURRENCES: Final = 4_096
MAX_TOTAL_INPUT_BYTES: Final = 256 * 1024 * 1024
PINNED_PASSED_FIT_COMMIT: Final = "78aef7cb932ceb3dbb9006dadb71c6c1f1fa1d00"
PINNED_PASSED_FIT_SOURCE_SHA256: Final = (
    "c7cd9bd5abfdcbc8f846b45be3478c679d1ddd03e2380de4e9e0e95217eccc65"
)
PINNED_PASSED_FIT_ALGORITHM_DIGEST: Final = (
    "sha256:eacc49c3304cbd3b8de4a6bb6208e25fe7d3878ed6fae49ed52fa9bc73b9151d"
)

_INFERENCE_PROTOCOL_LITERAL: Final = MappingProxyType(
    {
        "schema": "gkm.bongard-skeleton-graph-raw-inference-algorithm.v1",
        "deduplication": "exact_png_sha256_one_row_per_unique_payload",
        "row_order": "ascending_png_sha256",
        "feature_dtype": FEATURE_DTYPE,
        "feature_shape": FEATURE_SHAPE,
        "direct_pair_class_order": DIRECT_PAIR_CLASS_ORDER,
        "catalog_class_order": CATALOG_CLASS_ORDER,
        "probability_dtype": PROBABILITY_DTYPE,
        "raw_outputs_only": True,
        "fresh_verification": "exact_pixel_reextract_and_model_reinfer",
        "cold_replay": "canonical_record_and_digest_join_only",
    }
)

_SHA_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_ADDRESS = re.compile(r"[0-9a-f]{64}\Z")


class SkeletonGraphInferenceCustodyError(RuntimeError):
    """Raw inference bytes, authority, model output, or replay differs."""


def source_sha256() -> str:
    """Return the import-time source address, rejecting post-import drift."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_preflight() -> None:
    source_sha256()
    if (
        passed_fit_module.source_sha256() != PINNED_PASSED_FIT_SOURCE_SHA256
        or passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
        != PINNED_PASSED_FIT_ALGORITHM_DIGEST
        or core.source_sha256()
        != passed_fit_module.PINNED_DEVELOPMENT_SOURCE_SHA256
        or core.config_digest()
        != passed_fit_module.PINNED_DEVELOPMENT_CONFIG_DIGEST
    ):
        raise SkeletonGraphInferenceCustodyError(
            "pinned passed-fit or development authority differs"
        )


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def algorithm_digest() -> str:
    """Address the complete raw-inference algorithm and its exact core."""

    _authority_preflight()
    return "sha256:" + canonical_digest(
        {
            "inference_protocol": _plain(_INFERENCE_PROTOCOL_LITERAL),
            "inference_source_sha256": source_sha256(),
            "core_source_sha256": core.source_sha256(),
            "core_config_digest": core.config_digest(),
            "passed_fit_authority_source_sha256": (
                "sha256:" + passed_fit_module.source_sha256()
            ),
            "passed_fit_algorithm_digest": (
                passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
            ),
            "passed_fit_implementation_commit": PINNED_PASSED_FIT_COMMIT,
        }
    )


def _sha_address(value: object, label: str) -> str:
    if type(value) is not str or _SHA_ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphInferenceCustodyError(f"{label} is not a sha256: address")
    return value


def _source_address(value: object, label: str) -> str:
    if type(value) is not str or _SOURCE_ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphInferenceCustodyError(f"{label} is not a source address")
    return value


def _exact_int(value: object, label: str, *, lower: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int or value < lower:
        raise SkeletonGraphInferenceCustodyError(f"{label} is not an exact integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkeletonGraphInferenceCustodyError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SkeletonGraphInferenceCustodyError(f"{label} is not finite")
    return result


def _fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise SkeletonGraphInferenceCustodyError(f"{label} fields differ")
    return dict(value)


def _record_digest(body: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_digest(_plain(body))


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = _plain(body)
    return {**value, "record_digest": _record_digest(value)}


def _raw_address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _array_digest(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json(
        {"dtype": array.dtype.str, "name": name, "shape": list(array.shape)}
    )
    return "sha256:" + hashlib.sha256(
        header + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _feature_vector(value: object) -> np.ndarray:
    if not isinstance(value, (tuple, list)) or len(value) != FEATURE_SHAPE[0]:
        raise SkeletonGraphInferenceCustodyError("feature values cardinality differs")
    numbers = [_number(item, "feature value") for item in value]
    result = np.ascontiguousarray(numbers, dtype=FEATURE_DTYPE)
    if result.shape != FEATURE_SHAPE or not np.isfinite(result).all():
        raise SkeletonGraphInferenceCustodyError("feature vector differs")
    return result


def _probability_vector(value: object, size: int, label: str) -> np.ndarray:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise SkeletonGraphInferenceCustodyError(f"{label} cardinality differs")
    result = np.ascontiguousarray(
        [_number(item, label) for item in value], dtype=PROBABILITY_DTYPE
    )
    if (
        not np.isfinite(result).all()
        or np.any(result < 0.0)
        or np.any(result > 1.0)
        or not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise SkeletonGraphInferenceCustodyError(f"{label} is not a probability vector")
    return result


def _canonical_record_from_bytes(
    raw: bytes,
    *,
    schema: str,
    expected_file_sha256: str,
    expected_record_digest: str,
    label: str,
) -> dict[str, Any]:
    if type(raw) is not bytes or not raw:
        raise SkeletonGraphInferenceCustodyError(f"{label} is not archived bytes")
    if _raw_address(raw) != _sha_address(expected_file_sha256, f"{label} file"):
        raise SkeletonGraphInferenceCustodyError(f"{label} file address differs")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SkeletonGraphInferenceCustodyError(f"cannot decode {label}: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise SkeletonGraphInferenceCustodyError(
            f"{label} is not canonical JSON plus newline"
        )
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        value.get("schema") != schema
        or digest != _sha_address(expected_record_digest, f"{label} record")
        or digest != _record_digest(body)
    ):
        raise SkeletonGraphInferenceCustodyError(
            f"{label} schema or record digest differs"
        )
    return value


@dataclass(frozen=True)
class SkeletonGraphRawInferenceRow:
    """One content-keyed feature vector and two uncalibrated probability rows."""

    png_sha256: str
    png_size_bytes: int
    occurrence_count: int
    feature_values: tuple[float, ...]
    feature_digest: str
    direct_pair_probabilities: tuple[float, ...]
    direct_pair_probability_digest: str
    catalog_probabilities: tuple[float, ...]
    catalog_probability_digest: str
    core_source_sha256: str
    core_config_digest: str
    model_file_sha256: str
    passed_fit_protocol_record_digest: str
    passed_fit_authority_source_sha256: str
    passed_fit_algorithm_digest: str
    inference_source_sha256: str
    inference_algorithm_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _authority_preflight()
        _sha_address(self.png_sha256, "PNG")
        _exact_int(self.png_size_bytes, "PNG size", lower=1)
        _exact_int(self.occurrence_count, "occurrence count", lower=1)
        feature = _feature_vector(self.feature_values)
        pair = _probability_vector(
            self.direct_pair_probabilities, len(DIRECT_PAIR_CLASS_ORDER), "direct pair"
        )
        catalog = _probability_vector(
            self.catalog_probabilities, len(CATALOG_CLASS_ORDER), "catalog"
        )
        if self.feature_digest != _array_digest("feature_vector_f32", feature):
            raise SkeletonGraphInferenceCustodyError("feature digest differs")
        if self.direct_pair_probability_digest != _array_digest(
            "direct_pair_probabilities_f64", pair
        ):
            raise SkeletonGraphInferenceCustodyError(
                "direct pair probability digest differs"
            )
        if self.catalog_probability_digest != _array_digest(
            "catalog_probabilities_f64", catalog
        ):
            raise SkeletonGraphInferenceCustodyError(
                "catalog probability digest differs"
            )
        _source_address(self.core_source_sha256, "core source")
        _sha_address(self.core_config_digest, "core config")
        _sha_address(self.model_file_sha256, "model file")
        _sha_address(self.passed_fit_protocol_record_digest, "passed-fit protocol")
        _sha_address(
            self.passed_fit_authority_source_sha256, "passed-fit authority source"
        )
        _sha_address(self.passed_fit_algorithm_digest, "passed-fit algorithm")
        _source_address(self.inference_source_sha256, "inference source")
        _sha_address(self.inference_algorithm_digest, "inference algorithm")
        if (
            self.core_source_sha256 != core.source_sha256()
            or self.core_config_digest != core.config_digest()
            or self.passed_fit_authority_source_sha256
            != "sha256:" + passed_fit_module.source_sha256()
            or self.passed_fit_algorithm_digest
            != passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
            or self.inference_source_sha256 != source_sha256()
            or self.inference_algorithm_digest != algorithm_digest()
            or self.record_digest != _record_digest(self._body())
        ):
            raise SkeletonGraphInferenceCustodyError("raw inference row digest differs")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": ROW_SCHEMA,
            "png_sha256": self.png_sha256,
            "png_size_bytes": self.png_size_bytes,
            "occurrence_count": self.occurrence_count,
            "feature_dtype": FEATURE_DTYPE,
            "feature_shape": FEATURE_SHAPE,
            "feature_values": self.feature_values,
            "feature_digest": self.feature_digest,
            "direct_pair_class_order": DIRECT_PAIR_CLASS_ORDER,
            "direct_pair_probability_dtype": PROBABILITY_DTYPE,
            "direct_pair_probabilities": self.direct_pair_probabilities,
            "direct_pair_probability_digest": self.direct_pair_probability_digest,
            "catalog_class_order": CATALOG_CLASS_ORDER,
            "catalog_probability_dtype": PROBABILITY_DTYPE,
            "catalog_probabilities": self.catalog_probabilities,
            "catalog_probability_digest": self.catalog_probability_digest,
            "core_source_sha256": self.core_source_sha256,
            "core_config_digest": self.core_config_digest,
            "model_file_sha256": self.model_file_sha256,
            "passed_fit_protocol_record_digest": self.passed_fit_protocol_record_digest,
            "passed_fit_authority_source_sha256": (
                self.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": self.passed_fit_algorithm_digest,
            "inference_source_sha256": self.inference_source_sha256,
            "inference_algorithm_digest": self.inference_algorithm_digest,
        }

    def to_data(self) -> dict[str, Any]:
        return {**_plain(self._body()), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphRawInferenceRow":
        expected = {
            "schema", "png_sha256", "png_size_bytes", "occurrence_count",
            "feature_dtype", "feature_shape", "feature_values", "feature_digest",
            "direct_pair_class_order", "direct_pair_probability_dtype",
            "direct_pair_probabilities", "direct_pair_probability_digest",
            "catalog_class_order", "catalog_probability_dtype", "catalog_probabilities",
            "catalog_probability_digest", "core_source_sha256", "core_config_digest",
            "model_file_sha256", "passed_fit_protocol_record_digest",
            "passed_fit_authority_source_sha256", "passed_fit_algorithm_digest",
            "inference_source_sha256", "inference_algorithm_digest", "record_digest",
        }
        raw = _fields(value, expected, "raw inference row")
        if (
            raw["schema"] != ROW_SCHEMA
            or raw["feature_dtype"] != FEATURE_DTYPE
            or raw["feature_shape"] != list(FEATURE_SHAPE)
            or raw["direct_pair_class_order"] != list(DIRECT_PAIR_CLASS_ORDER)
            or raw["direct_pair_probability_dtype"] != PROBABILITY_DTYPE
            or raw["catalog_class_order"] != list(CATALOG_CLASS_ORDER)
            or raw["catalog_probability_dtype"] != PROBABILITY_DTYPE
        ):
            raise SkeletonGraphInferenceCustodyError("raw inference row policy differs")
        feature = _feature_vector(raw["feature_values"])
        pair = _probability_vector(
            raw["direct_pair_probabilities"], len(DIRECT_PAIR_CLASS_ORDER), "direct pair"
        )
        catalog = _probability_vector(
            raw["catalog_probabilities"], len(CATALOG_CLASS_ORDER), "catalog"
        )
        result = cls(
            png_sha256=raw["png_sha256"],
            png_size_bytes=raw["png_size_bytes"],
            occurrence_count=raw["occurrence_count"],
            feature_values=tuple(float(item) for item in feature),
            feature_digest=raw["feature_digest"],
            direct_pair_probabilities=tuple(float(item) for item in pair),
            direct_pair_probability_digest=raw["direct_pair_probability_digest"],
            catalog_probabilities=tuple(float(item) for item in catalog),
            catalog_probability_digest=raw["catalog_probability_digest"],
            core_source_sha256=raw["core_source_sha256"],
            core_config_digest=raw["core_config_digest"],
            model_file_sha256=raw["model_file_sha256"],
            passed_fit_protocol_record_digest=raw[
                "passed_fit_protocol_record_digest"
            ],
            passed_fit_authority_source_sha256=raw[
                "passed_fit_authority_source_sha256"
            ],
            passed_fit_algorithm_digest=raw["passed_fit_algorithm_digest"],
            inference_source_sha256=raw["inference_source_sha256"],
            inference_algorithm_digest=raw["inference_algorithm_digest"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != raw:
            raise SkeletonGraphInferenceCustodyError("raw inference row is not canonical")
        return result

    @classmethod
    def from_arrays(
        cls,
        *,
        png_sha256: str,
        png_size_bytes: int,
        occurrence_count: int,
        feature: np.ndarray,
        direct_pair_probabilities: np.ndarray,
        catalog_probabilities: np.ndarray,
        bindings: Mapping[str, str],
    ) -> "SkeletonGraphRawInferenceRow":
        feature_value = _feature_vector(feature.tolist())
        pair_value = _probability_vector(
            direct_pair_probabilities.tolist(), len(DIRECT_PAIR_CLASS_ORDER), "direct pair"
        )
        catalog_value = _probability_vector(
            catalog_probabilities.tolist(), len(CATALOG_CLASS_ORDER), "catalog"
        )
        body = {
            "schema": ROW_SCHEMA,
            "png_sha256": png_sha256,
            "png_size_bytes": png_size_bytes,
            "occurrence_count": occurrence_count,
            "feature_dtype": FEATURE_DTYPE,
            "feature_shape": FEATURE_SHAPE,
            "feature_values": tuple(float(item) for item in feature_value),
            "feature_digest": _array_digest("feature_vector_f32", feature_value),
            "direct_pair_class_order": DIRECT_PAIR_CLASS_ORDER,
            "direct_pair_probability_dtype": PROBABILITY_DTYPE,
            "direct_pair_probabilities": tuple(float(item) for item in pair_value),
            "direct_pair_probability_digest": _array_digest(
                "direct_pair_probabilities_f64", pair_value
            ),
            "catalog_class_order": CATALOG_CLASS_ORDER,
            "catalog_probability_dtype": PROBABILITY_DTYPE,
            "catalog_probabilities": tuple(float(item) for item in catalog_value),
            "catalog_probability_digest": _array_digest(
                "catalog_probabilities_f64", catalog_value
            ),
            **dict(bindings),
        }
        return cls.from_data(_seal(body))


@dataclass(frozen=True)
class SkeletonGraphRawInferenceBatch:
    """A sorted, deduplicated collection of role-free raw inference rows."""

    rows: tuple[SkeletonGraphRawInferenceRow, ...]
    input_occurrence_count: int
    unique_png_count: int
    input_png_size_bytes: int
    unique_png_size_bytes: int
    feature_matrix_digest: str
    direct_pair_probability_matrix_digest: str
    catalog_probability_matrix_digest: str
    core_source_sha256: str
    core_config_digest: str
    model_file_sha256: str
    passed_fit_protocol_record_digest: str
    passed_fit_authority_source_sha256: str
    passed_fit_algorithm_digest: str
    inference_source_sha256: str
    inference_algorithm_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(type(row) is not SkeletonGraphRawInferenceRow for row in self.rows)
        ):
            raise SkeletonGraphInferenceCustodyError("raw inference rows differ")
        digests = tuple(row.png_sha256 for row in self.rows)
        if digests != tuple(sorted(digests)) or len(set(digests)) != len(digests):
            raise SkeletonGraphInferenceCustodyError(
                "raw inference rows are not uniquely content-sorted"
            )
        _exact_int(self.input_occurrence_count, "input occurrence count", lower=1)
        _exact_int(self.unique_png_count, "unique PNG count", lower=1)
        _exact_int(self.input_png_size_bytes, "input PNG byte count", lower=1)
        _exact_int(self.unique_png_size_bytes, "unique PNG byte count", lower=1)
        if (
            self.unique_png_count != len(self.rows)
            or self.input_occurrence_count
            != sum(row.occurrence_count for row in self.rows)
            or self.unique_png_size_bytes
            != sum(row.png_size_bytes for row in self.rows)
            or self.input_png_size_bytes
            != sum(row.png_size_bytes * row.occurrence_count for row in self.rows)
        ):
            raise SkeletonGraphInferenceCustodyError("raw inference batch counts differ")
        common = self._binding_values()
        for row in self.rows:
            if {
                "core_source_sha256": row.core_source_sha256,
                "core_config_digest": row.core_config_digest,
                "model_file_sha256": row.model_file_sha256,
                "passed_fit_protocol_record_digest": (
                    row.passed_fit_protocol_record_digest
                ),
                "passed_fit_authority_source_sha256": (
                    row.passed_fit_authority_source_sha256
                ),
                "passed_fit_algorithm_digest": row.passed_fit_algorithm_digest,
                "inference_source_sha256": row.inference_source_sha256,
                "inference_algorithm_digest": row.inference_algorithm_digest,
            } != common:
                raise SkeletonGraphInferenceCustodyError(
                    "raw inference row authority differs from batch"
                )
        feature, pair, catalog = self._matrices()
        if self.feature_matrix_digest != _array_digest(
            "raw_inference_feature_matrix_f32", feature
        ):
            raise SkeletonGraphInferenceCustodyError("feature matrix digest differs")
        if self.direct_pair_probability_matrix_digest != _array_digest(
            "raw_inference_direct_pair_probability_matrix_f64", pair
        ):
            raise SkeletonGraphInferenceCustodyError(
                "direct pair probability matrix digest differs"
            )
        if self.catalog_probability_matrix_digest != _array_digest(
            "raw_inference_catalog_probability_matrix_f64", catalog
        ):
            raise SkeletonGraphInferenceCustodyError(
                "catalog probability matrix digest differs"
            )
        _source_address(self.core_source_sha256, "core source")
        _sha_address(self.core_config_digest, "core config")
        _sha_address(self.model_file_sha256, "model file")
        _sha_address(self.passed_fit_protocol_record_digest, "passed-fit protocol")
        _sha_address(
            self.passed_fit_authority_source_sha256, "passed-fit authority source"
        )
        _sha_address(self.passed_fit_algorithm_digest, "passed-fit algorithm")
        _source_address(self.inference_source_sha256, "inference source")
        _sha_address(self.inference_algorithm_digest, "inference algorithm")
        if (
            self.core_source_sha256 != core.source_sha256()
            or self.core_config_digest != core.config_digest()
            or self.passed_fit_authority_source_sha256
            != "sha256:" + passed_fit_module.source_sha256()
            or self.passed_fit_algorithm_digest
            != passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
            or self.inference_source_sha256 != source_sha256()
            or self.inference_algorithm_digest != algorithm_digest()
        ):
            raise SkeletonGraphInferenceCustodyError(
                "raw inference live algorithm binding differs"
            )
        if self.record_digest != _record_digest(self._body()):
            raise SkeletonGraphInferenceCustodyError("raw inference batch digest differs")

    def _binding_values(self) -> dict[str, str]:
        return {
            "core_source_sha256": self.core_source_sha256,
            "core_config_digest": self.core_config_digest,
            "model_file_sha256": self.model_file_sha256,
            "passed_fit_protocol_record_digest": self.passed_fit_protocol_record_digest,
            "passed_fit_authority_source_sha256": (
                self.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": self.passed_fit_algorithm_digest,
            "inference_source_sha256": self.inference_source_sha256,
            "inference_algorithm_digest": self.inference_algorithm_digest,
        }

    def _matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature = np.ascontiguousarray(
            [row.feature_values for row in self.rows], dtype=FEATURE_DTYPE
        )
        pair = np.ascontiguousarray(
            [row.direct_pair_probabilities for row in self.rows],
            dtype=PROBABILITY_DTYPE,
        )
        catalog = np.ascontiguousarray(
            [row.catalog_probabilities for row in self.rows],
            dtype=PROBABILITY_DTYPE,
        )
        return feature, pair, catalog

    def _body(self) -> dict[str, Any]:
        return {
            "schema": BATCH_SCHEMA,
            "inference_protocol": _INFERENCE_PROTOCOL_LITERAL,
            "rows": tuple(row.to_data() for row in self.rows),
            "input_occurrence_count": self.input_occurrence_count,
            "unique_png_count": self.unique_png_count,
            "input_png_size_bytes": self.input_png_size_bytes,
            "unique_png_size_bytes": self.unique_png_size_bytes,
            "feature_matrix_digest": self.feature_matrix_digest,
            "direct_pair_probability_matrix_digest": (
                self.direct_pair_probability_matrix_digest
            ),
            "catalog_probability_matrix_digest": (
                self.catalog_probability_matrix_digest
            ),
            **self._binding_values(),
        }

    def to_data(self) -> dict[str, Any]:
        return {**_plain(self._body()), "record_digest": self.record_digest}

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_data()) + b"\n"

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[SkeletonGraphRawInferenceRow],
    ) -> "SkeletonGraphRawInferenceBatch":
        values = tuple(rows)
        if not values:
            raise SkeletonGraphInferenceCustodyError("raw inference batch is empty")
        first = values[0]
        bindings = {
            "core_source_sha256": first.core_source_sha256,
            "core_config_digest": first.core_config_digest,
            "model_file_sha256": first.model_file_sha256,
            "passed_fit_protocol_record_digest": (
                first.passed_fit_protocol_record_digest
            ),
            "passed_fit_authority_source_sha256": (
                first.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": first.passed_fit_algorithm_digest,
            "inference_source_sha256": first.inference_source_sha256,
            "inference_algorithm_digest": first.inference_algorithm_digest,
        }
        feature = np.ascontiguousarray(
            [row.feature_values for row in values], dtype=FEATURE_DTYPE
        )
        pair = np.ascontiguousarray(
            [row.direct_pair_probabilities for row in values], dtype=PROBABILITY_DTYPE
        )
        catalog = np.ascontiguousarray(
            [row.catalog_probabilities for row in values], dtype=PROBABILITY_DTYPE
        )
        body = {
            "schema": BATCH_SCHEMA,
            "inference_protocol": _INFERENCE_PROTOCOL_LITERAL,
            "rows": tuple(row.to_data() for row in values),
            "input_occurrence_count": sum(row.occurrence_count for row in values),
            "unique_png_count": len(values),
            "input_png_size_bytes": sum(
                row.png_size_bytes * row.occurrence_count for row in values
            ),
            "unique_png_size_bytes": sum(row.png_size_bytes for row in values),
            "feature_matrix_digest": _array_digest(
                "raw_inference_feature_matrix_f32", feature
            ),
            "direct_pair_probability_matrix_digest": _array_digest(
                "raw_inference_direct_pair_probability_matrix_f64", pair
            ),
            "catalog_probability_matrix_digest": _array_digest(
                "raw_inference_catalog_probability_matrix_f64", catalog
            ),
            **bindings,
        }
        return cls.from_data(_seal(body))

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphRawInferenceBatch":
        expected = {
            "schema", "inference_protocol", "rows", "input_occurrence_count",
            "unique_png_count", "input_png_size_bytes", "unique_png_size_bytes",
            "feature_matrix_digest", "direct_pair_probability_matrix_digest",
            "catalog_probability_matrix_digest", "core_source_sha256",
            "core_config_digest", "model_file_sha256",
            "passed_fit_protocol_record_digest", "passed_fit_authority_source_sha256",
            "passed_fit_algorithm_digest", "inference_source_sha256",
            "inference_algorithm_digest", "record_digest",
        }
        raw = _fields(value, expected, "raw inference batch")
        if (
            raw["schema"] != BATCH_SCHEMA
            or raw["inference_protocol"] != _plain(_INFERENCE_PROTOCOL_LITERAL)
            or not isinstance(raw["rows"], list)
        ):
            raise SkeletonGraphInferenceCustodyError("raw inference batch policy differs")
        result = cls(
            rows=tuple(SkeletonGraphRawInferenceRow.from_data(row) for row in raw["rows"]),
            input_occurrence_count=raw["input_occurrence_count"],
            unique_png_count=raw["unique_png_count"],
            input_png_size_bytes=raw["input_png_size_bytes"],
            unique_png_size_bytes=raw["unique_png_size_bytes"],
            feature_matrix_digest=raw["feature_matrix_digest"],
            direct_pair_probability_matrix_digest=raw[
                "direct_pair_probability_matrix_digest"
            ],
            catalog_probability_matrix_digest=raw[
                "catalog_probability_matrix_digest"
            ],
            core_source_sha256=raw["core_source_sha256"],
            core_config_digest=raw["core_config_digest"],
            model_file_sha256=raw["model_file_sha256"],
            passed_fit_protocol_record_digest=raw[
                "passed_fit_protocol_record_digest"
            ],
            passed_fit_authority_source_sha256=raw[
                "passed_fit_authority_source_sha256"
            ],
            passed_fit_algorithm_digest=raw["passed_fit_algorithm_digest"],
            inference_source_sha256=raw["inference_source_sha256"],
            inference_algorithm_digest=raw["inference_algorithm_digest"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != raw:
            raise SkeletonGraphInferenceCustodyError(
                "raw inference batch is not canonical"
            )
        return result

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        expected_file_sha256: str,
        expected_record_digest: str,
    ) -> "SkeletonGraphRawInferenceBatch":
        return cls.from_data(
            _canonical_record_from_bytes(
                raw,
                schema=BATCH_SCHEMA,
                expected_file_sha256=expected_file_sha256,
                expected_record_digest=expected_record_digest,
                label="raw inference batch",
            )
        )


_RECOMPUTE_ISSUANCE_TOKEN = object()


@dataclass(frozen=True)
class SkeletonGraphInferenceRecomputeReceipt:
    """Content address issued only after exact pixel/model recomputation."""

    raw_batch_file_sha256: str
    raw_batch_record_digest: str
    input_occurrence_count: int
    unique_png_count: int
    input_png_size_bytes: int
    unique_png_size_bytes: int
    feature_matrix_digest: str
    direct_pair_probability_matrix_digest: str
    catalog_probability_matrix_digest: str
    model_file_sha256: str
    passed_fit_protocol_record_digest: str
    passed_fit_authority_source_sha256: str
    passed_fit_algorithm_digest: str
    inference_source_sha256: str
    inference_algorithm_digest: str
    feature_extraction_calls: int
    model_prediction_api_calls: int
    estimator_predict_proba_calls: int
    exact_recompute: bool
    record_digest: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_batch_file_sha256, "raw batch file"),
            (self.raw_batch_record_digest, "raw batch record"),
            (self.feature_matrix_digest, "feature matrix"),
            (self.direct_pair_probability_matrix_digest, "direct pair matrix"),
            (self.catalog_probability_matrix_digest, "catalog matrix"),
            (self.model_file_sha256, "model file"),
            (self.passed_fit_protocol_record_digest, "passed-fit protocol"),
            (self.passed_fit_algorithm_digest, "passed-fit algorithm"),
            (self.inference_algorithm_digest, "inference algorithm"),
        ):
            _sha_address(value, label)
        _sha_address(
            self.passed_fit_authority_source_sha256, "passed-fit authority source"
        )
        _source_address(self.inference_source_sha256, "inference source")
        for value, label, lower in (
            (self.input_occurrence_count, "input occurrence count", 1),
            (self.unique_png_count, "unique PNG count", 1),
            (self.input_png_size_bytes, "input PNG byte count", 1),
            (self.unique_png_size_bytes, "unique PNG byte count", 1),
            (self.feature_extraction_calls, "feature extraction calls", 1),
            (self.model_prediction_api_calls, "model prediction API calls", 1),
            (self.estimator_predict_proba_calls, "estimator predict-proba calls", 1),
        ):
            _exact_int(value, label, lower=lower)
        if self.exact_recompute is not True:
            raise SkeletonGraphInferenceCustodyError("recompute receipt did not pass")
        if (
            self.feature_extraction_calls != self.unique_png_count
            or self.model_prediction_api_calls != 2
            or self.estimator_predict_proba_calls != 4
            or self.inference_source_sha256 != source_sha256()
            or self.inference_algorithm_digest != algorithm_digest()
            or self.record_digest != _record_digest(self._body())
        ):
            raise SkeletonGraphInferenceCustodyError("recompute receipt policy differs")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": RECOMPUTE_RECEIPT_SCHEMA,
            "raw_batch_file_sha256": self.raw_batch_file_sha256,
            "raw_batch_record_digest": self.raw_batch_record_digest,
            "input_occurrence_count": self.input_occurrence_count,
            "unique_png_count": self.unique_png_count,
            "input_png_size_bytes": self.input_png_size_bytes,
            "unique_png_size_bytes": self.unique_png_size_bytes,
            "feature_matrix_digest": self.feature_matrix_digest,
            "direct_pair_probability_matrix_digest": (
                self.direct_pair_probability_matrix_digest
            ),
            "catalog_probability_matrix_digest": (
                self.catalog_probability_matrix_digest
            ),
            "model_file_sha256": self.model_file_sha256,
            "passed_fit_protocol_record_digest": self.passed_fit_protocol_record_digest,
            "passed_fit_authority_source_sha256": (
                self.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": self.passed_fit_algorithm_digest,
            "inference_source_sha256": self.inference_source_sha256,
            "inference_algorithm_digest": self.inference_algorithm_digest,
            "feature_extraction_calls": self.feature_extraction_calls,
            "model_prediction_api_calls": self.model_prediction_api_calls,
            "estimator_predict_proba_calls": self.estimator_predict_proba_calls,
            "exact_recompute": self.exact_recompute,
        }

    def to_data(self) -> dict[str, Any]:
        return {**_plain(self._body()), "record_digest": self.record_digest}

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_data()) + b"\n"

    @classmethod
    def _issue_after_exact_recompute(
        cls,
        batch: SkeletonGraphRawInferenceBatch,
        *,
        issuance_token: object,
    ) -> "SkeletonGraphInferenceRecomputeReceipt":
        if issuance_token is not _RECOMPUTE_ISSUANCE_TOKEN:
            raise SkeletonGraphInferenceCustodyError(
                "recompute receipt issuance requires fresh verification"
            )
        body = {
            "schema": RECOMPUTE_RECEIPT_SCHEMA,
            "raw_batch_file_sha256": _raw_address(batch.to_bytes()),
            "raw_batch_record_digest": batch.record_digest,
            "input_occurrence_count": batch.input_occurrence_count,
            "unique_png_count": batch.unique_png_count,
            "input_png_size_bytes": batch.input_png_size_bytes,
            "unique_png_size_bytes": batch.unique_png_size_bytes,
            "feature_matrix_digest": batch.feature_matrix_digest,
            "direct_pair_probability_matrix_digest": (
                batch.direct_pair_probability_matrix_digest
            ),
            "catalog_probability_matrix_digest": (
                batch.catalog_probability_matrix_digest
            ),
            "model_file_sha256": batch.model_file_sha256,
            "passed_fit_protocol_record_digest": (
                batch.passed_fit_protocol_record_digest
            ),
            "passed_fit_authority_source_sha256": (
                batch.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": batch.passed_fit_algorithm_digest,
            "inference_source_sha256": batch.inference_source_sha256,
            "inference_algorithm_digest": batch.inference_algorithm_digest,
            "feature_extraction_calls": batch.unique_png_count,
            "model_prediction_api_calls": 2,
            "estimator_predict_proba_calls": 4,
            "exact_recompute": True,
        }
        return cls.from_data(_seal(body))

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphInferenceRecomputeReceipt":
        expected = {
            "schema", "raw_batch_file_sha256", "raw_batch_record_digest",
            "input_occurrence_count", "unique_png_count", "input_png_size_bytes",
            "unique_png_size_bytes", "feature_matrix_digest",
            "direct_pair_probability_matrix_digest",
            "catalog_probability_matrix_digest", "model_file_sha256",
            "passed_fit_protocol_record_digest", "passed_fit_authority_source_sha256",
            "passed_fit_algorithm_digest", "inference_source_sha256",
            "inference_algorithm_digest", "feature_extraction_calls",
            "model_prediction_api_calls", "estimator_predict_proba_calls",
            "exact_recompute", "record_digest",
        }
        raw = _fields(value, expected, "inference recompute receipt")
        if raw["schema"] != RECOMPUTE_RECEIPT_SCHEMA:
            raise SkeletonGraphInferenceCustodyError("recompute receipt schema differs")
        result = cls(
            raw_batch_file_sha256=raw["raw_batch_file_sha256"],
            raw_batch_record_digest=raw["raw_batch_record_digest"],
            input_occurrence_count=raw["input_occurrence_count"],
            unique_png_count=raw["unique_png_count"],
            input_png_size_bytes=raw["input_png_size_bytes"],
            unique_png_size_bytes=raw["unique_png_size_bytes"],
            feature_matrix_digest=raw["feature_matrix_digest"],
            direct_pair_probability_matrix_digest=raw[
                "direct_pair_probability_matrix_digest"
            ],
            catalog_probability_matrix_digest=raw[
                "catalog_probability_matrix_digest"
            ],
            model_file_sha256=raw["model_file_sha256"],
            passed_fit_protocol_record_digest=raw[
                "passed_fit_protocol_record_digest"
            ],
            passed_fit_authority_source_sha256=raw[
                "passed_fit_authority_source_sha256"
            ],
            passed_fit_algorithm_digest=raw["passed_fit_algorithm_digest"],
            inference_source_sha256=raw["inference_source_sha256"],
            inference_algorithm_digest=raw["inference_algorithm_digest"],
            feature_extraction_calls=raw["feature_extraction_calls"],
            model_prediction_api_calls=raw["model_prediction_api_calls"],
            estimator_predict_proba_calls=raw["estimator_predict_proba_calls"],
            exact_recompute=raw["exact_recompute"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != raw:
            raise SkeletonGraphInferenceCustodyError(
                "inference recompute receipt is not canonical"
            )
        return result

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        expected_file_sha256: str,
        expected_record_digest: str,
    ) -> "SkeletonGraphInferenceRecomputeReceipt":
        return cls.from_data(
            _canonical_record_from_bytes(
                raw,
                schema=RECOMPUTE_RECEIPT_SCHEMA,
                expected_file_sha256=expected_file_sha256,
                expected_record_digest=expected_record_digest,
                label="inference recompute receipt",
            )
        )


@dataclass(frozen=True)
class _AuthenticatedInferenceAuthority:
    passed_fit: SkeletonGraphPassedFitProtocol
    model: core.VerifiedDevelopmentModel
    bindings: Mapping[str, str]


def _authenticate_inference_authority(
    passed_fit: SkeletonGraphPassedFitOutcome,
    *,
    development_precommit_path: Path,
    development_result_path: Path,
    development_replay_path: Path,
    model_path: Path,
    feature_artifact_path: Path,
    prediction_artifact_path: Path,
) -> _AuthenticatedInferenceAuthority:
    _authority_preflight()
    if type(passed_fit) is not SkeletonGraphPassedFitProtocol:
        raise SkeletonGraphInferenceCustodyError(
            "raw inference requires an exact passed-fit protocol, never a GAP"
        )
    try:
        verified = verify_skeleton_graph_passed_fit_protocol(
            passed_fit,
            development_precommit_path=development_precommit_path,
            development_result_path=development_result_path,
            development_replay_path=development_replay_path,
            model_path=model_path,
            feature_artifact_path=feature_artifact_path,
            prediction_artifact_path=prediction_artifact_path,
            expected_record_digest=passed_fit.record_digest,
        )
    except Exception as exc:
        raise SkeletonGraphInferenceCustodyError(
            "passed-fit fresh verification failed"
        ) from exc
    if type(verified) is not SkeletonGraphPassedFitProtocol or verified != passed_fit:
        raise SkeletonGraphInferenceCustodyError("passed-fit verification differs")
    protocol_data = passed_fit.to_data()
    if (
        protocol_data.get("passed_fit_authority_source_sha256")
        != "sha256:" + passed_fit_module.source_sha256()
        or protocol_data.get("passed_fit_algorithm_digest")
        != passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
        or protocol_data.get("development_source_sha256") != core.source_sha256()
        or protocol_data.get("development_config_digest") != core.config_digest()
        or tuple(protocol_data.get("required_heads", ()))
        != ("direct_pair", "catalog_three_class")
        or tuple(protocol_data.get("observed_pair_class_order", ()))
        != DIRECT_PAIR_CLASS_ORDER
        or tuple(protocol_data.get("catalog_class_order", ()))
        != CATALOG_CLASS_ORDER
        or protocol_data.get("both_named_development_heads_passed") is not True
        or protocol_data.get("promoted_heads_exact") is not True
        or protocol_data.get("full_six_file_chain_verified") is not True
    ):
        raise SkeletonGraphInferenceCustodyError("passed-fit inference policy differs")
    try:
        model = core.load_verified_development_model(
            precommit_path=development_precommit_path,
            expected_precommit_record_digest=protocol_data[
                "development_precommit_record_digest"
            ],
            expected_result_record_digest=protocol_data[
                "development_result_record_digest"
            ],
            required_heads=("direct_pair", "catalog_three_class"),
        )
    except Exception as exc:
        raise SkeletonGraphInferenceCustodyError(
            "both-head development model load failed"
        ) from exc
    if (
        model.model_file_sha256 != protocol_data.get("model_file_sha256")
        or model.precommit_record_digest
        != protocol_data.get("development_precommit_record_digest")
        or model.result_record_digest
        != protocol_data.get("development_result_record_digest")
        or model.promoted_heads != ("direct_pair", "catalog_three_class")
    ):
        raise SkeletonGraphInferenceCustodyError("verified model lineage differs")
    bindings = MappingProxyType(
        {
            "core_source_sha256": core.source_sha256(),
            "core_config_digest": core.config_digest(),
            "model_file_sha256": model.model_file_sha256,
            "passed_fit_protocol_record_digest": passed_fit.record_digest,
            "passed_fit_authority_source_sha256": (
                "sha256:" + passed_fit_module.source_sha256()
            ),
            "passed_fit_algorithm_digest": (
                passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
            ),
            "inference_source_sha256": source_sha256(),
            "inference_algorithm_digest": algorithm_digest(),
        }
    )
    return _AuthenticatedInferenceAuthority(passed_fit, model, bindings)


def _deduplicate_png_payloads(
    png_payloads: Sequence[bytes],
) -> tuple[tuple[str, bytes, int], ...]:
    if (
        isinstance(png_payloads, (bytes, bytearray, str))
        or not isinstance(png_payloads, Sequence)
        or not png_payloads
        or len(png_payloads) > MAX_INPUT_OCCURRENCES
    ):
        raise SkeletonGraphInferenceCustodyError("PNG payload inventory differs")
    total = 0
    by_digest: dict[str, tuple[bytes, int]] = {}
    for raw in png_payloads:
        if type(raw) is not bytes or not raw or len(raw) > 16 * 1024 * 1024:
            raise SkeletonGraphInferenceCustodyError("PNG payload byte count differs")
        total += len(raw)
        if total > MAX_TOTAL_INPUT_BYTES:
            raise SkeletonGraphInferenceCustodyError("PNG input byte cap exceeded")
        digest = _raw_address(raw)
        previous = by_digest.get(digest)
        if previous is not None and previous[0] != raw:
            raise SkeletonGraphInferenceCustodyError("PNG digest collision detected")
        by_digest[digest] = (raw, 1 if previous is None else previous[1] + 1)
    return tuple(
        (digest, by_digest[digest][0], by_digest[digest][1])
        for digest in sorted(by_digest)
    )


def _infer_with_authority(
    authority: _AuthenticatedInferenceAuthority,
    png_payloads: Sequence[bytes],
) -> SkeletonGraphRawInferenceBatch:
    unique = _deduplicate_png_payloads(png_payloads)
    features = np.ascontiguousarray(
        [core.extract_feature_vector(raw) for _digest, raw, _count in unique],
        dtype=FEATURE_DTYPE,
    )
    if features.shape != (len(unique), FEATURE_SHAPE[0]):
        raise SkeletonGraphInferenceCustodyError("extracted feature matrix differs")
    pair = np.ascontiguousarray(
        authority.model.predict(head="direct_pair", features=features),
        dtype=PROBABILITY_DTYPE,
    )
    catalog = np.ascontiguousarray(
        authority.model.predict(head="catalog_three_class", features=features),
        dtype=PROBABILITY_DTYPE,
    )
    if (
        pair.shape != (len(unique), len(DIRECT_PAIR_CLASS_ORDER))
        or catalog.shape != (len(unique), len(CATALOG_CLASS_ORDER))
    ):
        raise SkeletonGraphInferenceCustodyError("raw model output shape differs")
    rows = tuple(
        SkeletonGraphRawInferenceRow.from_arrays(
            png_sha256=digest,
            png_size_bytes=len(raw),
            occurrence_count=count,
            feature=features[index],
            direct_pair_probabilities=pair[index],
            catalog_probabilities=catalog[index],
            bindings=authority.bindings,
        )
        for index, (digest, raw, count) in enumerate(unique)
    )
    return SkeletonGraphRawInferenceBatch.from_rows(rows)


def create_raw_inference_batch(
    *,
    passed_fit: SkeletonGraphPassedFitOutcome,
    png_payloads: Sequence[bytes],
    development_precommit_path: Path,
    development_result_path: Path,
    development_replay_path: Path,
    model_path: Path,
    feature_artifact_path: Path,
    prediction_artifact_path: Path,
) -> SkeletonGraphRawInferenceBatch:
    """Authenticate the passed fit and infer one deduplicated anonymous batch."""

    authority = _authenticate_inference_authority(
        passed_fit,
        development_precommit_path=development_precommit_path,
        development_result_path=development_result_path,
        development_replay_path=development_replay_path,
        model_path=model_path,
        feature_artifact_path=feature_artifact_path,
        prediction_artifact_path=prediction_artifact_path,
    )
    return _infer_with_authority(authority, png_payloads)


def fresh_verify_raw_inference_batch(
    archived: SkeletonGraphRawInferenceBatch,
    *,
    passed_fit: SkeletonGraphPassedFitOutcome,
    png_payloads: Sequence[bytes],
    development_precommit_path: Path,
    development_result_path: Path,
    development_replay_path: Path,
    model_path: Path,
    feature_artifact_path: Path,
    prediction_artifact_path: Path,
) -> SkeletonGraphInferenceRecomputeReceipt:
    """Re-extract and re-infer exact bytes before issuing a sealed receipt."""

    if type(archived) is not SkeletonGraphRawInferenceBatch:
        raise TypeError("fresh verification needs an exact raw inference batch")
    restored = SkeletonGraphRawInferenceBatch.from_data(archived.to_data())
    if restored != archived:
        raise SkeletonGraphInferenceCustodyError(
            "archived raw inference batch canonical replay differs"
        )
    authority = _authenticate_inference_authority(
        passed_fit,
        development_precommit_path=development_precommit_path,
        development_result_path=development_result_path,
        development_replay_path=development_replay_path,
        model_path=model_path,
        feature_artifact_path=feature_artifact_path,
        prediction_artifact_path=prediction_artifact_path,
    )
    reconstructed = _infer_with_authority(authority, png_payloads)
    if reconstructed != archived or reconstructed.to_data() != archived.to_data():
        raise SkeletonGraphInferenceCustodyError(
            "fresh raw inference recomputation differs"
        )
    return SkeletonGraphInferenceRecomputeReceipt._issue_after_exact_recompute(
        archived, issuance_token=_RECOMPUTE_ISSUANCE_TOKEN
    )


def cold_replay_raw_inference(
    *,
    raw_batch_bytes: bytes,
    recompute_receipt_bytes: bytes,
    expected_raw_batch_file_sha256: str,
    expected_raw_batch_record_digest: str,
    expected_recompute_receipt_file_sha256: str,
    expected_recompute_receipt_record_digest: str,
) -> dict[str, Any]:
    """Verify archived records and their join with zero pixel/model calls."""

    batch = SkeletonGraphRawInferenceBatch.from_bytes(
        raw_batch_bytes,
        expected_file_sha256=expected_raw_batch_file_sha256,
        expected_record_digest=expected_raw_batch_record_digest,
    )
    receipt = SkeletonGraphInferenceRecomputeReceipt.from_bytes(
        recompute_receipt_bytes,
        expected_file_sha256=expected_recompute_receipt_file_sha256,
        expected_record_digest=expected_recompute_receipt_record_digest,
    )
    if (
        receipt.raw_batch_file_sha256 != _raw_address(raw_batch_bytes)
        or receipt.raw_batch_record_digest != batch.record_digest
        or receipt.input_occurrence_count != batch.input_occurrence_count
        or receipt.unique_png_count != batch.unique_png_count
        or receipt.input_png_size_bytes != batch.input_png_size_bytes
        or receipt.unique_png_size_bytes != batch.unique_png_size_bytes
        or receipt.feature_matrix_digest != batch.feature_matrix_digest
        or receipt.direct_pair_probability_matrix_digest
        != batch.direct_pair_probability_matrix_digest
        or receipt.catalog_probability_matrix_digest
        != batch.catalog_probability_matrix_digest
        or receipt.model_file_sha256 != batch.model_file_sha256
        or receipt.passed_fit_protocol_record_digest
        != batch.passed_fit_protocol_record_digest
        or receipt.passed_fit_authority_source_sha256
        != batch.passed_fit_authority_source_sha256
        or receipt.passed_fit_algorithm_digest != batch.passed_fit_algorithm_digest
        or receipt.inference_source_sha256 != batch.inference_source_sha256
        or receipt.inference_algorithm_digest != batch.inference_algorithm_digest
    ):
        raise SkeletonGraphInferenceCustodyError(
            "cold replay batch/receipt join differs"
        )
    return {
        "raw_batch_file_sha256": _raw_address(raw_batch_bytes),
        "raw_batch_record_digest": batch.record_digest,
        "recompute_receipt_file_sha256": _raw_address(recompute_receipt_bytes),
        "recompute_receipt_record_digest": receipt.record_digest,
        "unique_png_count": batch.unique_png_count,
        "input_occurrence_count": batch.input_occurrence_count,
        "pixel_reads": 0,
        "feature_extraction_calls": 0,
        "model_prediction_api_calls": 0,
        "estimator_predict_proba_calls": 0,
        "canonical_records_exact": True,
        "recompute_receipt_join_exact": True,
    }


__all__ = (
    "BATCH_SCHEMA",
    "CATALOG_CLASS_ORDER",
    "DIRECT_PAIR_CLASS_ORDER",
    "RECOMPUTE_RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "SkeletonGraphInferenceCustodyError",
    "SkeletonGraphInferenceRecomputeReceipt",
    "SkeletonGraphRawInferenceBatch",
    "SkeletonGraphRawInferenceRow",
    "algorithm_digest",
    "cold_replay_raw_inference",
    "create_raw_inference_batch",
    "fresh_verify_raw_inference_batch",
    "source_sha256",
)
