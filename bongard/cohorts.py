"""Metadata-only planning of historically clean ShapeBongard cohorts.

The historical seed records two importantly different kinds of exposure:

* exact official task identities; and
* semantic generator families (Basic shapes and Abstract attributes/pairs).

This module never turns absence from either list into a claim that official
panel bytes were unseen.  It only plans conservative drilling cohorts from a
``ShapeBongardCorpus`` inventory and its ``SplitIndex``.  In particular, none
of the functions below opens, hashes, decodes, or otherwise reads a PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .corpus import BongardTask, EXPECTED_REGIME_COUNTS, ShapeBongardCorpus
from .historical_exposure import (
    HistoricalExposureSeed,
    load_historical_exposure,
)


SCHEMA = "gkm.bongard-cohort-report.v1"
SUMMARY_SCHEMA = "gkm.shape-bongard-v2-cohort-summary.v1"

# These are the twelve Freeform program families in ShapeBongard_V2.  A loose
# ``ff_nact.*`` regex would silently accept generator names outside the
# official corpus vocabulary.
OFFICIAL_FREEFORM_FAMILIES = (
    "nact2_5",
    "nact3_3",
    "nact3_4",
    "nact3_5",
    "nact4",
    "nact4_4",
    "nact4_5",
    "nact5",
    "nact6",
    "nact7",
    "nact8",
    "nact9",
)

CLEAN_COHORTS = ("drill", "dev", "sealed")
FAMILIES = ("ff", "bd", "hd")
BYTE_EXPOSURE_QUALIFICATION = (
    "semantic cleanliness and absence of an exact-task record do not certify "
    "that official panel bytes were unseen"
)


class CohortError(ValueError):
    """A task id or requested cohort scope is not unambiguous and canonical."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _address(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _line_digest(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ParsedOfficialTaskId:
    """A task id parsed against the frozen official semantic vocabulary.

    Parsing establishes a canonical identifier shape; membership in a
    particular archive is supplied separately by ``ShapeBongardCorpus`` and
    ``SplitIndex``.
    """

    task_id: str
    family: str
    concepts: tuple[str, ...]
    instance: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "concepts": list(self.concepts),
            "instance": self.instance,
        }


_TASK_ID_RE = re.compile(r"^(ff|bd|hd)_(.+)_([0-9]{4})$")


def _unique_concept_parse(body: str, known: set[str], *, family: str) -> tuple[str, ...]:
    candidates: list[tuple[str, ...]] = []
    if body in known:
        candidates.append((body,))
    for index, character in enumerate(body):
        if character != "-":
            continue
        left, right = body[:index], body[index + 1 :]
        if left in known and right in known:
            candidates.append((left, right))

    if not candidates:
        raise CohortError(f"unknown {family} concept expression in task id: {body!r}")
    if len(candidates) != 1:
        raise CohortError(f"ambiguous {family} concept expression in task id: {body!r}")
    return candidates[0]


def parse_official_task_id(
    task_id: str,
    seed: HistoricalExposureSeed | None = None,
) -> ParsedOfficialTaskId:
    """Parse one exact ShapeBongard_V2 identifier or fail closed.

    Basic and Abstract names are resolved by exact membership in the frozen
    generator vocabulary, rather than by guessing where a hyphen belongs.
    """

    if not isinstance(task_id, str) or not task_id:
        raise CohortError("task id must be a non-empty string")
    match = _TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise CohortError(f"malformed ShapeBongard task id: {task_id!r}")
    family, body, suffix = match.groups()

    if family == "ff":
        if body not in OFFICIAL_FREEFORM_FAMILIES:
            raise CohortError(f"unknown Freeform program family in task id: {body!r}")
        concepts = (body,)
    else:
        historical = seed if seed is not None else load_historical_exposure()
        if family == "bd":
            known = set(historical.basic_shape_families) | set(
                historical.unused_basic_shape_families
            )
            concepts = _unique_concept_parse(body, known, family="Basic")
        else:
            concepts = _unique_concept_parse(
                body,
                set(historical.abstract_attributes),
                family="Abstract",
            )
            if len(concepts) == 2 and concepts not in set(
                historical.admissible_abstract_pairs
            ):
                raise CohortError(
                    f"unknown or non-canonical Abstract pair in task id: {body!r}"
                )

    instance = int(suffix)
    valid_instance = (
        (family == "bd" and instance == 0)
        or (family == "hd" and 0 <= instance < 20)
        or (family == "ff" and 0 <= instance < 300)
    )
    if not valid_instance:
        raise CohortError(
            f"instance suffix is outside the official {family} range: {suffix!r}"
        )

    return ParsedOfficialTaskId(
        task_id=task_id,
        family=family,
        concepts=concepts,
        instance=instance,
    )


@dataclass(frozen=True)
class TaskCohortRecord:
    """Independent exact-identity and semantic exposure decisions for a task."""

    parsed: ParsedOfficialTaskId
    split: str | None
    regime: str | None
    exact_task_exposure: str
    exact_panel_record_count: int
    semantic_exposure: str
    semantic_cohort: str | None
    historically_clean: bool
    reason: str

    @property
    def task_id(self) -> str:
        return self.parsed.task_id

    @property
    def family(self) -> str:
        return self.parsed.family

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.parsed.to_dict(),
            "split": self.split,
            "regime": self.regime,
            "exact_official_task_exposure": self.exact_task_exposure,
            "exact_official_panel_record_count": self.exact_panel_record_count,
            "semantic_family_exposure": self.semantic_exposure,
            "semantic_cohort": self.semantic_cohort,
            "historically_clean": self.historically_clean,
            "reason": self.reason,
        }


def _basic_semantics(
    concepts: tuple[str, ...],
    seed: HistoricalExposureSeed,
) -> tuple[str, str | None, str]:
    exposed = set(seed.basic_shape_families)
    if any(concept in exposed for concept in concepts):
        return (
            "historically_exposed",
            None,
            "at least one named Basic shape family is historically exposed",
        )

    partitions = {
        "drill": set(seed.partition.drill),
        "dev": set(seed.partition.dev),
        "sealed": set(seed.partition.sealed),
    }
    hits = [
        name
        for name, members in partitions.items()
        if all(concept in members for concept in concepts)
    ]
    if len(hits) == 1:
        cohort = hits[0]
        return (
            "unused_family_partition",
            cohort,
            f"all named Basic shape families belong to the frozen {cohort} partition",
        )
    if any(concept not in set(seed.unused_basic_shape_families) for concept in concepts):
        # A validated seed should make this unreachable; retain a fail-closed
        # disposition if a caller supplies a hand-built seed.
        return (
            "not_certified_clean",
            None,
            "a named Basic shape family is outside the frozen exposure universe",
        )
    return (
        "mixed_unused_partitions",
        None,
        "the named Basic shape families cross frozen drill/dev/sealed partitions",
    )


def _abstract_semantics(
    concepts: tuple[str, ...],
    seed: HistoricalExposureSeed,
) -> tuple[str, str | None, str]:
    if len(concepts) == 1:
        return (
            "historically_exposed",
            None,
            "all singleton Abstract attributes are historically exposed",
        )

    pair = (concepts[0], concepts[1])
    if pair in set(seed.abstract_pairs):
        return (
            "historically_exposed",
            None,
            "the exact ordered Abstract pair is historically exposed",
        )
    partitions = {
        "drill": set(seed.abstract_partition.drill),
        "dev": set(seed.abstract_partition.dev),
        "sealed": set(seed.abstract_partition.sealed),
    }
    hits = [name for name, members in partitions.items() if pair in members]
    if len(hits) == 1:
        cohort = hits[0]
        return (
            "unused_abstract_pair",
            cohort,
            f"the exact ordered Abstract pair belongs to the frozen {cohort} partition",
        )
    # ``parse_official_task_id`` rejects pairs outside the admissible ordered
    # universe, while the validated seed partitions that universe into exposed
    # and unused pairs.  Retain a fail-closed result for hand-built seeds.
    return ("not_certified_clean", None, "the pair is absent from the frozen pair sets")


def classify_task(
    task_id: str,
    seed: HistoricalExposureSeed | None = None,
    *,
    split: str | None = None,
    regime: str | None = None,
) -> TaskCohortRecord:
    """Classify a task without consulting its panel paths or bytes."""

    historical = seed if seed is not None else load_historical_exposure()
    parsed = parse_official_task_id(task_id, historical)
    exact_task_exposure = (
        "recorded" if task_id in set(historical.exact_official_task_ids) else "not_recorded"
    )
    panel_prefix = f"{parsed.family}/{task_id}/"
    exact_panel_record_count = sum(
        panel_id.startswith(panel_prefix) for panel_id in historical.exact_official_panel_ids
    )

    if parsed.family == "bd":
        semantic_exposure, semantic_cohort, reason = _basic_semantics(
            parsed.concepts, historical
        )
    elif parsed.family == "hd":
        semantic_exposure, semantic_cohort, reason = _abstract_semantics(
            parsed.concepts, historical
        )
    else:
        semantic_exposure, semantic_cohort, reason = (
            "indeterminate",
            None,
            "Freeform semantic exposure has no certified unused-family partition",
        )

    historically_clean = (
        exact_task_exposure == "not_recorded"
        and exact_panel_record_count == 0
        and semantic_cohort in CLEAN_COHORTS
    )
    if semantic_cohort is not None and not historically_clean:
        reason += "; an exact official task or panel identity is historically recorded"

    return TaskCohortRecord(
        parsed=parsed,
        split=split,
        regime=regime,
        exact_task_exposure=exact_task_exposure,
        exact_panel_record_count=exact_panel_record_count,
        semantic_exposure=semantic_exposure,
        semantic_cohort=semantic_cohort,
        historically_clean=historically_clean,
        reason=reason,
    )


@dataclass(frozen=True)
class CohortReport:
    """A deterministic, pixel-free report over a scoped corpus inventory."""

    seed_digest: str
    split_index_digest: str
    scope: tuple[tuple[str, str | None], ...]
    records: tuple[TaskCohortRecord, ...]
    counts: tuple[tuple[str, int], ...]
    membership_digests: tuple[tuple[str, str], ...]
    inventory_digest: str
    digest: str

    @property
    def count_map(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.counts))

    @property
    def membership_digest_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.membership_digests))

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "qualification": BYTE_EXPOSURE_QUALIFICATION,
            "seed_digest": self.seed_digest,
            "split_index_digest": self.split_index_digest,
            "scope": dict(self.scope),
            "inventory_digest": self.inventory_digest,
            "counts": dict(self.counts),
            "membership_digests": dict(self.membership_digests),
            "tasks": [record.to_dict() for record in self.records],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["digest"] = self.digest
        return result


def _normalise_split(split: str | None) -> str | None:
    if split is None:
        return None
    if not isinstance(split, str) or not split:
        raise CohortError("split filter must be a non-empty string")
    candidate = split.upper()
    return candidate if candidate in EXPECTED_REGIME_COUNTS else split.lower()


def _validate_scope(
    *,
    split: str | None,
    family: str | None,
    cohort: str | None,
) -> tuple[str | None, str | None, str | None]:
    normalised_split = _normalise_split(split)
    if family is not None and family not in FAMILIES:
        raise CohortError(f"unknown ShapeBongard family filter: {family!r}")
    if cohort is not None and cohort not in CLEAN_COHORTS + ("clean",):
        raise CohortError(f"unknown clean cohort filter: {cohort!r}")
    return normalised_split, family, cohort


def _records_for_corpus(
    corpus: ShapeBongardCorpus,
    seed: HistoricalExposureSeed,
) -> tuple[TaskCohortRecord, ...]:
    records: list[TaskCohortRecord] = []
    for task in corpus.tasks:
        parsed = parse_official_task_id(task.task_id, seed)
        if parsed.family != task.family:
            raise CohortError(
                f"task {task.task_id!r} is stored under {task.family!r}, "
                f"but its id declares {parsed.family!r}"
            )
        assignment = corpus.split.assignment(task.task_id)
        records.append(
            classify_task(
                task.task_id,
                seed,
                split=assignment.split,
                regime=assignment.regime,
            )
        )
    return tuple(sorted(records, key=lambda record: record.task_id))


def _scope_records(
    records: tuple[TaskCohortRecord, ...],
    *,
    split: str | None,
    family: str | None,
    cohort: str | None,
) -> tuple[TaskCohortRecord, ...]:
    def included(record: TaskCohortRecord) -> bool:
        if split is not None and split not in {record.split, record.regime}:
            return False
        if family is not None and record.family != family:
            return False
        if cohort == "clean" and not record.historically_clean:
            return False
        if cohort in CLEAN_COHORTS and not (
            record.historically_clean and record.semantic_cohort == cohort
        ):
            return False
        return True

    return tuple(record for record in records if included(record))


def build_cohort_report(
    corpus: ShapeBongardCorpus,
    seed: HistoricalExposureSeed | None = None,
    *,
    split: str | None = None,
    family: str | None = None,
    cohort: str | None = None,
) -> CohortReport:
    """Build a deterministic report from task ids and split metadata only."""

    historical = seed if seed is not None else load_historical_exposure()
    split, family, cohort = _validate_scope(split=split, family=family, cohort=cohort)
    if split is not None and split not in corpus.split.canonical_groups:
        raise CohortError(f"unknown split or regime filter: {split!r}")
    records = _scope_records(
        _records_for_corpus(corpus, historical),
        split=split,
        family=family,
        cohort=cohort,
    )

    count_values: dict[str, int] = {
        "tasks": len(records),
        "historically_clean": sum(record.historically_clean for record in records),
        "exact_task_recorded": sum(
            record.exact_task_exposure == "recorded" for record in records
        ),
        "ff": sum(record.family == "ff" for record in records),
        "bd": sum(record.family == "bd" for record in records),
        "hd": sum(record.family == "hd" for record in records),
    }
    for name in CLEAN_COHORTS:
        count_values[name] = sum(
            record.historically_clean and record.semantic_cohort == name
            for record in records
        )
    semantic_names = sorted({record.semantic_exposure for record in records})
    for name in semantic_names:
        count_values[f"semantic:{name}"] = sum(
            record.semantic_exposure == name for record in records
        )

    membership: dict[str, tuple[str, ...]] = {
        "all": tuple(record.task_id for record in records),
        "historically_clean": tuple(
            record.task_id for record in records if record.historically_clean
        ),
        "exact_task_recorded": tuple(
            record.task_id
            for record in records
            if record.exact_task_exposure == "recorded"
        ),
    }
    for name in CLEAN_COHORTS:
        membership[name] = tuple(
            record.task_id
            for record in records
            if record.historically_clean and record.semantic_cohort == name
        )

    inventory_lines = tuple(
        "\t".join(
            (
                record.task_id,
                record.family,
                record.split or "-",
                record.regime or "-",
            )
        )
        for record in records
    )
    split_index_digest = _address(corpus.split.to_manifest_dict())
    scope = (("split", split), ("family", family), ("cohort", cohort))
    counts = tuple(sorted(count_values.items()))
    membership_digests = tuple(
        (name, _line_digest(task_ids)) for name, task_ids in sorted(membership.items())
    )
    inventory_digest = _line_digest(inventory_lines)

    content = {
        "schema": SCHEMA,
        "qualification": BYTE_EXPOSURE_QUALIFICATION,
        "seed_digest": historical.seed_digest,
        "split_index_digest": split_index_digest,
        "scope": dict(scope),
        "inventory_digest": inventory_digest,
        "counts": dict(counts),
        "membership_digests": dict(membership_digests),
        "tasks": [record.to_dict() for record in records],
    }
    return CohortReport(
        seed_digest=historical.seed_digest,
        split_index_digest=split_index_digest,
        scope=scope,
        records=records,
        counts=counts,
        membership_digests=membership_digests,
        inventory_digest=inventory_digest,
        digest=_address(content),
    )


def build_cohort_summary(
    report: CohortReport,
    *,
    release_descriptor_digest: str,
) -> dict[str, object]:
    """Build the canonical compact summary from one complete-corpus report."""

    if not isinstance(report, CohortReport):
        raise TypeError("report must be a CohortReport")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", release_descriptor_digest) is None:
        raise CohortError("release descriptor digest must be a prefixed SHA-256")
    if report.scope != (("split", None), ("family", None), ("cohort", None)):
        raise CohortError("canonical summary requires an unscoped cohort report")
    if report.count_map.get("tasks") != 12_000:
        raise CohortError("canonical summary requires the complete 12,000-task corpus")
    return {
        "schema": SUMMARY_SCHEMA,
        "release_descriptor_digest": release_descriptor_digest,
        "historical_seed_digest": report.seed_digest,
        "qualification": BYTE_EXPOSURE_QUALIFICATION,
        "counts": dict(report.counts),
        "membership_digests": dict(report.membership_digests),
        "cohort_report_digest": report.digest,
    }


def select_tasks(
    corpus: ShapeBongardCorpus,
    seed: HistoricalExposureSeed | None = None,
    *,
    split: str | None = None,
    family: str | None = None,
    cohort: str | None = "clean",
) -> tuple[BongardTask, ...]:
    """Select task handles by split/family/clean cohort without reading pixels."""

    report = build_cohort_report(
        corpus,
        seed,
        split=split,
        family=family,
        cohort=cohort,
    )
    return tuple(corpus.task(record.task_id) for record in report.records)


__all__ = [
    "BYTE_EXPOSURE_QUALIFICATION",
    "CLEAN_COHORTS",
    "CohortError",
    "CohortReport",
    "OFFICIAL_FREEFORM_FAMILIES",
    "ParsedOfficialTaskId",
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "TaskCohortRecord",
    "build_cohort_report",
    "build_cohort_summary",
    "classify_task",
    "parse_official_task_id",
    "select_tasks",
]
