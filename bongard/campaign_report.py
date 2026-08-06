"""Tamper-evident aggregate reporting for small official Bongard campaigns.

This module summarizes *engineering smoke* attempts.  It intentionally keeps
stage yield separate from evaluation performance: when no attempt released a
query, the report contains no query- or image-accuracy metric.

Run records are supplied as already-decoded JSON.  Their byte SHA-256 and the
scope/disposition of any out-of-band cold verification are explicit inputs;
neither is inferred from a decoded value.  The builder independently checks
the record digest, exposure chain, support counts, receipt digests, and the
campaign-wide corpus/release/model/launcher/resolver identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from bongard.artifacts import canonical_digest
from bongard.exposure import ExposureEvent, ExposureIntegrityError
from bongard.release import OfficialReleaseDescriptor, ReleaseIdentityError
from bongard.transport import (
    CODEX_RECEIPT_SCHEMA,
    CodexProposerFailure,
    validate_codex_receipt,
)


CAMPAIGN_REPORT_SCHEMA = "gkm.bongard-headless-smoke-campaign-report.v1"
OUTER_RUN_SCHEMA = "gkm.bongard-episode-run.v5"
CAMPAIGN_TYPE = "engineering_smoke_infrastructure_and_support_fit"

UNVERIFIED = "unverified"
VERIFIED = "verified"
INFRASTRUCTURE_SCOPE = "infrastructure-failure-record-integrity-only"
EXACT_SUPPORT_SCOPE = (
    "exact-official-support-rejection-byte-preimage-gate-replay"
)

_ADDRESS_PREFIX = "sha256:"
_HEX_DIGITS = frozenset("0123456789abcdef")
_RECEIPT_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
_SUPPORT_DISPOSITIONS = frozenset(
    {"present", "certified_absent", "indeterminate", "error"}
)
_OUTER_RUN_FIELDS = frozenset(
    {
        "schema",
        "corpus_manifest_digest",
        "split_source_digest",
        "official_release",
        "plan",
        "episode",
        "vision",
        "run_archive",
        "support_commitment",
        "exposure",
        "record_digest",
    }
)


class CampaignReportError(ValueError):
    """A campaign input cannot support the claimed aggregate report."""


@dataclass(frozen=True, slots=True)
class CampaignRunInput:
    """One decoded run and its externally established byte/verification facts."""

    record: Mapping[str, Any]
    file_sha256: str
    verification_disposition: str
    verification_scope: str


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Canonical, content-addressed campaign report."""

    content: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return _ADDRESS_PREFIX + canonical_digest(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {**dict(self.content), "digest": self.digest}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CampaignReportError(f"{label} must be a JSON object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CampaignReportError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignReportError(f"{label} must be a non-negative integer")
    return value


def _hex(value: object, label: str, *, prefixed: bool = False) -> str:
    text = _text(value, label)
    payload = text.removeprefix(_ADDRESS_PREFIX) if prefixed else text
    if (
        len(payload) != 64
        or any(character not in _HEX_DIGITS for character in payload)
        or (prefixed and not text.startswith(_ADDRESS_PREFIX))
    ):
        form = "sha256: content address" if prefixed else "lowercase SHA-256"
        raise CampaignReportError(f"{label} must be a {form}")
    return text


def _single_identity(values: Sequence[str], label: str) -> str:
    unique = set(values)
    if len(unique) != 1:
        raise CampaignReportError(
            f"campaign {label} differs across runs: {sorted(unique)}"
        )
    return values[0]


def _walk_receipts(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("schema") == CODEX_RECEIPT_SCHEMA:
            yield value
        for child in value.values():
            yield from _walk_receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_receipts(child)


def _validated_unique_receipts(
    record: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for receipt in _walk_receipts(record):
        try:
            validate_codex_receipt(receipt)
        except CodexProposerFailure as exc:
            raise CampaignReportError(
                f"{label} has an invalid Codex receipt: {exc}"
            ) from exc
        digest = receipt["receipt_digest"]
        previous = result.setdefault(digest, receipt)
        if dict(previous) != dict(receipt):
            raise CampaignReportError(
                f"{label} reuses receipt digest {digest} for unequal receipts"
            )
    return result


def _support_counts(
    record: Mapping[str, Any], label: str
) -> tuple[dict[str, int], str] | None:
    vision = _mapping(record.get("vision"), f"{label}.vision")
    gate_value = vision.get("support_gate")
    if gate_value is None:
        return None
    gate = _mapping(gate_value, f"{label}.vision.support_gate")
    entries_value = gate.get("ordered_entries")
    if not isinstance(entries_value, list) or len(entries_value) != 12:
        raise CampaignReportError(f"{label} support replay must contain 12 entries")

    support = _mapping(
        record.get("support_commitment"), f"{label}.support_commitment"
    )
    support_items = support.get("support")
    if not isinstance(support_items, list) or len(support_items) != 12:
        raise CampaignReportError(f"{label} support commitment must contain 12 panels")
    committed: list[tuple[str, bool]] = []
    for index, item_value in enumerate(support_items):
        item = _mapping(item_value, f"{label}.support[{index}]")
        panel = _mapping(item.get("panel"), f"{label}.support[{index}].panel")
        slot_id = _text(panel.get("blob_id"), f"{label}.support[{index}].blob_id")
        positive = item.get("positive")
        if not isinstance(positive, bool):
            raise CampaignReportError(
                f"{label}.support[{index}].positive must be Boolean"
            )
        committed.append((slot_id, positive))

    replayed: list[tuple[str, bool]] = []
    counts = {
        "forward_matches": 0,
        "reverse_matches": 0,
        "present": 0,
        "nonmatch": 0,
        "indeterminate": 0,
        "error": 0,
        "transport_attempts": 0,
    }
    observer_receipts: set[str] = set()
    for index, entry_value in enumerate(entries_value):
        entry = _mapping(entry_value, f"{label}.support_gate[{index}]")
        slot_id = _text(entry.get("slot_id"), f"{label}.support_gate[{index}].slot_id")
        positive = entry.get("positive")
        if not isinstance(positive, bool):
            raise CampaignReportError(
                f"{label}.support_gate[{index}].positive must be Boolean"
            )
        replayed.append((slot_id, positive))
        evidence = _mapping(
            entry.get("evidence"), f"{label}.support_gate[{index}].evidence"
        )
        disposition = evidence.get("disposition")
        if disposition not in _SUPPORT_DISPOSITIONS:
            raise CampaignReportError(
                f"{label}.support_gate[{index}] has invalid disposition {disposition!r}"
            )
        count_key = "nonmatch" if disposition == "certified_absent" else disposition
        counts[count_key] += 1
        if (positive and disposition == "present") or (
            not positive and disposition == "certified_absent"
        ):
            counts["forward_matches"] += 1
        if (positive and disposition == "certified_absent") or (
            not positive and disposition == "present"
        ):
            counts["reverse_matches"] += 1
        attempted = entry.get("transport_attempted")
        if not isinstance(attempted, bool):
            raise CampaignReportError(
                f"{label}.support_gate[{index}].transport_attempted must be Boolean"
            )
        counts["transport_attempts"] += int(attempted)
        artifact = _mapping(
            entry.get("observer_artifact"),
            f"{label}.support_gate[{index}].observer_artifact",
        )
        receipt = _mapping(
            artifact.get("receipt"),
            f"{label}.support_gate[{index}].observer_artifact.receipt",
        )
        observer_receipts.add(
            _hex(receipt.get("receipt_digest"), "observer receipt digest")
        )
    if replayed != committed:
        raise CampaignReportError(
            f"{label} support replay order/labels differ from its commitment"
        )
    if len(observer_receipts) != 12:
        raise CampaignReportError(
            f"{label} support replay must use 12 distinct observer receipts"
        )

    archived_counts = _mapping(gate.get("counts"), f"{label}.support_gate.counts")
    if dict(archived_counts) != counts:
        raise CampaignReportError(
            f"{label} support counts do not reproduce from ordered entries"
        )
    gate_digest = _hex(gate.get("gate_digest"), f"{label}.support_gate.gate_digest")
    gate_content = {key: value for key, value in gate.items() if key != "gate_digest"}
    if canonical_digest(gate_content) != gate_digest:
        raise CampaignReportError(f"{label} support gate digest does not reproduce")
    result = _text(gate.get("result"), f"{label}.support_gate.result")
    return counts, result


def _verification_is_compatible(
    item: CampaignRunInput,
    *,
    status: str,
    support_replayed: bool,
    support_result: str | None,
    failure: Mapping[str, Any],
    vision: Mapping[str, Any],
    label: str,
) -> bool:
    pair = (item.verification_disposition, item.verification_scope)
    if pair == (UNVERIFIED, INFRASTRUCTURE_SCOPE):
        if (
            status != "proposal_error"
            or support_replayed
            or failure.get("stage") != "proposal"
            or failure.get("error_type") != "CodexProposerFailure"
            or vision.get("proposal") is not None
            or vision.get("rejected_proposal_attempt") is not None
        ):
            raise CampaignReportError(
                f"{label} infrastructure-only verification is incompatible with its run"
            )
        return False
    if pair == (VERIFIED, EXACT_SUPPORT_SCOPE):
        if (
            status != "support_rejected"
            or not support_replayed
            or support_result != "unsupported"
            or failure.get("stage") != "support_gate"
            or failure.get("error_type") != "SupportGateRejected"
            or failure.get("reason") != "unsupported"
        ):
            raise CampaignReportError(
                f"{label} exact support verification is incompatible with its run"
            )
        return True
    raise CampaignReportError(
        f"{label} has unsupported verification disposition/scope {pair!r}"
    )


def build_campaign_report(
    runs: Sequence[CampaignRunInput],
    *,
    campaign_id: str,
) -> CampaignReport:
    """Validate and aggregate a bounded sequence of official run records."""

    campaign_id = _text(campaign_id, "campaign_id")
    if not runs:
        raise CampaignReportError("campaign must contain at least one run")

    prepared: list[dict[str, Any]] = []
    for index, item in enumerate(runs):
        if not isinstance(item, CampaignRunInput):
            raise CampaignReportError(f"runs[{index}] must be CampaignRunInput")
        label = f"runs[{index}]"
        record = _mapping(item.record, f"{label}.record")
        if record.get("schema") != OUTER_RUN_SCHEMA:
            raise CampaignReportError(f"{label} is not a current official run record")
        if set(record) != _OUTER_RUN_FIELDS:
            raise CampaignReportError(f"{label} outer run fields differ from schema")
        file_sha256 = _hex(item.file_sha256, f"{label}.file_sha256", prefixed=True)
        embedded_digest = _hex(record.get("record_digest"), f"{label}.record_digest")
        content = {
            key: value for key, value in record.items() if key != "record_digest"
        }
        if canonical_digest(content) != embedded_digest:
            raise CampaignReportError(f"{label} record_digest does not reproduce")

        plan = _mapping(record.get("plan"), f"{label}.plan")
        episode = _mapping(record.get("episode"), f"{label}.episode")
        exposure = _mapping(record.get("exposure"), f"{label}.exposure")
        vision = _mapping(record.get("vision"), f"{label}.vision")
        task_id = _text(plan.get("task_id"), f"{label}.plan.task_id")
        run_id = _text(plan.get("run_id"), f"{label}.plan.run_id")
        plan_digest = canonical_digest(plan)
        for owner, value in (
            ("episode", episode.get("plan_digest")),
            ("exposure", exposure.get("plan_digest")),
        ):
            if value != plan_digest:
                raise CampaignReportError(f"{label} {owner} does not bind the plan")
        if episode.get("task_id") != task_id or exposure.get("task_id") != task_id:
            raise CampaignReportError(f"{label} task identity differs across layers")
        if episode.get("run_id") != run_id:
            raise CampaignReportError(f"{label} run identity differs across layers")

        corpus = _hex(
            record.get("corpus_manifest_digest"),
            f"{label}.corpus_manifest_digest",
            prefixed=True,
        )
        if plan.get("corpus_digest") != corpus.removeprefix(_ADDRESS_PREFIX):
            raise CampaignReportError(f"{label} plan corpus identity differs")
        if exposure.get("corpus_manifest_digest") != corpus:
            raise CampaignReportError(f"{label} exposure corpus identity differs")
        release_data = _mapping(
            record.get("official_release"), f"{label}.official_release"
        )
        try:
            release = OfficialReleaseDescriptor.from_dict(release_data)
        except (ReleaseIdentityError, TypeError, ValueError) as exc:
            raise CampaignReportError(
                f"{label} official release is invalid: {exc}"
            ) from exc
        if release.corpus_manifest_sha256 != corpus:
            raise CampaignReportError(
                f"{label} release and run corpus identities differ"
            )
        if record.get("split_source_digest") != release.split_sha256:
            raise CampaignReportError(
                f"{label} release and run split identities differ"
            )

        event_value = _mapping(exposure.get("event"), f"{label}.exposure.event")
        try:
            event = ExposureEvent.from_dict(event_value)
        except (ExposureIntegrityError, TypeError, ValueError) as exc:
            raise CampaignReportError(
                f"{label} exposure event is invalid: {exc}"
            ) from exc
        if exposure.get("event_digest") != event.digest:
            raise CampaignReportError(f"{label} exposure event digest differs")
        if event.task_ids != (task_id,) or event.panel_ids:
            raise CampaignReportError(
                f"{label} exposure event must name exactly its task and no query panels"
            )
        before_count = _integer(
            exposure.get("ledger_before_event_count"), f"{label}.before_event_count"
        )
        after_count = _integer(
            exposure.get("ledger_after_event_count"), f"{label}.after_event_count"
        )
        if event.sequence != before_count or after_count != before_count + 1:
            raise CampaignReportError(f"{label} exposure event count is not one-step")
        before_digest = _hex(
            exposure.get("ledger_before_digest"),
            f"{label}.ledger_before_digest",
            prefixed=True,
        )
        after_digest = _hex(
            exposure.get("ledger_after_digest"),
            f"{label}.ledger_after_digest",
            prefixed=True,
        )
        if exposure.get("successor_filename") != (
            after_digest.removeprefix(_ADDRESS_PREFIX) + ".exposure.json"
        ):
            raise CampaignReportError(f"{label} successor filename differs")
        unseen = _mapping(
            exposure.get("semantic_unseen_receipt"),
            f"{label}.semantic_unseen_receipt",
        )
        if unseen.get("ledger_digest") != before_digest:
            raise CampaignReportError(
                f"{label} semantic-unseen receipt does not bind the predecessor"
            )

        phases_value = episode.get("phases")
        if not isinstance(phases_value, list) or any(
            not isinstance(phase, str) for phase in phases_value
        ):
            raise CampaignReportError(f"{label}.episode.phases must be a string list")
        phases = tuple(phases_value)
        status = _text(episode.get("status"), f"{label}.episode.status")
        failure = _mapping(episode.get("failure"), f"{label}.episode.failure")
        proposer_success = vision.get("proposal") is not None
        support_data = _support_counts(record, label)
        support_replayed = support_data is not None
        if proposer_success != ("proposal_fixed" in phases):
            raise CampaignReportError(f"{label} proposer success markers disagree")
        if support_replayed != ("support_gate_replayed" in phases):
            raise CampaignReportError(f"{label} support replay markers disagree")
        support_result = support_data[1] if support_data is not None else None
        support_pass = support_result == "aligned"
        query_release = "query_released" in phases
        complete = status == "complete"
        if complete and not query_release:
            raise CampaignReportError(f"{label} complete status lacks query release")
        verified_rejection = _verification_is_compatible(
            item,
            status=status,
            support_replayed=support_replayed,
            support_result=support_result,
            failure=failure,
            vision=vision,
            label=label,
        )

        receipts = _validated_unique_receipts(vision, label)
        expected_receipts = int(proposer_success) + (12 if support_replayed else 0)
        if len(receipts) != expected_receipts:
            raise CampaignReportError(
                f"{label} archives {len(receipts)} unique successful receipts; "
                f"expected {expected_receipts}"
            )
        model = _text(exposure.get("model"), f"{label}.exposure.model")
        if event.actor != model:
            raise CampaignReportError(f"{label} exposure actor differs from model")
        if any(receipt["requested_model"] != model for receipt in receipts.values()):
            raise CampaignReportError(f"{label} receipt model differs from exposure")

        prepared.append(
            {
                "sort_key": before_count,
                "record": record,
                "file_sha256": file_sha256,
                "record_digest": embedded_digest,
                "task_id": task_id,
                "run_id": run_id,
                "plan_digest": plan_digest,
                "event": event,
                "before_digest": before_digest,
                "after_digest": after_digest,
                "before_count": before_count,
                "after_count": after_count,
                "corpus": corpus,
                "split": release.split_sha256,
                "release_digest": release.digest,
                "model": model,
                "resolver": _hex(
                    exposure.get("semantic_resolver_policy_digest"),
                    f"{label}.resolver",
                    prefixed=True,
                ),
                "receipts": receipts,
                "support_counts": support_data[0] if support_data else None,
                "status": status,
                "proposer_success": proposer_success,
                "support_replayed": support_replayed,
                "support_pass": support_pass,
                "query_release": query_release,
                "complete": complete,
                "verified_rejection": verified_rejection,
                "verification_disposition": item.verification_disposition,
                "verification_scope": item.verification_scope,
            }
        )

    prepared.sort(key=lambda value: value["sort_key"])
    unique_fields = {
        "run_id": [item["run_id"] for item in prepared],
        "task_id": [item["task_id"] for item in prepared],
        "plan_digest": [item["plan_digest"] for item in prepared],
        "event_digest": [item["event"].digest for item in prepared],
    }
    for label, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise CampaignReportError(f"campaign {label} values must be unique")

    for previous, current in zip(prepared, prepared[1:], strict=False):
        if (
            previous["after_digest"] != current["before_digest"]
            or previous["after_count"] != current["before_count"]
            or current["event"].previous_digest != previous["event"].digest
        ):
            raise CampaignReportError(
                "campaign exposure predecessor/successor chain is not exact"
            )

    corpus = _single_identity([item["corpus"] for item in prepared], "corpus")
    split = _single_identity([item["split"] for item in prepared], "split source")
    release_digest = _single_identity(
        [item["release_digest"] for item in prepared], "release descriptor"
    )
    model = _single_identity([item["model"] for item in prepared], "model")
    resolver = _single_identity([item["resolver"] for item in prepared], "resolver")

    all_receipts: dict[str, Mapping[str, Any]] = {}
    for item in prepared:
        for digest, receipt in item["receipts"].items():
            previous = all_receipts.setdefault(digest, receipt)
            if dict(previous) != dict(receipt):
                raise CampaignReportError(
                    f"campaign reuses receipt digest {digest} for unequal receipts"
                )
    if not all_receipts:
        raise CampaignReportError("campaign has no successful receipt to bind launcher")
    launcher = _single_identity(
        [receipt["codex_launcher_digest"] for receipt in all_receipts.values()],
        "Codex launcher",
    )
    cli_version = _single_identity(
        [receipt["codex_cli_version"] for receipt in all_receipts.values()],
        "Codex CLI version",
    )

    stages = {
        "attempts": len(prepared),
        "proposer_successes": sum(item["proposer_success"] for item in prepared),
        "support_gate_replays": sum(item["support_replayed"] for item in prepared),
        "support_gate_passes": sum(item["support_pass"] for item in prepared),
        "query_releases": sum(item["query_release"] for item in prepared),
        "completions": sum(item["complete"] for item in prepared),
    }
    support_totals = {
        key: sum(
            item["support_counts"][key]
            for item in prepared
            if item["support_counts"] is not None
        )
        for key in (
            "forward_matches",
            "reverse_matches",
            "present",
            "nonmatch",
            "indeterminate",
            "error",
            "transport_attempts",
        )
    }
    support_totals["panels"] = sum(
        sum(
            item["support_counts"][key]
            for key in ("present", "nonmatch", "indeterminate", "error")
        )
        for item in prepared
        if item["support_counts"] is not None
    )
    support_totals["verified_support_rejections"] = sum(
        item["verified_rejection"] for item in prepared
    )

    usage = {field: sum(receipt[field] for receipt in all_receipts.values())
             for field in _RECEIPT_TOKEN_FIELDS}
    usage = {"successful_receipts": len(all_receipts), **usage}

    run_summaries = []
    for item in prepared:
        run_summaries.append(
            {
                "task_id": item["task_id"],
                "run_id": item["run_id"],
                "file_sha256": item["file_sha256"],
                "record_digest": _ADDRESS_PREFIX + item["record_digest"],
                "plan_digest": _ADDRESS_PREFIX + item["plan_digest"],
                "event_digest": item["event"].digest,
                "status": item["status"],
                "stages": {
                    "proposer_success": item["proposer_success"],
                    "support_gate_replay": item["support_replayed"],
                    "support_gate_pass": item["support_pass"],
                    "query_release": item["query_release"],
                    "complete": item["complete"],
                },
                "successful_receipts": len(item["receipts"]),
                "verification": {
                    "disposition": item["verification_disposition"],
                    "scope": item["verification_scope"],
                    "outer_record_integrity_verified": True,
                    "cold_replay_verified": item["verified_rejection"],
                },
            }
        )

    content = {
        "schema": CAMPAIGN_REPORT_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_type": CAMPAIGN_TYPE,
        "interpretation": {
            "measures": [
                "headless proposer transport",
                "support predicate fit",
                "artifact and replay plumbing",
            ],
            "does_not_measure": [
                "query accuracy",
                "image accuracy",
                "full-benchmark performance",
            ],
            "reason": (
                "no campaign attempt released query panels; support outcomes are "
                "training-side gate diagnostics, not held-out predictions"
            ),
        },
        "identities": {
            "corpus_manifest_digest": corpus,
            "split_source_digest": split,
            "official_release_descriptor_digest": release_digest,
            "model": model,
            "codex_cli_version": cli_version,
            "codex_launcher_digest": _ADDRESS_PREFIX + launcher,
            "semantic_resolver_policy_digest": resolver,
        },
        "exposure_chain": {
            "initial_ledger_digest": prepared[0]["before_digest"],
            "final_ledger_digest": prepared[-1]["after_digest"],
            "initial_event_count": prepared[0]["before_count"],
            "final_event_count": prepared[-1]["after_count"],
            "event_digests": [item["event"].digest for item in prepared],
        },
        "stages": stages,
        "support_replay": support_totals,
        "transport_usage": usage,
        "runs": run_summaries,
    }
    if stages["query_releases"] == 0:
        # Deliberately no score/accuracy object in this branch.
        return CampaignReport(content)
    raise CampaignReportError(
        "query-releasing campaigns need a separate held-out scoring report schema"
    )
