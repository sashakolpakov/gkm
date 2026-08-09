"""Run the preregistered one-call contextual typed-count support probe.

The command can read only the twelve already-exposed support PNGs from the
support-gap archive.  It deterministically zooms every PNG, exposes the fixed
primary/contrast roles in twelve stable image names, and makes one exactly-once
journaled named-image call.  The model returns six typed scalar fields for each
view.  Fixed Python code, never the model, projects convexity and exact-four
straight-action intervals to one affirmative conjunction and writes either a
support pass or a typed support gap.  There is no query input or query API.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
)
from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SOURCE_ARCHIVE,
    _read_source,
    _record,
    _runtime,
    _write_once_or_verify,
)
from bongard.panel_positive_prose_exposed_probe_command import (
    _call,
    _component_conjunction_disposition,
    _count_four_interval,
    _count_interval,
    _deterministic_ink_zoom,
    _interval,
    _load_ink_zoom_policy,
    _load_typed_count_policy,
    positive_prose_exposed_probe_source_digest,
)
from bongard.transport import (
    run_codex_named_images_structured,
    validate_codex_strict_output_schema,
)


CONTEXT_POLICY_SCHEMA = (
    "gkm.bongard-contextual-typed-count-support-probe-preregistration.v1"
)
AUTHORIZATION_SCHEMA = (
    "gkm.bongard-contextual-typed-count-support-authorization.v1"
)
PRECOMMIT_SCHEMA = "gkm.bongard-contextual-typed-count-support-precommit.v1"
COMPLETION_SCHEMA = "gkm.bongard-contextual-typed-count-support-completion.v1"
ROW_SCHEMA = "gkm.bongard-contextual-typed-count-support-row.v1"
TARGET_TASK_ID = "hd_convex-has_four_straight_lines_0001"
CONTEXT_POLICY_RECORD_DIGEST = (
    "sha256:f4e8f4e3d44a91cf96ce64ccdaf6bbbb951e48a02eb50e19a9a5a47f9075591f"
)
TYPED_COUNT_POLICY_RECORD_DIGEST = (
    "sha256:3ac70952b2fa0c94a4b4afe87e0dce7448f86cef21bf70f6952cd33e68164bae"
)
INK_ZOOM_POLICY_RECORD_DIGEST = (
    "sha256:25a602455aab02b0d5cbcb05f18bd283e9b7ce43e88c343933ab2a4b2798d564"
)
PARENT_ISOLATED_COMPLETION_DIGEST = (
    "sha256:e867f44aa88f853dcda917fc19455bfb6ae86ab28e44eaf7450257cc6d2c127c"
)

DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_CONTEXT_POLICY = (
    DATA_ROOT / "panel_positive_contextual_typed_count_probe_20260809_v1.json"
)
DEFAULT_TYPED_COUNT_POLICY = (
    DATA_ROOT / "panel_positive_straight_action_count_probe_20260809_v1.json"
)
DEFAULT_INK_ZOOM_POLICY = (
    DATA_ROOT / "panel_positive_prose_ink_zoom_20260809_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_positive_contextual_typed_count_probe_20260809_v1"
)

ROLE_VIEW_NAMES = tuple(
    [f"primary_{index:02d}.png" for index in range(6)]
    + [f"contrast_{index:02d}.png" for index in range(6)]
)
MEASUREMENT_FIELDS = (
    "convex_score_lower",
    "convex_score_upper",
    "straight_action_count_lower",
    "straight_action_count_upper",
    "curved_action_count_lower",
    "curved_action_count_upper",
)
MINIMUM_DECISIVE_PER_SIDE = 5


class ContextualTypedCountProbeError(RuntimeError):
    """The frozen policy, support pixels, model result, or journal differs."""


def contextual_typed_count_probe_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _load_context_policy(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(os.path.abspath(os.fspath(path)))
    if source.is_symlink() or not source.is_file():
        raise ContextualTypedCountProbeError("context policy file is unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContextualTypedCountProbeError("context policy is malformed") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise ContextualTypedCountProbeError("context policy is not canonical")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        digest != "sha256:" + canonical_digest(body)
        or digest != CONTEXT_POLICY_RECORD_DIGEST
    ):
        raise ContextualTypedCountProbeError("context policy digest differs")
    expected = {
        "schema": CONTEXT_POLICY_SCHEMA,
        "change_reason": (
            "isolated panel observer cannot infer latent action boundaries under "
            "arc and texture decoration"
        ),
        "parent_typed_count_policy_record_digest": TYPED_COUNT_POLICY_RECORD_DIGEST,
        "parent_isolated_count_completion_digest": PARENT_ISOLATED_COMPLETION_DIGEST,
        "ink_zoom_policy_record_digest": INK_ZOOM_POLICY_RECORD_DIGEST,
        "physical_call_plan": {"support_context_batch_observer": 1, "query": 0},
        "support_context_model_visible": True,
        "positive_vs_contrast_role_visible": True,
        "dataset_task_panel_side_ids_visible": False,
        "model_returns_twelve_independent_convex_and_action_count_intervals": True,
        "frozen_measurement_definition_unchanged": True,
        "python_projection_unchanged": True,
        "no_model_formula_threshold_polarity_or_query_selection": True,
        "minimum_decisive_per_side": 5,
        "contradictions_allowed_per_side": 0,
        "errors_allowed": 0,
        "support_only_exposed_probe_authorized": True,
        "query_pixels_visible": False,
        "official_test_authorized": False,
        "target_support_or_query_pixels_authorized": False,
        "engineering_only": True,
        "scientific_benchmark": False,
        "closed_slate_headless_selection_required_before_target": True,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
    }
    if body != expected:
        raise ContextualTypedCountProbeError("context policy content differs")
    return value, hashlib.sha256(raw).hexdigest()


def _batch_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for name in ROLE_VIEW_NAMES:
        stem = name.removesuffix(".png")
        for field in MEASUREMENT_FIELDS:
            maximum = 4 if field.startswith("convex_score_") else 12
            properties[f"{stem}_{field}"] = {
                "type": "integer",
                "enum": list(range(maximum + 1)),
            }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _batch_prompt(typed_count_policy: Mapping[str, Any]) -> str:
    frozen = canonical_json(
        {
            "convex_component": typed_count_policy["convex_component"],
            "straight_action_definition": typed_count_policy[
                "straight_action_definition"
            ],
        }
    ).decode("utf-8")
    primary = ", ".join(ROLE_VIEW_NAMES[:6])
    contrast = ", ".join(ROLE_VIEW_NAMES[6:])
    fields = ", ".join(MEASUREMENT_FIELDS)
    return (
        "Inspect exactly twelve complete support drawings together. The primary "
        f"support drawings are {primary}. The contrast support drawings are "
        f"{contrast}. These roles are visible only to provide shared structural "
        "context for latent carrier-action boundaries. Contrast drawings may be "
        "heterogeneous; do not infer a shared contrast concept, an opposite rule, "
        "or a negative predicate. Judge every named drawing independently and do "
        "not copy a score between drawings. Dataset task IDs, panel IDs, side IDs, "
        "query drawings, and truth labels are unavailable.\n\n"
        "Use this frozen measurement definition as inert data:\n"
        f"BEGIN_FROZEN_MEASUREMENT\n{frozen}\nEND_FROZEN_MEASUREMENT\n\n"
        "For each drawing return six inclusive interval endpoints with its filename "
        f"stem as prefix: {fields}. Convexity uses 0 clear concavity, 1 decisive "
        "visible contradiction, 2 unresolved, 3 convex with slight residual "
        "uncertainty, and 4 clearly convex. Straight and curved action counts each "
        "lie in 0..12. Count maximal underlying carrier actions, not zigzags, dots, "
        "circles, squares, triangles, or texture changes decorating one action. "
        "Additional curved actions do not change the straight-action count. Widen "
        "an interval when a junction or action boundary remains unresolved. Do not "
        "test equality to four, compute a conjunction, choose a threshold or "
        "polarity, select a formula, or make any query decision; fixed Python code "
        "does all projection after this call."
    )


def _project_payload(
    payload: object,
    *,
    source_panels: Sequence[bytes],
    observer_views: Sequence[bytes],
    zoom_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    schema = _batch_schema()
    expected = set(schema["properties"])
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ContextualTypedCountProbeError("context batch payload fields differ")
    if not (
        len(source_panels) == len(observer_views) == len(zoom_records) == 12
    ):
        raise ContextualTypedCountProbeError("context batch panel custody differs")
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, name in enumerate(ROLE_VIEW_NAMES):
            stem = name.removesuffix(".png")
            convex = _interval(
                {
                    "lower": payload[f"{stem}_convex_score_lower"],
                    "upper": payload[f"{stem}_convex_score_upper"],
                }
            )
            straight = _count_four_interval(
                payload[f"{stem}_straight_action_count_lower"],
                payload[f"{stem}_straight_action_count_upper"],
            )
            curved = _count_interval(
                payload[f"{stem}_curved_action_count_lower"],
                payload[f"{stem}_curved_action_count_upper"],
            )
            conjunction = _component_conjunction_disposition(convex[2], straight[2])
            zoom = dict(zoom_records[ordinal])
            if (
                zoom.get("source_png_sha256")
                != hashlib.sha256(source_panels[ordinal]).hexdigest()
                or zoom.get("observer_view_png_sha256")
                != hashlib.sha256(observer_views[ordinal]).hexdigest()
            ):
                raise ContextualTypedCountProbeError("zoom record pixels differ")
            rows.append(
                _record(
                    {
                        "schema": ROW_SCHEMA,
                        "ordinal": ordinal,
                        "role": "primary" if ordinal < 6 else "contrast",
                        "role_ordinal": ordinal if ordinal < 6 else ordinal - 6,
                        "model_visible_name": name,
                        "source_png_sha256": hashlib.sha256(
                            source_panels[ordinal]
                        ).hexdigest(),
                        "observer_view_png_sha256": hashlib.sha256(
                            observer_views[ordinal]
                        ).hexdigest(),
                        "ink_zoom_record_digest": zoom["record_digest"],
                        "convex_score_lower": convex[0],
                        "convex_score_upper": convex[1],
                        "convexity_disposition": convex[2].value,
                        "straight_action_count_lower": straight[0],
                        "straight_action_count_upper": straight[1],
                        "straight_count_four_disposition": straight[2].value,
                        "curved_action_count_lower": curved[0],
                        "curved_action_count_upper": curved[1],
                        "disposition": conjunction.value,
                        "failed_fit_counts_as_absence": False,
                    }
                )
            )
    except ContextualTypedCountProbeError:
        raise
    except Exception as exc:
        raise ContextualTypedCountProbeError(
            "context batch typed projection failed"
        ) from exc
    return tuple(rows)


def _profile(dispositions: Sequence[str]) -> dict[str, int]:
    counts = Counter(dispositions)
    return {item.value: counts[item.value] for item in Disposition}


def _support_result(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, int], dict[str, int], tuple[str, ...]]:
    if len(rows) != 12:
        raise ContextualTypedCountProbeError("support result needs twelve rows")
    primary = _profile([row["disposition"] for row in rows[:6]])
    contrast = _profile([row["disposition"] for row in rows[6:]])
    reasons: list[str] = []
    if primary[Disposition.PRESENT.value] < MINIMUM_DECISIVE_PER_SIDE:
        reasons.append("primary_present_below_five")
    if contrast[Disposition.CERTIFIED_ABSENT.value] < MINIMUM_DECISIVE_PER_SIDE:
        reasons.append("contrast_certified_absent_below_five")
    if primary[Disposition.CERTIFIED_ABSENT.value]:
        reasons.append("primary_certified_absent_contradiction")
    if contrast[Disposition.PRESENT.value]:
        reasons.append("contrast_present_contradiction")
    if primary[Disposition.ERROR.value] or contrast[Disposition.ERROR.value]:
        reasons.append("support_observer_error")
    if primary[Disposition.INDETERMINATE.value] > 1:
        reasons.append("primary_indeterminate_above_one")
    if contrast[Disposition.INDETERMINATE.value] > 1:
        reasons.append("contrast_indeterminate_above_one")
    return not reasons, primary, contrast, tuple(reasons)


def _authorization_and_precommit(
    *,
    task: object,
    panel_ids: Sequence[str],
    source_panels: Sequence[bytes],
    observer_views: Sequence[bytes],
    zoom_records: Sequence[Mapping[str, Any]],
    source_archive_sha256: str,
    context_policy: Mapping[str, Any],
    context_policy_file_sha256: str,
    typed_count_policy: Mapping[str, Any],
    typed_count_policy_file_sha256: str,
    ink_zoom_policy: Mapping[str, Any],
    ink_zoom_policy_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(task, "task_id", None) != TARGET_TASK_ID:
        raise ContextualTypedCountProbeError("context policy is bound to another task")
    if not (
        len(panel_ids) == len(source_panels) == len(observer_views)
        == len(zoom_records) == 12
    ):
        raise ContextualTypedCountProbeError("support context input count differs")
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": contextual_typed_count_probe_source_digest(),
            "helper_source_digest": positive_prose_exposed_probe_source_digest(),
            "source_archive_sha256": source_archive_sha256,
            "task_plan": task.to_data(),
            "support_panel_ids": list(panel_ids),
            "source_png_sha256": [
                hashlib.sha256(item).hexdigest() for item in source_panels
            ],
            "model_visible_names": list(ROLE_VIEW_NAMES),
            "model_visible_roles": ["primary"] * 6 + ["contrast"] * 6,
            "observer_view_png_sha256": [
                hashlib.sha256(item).hexdigest() for item in observer_views
            ],
            "ink_zoom_records": [dict(item) for item in zoom_records],
            "context_policy_record_digest": context_policy["record_digest"],
            "context_policy_file_sha256": context_policy_file_sha256,
            "typed_count_policy_record_digest": typed_count_policy["record_digest"],
            "typed_count_policy_file_sha256": typed_count_policy_file_sha256,
            "ink_zoom_policy_record_digest": ink_zoom_policy["record_digest"],
            "ink_zoom_policy_file_sha256": ink_zoom_policy_file_sha256,
            "support_context_model_visible": True,
            "positive_vs_contrast_role_visible": True,
            "dataset_task_panel_side_ids_model_visible": False,
            "query_pixels_available_to_command": False,
            "query_release_or_observation_authorized": False,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    precommit = _record(
        {
            "schema": PRECOMMIT_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "physical_call_plan": {
                "support_context_batch_observer": 1,
                "query": 0,
            },
            "model_visible_image_order": list(ROLE_VIEW_NAMES),
            "output_field_count": 12 * len(MEASUREMENT_FIELDS),
            "convex_present_when_lower_at_least": 3,
            "convex_certified_absent_when_upper_at_most": 1,
            "count_four_present_when": "lower_equals_upper_equals_4",
            "count_four_certified_absent_when": (
                "upper_less_than_4_or_lower_greater_than_4"
            ),
            "conjunction_present_when": "convex_present_and_count_four_present",
            "conjunction_absent_when": "either_component_certified_absent",
            "minimum_decisive_per_side": MINIMUM_DECISIVE_PER_SIDE,
            "maximum_indeterminate_per_side": 1,
            "contradictions_allowed_per_side": 0,
            "errors_allowed": 0,
            "formula_threshold_polarity_selected_by_model": False,
            "exactly_once_journal_required": True,
            "query_pixels_available_to_command": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    return authorization, precommit


def run_contextual_typed_count_support_probe(
    *,
    source_archive: str | Path = DEFAULT_SOURCE_ARCHIVE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    context_policy_file: str | Path = DEFAULT_CONTEXT_POLICY,
    typed_count_policy_file: str | Path = DEFAULT_TYPED_COUNT_POLICY,
    ink_zoom_policy_file: str | Path = DEFAULT_INK_ZOOM_POLICY,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 20,
    executable: str = "codex",
    launcher_sha256: str = DEFAULT_LAUNCHER_SHA256,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute one support-only contextual batch call and persist its result."""

    source = Path(os.path.abspath(os.fspath(source_archive)))
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ContextualTypedCountProbeError("output root is unsafe")
    context_policy, context_file_sha = _load_context_policy(context_policy_file)
    typed_policy, typed_file_sha = _load_typed_count_policy(typed_count_policy_file)
    zoom_policy, zoom_file_sha = _load_ink_zoom_policy(ink_zoom_policy_file)
    if (
        typed_policy["record_digest"] != context_policy[
            "parent_typed_count_policy_record_digest"
        ]
        or zoom_policy["record_digest"] != context_policy[
            "ink_zoom_policy_record_digest"
        ]
    ):
        raise ContextualTypedCountProbeError("context parent policy differs")
    task, panel_ids, panels, source_digest = _read_source(source)
    if task.task_id != TARGET_TASK_ID:
        raise ContextualTypedCountProbeError("source archive task differs")
    transformed = tuple(_deterministic_ink_zoom(panel, zoom_policy) for panel in panels)
    views = tuple(item[0] for item in transformed)
    zoom_records = tuple(item[1] for item in transformed)
    authorization, precommit = _authorization_and_precommit(
        task=task,
        panel_ids=panel_ids,
        source_panels=panels,
        observer_views=views,
        zoom_records=zoom_records,
        source_archive_sha256=source_digest,
        context_policy=context_policy,
        context_policy_file_sha256=context_file_sha,
        typed_count_policy=typed_policy,
        typed_count_policy_file_sha256=typed_file_sha,
        ink_zoom_policy=zoom_policy,
        ink_zoom_policy_file_sha256=zoom_file_sha,
    )
    _write_once_or_verify(root / "authorization.json", authorization)
    _write_once_or_verify(root / "execution_precommit.json", precommit)
    runtime, runtime_evidence = _runtime(
        output_root=root,
        authorization=authorization,
        precommit=precommit,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        executable=executable,
        launcher_sha256=launcher_sha256,
        verbose=verbose,
    )
    prompt = _batch_prompt(typed_policy)
    schema = _batch_schema()
    images = tuple(zip(ROLE_VIEW_NAMES, views, strict=True))
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / "support_context_batch",
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit["record_digest"],
        task_id=task.task_id,
        turn_kind="positive_typed_count_context_batch",
        expected_prompt=prompt,
        expected_images=images,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    payload, receipt = _call(
        images, prompt=prompt, schema=schema, journal=journal, runtime=runtime
    )
    terminal = journal.verify()
    if terminal.terminal_status != "success":
        raise ContextualTypedCountProbeError("context batch journal is not successful")
    rows = _project_payload(
        payload,
        source_panels=panels,
        observer_views=views,
        zoom_records=zoom_records,
    )
    support_pass, primary_counts, contrast_counts, gap_reasons = _support_result(rows)
    completion = _record(
        {
            "schema": COMPLETION_SCHEMA,
            "command_source_digest": contextual_typed_count_probe_source_digest(),
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "task_id": task.task_id,
            "context_policy_record_digest": context_policy["record_digest"],
            "typed_count_policy_record_digest": typed_policy["record_digest"],
            "ink_zoom_policy_record_digest": zoom_policy["record_digest"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "receipt_digest": receipt.receipt_digest,
            "journal_terminal": terminal.to_data(),
            "rows": list(rows),
            "primary_counts": primary_counts,
            "contrast_counts": contrast_counts,
            "support_rule": {
                "minimum_decisive_per_side": MINIMUM_DECISIVE_PER_SIDE,
                "maximum_indeterminate_per_side": 1,
                "contradictions_allowed_per_side": 0,
                "errors_allowed": 0,
            },
            "support_consistent": support_pass,
            "gap_reasons": list(gap_reasons),
            "status": "support_pass" if support_pass else "support_gap",
            "physical_model_calls": 1,
            "support_context_batch_observer_calls": 1,
            "query_observer_calls": 0,
            "query_release_calls": 0,
            "query_pixels_available_to_command": False,
            "model_selected_formula_threshold_or_polarity": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_benchmark": False,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    _write_once_or_verify(root / "completion.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", default=str(DEFAULT_SOURCE_ARCHIVE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--context-policy-file", default=str(DEFAULT_CONTEXT_POLICY))
    parser.add_argument(
        "--typed-count-policy-file", default=str(DEFAULT_TYPED_COUNT_POLICY)
    )
    parser.add_argument("--ink-zoom-policy-file", default=str(DEFAULT_INK_ZOOM_POLICY))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    result = run_contextual_typed_count_support_probe(
        source_archive=args.source_archive,
        output_root=args.output_root,
        context_policy_file=args.context_policy_file,
        typed_count_policy_file=args.typed_count_policy_file,
        ink_zoom_policy_file=args.ink_zoom_policy_file,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        minutes=args.minutes,
        executable=args.executable,
        launcher_sha256=args.launcher_sha256,
        verbose=args.verbose,
    )
    print(result["record_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
