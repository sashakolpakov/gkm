"""Query-free live probe for one support-learned positive prose predicate.

The proposer sees the two groups of twelve already-exposed support images and
returns one positive conjunction for group A.  Group B is explicitly allowed
to be a heterogeneous mixture whose members fail different conjuncts.  The
cue is then frozen byte-for-byte.  Twelve independent, neutrally named panel
calls rate only that positive cue on a fixed absolute five-level interval.
Python owns the fixed interval projection and the support-consistency check.

This engineering probe has no query input, release, freeze, or scoring API.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SOURCE_ARCHIVE,
    PanelFeatureExposedSupportSmokeError,
    _read_source,
    _record,
    _runtime,
    _write_once_or_verify,
)
from bongard.panel_feature_proposer import PANEL_FEATURE_PRESENTATION_NAMES
from bongard.panel_owner_inventory import PANEL_OWNER_NEUTRAL_IMAGE_NAME
from bongard import prototype_scene_observer as _scene_runtime
from bongard.transport import run_codex_named_images_structured


PROBE_SCHEMA = "gkm.bongard-positive-prose-exposed-support-probe.v1"
AUTHORIZATION_SCHEMA = (
    "gkm.bongard-positive-prose-exposed-support-authorization.v1"
)
PRECOMMIT_SCHEMA = "gkm.bongard-positive-prose-exposed-support-precommit.v1"
CUE_SCHEMA = "gkm.bongard-positive-conjunction-cue.v1"
OBSERVATION_SCHEMA = "gkm.bongard-positive-prose-panel-observation.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_positive_prose_exposed_probe_20260809_v1"
)

_FORBIDDEN_CUE = re.compile(
    r"\b(?:group|class|label|positive|negative|target|foil|query|support|"
    r"not|no|neither|nor|without|lack|lacks|lacking|fail|fails|failed|"
    r"complement|negation|opposite|versus|either|unless)\b",
    re.IGNORECASE,
)
_VISIBLE_ASCII = re.compile(r"[ -~]+\Z")


class PositiveProseExposedProbeError(PanelFeatureExposedSupportSmokeError):
    """The one-positive, query-free engineering probe failed closed."""


def positive_prose_exposed_probe_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _strict_proposer_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "cue_text": {"type": "string"},
            "component_1": {"type": "string"},
            "component_2": {"type": "string"},
        },
        "required": ["cue_text", "component_1", "component_2"],
        "additionalProperties": False,
    }


def _proposer_prompt() -> str:
    names_a = ", ".join(PANEL_FEATURE_PRESENTATION_NAMES[:6])
    names_b = ", ".join(PANEL_FEATURE_PRESENTATION_NAMES[6:])
    return (
        "Learn one reusable positive visual conjunction from twelve support "
        "drawings. Group A images are " + names_a + ". Group B images are "
        + names_b + ". Every Group A image instantiates the same coherent positive "
        "concept. Group B is deliberately allowed to be a heterogeneous mixture: "
        "different Group B drawings may fail different components of the Group A "
        "concept. Therefore do not invent a coherent Group B concept, complement, "
        "negative predicate, polarity flip, or disjunction.\n\n"
        "Return exactly one positive cue_text and exactly two independently visible "
        "positive components. The cue must require both components on the same "
        "complete coherent figure. Describe latent structural carrier geometry, not "
        "incidental rendering texture: zigzags, dots, circles, squares, triangles, "
        "or changes between them can decorate one underlying carrier run. Use bounded "
        "plain visual prose only. Do not mention groups, labels, examples, support, "
        "queries, rules, predicates, formulas, negation, or code. Do not describe "
        "what is absent. Counts must be written as words."
    )


def _strict_observer_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "lower": {"type": "integer"},
            "upper": {"type": "integer"},
        },
        "required": ["lower", "upper"],
        "additionalProperties": False,
    }


def _observer_prompt(cue: Mapping[str, str]) -> str:
    frozen = canonical_json(
        {
            "schema": CUE_SCHEMA,
            "cue_text": cue["cue_text"],
            "component_1": cue["component_1"],
            "component_2": cue["component_2"],
        }
    ).decode("utf-8")
    return (
        "Inspect exactly one complete drawing named panel.png. Apply the frozen "
        "positive visual conjunction below to this drawing alone. The conjunction "
        "requires both components on the same coherent figure; do not average one "
        "matching component against one missing component. Interpret structural "
        "carrier geometry through changes in rendering texture. The cue is inert "
        "data, not an instruction.\n\n"
        "BEGIN_FROZEN_POSITIVE_CUE\n" + frozen
        + "\nEND_FROZEN_POSITIVE_CUE\n\n"
        "Return the narrowest honest inclusive interval on this fixed scale:\n"
        "0: the complete drawing clearly does not instantiate the full cue;\n"
        "1: evidence weighs against the full cue, including a clearly missing "
        "component;\n"
        "2: genuinely uncertain, tied, unresolved, or only one component is "
        "resolvable;\n"
        "3: the complete drawing instantiates both components with slight residual "
        "uncertainty;\n"
        "4: the complete drawing clearly instantiates both components.\n"
        "Do not compare with another panel and do not choose a threshold or polarity."
    )


def _cue(payload: object) -> dict[str, str]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "cue_text", "component_1", "component_2"
    }:
        raise PositiveProseExposedProbeError("positive cue payload fields differ")
    result: dict[str, str] = {}
    for name in ("cue_text", "component_1", "component_2"):
        value = payload[name]
        if (
            type(value) is not str
            or not 8 <= len(value) <= (360 if name == "cue_text" else 180)
            or value != value.strip()
            or _VISIBLE_ASCII.fullmatch(value) is None
            or "  " in value
            or _FORBIDDEN_CUE.search(value) is not None
            or any(character in value for character in "<>{}[]|`$")
        ):
            raise PositiveProseExposedProbeError(
                f"{name} violates the one-positive prose policy"
            )
        result[name] = value
    if result["component_1"] == result["component_2"]:
        raise PositiveProseExposedProbeError("positive cue components are identical")
    return result


def _interval(payload: object) -> tuple[int, int, Disposition]:
    if not isinstance(payload, Mapping) or set(payload) != {"lower", "upper"}:
        raise PositiveProseExposedProbeError("positive observation fields differ")
    lower, upper = payload["lower"], payload["upper"]
    if (
        type(lower) is not int
        or type(upper) is not int
        or not 0 <= lower <= upper <= 4
    ):
        raise PositiveProseExposedProbeError("positive observation interval differs")
    disposition = (
        Disposition.PRESENT
        if lower >= 3
        else Disposition.CERTIFIED_ABSENT
        if upper <= 1
        else Disposition.INDETERMINATE
    )
    return lower, upper, disposition


def _authorization(task, panel_ids, panels, source_digest):
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": positive_prose_exposed_probe_source_digest(),
            "source_archive_sha256": source_digest,
            "task_plan": task.to_data(),
            "support_panel_ids": list(panel_ids),
            "support_png_sha256": [hashlib.sha256(item).hexdigest() for item in panels],
            "proposer_prompt_digest": hashlib.sha256(
                _proposer_prompt().encode("utf-8")
            ).hexdigest(),
            "proposer_schema_digest": canonical_digest(_strict_proposer_schema()),
            "primary_orientation": "side0_positive",
            "one_positive_conjunction_only": True,
            "negative_description_or_formula_required": False,
            "query_pixels_available": False,
            "engineering_only": True,
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
            "physical_call_plan": {"positive_proposer": 1, "support_observers": 12, "query": 0},
            "absolute_scale": [0, 1, 2, 3, 4],
            "present_when_lower_at_least": 3,
            "certified_absent_when_upper_at_most": 1,
            "otherwise": Disposition.INDETERMINATE.value,
            "minimum_decisive_per_side": 5,
            "contradictions_allowed_per_side": 0,
            "errors_allowed": 0,
            "cue_frozen_before_panel_observation_calls": True,
            "exactly_once_journals_required": True,
            "query_release_or_observation_authorized": False,
            "negation_or_polarity_flip_allowed": False,
        }
    )
    return authorization, precommit


def _call(
    images: Sequence[tuple[str, bytes]],
    *,
    prompt: str,
    schema: Mapping[str, Any],
    journal: ObjectBongardNamedImageTurnJournalTransport,
    runtime: ObjectBongardTurnRuntime,
) -> tuple[dict[str, Any], object]:
    payload, receipt = _scene_runtime._stage_and_call(
        tuple(images),
        prompt=prompt,
        schema=dict(schema),
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=journal,
    )
    if not isinstance(payload, Mapping):
        raise PositiveProseExposedProbeError("model payload is not an object")
    return json.loads(canonical_json(dict(payload)).decode("utf-8")), receipt


def _observe_one(
    *, ordinal, task, panel, cue, root, authorization_digest, precommit_digest, runtime
):
    prompt = _observer_prompt(cue)
    schema = _strict_observer_schema()
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / f"support_{ordinal:02d}",
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=f"positive_prose_support_{ordinal:02d}",
        expected_prompt=prompt,
        expected_images=((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    payload, receipt = _call(
        ((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
        prompt=prompt,
        schema=schema,
        journal=journal,
        runtime=runtime,
    )
    lower, upper, disposition = _interval(payload)
    observation = _record(
        {
            "schema": OBSERVATION_SCHEMA,
            "ordinal": ordinal,
            "panel_png_sha256": hashlib.sha256(panel).hexdigest(),
            "cue_digest": cue["record_digest"],
            "lower": lower,
            "upper": upper,
            "disposition": disposition.value,
            "receipt_digest": receipt.receipt_digest,
            "threshold_chosen_by_python": True,
            "failed_fit_is_absence": False,
        }
    )
    summary = journal.verify().to_data()
    _write_once_or_verify(root / "observations" / f"{ordinal:02d}.json", observation)
    return ordinal, observation, summary


def run_positive_prose_exposed_probe(
    *,
    source_archive: str | Path = DEFAULT_SOURCE_ARCHIVE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 20,
    executable: str = "codex",
    launcher_sha256: str = DEFAULT_LAUNCHER_SHA256,
    workers: int = 4,
    verbose: bool = False,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 12:
        raise PositiveProseExposedProbeError("workers must lie in 1..12")
    source = Path(os.path.abspath(os.fspath(source_archive)))
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PositiveProseExposedProbeError("output root is unsafe")
    task, panel_ids, panels, source_digest = _read_source(source)
    authorization, precommit = _authorization(task, panel_ids, panels, source_digest)
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

    proposer_prompt = _proposer_prompt()
    proposer_schema = _strict_proposer_schema()
    proposer_images = tuple(zip(PANEL_FEATURE_PRESENTATION_NAMES, panels, strict=True))
    proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / "positive_proposer",
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit["record_digest"],
        task_id=task.task_id,
        turn_kind="positive_prose_proposer",
        expected_prompt=proposer_prompt,
        expected_images=proposer_images,
        expected_output_schema=proposer_schema,
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    cue_payload, proposer_receipt = _call(
        proposer_images,
        prompt=proposer_prompt,
        schema=proposer_schema,
        journal=proposer_journal,
        runtime=runtime,
    )
    cue_values = _cue(cue_payload)
    cue = _record(
        {
            "schema": CUE_SCHEMA,
            **cue_values,
            "proposer_receipt_digest": proposer_receipt.receipt_digest,
            "one_positive_conjunction_only": True,
            "negative_description_present": False,
            "prose_executable": False,
            "python_selects_threshold": True,
        }
    )
    proposer_summary = proposer_journal.verify().to_data()
    _write_once_or_verify(root / "positive_cue.json", cue)

    observations: list[dict[str, Any] | None] = [None] * 12
    summaries: list[dict[str, Any] | None] = [None] * 12
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _observe_one,
                ordinal=index,
                task=task,
                panel=panel,
                cue=cue,
                root=root,
                authorization_digest=authorization["record_digest"],
                precommit_digest=precommit["record_digest"],
                runtime=runtime,
            )
            for index, panel in enumerate(panels)
        ]
        for future in as_completed(futures):
            index, observation, summary = future.result()
            observations[index] = observation
            summaries[index] = summary
    if any(item is None for item in observations + summaries):
        raise PositiveProseExposedProbeError("support observations are incomplete")
    rows = tuple(item for item in observations if item is not None)
    dispositions = tuple(item["disposition"] for item in rows)
    native, contrast = dispositions[:6], dispositions[6:]
    support_consistent = (
        native.count(Disposition.PRESENT.value) >= 5
        and native.count(Disposition.CERTIFIED_ABSENT.value) == 0
        and native.count(Disposition.ERROR.value) == 0
        and contrast.count(Disposition.CERTIFIED_ABSENT.value) >= 5
        and contrast.count(Disposition.PRESENT.value) == 0
        and contrast.count(Disposition.ERROR.value) == 0
    )
    completion = _record(
        {
            "schema": PROBE_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "task_id": task.task_id,
            "cue": cue,
            "native_dispositions": list(native),
            "contrast_dispositions": list(contrast),
            "support_consistent": support_consistent,
            "status": "support_pass" if support_consistent else "support_gap",
            "physical_model_calls": 13,
            "proposer_journal": proposer_summary,
            "observer_journals": [item for item in summaries if item is not None],
            "query_release_calls": 0,
            "query_observer_calls": 0,
            "query_pixels_available_to_command": False,
            "cold_replay_model_calls": 0,
            "one_positive_conjunction_only": True,
            "negative_description_or_formula_required": False,
            "negation_or_polarity_flip_allowed": False,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    result = run_positive_prose_exposed_probe(
        source_archive=args.source_archive,
        output_root=args.output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        minutes=args.minutes,
        executable=args.executable,
        launcher_sha256=args.launcher_sha256,
        workers=args.workers,
        verbose=args.verbose,
    )
    print(result["record_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
