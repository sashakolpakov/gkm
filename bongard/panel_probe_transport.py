"""Neutral named-image transport adapter for active panel observers."""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import json
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard import prototype_scene_observer as _scene_runtime
from bongard.transport import CodexReceipt


class PanelProbeTransportError(RuntimeError):
    """A receipted named-image call returned a non-object payload."""


def panel_probe_transport_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def call_panel_probe(
    images: Sequence[tuple[str, bytes]],
    *,
    prompt: str,
    schema: Mapping[str, Any],
    journal: ObjectBongardNamedImageTurnJournalTransport,
    runtime: ObjectBongardTurnRuntime,
) -> tuple[dict[str, Any], CodexReceipt]:
    """Stage exact images and return one canonical, journaled object payload."""

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
        raise PanelProbeTransportError("model payload is not an object")
    return json.loads(canonical_json(dict(payload)).decode("utf-8")), receipt


__all__ = (
    "PanelProbeTransportError",
    "call_panel_probe",
    "panel_probe_transport_source_digest",
)
