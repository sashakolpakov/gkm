"""Decision-only complete panel inventory for object-scene predicates.

The extraction catalog is deliberately rich replay provenance.  Predicate
evaluation needs a much smaller statement: which panel and inventory were
frozen, and the ordered decision manifest for every inventoried object.  This
module projects that statement only after a cold catalog replay and provides a
second cold verifier for downstream observer and version-space artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_catalog import (
    ObjectSceneAnchorCatalog,
    ObjectSceneAnchorCatalogError,
    ObjectSceneAnchorDecisionManifest,
    verify_object_scene_anchor_catalog,
)
from bongard.object_scene_visual_frontend import ObjectSceneProposalInventory
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA = (
    "gkm.object-scene-anchor-panel-decision-manifest.v1"
)
OBJECT_SCENE_ANCHOR_PANEL_DECISION_PROJECTION_ID = (
    "bongard.object-scene-anchor-panel-decision/complete-inventory-v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")


class ObjectSceneAnchorPanelManifestError(ValueError):
    """A panel decision projection is malformed or fails cold replay."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorPanelManifestError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorPanelManifestError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectSceneAnchorPanelManifestError(f"{label} fields differ")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "complete_object_inventory_required": True,
        "object_omission_allowed": False,
        "raw_graph_decision_bearing": False,
        "audit_graph_decision_bearing": False,
    }


def _manifest_content(
    value: "ObjectSceneAnchorPanelDecisionManifest",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA,
        "projection_id": OBJECT_SCENE_ANCHOR_PANEL_DECISION_PROJECTION_ID,
        "panel_digest": value.panel_digest,
        "width_pixels": value.width_pixels,
        "height_pixels": value.height_pixels,
        "inventory_digest": value.inventory_digest,
        "proposal_count": value.proposal_count,
        "object_ids": list(value.object_ids),
        "object_decisions": [item.to_data() for item in value.object_decisions],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPanelDecisionManifest:
    """Every object-level decision for one exact panel, in inventory order."""

    panel_digest: str
    width_pixels: int
    height_pixels: int
    inventory_digest: str
    proposal_count: int
    object_ids: tuple[str, ...]
    object_decisions: tuple[ObjectSceneAnchorDecisionManifest, ...]
    manifest_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "panel digest")
        _digest(self.inventory_digest, "inventory digest")
        _integer(self.width_pixels, "panel width", minimum=2)
        _integer(self.height_pixels, "panel height", minimum=2)
        _integer(self.proposal_count, "proposal count")
        if type(self.object_ids) is not tuple or type(
            self.object_decisions
        ) is not tuple:
            raise TypeError("panel object inventory must use exact tuples")
        expected_ids = tuple(
            f"object_{index:04d}" for index in range(self.proposal_count)
        )
        if (
            self.object_ids != expected_ids
            or len(self.object_decisions) != self.proposal_count
            or any(
                not isinstance(object_id, str)
                or _OBJECT_ID.fullmatch(object_id) is None
                for object_id in self.object_ids
            )
            or any(
                type(item) is not ObjectSceneAnchorDecisionManifest
                for item in self.object_decisions
            )
            or tuple(item.object_id for item in self.object_decisions)
            != self.object_ids
        ):
            raise ObjectSceneAnchorPanelManifestError(
                "panel decision manifest does not exhaust inventory in order"
            )
        _digest(self.manifest_digest, "panel decision manifest digest")
        if self.manifest_digest != canonical_digest(_manifest_content(self)):
            raise ObjectSceneAnchorPanelManifestError(
                "panel decision manifest digest differs"
            )

    @property
    def by_object_id(self) -> dict[str, ObjectSceneAnchorDecisionManifest]:
        return dict(zip(self.object_ids, self.object_decisions, strict=True))

    def to_data(self) -> dict[str, object]:
        return {**_manifest_content(self), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPanelDecisionManifest":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "projection_id",
                    "panel_digest",
                    "width_pixels",
                    "height_pixels",
                    "inventory_digest",
                    "proposal_count",
                    "object_ids",
                    "object_decisions",
                    *tuple(_authority_data()),
                    "manifest_digest",
                )
            ),
            "panel decision manifest",
        )
        if (
            raw["schema"]
            != OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA
            or raw["projection_id"]
            != OBJECT_SCENE_ANCHOR_PANEL_DECISION_PROJECTION_ID
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["object_ids"], list)
            or not isinstance(raw["object_decisions"], list)
        ):
            raise ObjectSceneAnchorPanelManifestError(
                "panel decision manifest policy differs"
            )
        try:
            decisions = tuple(
                ObjectSceneAnchorDecisionManifest.from_data(item)
                for item in raw["object_decisions"]
            )
        except (ObjectSceneAnchorCatalogError, TypeError) as exc:
            raise ObjectSceneAnchorPanelManifestError(
                "panel object decision differs"
            ) from exc
        result = cls(
            panel_digest=raw["panel_digest"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            inventory_digest=raw["inventory_digest"],
            proposal_count=raw["proposal_count"],
            object_ids=tuple(raw["object_ids"]),
            object_decisions=decisions,
            manifest_digest=raw["manifest_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorPanelManifestError(
                "panel decision manifest is not canonical"
            )
        return result


def _project_verified_catalog(
    catalog: ObjectSceneAnchorCatalog,
) -> ObjectSceneAnchorPanelDecisionManifest:
    values = {
        "panel_digest": catalog.panel_digest,
        "width_pixels": catalog.width_pixels,
        "height_pixels": catalog.height_pixels,
        "inventory_digest": catalog.inventory_digest,
        "proposal_count": catalog.proposal_count,
        "object_ids": catalog.object_ids,
        "object_decisions": tuple(
            item.decision_manifest for item in catalog.entries
        ),
    }
    provisional = object.__new__(ObjectSceneAnchorPanelDecisionManifest)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPanelDecisionManifest(
        **values,
        manifest_digest=canonical_digest(_manifest_content(provisional)),
    )


def build_object_scene_anchor_panel_decision_manifest(
    catalog: ObjectSceneAnchorCatalog,
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    *,
    expected_catalog_digest: str | None = None,
) -> ObjectSceneAnchorPanelDecisionManifest:
    """Cold-verify a full catalog, then expose only its complete decision view."""

    verified = verify_object_scene_anchor_catalog(
        catalog,
        png_bytes,
        inventory,
        expected_catalog_digest=expected_catalog_digest,
    )
    return _project_verified_catalog(verified)


def verify_object_scene_anchor_panel_decision_manifest(
    manifest: ObjectSceneAnchorPanelDecisionManifest,
    catalog: ObjectSceneAnchorCatalog,
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    *,
    expected_manifest_digest: str | None = None,
    expected_catalog_digest: str | None = None,
) -> ObjectSceneAnchorPanelDecisionManifest:
    """Cold-replay the complete inventory projection from pixels."""

    if type(manifest) is not ObjectSceneAnchorPanelDecisionManifest:
        raise TypeError(
            "manifest must be exact ObjectSceneAnchorPanelDecisionManifest"
        )
    restored = ObjectSceneAnchorPanelDecisionManifest.from_data(
        manifest.to_data()
    )
    if expected_manifest_digest is not None and restored.manifest_digest != _digest(
        expected_manifest_digest, "expected panel decision manifest digest"
    ):
        raise ObjectSceneAnchorPanelManifestError(
            "panel decision manifest differs from commitment"
        )
    replayed = build_object_scene_anchor_panel_decision_manifest(
        catalog,
        png_bytes,
        inventory,
        expected_catalog_digest=expected_catalog_digest,
    )
    if replayed != restored:
        raise ObjectSceneAnchorPanelManifestError(
            "panel decision manifest differs from exact catalog replay"
        )
    return restored


__all__ = (
    "OBJECT_SCENE_ANCHOR_PANEL_DECISION_MANIFEST_SCHEMA",
    "OBJECT_SCENE_ANCHOR_PANEL_DECISION_PROJECTION_ID",
    "ObjectSceneAnchorPanelDecisionManifest",
    "ObjectSceneAnchorPanelManifestError",
    "build_object_scene_anchor_panel_decision_manifest",
    "verify_object_scene_anchor_panel_decision_manifest",
)
