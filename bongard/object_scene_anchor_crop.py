"""Exact full-style object crops for anchor-based vision judgments.

Anchor atlases make the binding explicit but intentionally discard stroke and
marker style.  This module replays the canonical masked-strength crop for one
catalog entry and encodes it as a deterministic grayscale PNG.  The observer
therefore sees both the original visual style and the exact anchor map.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from bongard import object_scene_visual_frontend as _frontend
from bongard import prototype_object_hypotheses as _hypotheses
from bongard.object_scene_anchor_atlas import _encode_grayscale_png
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorCatalogEntry
from bongard.object_scene_visual_frontend import (
    ObjectSceneProposalInventory,
    verify_object_scene_proposal_inventory,
)


OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID: Final[str] = (
    "bongard.object-scene-anchor-crop/canonical-masked-strength-l-png-v1"
)


class ObjectSceneAnchorCropError(ValueError):
    """An object crop does not replay from its exact pixels and receipt."""


def _canonical_crop_strength(
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    entry: ObjectSceneAnchorCatalogEntry,
) -> np.ndarray:
    if not isinstance(png_bytes, bytes):
        raise TypeError("panel input must be exact PNG bytes")
    if type(inventory) is not ObjectSceneProposalInventory:
        raise TypeError("inventory must be exact ObjectSceneProposalInventory")
    if type(entry) is not ObjectSceneAnchorCatalogEntry:
        raise TypeError("entry must be exact ObjectSceneAnchorCatalogEntry")
    verified_inventory = verify_object_scene_proposal_inventory(
        inventory,
        png_bytes,
        expected_inventory_digest=inventory.inventory_digest,
    )
    frozen_entry = ObjectSceneAnchorCatalogEntry.from_data(entry.to_data())
    if frozen_entry.inventory_index >= len(verified_inventory.objects):
        raise ObjectSceneAnchorCropError("catalog entry is outside the inventory")
    receipt = verified_inventory.objects[frozen_entry.inventory_index]
    if (
        receipt.object_id != frozen_entry.object_id
        or receipt.receipt_digest != frozen_entry.crop_receipt_digest
        or receipt.lineage_id != frozen_entry.lineage_id
        or receipt.lineage_digest != frozen_entry.lineage_digest
        or receipt.scenario_id != frozen_entry.scenario_id
        or receipt.hypothesis_id != frozen_entry.hypothesis_id
        or receipt.hypothesis_digest != frozen_entry.hypothesis_digest
        or receipt.masked_crop_pixel_digest
        != frozen_entry.masked_crop_pixel_digest
    ):
        raise ObjectSceneAnchorCropError(
            "catalog entry differs from the exact inventory receipt"
        )
    packet = _hypotheses.extract_object_hypothesis_packet(png_bytes)
    _hypotheses.verify_object_hypothesis_packet(packet, png_bytes)
    if packet.digest() != verified_inventory.hypothesis_packet_digest:
        raise ObjectSceneAnchorCropError(
            "crop hypothesis packet differs from the inventory"
        )
    crop = _frontend._hypothesis_crop_map(png_bytes, packet).get(
        (frozen_entry.scenario_id, frozen_entry.hypothesis_id)
    )
    if (
        type(crop) is not np.ndarray
        or crop.dtype != np.uint8
        or crop.ndim != 2
        or crop.shape
        != (frozen_entry.crop_height_pixels, frozen_entry.crop_width_pixels)
        or _hypotheses._crop_pixel_digest(crop)
        != frozen_entry.masked_crop_pixel_digest
    ):
        raise ObjectSceneAnchorCropError(
            "canonical crop pixels differ from the catalog entry"
        )
    return np.ascontiguousarray(crop, dtype=np.uint8)


def render_object_scene_anchor_object_crop(
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    entry: ObjectSceneAnchorCatalogEntry,
) -> bytes:
    """Return one native-resolution, full-style deterministic grayscale PNG."""

    strength = _canonical_crop_strength(png_bytes, inventory, entry)
    luminance = np.ascontiguousarray(255 - strength, dtype=np.uint8)
    return _encode_grayscale_png(luminance)


def verify_object_scene_anchor_object_crop(
    crop_png_bytes: bytes,
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    entry: ObjectSceneAnchorCatalogEntry,
) -> bytes:
    """Cold-replay one presented object crop from the original panel pixels."""

    if not isinstance(crop_png_bytes, bytes):
        raise TypeError("crop input must be exact PNG bytes")
    replayed = render_object_scene_anchor_object_crop(
        png_bytes, inventory, entry
    )
    if replayed != crop_png_bytes:
        raise ObjectSceneAnchorCropError(
            "object crop differs from exact panel replay"
        )
    return crop_png_bytes


__all__ = (
    "OBJECT_SCENE_ANCHOR_CROP_RENDERER_ID",
    "ObjectSceneAnchorCropError",
    "render_object_scene_anchor_object_crop",
    "verify_object_scene_anchor_object_crop",
)
