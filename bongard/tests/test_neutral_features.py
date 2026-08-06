from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import inspect

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from bongard.evidence import Disposition
from bongard.legs.neutral_features import (
    FEATURE_GROUP_IDS,
    NeutralFeatureReceipt,
    extract_neutral_features,
    feature_group_catalog,
    feature_group_catalog_digest,
    feature_space_for_group,
    feature_space_for_groups,
    full_neutral_feature_space,
    project_neutral_feature_extraction,
    verify_neutral_feature_extraction,
)


def _png(panel: np.ndarray, *, note: str | None = None) -> bytes:
    output = BytesIO()
    metadata = None
    if note is not None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("receipt-test", note)
    Image.fromarray(panel, mode="L").save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def _panel() -> np.ndarray:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    panel[20:72, 26:65] = 0
    panel[36:56, 38:53] = 255
    return panel


def _values(extraction: object) -> dict[str, tuple[float, float]]:
    packet = extraction.evidence.unwrap()  # type: ignore[attr-defined]
    return {item.name: (item.lower, item.upper) for item in packet.values}


def test_catalog_is_finite_content_addressed_and_uses_shared_extractor() -> None:
    assert FEATURE_GROUP_IDS == (
        "prototype.topology",
        "prototype.global_geometry",
        "prototype.moments_symmetry",
        "prototype.boundary_angle",
    )
    catalog = feature_group_catalog()
    assert tuple(item.group_id for item in catalog) == FEATURE_GROUP_IDS
    assert len({name for item in catalog for name in item.dimension_names}) == 18
    assert all(item.description for item in catalog)
    assert len(feature_group_catalog_digest()) == 64

    spaces = tuple(feature_space_for_group(group_id) for group_id in FEATURE_GROUP_IDS)
    assert len({space.digest() for space in spaces}) == 4
    assert len({space.extractor_id for space in spaces}) == 1
    assert len({space.extractor_version for space in spaces}) == 1
    assert len({space.extractor_artifact_digest for space in spaces}) == 1
    assert len({space.receipt_protocol_digest for space in spaces}) == 1


def test_raw_extractor_has_only_panel_input_and_is_exactly_deterministic(tmp_path) -> None:
    assert tuple(inspect.signature(extract_neutral_features).parameters) == ("panel",)
    raw = _png(_panel())
    path = tmp_path / "panel.png"
    path.write_bytes(raw)

    first = extract_neutral_features(raw)
    second = extract_neutral_features(raw)
    from_path = extract_neutral_features(path)

    assert first == second == from_path
    assert first.evidence.disposition is Disposition.PRESENT
    packet = first.evidence.unwrap()
    assert packet.panel_digest == hashlib.sha256(raw).hexdigest()
    assert packet.feature_space_digest == full_neutral_feature_space().digest()
    assert packet.extractor_receipt_digest == first.receipt.digest()
    assert first.receipt.input_identity.byte_count == len(raw)
    assert first.receipt.input_identity.media_type == "image/png"
    assert verify_neutral_feature_extraction(first, raw) == packet


def test_exact_container_bytes_are_bound_even_when_pixels_match() -> None:
    first_raw = _png(_panel(), note="first")
    second_raw = _png(_panel(), note="second")
    first = extract_neutral_features(first_raw)
    second = extract_neutral_features(second_raw)

    assert first_raw != second_raw
    assert _values(first) == _values(second)
    assert first.evidence.unwrap().panel_digest != second.evidence.unwrap().panel_digest
    assert first.receipt.digest() != second.receipt.digest()


def test_topology_coordinates_record_components_and_holes_without_semantics() -> None:
    ring = np.full((96, 96), 255, dtype=np.uint8)
    ring[16:72, 18:74] = 0
    ring[26:62, 28:64] = 255
    blocks = np.full((96, 96), 255, dtype=np.uint8)
    blocks[18:42, 16:40] = 0
    blocks[54:78, 56:80] = 0

    ring_values = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(ring)), "prototype.topology"
        )
    )
    block_values = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(blocks)), "prototype.topology"
        )
    )

    assert ring_values["component_count"] == (1.0, 1.0)
    assert ring_values["hole_count"] == (1.0, 1.0)
    assert ring_values["euler_characteristic"] == (0.0, 0.0)
    assert block_values["component_count"] == (2.0, 2.0)
    assert block_values["hole_count"] == (0.0, 0.0)
    assert block_values["euler_characteristic"] == (2.0, 2.0)


def test_fixed_threshold_ensemble_produces_intervals_not_chosen_points() -> None:
    panel = _panel()
    panel[24:68, 65:78] = 200  # white-distance 55: present only at threshold 32.
    extraction = extract_neutral_features(_png(panel))
    values = _values(extraction)

    assert extraction.evidence.disposition is Disposition.PRESENT
    lower, upper = values["foreground_area_fraction"]
    assert lower < upper
    assert values["bbox_width_fraction"][0] < values["bbox_width_fraction"][1]


def test_boundary_angle_and_moment_coordinates_are_geometric_proxies() -> None:
    axis = np.full((96, 96), 255, dtype=np.uint8)
    axis[26:70, 22:74] = 0
    diagonal = np.full((96, 96), 255, dtype=np.uint8)
    yy, xx = np.indices(diagonal.shape)
    diagonal[(np.abs(yy - xx) <= 4) & (xx >= 16) & (xx <= 78)] = 0

    axis_boundary = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(axis)), "prototype.boundary_angle"
        )
    )
    diagonal_boundary = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(diagonal)), "prototype.boundary_angle"
        )
    )
    axis_moments = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(axis)), "prototype.moments_symmetry"
        )
    )
    diagonal_moments = _values(
        project_neutral_feature_extraction(
            extract_neutral_features(_png(diagonal)), "prototype.moments_symmetry"
        )
    )

    assert diagonal_boundary["diagonal_gradient_fraction"][0] > axis_boundary[
        "diagonal_gradient_fraction"
    ][1]
    assert diagonal_moments["principal_axis_obliqueness_fraction"][0] > 0.9
    assert axis_moments["principal_axis_obliqueness_fraction"][1] < 0.1


def test_all_four_dispositions_are_distinct_and_failure_is_never_negative(
    tmp_path,
) -> None:
    present = extract_neutral_features(_png(_panel()))
    blank = extract_neutral_features(_png(np.full((64, 64), 255, np.uint8)))
    tiny_panel = np.full((64, 64), 255, np.uint8)
    tiny_panel[30:34, 30:34] = 0
    tiny = extract_neutral_features(_png(tiny_panel))
    clipped_panel = np.full((64, 64), 255, np.uint8)
    clipped_panel[:12, 20:44] = 0
    clipped = extract_neutral_features(_png(clipped_panel))
    malformed = extract_neutral_features(b"not a png")
    missing = extract_neutral_features(tmp_path / "missing.png")

    assert present.evidence.disposition is Disposition.PRESENT
    assert blank.evidence.disposition is Disposition.CERTIFIED_ABSENT
    assert "zero foreground" in (blank.evidence.certificate or "")
    assert tiny.evidence.disposition is Disposition.INDETERMINATE
    assert clipped.evidence.disposition is Disposition.INDETERMINATE
    assert malformed.evidence.disposition is Disposition.ERROR
    assert missing.evidence.disposition is Disposition.ERROR
    assert malformed.evidence.disposition is not Disposition.CERTIFIED_ABSENT
    assert missing.evidence.disposition is not Disposition.CERTIFIED_ABSENT
    assert malformed.receipt.verify() is None
    assert missing.receipt.verify() is None


def test_receipt_round_trip_reconstructs_packet_and_rejects_tampering() -> None:
    extraction = project_neutral_feature_extraction(
        extract_neutral_features(_png(_panel())), "prototype.topology"
    )
    receipt = extraction.receipt
    decoded = NeutralFeatureReceipt.from_data(receipt.to_data())

    assert decoded == receipt
    assert decoded.digest() == receipt.digest()
    assert decoded.verify() == extraction.evidence.unwrap()

    tampered = deepcopy(receipt.to_data())
    tampered["packet_commitment"]["values"][0]["upper"] += 1  # type: ignore[index]
    with pytest.raises(ValueError, match="commitment digest drift"):
        NeutralFeatureReceipt.from_data(tampered)

    with pytest.raises(ValueError, match="panel bytes differ"):
        verify_neutral_feature_extraction(extraction, _png(np.fliplr(_panel())))


def test_projection_is_closed_and_binds_exact_ordered_group_tuple() -> None:
    full = extract_neutral_features(_png(_panel()))
    topology = project_neutral_feature_extraction(full, "prototype.topology")
    ordered = project_neutral_feature_extraction(
        full, ("prototype.topology", "prototype.global_geometry")
    )
    reversed_order = project_neutral_feature_extraction(
        full, ("prototype.global_geometry", "prototype.topology")
    )

    assert topology.receipt.feature_group_ids == ("prototype.topology",)
    assert topology.receipt.parent_receipt_digest == full.receipt.digest()
    assert tuple(item.name for item in topology.evidence.unwrap().values) == tuple(
        item.name for item in feature_space_for_group("prototype.topology").dimensions
    )
    assert ordered.receipt.feature_group_ids == (
        "prototype.topology",
        "prototype.global_geometry",
    )
    assert ordered.receipt.digest() != reversed_order.receipt.digest()
    assert feature_space_for_groups(ordered.receipt.feature_group_ids).digest() != (
        feature_space_for_groups(reversed_order.receipt.feature_group_ids).digest()
    )
    with pytest.raises(ValueError, match="unique"):
        project_neutral_feature_extraction(
            full, ("prototype.topology", "prototype.topology")
        )
    with pytest.raises(ValueError, match="unknown"):
        project_neutral_feature_extraction(full, "prototype.semantic_bird")


def test_receipt_schema_contains_no_task_side_role_prose_or_candidate_fields() -> None:
    receipt = extract_neutral_features(_png(_panel())).receipt.to_data()
    forbidden = {
        "task_id",
        "side",
        "query_role",
        "prose",
        "proposal",
        "formula",
        "candidate",
        "label",
    }
    assert forbidden.isdisjoint(receipt)
    assert forbidden.isdisjoint(receipt["input_identity"])  # type: ignore[arg-type]
    assert forbidden.isdisjoint(receipt["packet_commitment"])  # type: ignore[arg-type]
