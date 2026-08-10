from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_ROOT = PROJECT_ROOT / "references" / "hardware"
XACRO_PATH = HARDWARE_ROOT / "roarm_m2.xacro"
TRANSFORMS_PATH = HARDWARE_ROOT / "transforms.json"


def split_vector(element: ET.Element, attribute: str) -> list[str]:
    return element.attrib[attribute].split()


def extracted_joint(joint: ET.Element) -> dict[str, object]:
    origin = joint.find("origin")
    parent = joint.find("parent")
    child = joint.find("child")
    axis = joint.find("axis")
    limit = joint.find("limit")
    assert origin is not None
    assert parent is not None
    assert child is not None
    assert axis is not None

    return {
        "name": joint.attrib["name"],
        "type": joint.attrib["type"],
        "parent": parent.attrib["link"],
        "child": child.attrib["link"],
        "origin_xyz": split_vector(origin, "xyz"),
        "origin_rpy": split_vector(origin, "rpy"),
        "axis_xyz": split_vector(axis, "xyz"),
        "limit": (
            {
                "lower": limit.attrib["lower"],
                "upper": limit.attrib["upper"],
            }
            if limit is not None
            else None
        ),
    }


def test_preserved_xacro_digest_matches_manifest() -> None:
    manifest = json.loads(TRANSFORMS_PATH.read_text(encoding="utf-8"))
    observed = hashlib.sha256(XACRO_PATH.read_bytes()).hexdigest()
    assert observed == manifest["source"]["sha256"]
    assert len(manifest["source"]["commit"]) == 40
    assert manifest["source"]["commit"] in (
        HARDWARE_ROOT / "PROVENANCE.md"
    ).read_text(encoding="utf-8")


def test_transform_fixture_is_exact_xacro_extraction() -> None:
    manifest = json.loads(TRANSFORMS_PATH.read_text(encoding="utf-8"))
    root = ET.parse(XACRO_PATH).getroot()
    observed = [extracted_joint(joint) for joint in root.findall("joint")]
    assert observed == manifest["joints"]


def test_exact_articulated_chain_is_present() -> None:
    manifest = json.loads(TRANSFORMS_PATH.read_text(encoding="utf-8"))
    assert [joint["name"] for joint in manifest["joints"]] == [
        "world_to_base_link",
        "base_link_to_link1",
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_gripper_link",
        "link3_to_hand_tcp",
    ]
