"""Sealed replay calibration for contrastive Bongard rubric predicates.

This module deliberately reuses only twelve panels that were durably released
by the 2026-08-07 object campaign.  It verifies their exact historical JSON and
PNG commitments, extracts the successful historical hypothesis catalogs, and
recomputes the current candidate-independent object lineages.  It never opens
the new broad cohort.

Vision is run before either support-side label is selected.  Each vision turn
jointly judges one frozen ordered target-versus-foil description; independently
scored adjectives are never subtracted downstream.  The only downstream
predicates are the two closed pure-Python rubric candidates from
``object_bongard_rubric_version_space``.  Lean is not imported or required.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard import prototype_object_scene_observer as _object_observer
from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    object_bongard_catalog_contrast_rubric,
    object_bongard_rubric_observer_output_schema,
    object_bongard_rubric_observer_prompt,
    observe_object_bongard_rubric,
    verify_object_bongard_rubric_observer_artifact,
)
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricCandidate,
    ObjectBongardRubricSupportVersionSpace,
    RUBRIC_SUPPORT_PANELS_PER_SIDE,
    RubricSupportGapKind,
    RubricSupportSide,
    build_object_bongard_rubric_support_version_space,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
    verify_object_bongard_turn_journal,
)
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    extract_object_hypothesis_packet,
    render_object_hypothesis_atlas,
)
from bongard.prototype_object_lineages import (
    ObjectLineagePacket,
    extract_object_lineage_packet,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CodexStructuredResult,
    run_codex_named_images_structured,
)


OBJECT_RUBRIC_CALIBRATION_SOURCE_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-source.v4"
)
OBJECT_RUBRIC_CALIBRATION_LIVE_OBSERVATION_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-live-observation.v1"
)
OBJECT_RUBRIC_CALIBRATION_OBSERVATION_BATCH_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-observation-batch.v2"
)
OBJECT_RUBRIC_CALIBRATION_COUNTS_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-candidate-counts.v1"
)
OBJECT_RUBRIC_CALIBRATION_SPEC_ASSESSMENT_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-spec-assessment.v1"
)
OBJECT_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-assessment.v2"
)
OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID = (
    "bongard.object-rubric-calibration/exact-released-12-single-signed-v4"
)

DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE = Path(
    "downloads/ShapeBongard_V2_full/"
    "prototype_pair_python_campaign_20260807_object_v1/objects"
)

CALIBRATION_GROUP_A_ORDINALS = (0, 1, 3, 4, 5, 7)
CALIBRATION_GROUP_B_ORDINALS = (14, 17, 18, 19, 20, 21)
CALIBRATION_FIT_GROUP_A_ORDINALS = (0, 1, 3)
CALIBRATION_FIT_GROUP_B_ORDINALS = (14, 17, 18)
CALIBRATION_CONFIRM_GROUP_A_ORDINALS = (4, 5, 7)
CALIBRATION_CONFIRM_GROUP_B_ORDINALS = (19, 20, 21)
CALIBRATION_SELECTED_ORDINALS = (
    CALIBRATION_GROUP_A_ORDINALS + CALIBRATION_GROUP_B_ORDINALS
)

_PLAN_FILE_SHA256 = (
    "0447225b24bd440299f7976d29332d9bcd98a5f3d3d10b4fd453eb1ca634dc2c"
)
_PLAN_RECORD_DIGEST = (
    "sha256:d5643e8efc0fefaddeecd1fe90c2d47dfe25fc49aa401a06a6ba68615560102d"
)
_DESCRIPTION_FILE_SHA256 = (
    "b4746e97a07887f3a646b68fd02ba36a49bdb50a4a1c765e6cf7e19a03229a48"
)
_DESCRIPTION_ARTIFACT_DIGEST = (
    "3e0f8999aa0d9dd23dd3f662ab86fedbcf4b29f8761b4ddd71d6a58e58881902"
)

_RUBRIC_ROWS = (
    (
        "A joined pair of visibly mismatched sector-like subshapes recurs.",
        "paired_sector_mismatch_support_ppm",
    ),
    (
        "A rounded, bird-like contour arrangement recurs.",
        "bird_like_support_ppm",
    ),
)

_CALIBRATION_SIGNED_CUE_PAIR = (_RUBRIC_ROWS[0][1], _RUBRIC_ROWS[1][1])

# ordinal -> (task_id, panel_id, released file SHA-256, released record digest,
# PNG SHA-256, old observer file SHA-256, old observer artifact digest,
# historical hypothesis-packet digest).  Historical packets are provenance,
# not current geometry authority: the loader recomputes a current packet from
# the pinned PNG.  A matching panel name is not sufficient admission.
_PINNED_PANELS: dict[int, tuple[str, ...]] = {
    0: (
        "bd_two_mismatch_sectors8-thin_seven_lines2_0000",
        "bd/bd_two_mismatch_sectors8-thin_seven_lines2_0000/1/0.png",
        "d0348829244ca07a387b9621f664fdbe6fc639ca43018caaefd43592a10928e6",
        "sha256:62c612838ecc986ceeb8e78b63d81b513d0ed0586c2697134c901ca7bf070d68",
        "71308a64182c5c193b46a7f3bd285194e5d71e42f44501b1b8180bbc8269c5d2",
        "c96867722ebf00c0d36fdcb8c3c4fc2a901a86c740c4cc11547f59f6266d437c",
        "0c49c2ec8299a5d12452b879cb6d3d6af1862799ebf5987fea87804d085e3793",
        "0bf7d8ad7748f8e97433ed096bbf6351a1d9670c9a5cc237208d5f06b87c0750",
    ),
    1: (
        "bd_two_mismatch_sectors8-thin_seven_lines1_0000",
        "bd/bd_two_mismatch_sectors8-thin_seven_lines1_0000/1/2.png",
        "48cb46ab86d09ff1fad6f5f79da72b6fb9fb7ad72ee81c6af17479b049a6c36d",
        "sha256:7204fb38d6d97f3b7ab96cb713cbc8a7527d650905badbe1d2830b54999b9537",
        "328bb589fbefa3bacbd0118fcf45985d8973a2d72d626a9212a047f9e5dbe809",
        "f48e017b6c6af30e88c2f6d1a12ffd8141599da58aa6f06fd907497dc2903011",
        "102e59903fa8e95e36cdd6d5f6c02a41cd9ee536ddb54abbd38d1afa044a71b9",
        "b762ef45beeeb3bf19177fe92ad18fc0316af0c425a5748b357b0f271fc7ab56",
    ),
    3: (
        "bd_two_mismatch_sectors8-advanced_lamp5_0000",
        "bd/bd_two_mismatch_sectors8-advanced_lamp5_0000/1/0.png",
        "36ccb57ff6c3d8f76b64f03c05ad4d98aed9bc20217cd1cd945f085ce8b6e45b",
        "sha256:79d9c48d8779027f17e895121f57e676786836465d642b360584dce6c05df159",
        "7422d3b033451f92486091739e3ad593b97889a0bd9746938f82b1461c73c29a",
        "9145fdc4d1b99f8e8cb4765e76c1e1030a430e57f8d66aee4266667526702032",
        "b9f8e533e40c1b8d3268b9fbd5f65239e79e3bbbd4c403e74638808393b2a13f",
        "6e45cb7c0336e84d5946a6a44bcbfd11b556f61684bdab395614f6feaa6600a2",
    ),
    4: (
        "bd_two_mismatch_sectors8-three_mismatch_triangles4_0000",
        "bd/bd_two_mismatch_sectors8-three_mismatch_triangles4_0000/1/2.png",
        "ff529a338fd2ba4cf037aeb92474df314b85e27e19d96dd59cbd9b470ae7e39e",
        "sha256:d14c18c14e3b7c38d2c61ec4859cb515bec3b25cae20824af41186e6b280ec5b",
        "a818d3f3d730ddfeed399075ed1d3bf23f91c876c2bdc5bae046fcd8a7c85bf8",
        "661636fe7fddbc33846296ad56e89abc6b858a56b84b3eea17297afa5b3b6ce9",
        "3631011efe62e55095b2dc5f645d6a66c17aafc803f2ca86ecb7332ead6169d5",
        "b21c76911f0a1684a1a761c70767579275954052c233212ed18d278acd5c13b9",
    ),
    5: (
        "bd_open_line_arc5-two_mismatch_sectors8_0000",
        "bd/bd_open_line_arc5-two_mismatch_sectors8_0000/1/6.png",
        "b044be705b4d65d8cb16af956f14d5b9f6fe9014e7a653d04ca15385dc1c8cbe",
        "sha256:cf22b0feac983d8170939babe0c0058322858ac63785336c976729c4d3033d87",
        "6ee727aeded6b2c20c0ef1d9f34b9faed2fe22e7872019101ecc565eca7b8919",
        "c4196d475278e8338fb97b86c4296675b8f2048e64297f6696a910430887bad3",
        "d4fe076df72554012db20537dd8ddb0967dc67b6fa0f69f30241eaf199e4dbc6",
        "a368aa88bd38754b89e115e784ea0ab218a6a9036ef0224d4de938ffd764f969",
    ),
    7: (
        "bd_thin_parallel_bridge-two_mismatch_sectors8_0000",
        "bd/bd_thin_parallel_bridge-two_mismatch_sectors8_0000/1/2.png",
        "bf4a4430fe1e5beac31468cf78c82c5d34eba615aa4d262af3d26acbc77ffb73",
        "sha256:7d416534b9255e4ad8c097502ce29634716c97891385caecc7b74946c4e73501",
        "0ea165d64ac9935d73237f78a14f948a7d6a84e32e64127dcf4e5475f8ccf64c",
        "3bd63eb570e46dd4582356cfce1528cd14eb25babb23f4c0d6786106304baf0b",
        "d443ea80904fd4b74c12ae6c71228b0624978844d66df6360acde0c132d7f714",
        "8107155d53d405e4d26aed0dfad1c3703c630d5f0e523cc6fb75b51ec2754f31",
    ),
    14: (
        "bd_acute_nonequi_triangle-exist_triangle_three_lines4_0000",
        "bd/bd_acute_nonequi_triangle-exist_triangle_three_lines4_0000/1/2.png",
        "660a8dd1c80a692d68ae35e10f9b939d3ca536f7a83c9704c9d694c21690b957",
        "sha256:7130787849334e9f2c45b6f29ae4920241cb42749e44d9b6e6d9245252ecab14",
        "14847a726f5e80fdd68ff82515007ddee47108ebc0c7c84e2dc1e94e08a8fd34",
        "0b8d582d9e75b814497918ac03960b2420a3b61cc0ef4efcd915c5d719bc1e5e",
        "062cb476f7997d89f464d821ea93bbc3388bab538fd44760ce4c032d860825d4",
        "11460fc047f71192c7c05bd385f3380ddab4ed46e6c42e78241997fe6c2ffaae",
    ),
    17: (
        "bd_two_mismatch_triangles6-exist_triangle_three_lines4_0000",
        "bd/bd_two_mismatch_triangles6-exist_triangle_three_lines4_0000/1/6.png",
        "af7f676535cca1bf15051ec23d9ba6de0b7158ec2f5fd2acf69fe6ecab8a81eb",
        "sha256:dd835e1855a2c07210f416aec1cbdac091545e4a321d9e6024db131f6f6e99fd",
        "1f35c3c4bf1e138cc25beec5615cee59baddab98a69a244147e121dd1ac3605d",
        "742c782a44a3cb6f90c80cd113d57d161dd4001ea682b95e28b12460f6e4c9dc",
        "647eea54e209ca984624714422a7bd47b3c31a06fffa4f38f767a47199280732",
        "29b09ba84b8077b5d9e0314ac7874a3d5b2fe2d94986b8b7029047c64b062efc",
    ),
    18: (
        "bd_thin_symm_band-exist_triangle_three_lines4_0000",
        "bd/bd_thin_symm_band-exist_triangle_three_lines4_0000/1/1.png",
        "18739494e35a4fbefcc4468996399c46ec304cd0abf4b97beef06c8a4b81cef3",
        "sha256:18f0b91009dcbd06e5a3788f630b87d06feb35f661086f3963400d63333ecc9a",
        "95ac8cd158f36268e1e96e863d49fb5a8c0f38a8ab05955d4667020f67e701bf",
        "14e6a54a2823e7d11d95ff4e53081a870923e8bc4f33fa921ac16688d17e9929",
        "87a50118bf08e593d0b71121b6195cc2fa3b5bda5acd5a81bbffff30980418b8",
        "569656cfbefbcc47bc83b28410e19619ab23d353e3157503b7f3262a823c0efe",
    ),
    19: (
        "bd_dagger_sector1-exist_triangle_three_lines4_0000",
        "bd/bd_dagger_sector1-exist_triangle_three_lines4_0000/1/5.png",
        "cf3235e51c7d84cd280ab7a74df9a5cb40e7c19d605f9dffe30913363cce0fd0",
        "sha256:3b98dadc3955b264e7af1ebb3ebc27a879d2f1d9937b4063799ecada073c4394",
        "d2c77806e01ffff998969901529344e34786fd408aaed3712920d8932cacb8ab",
        "313df2826d00c7f99bb356ef436cfd78fdfff0e3de1e37feeffbcd23c54573ff",
        "bfea505d13e146863b586d08577cce6e3b996c863a0d157802b91a765263d74f",
        "431e9ec8d36de7a186ccea262efa8e1626f771468fb5542988c19c59c87c1419",
    ),
    20: (
        "bd_thin_regular_lamp-exist_triangle_three_lines4_0000",
        "bd/bd_thin_regular_lamp-exist_triangle_three_lines4_0000/1/2.png",
        "6174740819ba35114a070aadfcd0770e63f87df171e42fb5a926e7a358c64915",
        "sha256:273cafea58880aee46dac8592e0a3a39c3bc005b98aece5c69c55f49306d4989",
        "d27b9a673ce1768729f3a533321f05c4a2addd1619011df85103d5d43f55a933",
        "cf5b5f3a67ba9e7135bc5eb1bfd2fb300a5be4751cf0be5fd9cf0972be55ab13",
        "e9e5b60228621978dcbf09076c406df6c7142ce36cce7bf3e4e3311ae5169a55",
        "7a0c645d22cb132194c774627d7672bdc222832a88c56a799ea26b08d6078750",
    ),
    21: (
        "bd_exist_triangle_three_lines4-thin_seven_lines3_0000",
        "bd/bd_exist_triangle_three_lines4-thin_seven_lines3_0000/1/5.png",
        "15eb3f56bdb656a59f9d64cc30995176d03d2b017f74aa9844ea9be383956901",
        "sha256:f42d0b372be5993525fee462644534c996c4adfddedd215d692bec1b006cbc27",
        "a348487f86bf2f1c306789ef3f386934d333a9fcf7337c3888d6c9762b834859",
        "d1bc21c7e4a9e35f2c2cc48d6f73611e52b1f8cac1f5426a8928d74a8bc8735a",
        "d36c6e7a0cb31eb3e687f0360a4f7e459b1873c8338c490bd4ea4a911bac6a1b",
        "897fb2574609b3a29e78a4dd7d0f66abb07ca6820d247ac2c13e60c675fe2bcf",
    ),
}

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_SOURCE_RECORD_BYTES = 8 * 1024 * 1024


class ObjectBongardRubricCalibrationError(ValueError):
    """A source commitment, live turn, assessment, or replay is invalid."""


class ObjectBongardRubricCalibrationGroup(str, Enum):
    GROUP_A = "group_a"
    GROUP_B = "group_b"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _calibration_rubric_specs() -> tuple[ObjectBongardRubricSpec, ...]:
    """Legacy source's single canonical group-A-over-group-B orientation."""

    target, foil = _CALIBRATION_SIGNED_CUE_PAIR
    return (
        ObjectBongardRubricSpec.create(
            _DESCRIPTION_ARTIFACT_DIGEST,
            object_bongard_catalog_contrast_rubric(target, foil),
            (target, foil),
        ),
    )


def _nominated_rubric_specs(
    artifact: ObjectBongardSemanticArtifact,
) -> tuple[ObjectBongardRubricSpec, ...]:
    """Derive one canonical group-0-over-group-1 signed comparison."""

    forward = ObjectBongardRubricSpec.from_semantic_artifact(
        artifact, expected_artifact_digest=artifact.artifact_digest
    )
    return (forward,)


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricCalibrationError(
            f"{label} must be a lowercase raw SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricCalibrationError(
            f"{label} must be a sha256: address"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricCalibrationError(f"{label} fields differ")
    return value


def object_bongard_rubric_calibration_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _read_exact_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    """Read one pinned canonical-JSON-plus-newline record without symlinks."""

    digest = _raw_digest(expected_file_sha256, "source file digest")
    if path.name != f"{digest}.json":
        raise ObjectBongardRubricCalibrationError(
            "source record filename differs from its commitment"
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationError(
            "pinned source record is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not 0 < before.st_size <= _MAX_SOURCE_RECORD_BYTES
    ):
        raise ObjectBongardRubricCalibrationError(
            "pinned source record is not a bounded regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObjectBongardRubricCalibrationError(
            "cannot open pinned source record"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ObjectBongardRubricCalibrationError(
                "pinned source record changed while opening"
            )
        payload = bytearray()
        while len(payload) < opened.st_size:
            chunk = os.read(descriptor, min(65536, opened.st_size - len(payload)))
            if not chunk:
                raise ObjectBongardRubricCalibrationError(
                    "pinned source record was truncated"
                )
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ObjectBongardRubricCalibrationError(
                "pinned source record changed while reading"
            )
    finally:
        os.close(descriptor)
    frozen = bytes(payload)
    if hashlib.sha256(frozen).hexdigest() != digest:
        raise ObjectBongardRubricCalibrationError(
            "pinned source record bytes differ from their commitment"
        )
    try:
        decoded = json.loads(frozen.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricCalibrationError(
            "pinned source record is malformed JSON"
        ) from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != frozen:
        raise ObjectBongardRubricCalibrationError(
            "pinned source record is not canonical JSON plus one newline"
        )
    return decoded


def _verify_record_digest(
    value: Mapping[str, Any], *, field: str, addressed: bool, label: str
) -> None:
    if field not in value:
        raise ObjectBongardRubricCalibrationError(f"{label} lacks {field}")
    expected = canonical_digest({key: item for key, item in value.items() if key != field})
    claimed = value[field]
    if claimed != ("sha256:" + expected if addressed else expected):
        raise ObjectBongardRubricCalibrationError(
            f"{label} canonical record digest differs"
        )


def _group_for_ordinal(ordinal: int) -> ObjectBongardRubricCalibrationGroup:
    if ordinal in CALIBRATION_GROUP_A_ORDINALS:
        return ObjectBongardRubricCalibrationGroup.GROUP_A
    if ordinal in CALIBRATION_GROUP_B_ORDINALS:
        return ObjectBongardRubricCalibrationGroup.GROUP_B
    raise ObjectBongardRubricCalibrationError(
        "ordinal is outside the exact calibration selection"
    )


def _panel_content(value: "ObjectBongardRubricCalibrationPanel") -> dict[str, object]:
    return {
        "ordinal": value.ordinal,
        "group": value.group.value,
        "task_id": value.task_id,
        "panel_id": value.panel_id,
        "released_file_sha256": value.released_file_sha256,
        "released_record_digest": value.released_record_digest,
        "png_sha256": value.png_sha256,
        "source_observer_file_sha256": value.source_observer_file_sha256,
        "source_observer_artifact_digest": value.source_observer_artifact_digest,
        "historical_hypothesis_packet_digest": (
            value.historical_hypothesis_packet_digest
        ),
        "current_hypothesis_packet_digest": value.hypothesis_packet.digest(),
        "lineage_packet_digest": value.lineage_packet.digest(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationPanel:
    ordinal: int
    group: ObjectBongardRubricCalibrationGroup
    task_id: str
    panel_id: str
    released_file_sha256: str
    released_record_digest: str
    png_sha256: str
    source_observer_file_sha256: str
    source_observer_artifact_digest: str
    historical_hypothesis_packet_digest: str
    exact_png_bytes: bytes
    hypothesis_packet: ObjectHypothesisPacket
    lineage_packet: ObjectLineagePacket
    panel_binding_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal not in CALIBRATION_SELECTED_ORDINALS
            or self.group is not _group_for_ordinal(self.ordinal)
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration panel ordinal/group differs"
            )
        pinned = _PINNED_PANELS[self.ordinal]
        if (
            (self.task_id, self.panel_id) != pinned[:2]
            or self.released_file_sha256 != pinned[2]
            or self.released_record_digest != pinned[3]
            or self.png_sha256 != pinned[4]
            or self.source_observer_file_sha256 != pinned[5]
            or self.source_observer_artifact_digest != pinned[6]
            or self.historical_hypothesis_packet_digest != pinned[7]
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration panel differs from exact historical commitment"
            )
        for name in (
            "released_file_sha256",
            "png_sha256",
            "source_observer_file_sha256",
            "source_observer_artifact_digest",
            "historical_hypothesis_packet_digest",
            "panel_binding_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.released_record_digest, "released record digest")
        if (
            not isinstance(self.exact_png_bytes, bytes)
            or hashlib.sha256(self.exact_png_bytes).hexdigest() != self.png_sha256
            or self.hypothesis_packet.panel_digest != self.png_sha256
            or self.lineage_packet.panel_digest != self.png_sha256
            or self.lineage_packet.hypothesis_packet_digest
            != self.hypothesis_packet.digest()
            or self.panel_binding_digest != canonical_digest(_panel_content(self))
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration panel byte/geometry binding differs"
            )

    def commitment_data(self) -> dict[str, object]:
        return {**_panel_content(self), "panel_binding_digest": self.panel_binding_digest}


def _source_content(value: "ObjectBongardRubricCalibrationSource") -> dict[str, object]:
    nomination = value.nomination_artifact
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_SOURCE_SCHEMA,
        "algorithm_id": OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "implementation_source_sha256": object_bongard_rubric_calibration_source_digest(),
        "historical_plan_file_sha256": value.historical_plan_file_sha256,
        "historical_plan_record_digest": value.historical_plan_record_digest,
        "historical_description_file_sha256": value.historical_description_file_sha256,
        "historical_description_artifact_digest": value.historical_description_artifact_digest,
        "selected_ordinals": list(CALIBRATION_SELECTED_ORDINALS),
        "selection_policy": "ordinal-first-prior-geometry-success-exactly-six-per-group",
        "rubric_derivation_policy": (
            "verified-historical-description-cues/canonical-group-a-over-group-b/v4"
            if nomination is None
            else "one-sealed-joint-contrastive-semantic-nomination-then-one-"
            "canonical-group-0-over-group-1-signed-orientation/v2"
        ),
        "historical_description_used_for_rubric_derivation": nomination is None,
        "nomination_binding": (
            None
            if nomination is None
            else {
                "artifact_digest": nomination.artifact_digest,
                "authorization_digest": value.nomination_authorization_digest,
                "execution_precommit_digest": value.nomination_precommit_digest,
                "cold_replay_digest": value.nomination_replay_digest,
                "command_result_digest": value.nomination_result_digest,
                "context_task_id_policy": (
                    "lowest-selected-ordinal-task-id-is-transport-context-only"
                ),
                "neutral_group_0_panel_ids": sorted(
                    item.panel_id for item in value.group_a_panels
                ),
                "neutral_group_1_panel_ids": sorted(
                    item.panel_id for item in value.group_b_panels
                ),
            }
        ),
        "vision_judgment_policy": "joint-same-turn-target-versus-foil",
        "panels": [item.commitment_data() for item in value.panels],
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "labels_consumed_while_observing": False,
        "fresh_broad_cohort_pixels_opened": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationSource:
    historical_plan_file_sha256: str
    historical_plan_record_digest: str
    historical_description_file_sha256: str
    historical_description_artifact_digest: str
    panels: tuple[ObjectBongardRubricCalibrationPanel, ...]
    rubric_specs: tuple[ObjectBongardRubricSpec, ...]
    nomination_artifact: ObjectBongardSemanticArtifact | None
    nomination_authorization_digest: str | None
    nomination_precommit_digest: str | None
    nomination_replay_digest: str | None
    nomination_result_digest: str | None
    source_digest: str

    def __post_init__(self) -> None:
        if (
            self.historical_plan_file_sha256 != _PLAN_FILE_SHA256
            or self.historical_plan_record_digest != _PLAN_RECORD_DIGEST
            or self.historical_description_file_sha256 != _DESCRIPTION_FILE_SHA256
            or self.historical_description_artifact_digest
            != _DESCRIPTION_ARTIFACT_DIGEST
        ):
            raise ObjectBongardRubricCalibrationError(
                "historical plan/description commitment differs"
            )
        if (
            not isinstance(self.panels, tuple)
            or tuple(item.ordinal for item in self.panels)
            != CALIBRATION_SELECTED_ORDINALS
            or len({item.panel_id for item in self.panels}) != len(self.panels)
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration source panel selection differs"
            )
        nomination_values = (
            self.nomination_artifact,
            self.nomination_authorization_digest,
            self.nomination_precommit_digest,
            self.nomination_replay_digest,
            self.nomination_result_digest,
        )
        if self.nomination_artifact is None:
            if any(item is not None for item in nomination_values[1:]):
                raise ObjectBongardRubricCalibrationError(
                    "legacy calibration source has partial nomination parents"
                )
            expected_specs = _calibration_rubric_specs()
        else:
            if any(item is None for item in nomination_values[1:]):
                raise ObjectBongardRubricCalibrationError(
                    "nominated calibration source lacks a nomination parent"
                )
            assert self.nomination_precommit_digest is not None
            assert self.nomination_authorization_digest is not None
            assert self.nomination_replay_digest is not None
            assert self.nomination_result_digest is not None
            _address(
                self.nomination_authorization_digest,
                "nomination authorization digest",
            )
            _address(
                self.nomination_precommit_digest,
                "nomination execution precommit digest",
            )
            _address(self.nomination_replay_digest, "nomination replay digest")
            _address(self.nomination_result_digest, "nomination result digest")
            artifact = ObjectBongardSemanticArtifact.from_data(
                self.nomination_artifact.to_data(),
                expected_artifact_digest=self.nomination_artifact.artifact_digest,
            )
            expected_group_0 = tuple(
                sorted(item.panel_id for item in self.group_a_panels)
            )
            expected_group_1 = tuple(
                sorted(item.panel_id for item in self.group_b_panels)
            )
            if (
                artifact != self.nomination_artifact
                or artifact.task_id != self.panels[0].task_id
                or artifact.observation_context_digest
                != self.nomination_precommit_digest
                or artifact.group_panel_ids
                != (expected_group_0, expected_group_1)
            ):
                raise ObjectBongardRubricCalibrationError(
                    "semantic nomination context or exact neutral groups differ"
                )
            support_png = {
                item.panel_id: item.exact_png_bytes for item in self.panels
            }
            verify_object_bongard_semantic_artifact(
                artifact,
                support_png_by_panel_id=support_png,
                expected_task_id=self.panels[0].task_id,
                expected_observation_context_digest=(
                    self.nomination_precommit_digest
                ),
                expected_artifact_digest=artifact.artifact_digest,
            )
            expected_specs = _nominated_rubric_specs(artifact)
        if self.rubric_specs != expected_specs:
            raise ObjectBongardRubricCalibrationError(
                "calibration rubric spec differs from the canonical frozen orientation"
            )
        _raw_digest(self.source_digest, "calibration source digest")
        if self.source_digest != canonical_digest(_source_content(self)):
            raise ObjectBongardRubricCalibrationError(
                "calibration source digest differs"
            )

    @property
    def group_a_panels(self) -> tuple[ObjectBongardRubricCalibrationPanel, ...]:
        return tuple(item for item in self.panels if item.group is ObjectBongardRubricCalibrationGroup.GROUP_A)

    @property
    def group_b_panels(self) -> tuple[ObjectBongardRubricCalibrationPanel, ...]:
        return tuple(item for item in self.panels if item.group is ObjectBongardRubricCalibrationGroup.GROUP_B)

    def panel_by_id(self, panel_id: str) -> ObjectBongardRubricCalibrationPanel:
        matches = tuple(item for item in self.panels if item.panel_id == panel_id)
        if len(matches) != 1:
            raise ObjectBongardRubricCalibrationError(
                "panel is outside the exact calibration source"
            )
        return matches[0]

    def to_data(self) -> dict[str, object]:
        return {**_source_content(self), "source_digest": self.source_digest}


def _bind_object_bongard_rubric_calibration_nomination_content(
    source: ObjectBongardRubricCalibrationSource,
    artifact: ObjectBongardSemanticArtifact,
    *,
    nomination_authorization_digest: str,
    nomination_precommit_digest: str,
    nomination_replay_digest: str,
    nomination_result_digest: str,
) -> ObjectBongardRubricCalibrationSource:
    """Mechanically bind content already verified by the command boundary."""

    if not isinstance(source, ObjectBongardRubricCalibrationSource):
        raise TypeError("source must be ObjectBongardRubricCalibrationSource")
    if source.nomination_artifact is not None:
        raise ObjectBongardRubricCalibrationError(
            "calibration source already has a semantic nomination"
        )
    frozen = ObjectBongardSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=artifact.artifact_digest
    )
    values = {
        "historical_plan_file_sha256": source.historical_plan_file_sha256,
        "historical_plan_record_digest": source.historical_plan_record_digest,
        "historical_description_file_sha256": (
            source.historical_description_file_sha256
        ),
        "historical_description_artifact_digest": (
            source.historical_description_artifact_digest
        ),
        "panels": source.panels,
        "rubric_specs": _nominated_rubric_specs(frozen),
        "nomination_artifact": frozen,
        "nomination_authorization_digest": nomination_authorization_digest,
        "nomination_precommit_digest": nomination_precommit_digest,
        "nomination_replay_digest": nomination_replay_digest,
        "nomination_result_digest": nomination_result_digest,
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationSource)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationSource(
        **values,
        source_digest=canonical_digest(_source_content(provisional)),
    )


def _verify_plan_and_description(
    plan: Mapping[str, Any], description: Mapping[str, Any]
) -> tuple[ObjectBongardRubricSpec, ...]:
    _verify_record_digest(
        plan, field="record_digest", addressed=True, label="historical calibration plan"
    )
    if (
        plan.get("schema") != "gkm.bongard-prototype-scene-calibration-plan.v1"
        or plan.get("record_digest") != _PLAN_RECORD_DIGEST
        or not isinstance(plan.get("scenes"), list)
        or len(plan["scenes"]) != 28
        or [item.get("ordinal") for item in plan["scenes"]] != list(range(28))
    ):
        raise ObjectBongardRubricCalibrationError(
            "historical calibration plan identity/scene inventory differs"
        )
    _verify_record_digest(
        description,
        field="artifact_digest",
        addressed=False,
        label="historical description artifact",
    )
    if (
        description.get("schema")
        != "gkm.bongard-object-profile-description-artifact.v1"
        or description.get("status") != "success"
        or description.get("artifact_digest") != _DESCRIPTION_ARTIFACT_DIGEST
        or not isinstance(description.get("rubrics"), list)
        or not isinstance(description.get("profiles"), list)
        or len(description["rubrics"]) != 2
        or len(description["profiles"]) != 2
    ):
        raise ObjectBongardRubricCalibrationError(
            "historical description identity/inventory differs"
        )
    for index, (prose, feature) in enumerate(_RUBRIC_ROWS):
        rubric = description["rubrics"][index]
        profile = description["profiles"][index]
        atoms = profile.get("atoms") if isinstance(profile, Mapping) else None
        if (
            not isinstance(rubric, Mapping)
            or rubric.get("state") != "defined"
            or rubric.get("prose") != prose
            or rubric.get("tag_id") != f"opaque_visual_tag_{index}"
            or rubric.get("group_id") != f"group_{index}"
            or not isinstance(atoms, list)
            or len(atoms) != 1
            or not isinstance(atoms[0], Mapping)
            or atoms[0].get("feature_id") != feature
            or atoms[0].get("operator") != "at_least"
        ):
            raise ObjectBongardRubricCalibrationError(
                "historical prose/profile row differs"
            )
    return _calibration_rubric_specs()


def load_object_bongard_rubric_calibration_source(
    directory: str | os.PathLike[str] = DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
) -> ObjectBongardRubricCalibrationSource:
    """Load and pixel-replay the exact already-exposed twelve-panel source."""

    root = Path(directory)
    plan = _read_exact_json(
        root / "calibration_plan" / f"{_PLAN_FILE_SHA256}.json",
        _PLAN_FILE_SHA256,
    )
    description = _read_exact_json(
        root / "description_artifact" / f"{_DESCRIPTION_FILE_SHA256}.json",
        _DESCRIPTION_FILE_SHA256,
    )
    specs = _verify_plan_and_description(plan, description)
    scenes = {item["ordinal"]: item for item in plan["scenes"]}
    panels: list[ObjectBongardRubricCalibrationPanel] = []
    cohort = plan.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ObjectBongardRubricCalibrationError("historical plan cohort is malformed")
    for ordinal in CALIBRATION_SELECTED_ORDINALS:
        (
            task_id,
            panel_id,
            released_file,
            released_digest,
            png_digest,
            observer_file,
            observer_digest,
            hypothesis_digest,
        ) = _PINNED_PANELS[ordinal]
        scene = scenes[ordinal]
        expected_states = (
            ("present", "absent")
            if ordinal in CALIBRATION_GROUP_A_ORDINALS
            else ("absent", "present")
        )
        states = scene.get("expected_tag_states")
        if (
            scene.get("task_id") != task_id
            or scene.get("panel_id") != panel_id
            or not isinstance(states, list)
            or tuple(item.get("state") for item in states) != expected_states
            or tuple(item.get("tag_id") for item in states)
            != ("opaque_visual_tag_0", "opaque_visual_tag_1")
        ):
            raise ObjectBongardRubricCalibrationError(
                "historical selected scene or support direction differs"
            )
        released_raw = _read_exact_json(
            root / "released_panel" / f"{released_file}.json", released_file
        )
        released = ReleasedOfficialPanel.from_data(released_raw)
        if (
            released.panel_id != panel_id
            or released.record_digest != released_digest
            or released.exact_png_digest != "sha256:" + png_digest
        ):
            raise ObjectBongardRubricCalibrationError(
                "historical released-panel binding differs"
            )
        observer = _read_exact_json(
            root / "observer_artifact" / f"{observer_file}.json", observer_file
        )
        _verify_record_digest(
            observer,
            field="artifact_digest",
            addressed=False,
            label="historical observer artifact",
        )
        if (
            observer.get("schema")
            != "gkm.bongard-object-scene-observer-artifact.v1"
            or observer.get("status") != "success"
            or observer.get("artifact_digest") != observer_digest
            or observer.get("plan_digest") != cohort.get("plan_digest")
            or observer.get("observation_context_digest") != _PLAN_RECORD_DIGEST
            or observer.get("rubric_description_digest")
            != _DESCRIPTION_ARTIFACT_DIGEST
            or observer.get("scene_task_id") != task_id
            or observer.get("scene_panel_id") != panel_id
            or observer.get("scene_digest") != png_digest
            or not isinstance(observer.get("hypothesis_packet"), Mapping)
        ):
            raise ObjectBongardRubricCalibrationError(
                "historical successful observer binding differs"
            )
        historical_hypothesis_packet_digest = canonical_digest(
            observer["hypothesis_packet"]
        )
        if historical_hypothesis_packet_digest != hypothesis_digest:
            raise ObjectBongardRubricCalibrationError(
                "historical hypothesis packet content digest differs"
            )
        # Historical packet metadata binds an old extractor implementation.  It
        # remains exact provenance, but is never admitted as current geometry.
        hypothesis_packet = extract_object_hypothesis_packet(
            released.exact_png_bytes
        )
        lineage_packet = extract_object_lineage_packet(
            released.exact_png_bytes, hypothesis_packet
        )
        # ``extract_object_lineage_packet`` is the current deterministic pixel
        # replay.  A second call through ``verify_object_lineage_packet`` would
        # repeat both hypothesis and lineage extraction without adding a new
        # commitment check; canonical round-trip is sufficient here.
        if ObjectLineagePacket.from_data(lineage_packet.to_data()) != lineage_packet:
            raise ObjectBongardRubricCalibrationError(
                "current lineage packet is not canonical"
            )
        values = {
            "ordinal": ordinal,
            "group": _group_for_ordinal(ordinal),
            "task_id": task_id,
            "panel_id": panel_id,
            "released_file_sha256": released_file,
            "released_record_digest": released_digest,
            "png_sha256": png_digest,
            "source_observer_file_sha256": observer_file,
            "source_observer_artifact_digest": observer_digest,
            "historical_hypothesis_packet_digest": (
                historical_hypothesis_packet_digest
            ),
            "exact_png_bytes": released.exact_png_bytes,
            "hypothesis_packet": hypothesis_packet,
            "lineage_packet": lineage_packet,
        }
        provisional = object.__new__(ObjectBongardRubricCalibrationPanel)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        panels.append(
            ObjectBongardRubricCalibrationPanel(
                **values,  # type: ignore[arg-type]
                panel_binding_digest=canonical_digest(_panel_content(provisional)),
            )
        )
    values = {
        "historical_plan_file_sha256": _PLAN_FILE_SHA256,
        "historical_plan_record_digest": _PLAN_RECORD_DIGEST,
        "historical_description_file_sha256": _DESCRIPTION_FILE_SHA256,
        "historical_description_artifact_digest": _DESCRIPTION_ARTIFACT_DIGEST,
        "panels": tuple(panels),
        "rubric_specs": specs,
        "nomination_artifact": None,
        "nomination_authorization_digest": None,
        "nomination_precommit_digest": None,
        "nomination_replay_digest": None,
        "nomination_result_digest": None,
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationSource)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationSource(
        **values,
        source_digest=canonical_digest(_source_content(provisional)),
    )


class ObjectBongardRubricJournalDispatcher:
    """Route each atlas-sheet call to its own exactly-once turn journal."""

    def __init__(
        self,
        journal_root: str | os.PathLike[str],
        *,
        panel: ObjectBongardRubricCalibrationPanel,
        rubric_spec: ObjectBongardRubricSpec,
        authorization_digest: str,
        execution_precommit_digest: str,
        runtime: ObjectBongardTurnRuntime,
        underlying_transport: Callable[..., CodexStructuredResult] = (
            run_codex_named_images_structured
        ),
    ) -> None:
        if not isinstance(panel, ObjectBongardRubricCalibrationPanel):
            raise TypeError("panel must be a typed calibration panel")
        if not isinstance(rubric_spec, ObjectBongardRubricSpec):
            raise TypeError("rubric_spec must be ObjectBongardRubricSpec")
        if not isinstance(runtime, ObjectBongardTurnRuntime):
            raise TypeError("runtime must be ObjectBongardTurnRuntime")
        if runtime.transport_source_digest != (
            _object_observer.prototype_scene_transport_source_digest()
        ):
            raise ObjectBongardRubricCalibrationError(
                "journal runtime transport source differs from rubric observer"
            )
        _raw_or_address(authorization_digest, "authorization digest")
        _raw_or_address(execution_precommit_digest, "execution precommit digest")
        rendered = dict(
            render_object_hypothesis_atlas(
                panel.hypothesis_packet, panel.exact_png_bytes
            )
        )
        schema = object_bongard_rubric_observer_output_schema()
        base = (
            Path(journal_root)
            / panel.task_id
            / rubric_spec.spec_digest
        )
        journals: list[ObjectBongardNamedImageTurnJournalTransport] = []
        keys: list[tuple[str, tuple[str, ...]]] = []
        for sheet in panel.hypothesis_packet.atlas_sheets:
            prompt = object_bongard_rubric_observer_prompt(rubric_spec, sheet)
            names = ("scene.png", sheet.name)
            journal = ObjectBongardNamedImageTurnJournalTransport(
                base / f"sheet_{sheet.sheet_index:03d}",
                authorization_digest=authorization_digest,
                execution_precommit_digest=execution_precommit_digest,
                task_id=panel.task_id,
                turn_kind=(
                    f"rubric_{rubric_spec.spec_digest[:12]}_"
                    f"sheet_{sheet.sheet_index:03d}"
                ),
                expected_prompt=prompt,
                expected_images=(
                    ("scene.png", panel.exact_png_bytes),
                    (sheet.name, rendered[sheet.name]),
                ),
                expected_output_schema=schema,
                runtime=runtime,
                underlying_transport=underlying_transport,
            )
            journals.append(journal)
            keys.append((prompt, names))
        if len(keys) != len(set(keys)):
            raise ObjectBongardRubricCalibrationError(
                "rubric journal dispatch keys are not unique"
            )
        self.panel = panel
        self.rubric_spec = rubric_spec
        self.runtime = runtime
        self._journals = tuple(journals)
        self._by_key = dict(zip(keys, self._journals, strict=True))

    @property
    def journal_count(self) -> int:
        return len(self._journals)

    @property
    def fresh_call_count(self) -> int:
        return sum(item.fresh_call_count for item in self._journals)

    @property
    def reused_call_count(self) -> int:
        return sum(item.reused_call_count for item in self._journals)

    @property
    def refused_call_count(self) -> int:
        return sum(item.refused_call_count for item in self._journals)

    def __call__(
        self,
        task: str,
        image_png_paths: Sequence[str],
        image_names: Sequence[str],
        output_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> CodexStructuredResult:
        key = (task, tuple(image_names))
        journal = self._by_key.get(key)
        if journal is None:
            raise ObjectBongardRubricCalibrationError(
                "rubric observer call is outside the committed sheet inventory"
            )
        return journal(
            task,
            image_png_paths,
            image_names,
            output_schema,
            **kwargs,
        )

    def verify(self) -> tuple[ObjectBongardTurnJournalSummary, ...]:
        return tuple(
            verify_object_bongard_turn_journal(item) for item in self._journals
        )


def _raw_or_address(value: object, label: str) -> str:
    if not isinstance(value, str) or (
        _RAW_DIGEST.fullmatch(value) is None and _ADDRESS.fullmatch(value) is None
    ):
        raise ObjectBongardRubricCalibrationError(
            f"{label} must be raw or sha256:-addressed SHA-256"
        )
    return value


def create_object_bongard_rubric_journal_dispatcher(
    journal_root: str | os.PathLike[str],
    *,
    panel: ObjectBongardRubricCalibrationPanel,
    rubric_spec: ObjectBongardRubricSpec,
    authorization_digest: str,
    execution_precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> ObjectBongardRubricJournalDispatcher:
    return ObjectBongardRubricJournalDispatcher(
        journal_root,
        panel=panel,
        rubric_spec=rubric_spec,
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
        runtime=runtime,
        underlying_transport=underlying_transport,
    )


def _journal_summary_from_data(value: object) -> ObjectBongardTurnJournalSummary:
    raw = _fields(
        value,
        {
            "schema",
            "manifest_digest",
            "turn_key",
            "terminal_status",
            "claim_digest",
            "result_digest",
            "outcome_digest",
            "record_digest",
            "predicate_authority_id",
            "python_is_canonical_authority",
            "lean_present",
            "lean_required",
            "lean_removable",
            "lean_affects_identity_or_replay",
        },
        "turn journal summary",
    )
    if (
        raw["schema"] != "gkm.bongard-codex-turn-journal-summary.v1"
        or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
        or raw["python_is_canonical_authority"] is not True
        or raw["lean_present"] is not False
        or raw["lean_required"] is not False
        or raw["lean_removable"] is not True
        or raw["lean_affects_identity_or_replay"] is not False
        or raw["record_digest"]
        != "sha256:"
        + canonical_digest(
            {key: item for key, item in raw.items() if key != "record_digest"}
        )
    ):
        raise ObjectBongardRubricCalibrationError(
            "turn journal summary policy/digest differs"
        )
    for name in (
        "manifest_digest",
        "turn_key",
        "claim_digest",
        "result_digest",
        "outcome_digest",
        "record_digest",
    ):
        item = raw[name]
        if item is not None:
            _raw_or_address(item, f"journal summary {name}")
    if raw["terminal_status"] not in {"success", "failure", "unclaimed"}:
        raise ObjectBongardRubricCalibrationError(
            "turn journal summary status differs"
        )
    result = ObjectBongardTurnJournalSummary(
        raw["manifest_digest"],
        raw["turn_key"],
        raw["terminal_status"],
        raw["claim_digest"],
        raw["result_digest"],
        raw["outcome_digest"],
        raw["record_digest"],
    )
    if result.to_data() != dict(raw):
        raise ObjectBongardRubricCalibrationError(
            "turn journal summary is not canonical"
        )
    return result


def _live_run_content(value: "ObjectBongardRubricLiveObservation") -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_LIVE_OBSERVATION_SCHEMA,
        "panel_binding_digest": value.panel_binding_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "observer_artifact": value.artifact.to_data(),
        "journal_summaries": [item.to_data() for item in value.journal_summaries],
        "fresh_call_count": value.fresh_call_count,
        "reused_call_count": value.reused_call_count,
        "labels_visible_to_observer": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricLiveObservation:
    panel_binding_digest: str
    rubric_spec_digest: str
    artifact: ObjectBongardRubricObserverArtifact
    journal_summaries: tuple[ObjectBongardTurnJournalSummary, ...]
    fresh_call_count: int
    reused_call_count: int
    run_digest: str

    def __post_init__(self) -> None:
        for name in (
            "panel_binding_digest",
            "rubric_spec_digest",
            "run_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if (
            not isinstance(self.artifact, ObjectBongardRubricObserverArtifact)
            or self.artifact.rubric_spec.spec_digest != self.rubric_spec_digest
            or not isinstance(self.journal_summaries, tuple)
            or len(self.journal_summaries) != self.artifact.physical_call_count
            or any(
                not isinstance(item, ObjectBongardTurnJournalSummary)
                for item in self.journal_summaries
            )
            or any(item.terminal_status not in {"success", "failure"} for item in self.journal_summaries)
            or isinstance(self.fresh_call_count, bool)
            or not isinstance(self.fresh_call_count, int)
            or self.fresh_call_count < 0
            or isinstance(self.reused_call_count, bool)
            or not isinstance(self.reused_call_count, int)
            or self.reused_call_count < 0
            or self.fresh_call_count + self.reused_call_count
            != len(self.journal_summaries)
            or self.run_digest != canonical_digest(_live_run_content(self))
        ):
            raise ObjectBongardRubricCalibrationError(
                "live rubric observation/journal binding differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_live_run_content(self), "run_digest": self.run_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricLiveObservation":
        raw = _fields(
            value,
            {
                "schema",
                "panel_binding_digest",
                "rubric_spec_digest",
                "observer_artifact",
                "journal_summaries",
                "fresh_call_count",
                "reused_call_count",
                "labels_visible_to_observer",
                *_authority_data(),
                "run_digest",
            },
            "live rubric observation",
        )
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_LIVE_OBSERVATION_SCHEMA
            or raw["labels_visible_to_observer"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["journal_summaries"], list)
        ):
            raise ObjectBongardRubricCalibrationError(
                "live rubric observation policy differs"
            )
        result = cls(
            raw["panel_binding_digest"],
            raw["rubric_spec_digest"],
            ObjectBongardRubricObserverArtifact.from_data(
                raw["observer_artifact"]
            ),
            tuple(
                _journal_summary_from_data(item)
                for item in raw["journal_summaries"]
            ),
            raw["fresh_call_count"],
            raw["reused_call_count"],
            raw["run_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationError(
                "live rubric observation is not canonical"
            )
        return result


RubricJournalDispatcherFactory = Callable[..., ObjectBongardRubricJournalDispatcher]


def run_object_bongard_rubric_calibration_observation(
    panel: ObjectBongardRubricCalibrationPanel,
    rubric_spec: ObjectBongardRubricSpec,
    *,
    runtime: ObjectBongardTurnRuntime,
    journal_root: str | os.PathLike[str],
    authorization_digest: str,
    execution_precommit_digest: str,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
    dispatcher_factory: RubricJournalDispatcherFactory = (
        create_object_bongard_rubric_journal_dispatcher
    ),
) -> ObjectBongardRubricLiveObservation:
    """Observe one panel/rubric through resumable per-sheet journals."""

    if not isinstance(panel, ObjectBongardRubricCalibrationPanel):
        raise TypeError("panel must be a typed calibration panel")
    if not isinstance(rubric_spec, ObjectBongardRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardRubricSpec")
    if not isinstance(runtime, ObjectBongardTurnRuntime):
        raise TypeError("runtime must be ObjectBongardTurnRuntime")
    context = "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-calibration-observation-context.v1",
            "panel_binding_digest": panel.panel_binding_digest,
            "rubric_spec_digest": rubric_spec.spec_digest,
            "authorization_digest": _raw_or_address(
                authorization_digest, "authorization digest"
            ),
            "execution_precommit_digest": _raw_or_address(
                execution_precommit_digest, "execution precommit digest"
            ),
            "labels_visible_to_observer": False,
        }
    )
    dispatcher = dispatcher_factory(
        journal_root,
        panel=panel,
        rubric_spec=rubric_spec,
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
        runtime=runtime,
        underlying_transport=underlying_transport,
    )
    if not isinstance(dispatcher, ObjectBongardRubricJournalDispatcher):
        raise TypeError("dispatcher_factory must return the typed journal dispatcher")
    artifact = observe_object_bongard_rubric(
        panel.exact_png_bytes,
        panel_id=panel.panel_id,
        rubric_spec=rubric_spec,
        hypothesis_packet=panel.hypothesis_packet,
        lineage_packet=panel.lineage_packet,
        expected_scene_sha256=panel.png_sha256,
        expected_rubric_spec_digest=rubric_spec.spec_digest,
        expected_hypothesis_packet_digest=panel.hypothesis_packet.digest(),
        expected_lineage_packet_digest=panel.lineage_packet.digest(),
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=dispatcher,
        observation_context_digest=context,
    )
    # The live observer already pixel-replays the packet/atlas before its calls
    # and its artifact constructor replays every raw row and receipt binding.
    # Keep live launch latency bounded by doing only a canonical typed reload
    # here; the public cold-replay path performs the independent pixel replay.
    if ObjectBongardRubricObserverArtifact.from_data(artifact.to_data()) != artifact:
        raise ObjectBongardRubricCalibrationError(
            "live observer artifact canonical reload differs"
        )
    summaries = dispatcher.verify()
    values = {
        "panel_binding_digest": panel.panel_binding_digest,
        "rubric_spec_digest": rubric_spec.spec_digest,
        "artifact": artifact,
        "journal_summaries": summaries,
        "fresh_call_count": dispatcher.fresh_call_count,
        "reused_call_count": dispatcher.reused_call_count,
    }
    provisional = object.__new__(ObjectBongardRubricLiveObservation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricLiveObservation(
        **values,
        run_digest=canonical_digest(_live_run_content(provisional)),
    )


def _batch_content(value: "ObjectBongardRubricObservationBatch") -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_OBSERVATION_BATCH_SCHEMA,
        "source_digest": value.source_digest,
        "parallel_workers": value.parallel_workers,
        "runs": [item.to_data() for item in value.runs],
        "run_order": "rubric-spec-order-then-source-ordinal-order",
        "journal_directories_disjoint_by_task-spec-sheet": True,
        "labels_visible_to_observer": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricObservationBatch:
    source_digest: str
    parallel_workers: int
    runs: tuple[ObjectBongardRubricLiveObservation, ...]
    batch_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.source_digest, "batch source digest")
        if (
            isinstance(self.parallel_workers, bool)
            or not isinstance(self.parallel_workers, int)
            or not 1 <= self.parallel_workers <= 32
            or not isinstance(self.runs, tuple)
            or len(self.runs) != 12
            or any(
                not isinstance(item, ObjectBongardRubricLiveObservation)
                for item in self.runs
            )
            or len(
                {
                    (item.panel_binding_digest, item.rubric_spec_digest)
                    for item in self.runs
                }
            )
            != len(self.runs)
            or tuple(item.rubric_spec_digest for item in self.runs)
            != (self.runs[0].rubric_spec_digest,) * 12
            or len({item.panel_binding_digest for item in self.runs}) != 12
            or self.batch_digest != canonical_digest(_batch_content(self))
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration observation batch differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_batch_content(self), "batch_digest": self.batch_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricObservationBatch":
        raw = _fields(
            value,
            {
                "schema",
                "source_digest",
                "parallel_workers",
                "runs",
                "run_order",
                "journal_directories_disjoint_by_task-spec-sheet",
                "labels_visible_to_observer",
                *_authority_data(),
                "batch_digest",
            },
            "rubric observation batch",
        )
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_OBSERVATION_BATCH_SCHEMA
            or raw["run_order"]
            != "rubric-spec-order-then-source-ordinal-order"
            or raw["journal_directories_disjoint_by_task-spec-sheet"] is not True
            or raw["labels_visible_to_observer"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["runs"], list)
        ):
            raise ObjectBongardRubricCalibrationError(
                "rubric observation batch policy differs"
            )
        result = cls(
            raw["source_digest"],
            raw["parallel_workers"],
            tuple(
                ObjectBongardRubricLiveObservation.from_data(item)
                for item in raw["runs"]
            ),
            raw["batch_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationError(
                "rubric observation batch is not canonical"
            )
        return result

    def artifacts_by_spec_digest(
        self,
    ) -> dict[str, tuple[ObjectBongardRubricObserverArtifact, ...]]:
        specs: dict[str, list[ObjectBongardRubricObserverArtifact]] = {}
        for run in self.runs:
            specs.setdefault(run.rubric_spec_digest, []).append(run.artifact)
        return {key: tuple(values) for key, values in specs.items()}


def run_object_bongard_rubric_calibration_observations(
    source: ObjectBongardRubricCalibrationSource,
    *,
    runtime: ObjectBongardTurnRuntime,
    journal_root: str | os.PathLike[str],
    authorization_digest: str,
    execution_precommit_digest: str,
    parallel_workers: int = 4,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
    dispatcher_factory: RubricJournalDispatcherFactory = (
        create_object_bongard_rubric_journal_dispatcher
    ),
) -> ObjectBongardRubricObservationBatch:
    """Observe the frozen signed rubric on all twelve panels, four at a time.

    Jobs are created in rubric-spec order and source-ordinal order.  Their
    journal paths include task ID, full spec digest, and sheet index, so the
    default four concurrent workers never share a mutable journal directory.
    No group or support-side label is passed into a job.
    """

    if not isinstance(source, ObjectBongardRubricCalibrationSource):
        raise TypeError("source must be ObjectBongardRubricCalibrationSource")
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= 32
    ):
        raise ObjectBongardRubricCalibrationError(
            "parallel_workers must lie in 1..32"
        )
    jobs = tuple(
        (panel, spec)
        for spec in source.rubric_specs
        for panel in source.panels
    )

    def run_job(
        job: tuple[ObjectBongardRubricCalibrationPanel, ObjectBongardRubricSpec]
    ) -> ObjectBongardRubricLiveObservation:
        panel, spec = job
        return run_object_bongard_rubric_calibration_observation(
            panel,
            spec,
            runtime=runtime,
            journal_root=journal_root,
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            underlying_transport=underlying_transport,
            dispatcher_factory=dispatcher_factory,
        )

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        runs = tuple(executor.map(run_job, jobs))
    values = {
        "source_digest": source.source_digest,
        "parallel_workers": parallel_workers,
        "runs": runs,
    }
    provisional = object.__new__(ObjectBongardRubricObservationBatch)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricObservationBatch(
        **values,
        batch_digest=canonical_digest(_batch_content(provisional)),
    )


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricDispositionCounts:
    present: int
    certified_absent: int
    indeterminate: int
    error: int

    def __post_init__(self) -> None:
        values = (
            self.present,
            self.certified_absent,
            self.indeterminate,
            self.error,
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ) or sum(values) != RUBRIC_SUPPORT_PANELS_PER_SIDE:
            raise ObjectBongardRubricCalibrationError(
                "disposition counts must exhaust exactly one six-panel side"
            )

    @classmethod
    def from_states(
        cls, values: Sequence[Disposition]
    ) -> "ObjectBongardRubricDispositionCounts":
        frozen = tuple(values)
        if len(frozen) != RUBRIC_SUPPORT_PANELS_PER_SIDE or any(
            not isinstance(item, Disposition) for item in frozen
        ):
            raise ObjectBongardRubricCalibrationError(
                "counted dispositions must be one exact support side"
            )
        return cls(
            frozen.count(Disposition.PRESENT),
            frozen.count(Disposition.CERTIFIED_ABSENT),
            frozen.count(Disposition.INDETERMINATE),
            frozen.count(Disposition.ERROR),
        )

    def to_data(self) -> dict[str, int]:
        return {
            "present": self.present,
            "certified_absent": self.certified_absent,
            "indeterminate": self.indeterminate,
            "error": self.error,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricDispositionCounts":
        raw = _fields(
            value,
            {"present", "certified_absent", "indeterminate", "error"},
            "disposition counts",
        )
        return cls(
            raw["present"],
            raw["certified_absent"],
            raw["indeterminate"],
            raw["error"],
        )


def _candidate_counts_content(
    value: "ObjectBongardRubricCandidateCalibrationCounts",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_COUNTS_SCHEMA,
        "candidate_digest": value.candidate_digest,
        "candidate_id": value.candidate_id,
        "positive": value.positive.to_data(),
        "negative": value.negative.to_data(),
        "support_consistent": value.support_consistent,
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCandidateCalibrationCounts:
    candidate_digest: str
    candidate_id: str
    positive: ObjectBongardRubricDispositionCounts
    negative: ObjectBongardRubricDispositionCounts
    support_consistent: bool
    counts_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.candidate_digest, "count candidate digest")
        _raw_digest(self.counts_digest, "candidate counts digest")
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not isinstance(self.positive, ObjectBongardRubricDispositionCounts)
            or not isinstance(self.negative, ObjectBongardRubricDispositionCounts)
            or not isinstance(self.support_consistent, bool)
            or self.counts_digest
            != canonical_digest(_candidate_counts_content(self))
        ):
            raise ObjectBongardRubricCalibrationError(
                "candidate assessment counts differ"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_candidate_counts_content(self),
            "counts_digest": self.counts_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCandidateCalibrationCounts":
        raw = _fields(
            value,
            {
                "schema",
                "candidate_digest",
                "candidate_id",
                "positive",
                "negative",
                "support_consistent",
                "counts_digest",
            },
            "candidate calibration counts",
        )
        if raw["schema"] != OBJECT_RUBRIC_CALIBRATION_COUNTS_SCHEMA:
            raise ObjectBongardRubricCalibrationError(
                "candidate counts schema differs"
            )
        result = cls(
            raw["candidate_digest"],
            raw["candidate_id"],
            ObjectBongardRubricDispositionCounts.from_data(raw["positive"]),
            ObjectBongardRubricDispositionCounts.from_data(raw["negative"]),
            raw["support_consistent"],
            raw["counts_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationError(
                "candidate counts are not canonical"
            )
        return result


def _candidate_counts(
    candidate: ObjectBongardRubricCandidate,
    row: Sequence[Disposition],
    survivor_candidate_digests: Sequence[str],
) -> ObjectBongardRubricCandidateCalibrationCounts:
    states = tuple(row)
    if len(states) != RUBRIC_SUPPORT_PANELS_PER_SIDE * 2:
        raise ObjectBongardRubricCalibrationError(
            "candidate row does not contain exact 6+6 support"
        )
    values = {
        "candidate_digest": candidate.candidate_digest,
        "candidate_id": candidate.candidate_id,
        "positive": ObjectBongardRubricDispositionCounts.from_states(
            states[:RUBRIC_SUPPORT_PANELS_PER_SIDE]
        ),
        "negative": ObjectBongardRubricDispositionCounts.from_states(
            states[RUBRIC_SUPPORT_PANELS_PER_SIDE:]
        ),
        "support_consistent": candidate.candidate_digest
        in tuple(survivor_candidate_digests),
    }
    provisional = object.__new__(ObjectBongardRubricCandidateCalibrationCounts)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCandidateCalibrationCounts(
        **values,
        counts_digest=canonical_digest(_candidate_counts_content(provisional)),
    )


def _spec_assessment_content(
    value: "ObjectBongardRubricCalibrationSpecAssessment",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_SPEC_ASSESSMENT_SCHEMA,
        "rubric_spec": value.rubric_spec.to_data(),
        "version_space": value.version_space.to_data(),
        "candidate_counts": [item.to_data() for item in value.candidate_counts],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "gap_kind": None if value.gap_kind is None else value.gap_kind.value,
        "gap_digest": value.gap_digest,
        "labels_introduced_after_all_observations": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationSpecAssessment:
    rubric_spec: ObjectBongardRubricSpec
    version_space: ObjectBongardRubricSupportVersionSpace
    candidate_counts: tuple[ObjectBongardRubricCandidateCalibrationCounts, ...]
    survivor_candidate_digests: tuple[str, ...]
    gap_kind: RubricSupportGapKind | None
    gap_digest: str | None
    assessment_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rubric_spec, ObjectBongardRubricSpec)
            or not isinstance(
                self.version_space, ObjectBongardRubricSupportVersionSpace
            )
            or self.version_space.rubric_spec_digest != self.rubric_spec.spec_digest
        ):
            raise ObjectBongardRubricCalibrationError(
                "spec assessment rubric/version-space binding differs"
            )
        expected_counts = tuple(
            _candidate_counts(
                candidate,
                row,
                self.version_space.survivor_candidate_digests,
            )
            for candidate, row in zip(
                self.version_space.candidates,
                self.version_space.rows,
                strict=True,
            )
        )
        gap = self.version_space.gap
        if (
            self.candidate_counts != expected_counts
            or self.survivor_candidate_digests
            != self.version_space.survivor_candidate_digests
            or self.gap_kind != (None if gap is None else gap.kind)
            or self.gap_digest != (None if gap is None else gap.gap_digest)
        ):
            raise ObjectBongardRubricCalibrationError(
                "spec assessment counts/survivors/gap differ from replay"
            )
        _raw_digest(self.assessment_digest, "spec assessment digest")
        if self.assessment_digest != canonical_digest(_spec_assessment_content(self)):
            raise ObjectBongardRubricCalibrationError(
                "spec assessment digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_spec_assessment_content(self),
            "assessment_digest": self.assessment_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationSpecAssessment":
        raw = _fields(
            value,
            {
                "schema",
                "rubric_spec",
                "version_space",
                "candidate_counts",
                "survivor_candidate_digests",
                "gap_kind",
                "gap_digest",
                "labels_introduced_after_all_observations",
                *_authority_data(),
                "assessment_digest",
            },
            "rubric calibration spec assessment",
        )
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_SPEC_ASSESSMENT_SCHEMA
            or raw["labels_introduced_after_all_observations"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["candidate_counts"], list)
            or not isinstance(raw["survivor_candidate_digests"], list)
        ):
            raise ObjectBongardRubricCalibrationError(
                "spec assessment policy differs"
            )
        gap_kind = (
            None
            if raw["gap_kind"] is None
            else RubricSupportGapKind(raw["gap_kind"])
        )
        result = cls(
            ObjectBongardRubricSpec.from_data(raw["rubric_spec"]),
            ObjectBongardRubricSupportVersionSpace.from_data(
                raw["version_space"]
            ),
            tuple(
                ObjectBongardRubricCandidateCalibrationCounts.from_data(item)
                for item in raw["candidate_counts"]
            ),
            tuple(raw["survivor_candidate_digests"]),
            gap_kind,
            raw["gap_digest"],
            raw["assessment_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationError(
                "spec assessment is not canonical"
            )
        return result


def _make_spec_assessment(
    spec: ObjectBongardRubricSpec,
    version_space: ObjectBongardRubricSupportVersionSpace,
) -> ObjectBongardRubricCalibrationSpecAssessment:
    gap = version_space.gap
    values = {
        "rubric_spec": spec,
        "version_space": version_space,
        "candidate_counts": tuple(
            _candidate_counts(
                candidate, row, version_space.survivor_candidate_digests
            )
            for candidate, row in zip(
                version_space.candidates, version_space.rows, strict=True
            )
        ),
        "survivor_candidate_digests": version_space.survivor_candidate_digests,
        "gap_kind": None if gap is None else gap.kind,
        "gap_digest": None if gap is None else gap.gap_digest,
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationSpecAssessment)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationSpecAssessment(
        **values,
        assessment_digest=canonical_digest(_spec_assessment_content(provisional)),
    )


def _assessment_content(
    value: "ObjectBongardRubricCalibrationAssessment",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA,
        "algorithm_id": OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "implementation_source_sha256": object_bongard_rubric_calibration_source_digest(),
        "source_digest": value.source_digest,
        "spec_assessments": [item.to_data() for item in value.spec_assessments],
        "spec_count": 1,
        "support_panels_per_side": RUBRIC_SUPPORT_PANELS_PER_SIDE,
        "labels_visible_to_observer": False,
        "cold_replay_calls_model": False,
        "ranker_used": False,
        "query_pixels_used": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationAssessment:
    source_digest: str
    spec_assessments: tuple[ObjectBongardRubricCalibrationSpecAssessment, ...]
    assessment_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.source_digest, "assessment source digest")
        if (
            not isinstance(self.spec_assessments, tuple)
            or len(self.spec_assessments) != 1
            or any(
                not isinstance(item, ObjectBongardRubricCalibrationSpecAssessment)
                for item in self.spec_assessments
            )
            or len(
                {item.rubric_spec.spec_digest for item in self.spec_assessments}
            )
            != 1
        ):
            raise ObjectBongardRubricCalibrationError(
                "assessment must contain the single frozen signed rubric spec"
            )
        _raw_digest(self.assessment_digest, "calibration assessment digest")
        if self.assessment_digest != canonical_digest(_assessment_content(self)):
            raise ObjectBongardRubricCalibrationError(
                "calibration assessment digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_assessment_content(self),
            "assessment_digest": self.assessment_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationAssessment":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "implementation_source_sha256",
                "source_digest",
                "spec_assessments",
                "spec_count",
                "support_panels_per_side",
                "labels_visible_to_observer",
                "cold_replay_calls_model",
                "ranker_used",
                "query_pixels_used",
                *_authority_data(),
                "assessment_digest",
            },
            "rubric calibration assessment",
        )
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA
            or raw["algorithm_id"] != OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID
            or raw["implementation_source_sha256"]
            != object_bongard_rubric_calibration_source_digest()
            or raw["spec_count"] != 1
            or raw["support_panels_per_side"]
            != RUBRIC_SUPPORT_PANELS_PER_SIDE
            or raw["labels_visible_to_observer"] is not False
            or raw["cold_replay_calls_model"] is not False
            or raw["ranker_used"] is not False
            or raw["query_pixels_used"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["spec_assessments"], list)
        ):
            raise ObjectBongardRubricCalibrationError(
                "calibration assessment policy differs"
            )
        result = cls(
            raw["source_digest"],
            tuple(
                ObjectBongardRubricCalibrationSpecAssessment.from_data(item)
                for item in raw["spec_assessments"]
            ),
            raw["assessment_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationError(
                "calibration assessment is not canonical"
            )
        return result


def _canonical_observations(
    source: ObjectBongardRubricCalibrationSource,
    observations: (
        Mapping[str, Sequence[ObjectBongardRubricObserverArtifact]]
        | ObjectBongardRubricObservationBatch
    ),
) -> dict[str, tuple[ObjectBongardRubricObserverArtifact, ...]]:
    if isinstance(observations, ObjectBongardRubricObservationBatch):
        if observations.source_digest != source.source_digest:
            raise ObjectBongardRubricCalibrationError(
                "observation batch source differs"
            )
        values = observations.artifacts_by_spec_digest()
    else:
        if not isinstance(observations, Mapping) or any(
            not isinstance(key, str) for key in observations
        ):
            raise TypeError("observations must map spec digests to artifacts")
        values = dict(observations)
    spec_digests = tuple(item.spec_digest for item in source.rubric_specs)
    if set(values) != set(spec_digests):
        raise ObjectBongardRubricCalibrationError(
            "observations do not exhaust the single exact rubric spec"
        )
    expected_panel_ids = tuple(item.panel_id for item in source.panels)
    result: dict[str, tuple[ObjectBongardRubricObserverArtifact, ...]] = {}
    # Complete and cold-verify the label-free observation inventory before
    # selecting its preregistered group-0/group-A support direction.
    for spec in source.rubric_specs:
        by_panel: dict[str, ObjectBongardRubricObserverArtifact] = {}
        for artifact in values[spec.spec_digest]:
            if not isinstance(artifact, ObjectBongardRubricObserverArtifact):
                raise TypeError("calibration observation has the wrong type")
            frozen = ObjectBongardRubricObserverArtifact.from_data(
                artifact.to_data()
            )
            if frozen.panel_id in by_panel:
                raise ObjectBongardRubricCalibrationError(
                    "calibration observation panel is duplicated"
                )
            panel = source.panel_by_id(frozen.panel_id)
            if (
                frozen.rubric_spec.spec_digest != spec.spec_digest
                or frozen.panel_digest != panel.png_sha256
                or frozen.hypothesis_packet != panel.hypothesis_packet
                or frozen.lineage_packet != panel.lineage_packet
            ):
                raise ObjectBongardRubricCalibrationError(
                    "calibration observation source/spec binding differs"
                )
            verify_object_bongard_rubric_observer_artifact(
                frozen,
                panel.exact_png_bytes,
                panel_id=panel.panel_id,
                rubric_spec=spec,
                hypothesis_packet=panel.hypothesis_packet,
                lineage_packet=panel.lineage_packet,
                expected_artifact_digest=frozen.artifact_digest,
            )
            by_panel[frozen.panel_id] = frozen
        if set(by_panel) != set(expected_panel_ids):
            raise ObjectBongardRubricCalibrationError(
                "calibration observations do not exhaust the exact twelve panels"
            )
        result[spec.spec_digest] = tuple(
            by_panel[panel_id] for panel_id in expected_panel_ids
        )
    return result


def assess_object_bongard_rubric_calibration(
    source: ObjectBongardRubricCalibrationSource,
    observations: (
        Mapping[str, Sequence[ObjectBongardRubricObserverArtifact]]
        | ObjectBongardRubricObservationBatch
    ),
) -> ObjectBongardRubricCalibrationAssessment:
    """Build the canonical exact 6+6 version space after vision is frozen."""

    if not isinstance(source, ObjectBongardRubricCalibrationSource):
        raise TypeError("source must be ObjectBongardRubricCalibrationSource")
    frozen = _canonical_observations(source, observations)
    assessments: list[ObjectBongardRubricCalibrationSpecAssessment] = []
    for spec in source.rubric_specs:
        artifacts = frozen[spec.spec_digest]
        by_panel = {item.panel_id: item for item in artifacts}
        version_space = build_object_bongard_rubric_support_version_space(
            spec,
            tuple(by_panel[item.panel_id] for item in source.group_a_panels),
            tuple(by_panel[item.panel_id] for item in source.group_b_panels),
        )
        assessments.append(_make_spec_assessment(spec, version_space))
    values = {
        "source_digest": source.source_digest,
        "spec_assessments": tuple(assessments),
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationAssessment)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationAssessment(
        **values,
        assessment_digest=canonical_digest(_assessment_content(provisional)),
    )


def cold_verify_object_bongard_rubric_calibration(
    assessment: ObjectBongardRubricCalibrationAssessment,
    source: ObjectBongardRubricCalibrationSource,
    observations: (
        Mapping[str, Sequence[ObjectBongardRubricObserverArtifact]]
        | ObjectBongardRubricObservationBatch
    ),
) -> ObjectBongardRubricCalibrationAssessment:
    """Replay pixels, receipts, projections, candidates, counts, and gaps.

    No callable transport is accepted here, so this path cannot issue a model
    call.  It detects changes in historical source records when ``source`` is
    freshly loaded and detects any supplied artifact or assessment tampering.
    """

    if not isinstance(assessment, ObjectBongardRubricCalibrationAssessment):
        raise TypeError("assessment must be a typed calibration assessment")
    decoded = ObjectBongardRubricCalibrationAssessment.from_data(
        assessment.to_data()
    )
    if decoded.source_digest != source.source_digest:
        raise ObjectBongardRubricCalibrationError(
            "calibration assessment source differs"
        )
    replayed = assess_object_bongard_rubric_calibration(source, observations)
    if decoded != replayed:
        raise ObjectBongardRubricCalibrationError(
            "cold calibration replay differs"
        )
    return decoded


__all__ = (
    "CALIBRATION_GROUP_A_ORDINALS",
    "CALIBRATION_GROUP_B_ORDINALS",
    "CALIBRATION_SELECTED_ORDINALS",
    "CALIBRATION_FIT_GROUP_A_ORDINALS",
    "CALIBRATION_FIT_GROUP_B_ORDINALS",
    "CALIBRATION_CONFIRM_GROUP_A_ORDINALS",
    "CALIBRATION_CONFIRM_GROUP_B_ORDINALS",
    "DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE",
    "OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID",
    "ObjectBongardRubricCalibrationAssessment",
    "ObjectBongardRubricCalibrationError",
    "ObjectBongardRubricCalibrationGroup",
    "ObjectBongardRubricCalibrationPanel",
    "ObjectBongardRubricCalibrationSource",
    "ObjectBongardRubricCalibrationSpecAssessment",
    "ObjectBongardRubricCandidateCalibrationCounts",
    "ObjectBongardRubricDispositionCounts",
    "ObjectBongardRubricJournalDispatcher",
    "ObjectBongardRubricLiveObservation",
    "ObjectBongardRubricObservationBatch",
    "assess_object_bongard_rubric_calibration",
    "cold_verify_object_bongard_rubric_calibration",
    "create_object_bongard_rubric_journal_dispatcher",
    "load_object_bongard_rubric_calibration_source",
    "object_bongard_rubric_calibration_source_digest",
    "run_object_bongard_rubric_calibration_observation",
    "run_object_bongard_rubric_calibration_observations",
)
