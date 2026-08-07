"""Canonical typed leg contracts for Bongard experiments."""

from importlib import import_module

from .contracts import (
    AffirmativeRelation,
    BOOLEAN_WITNESS,
    FROZEN_VISUAL_SCORE,
    OBJECT,
    PANEL,
    SOFT_SEMANTIC,
    WITNESS,
    ContractViolation,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
    Transform,
    TypedValue,
    Unit,
    ValueType,
    implementation_sha256,
)
from .bilateral_symmetry import (
    ALGORITHM_ID as BILATERAL_SYMMETRY_ALGORITHM_ID,
    BILATERAL_SYMMETRY_SCORE,
    BilateralSymmetryInputError,
    BilateralSymmetryObservation,
    bilateral_symmetry_contract,
    bilateral_symmetry_score,
    measure_bilateral_symmetry,
    operation_digest as bilateral_symmetry_operation_digest,
    register_bilateral_symmetry_leg,
)
_NEUTRAL_FEATURE_EXPORTS = {
    "ALGORITHM_ID": "NEUTRAL_FEATURE_ALGORITHM_ID",
    "EXTRACTOR_VERSION": "NEUTRAL_FEATURE_EXTRACTOR_VERSION",
    "FEATURE_GROUP_IDS": "FEATURE_GROUP_IDS",
    "NeutralFeatureExtraction": "NeutralFeatureExtraction",
    "NeutralFeatureGroup": "NeutralFeatureGroup",
    "NeutralFeatureReceipt": "NeutralFeatureReceipt",
    "NeutralInputIdentity": "NeutralInputIdentity",
    "NeutralPacketCommitment": "NeutralPacketCommitment",
    "extract_neutral_features": "extract_neutral_features",
    "feature_group_catalog": "feature_group_catalog",
    "feature_group_catalog_digest": "feature_group_catalog_digest",
    "feature_space_for_group": "feature_space_for_group",
    "feature_space_for_groups": "feature_space_for_groups",
    "full_neutral_feature_space": "full_neutral_feature_space",
    "project_neutral_feature_extraction": "project_neutral_feature_extraction",
    "verify_neutral_feature_extraction": "verify_neutral_feature_extraction",
}
_NEUTRAL_FEATURE_PUBLIC_TO_SOURCE = {
    public: source for source, public in _NEUTRAL_FEATURE_EXPORTS.items()
}

__all__ = (
    "AffirmativeRelation",
    "BOOLEAN_WITNESS",
    "FROZEN_VISUAL_SCORE",
    "OBJECT",
    "PANEL",
    "SOFT_SEMANTIC",
    "WITNESS",
    "ContractViolation",
    "InvarianceContract",
    "LegContract",
    "LegReference",
    "LegRegistry",
    "LegSemantics",
    "Transform",
    "TypedValue",
    "Unit",
    "ValueType",
    "implementation_sha256",
    "BILATERAL_SYMMETRY_ALGORITHM_ID",
    "BILATERAL_SYMMETRY_SCORE",
    "BilateralSymmetryInputError",
    "BilateralSymmetryObservation",
    "bilateral_symmetry_contract",
    "bilateral_symmetry_operation_digest",
    "bilateral_symmetry_score",
    "measure_bilateral_symmetry",
    "register_bilateral_symmetry_leg",
    "NEUTRAL_FEATURE_ALGORITHM_ID",
    "NEUTRAL_FEATURE_EXTRACTOR_VERSION",
    "FEATURE_GROUP_IDS",
    "NeutralFeatureExtraction",
    "NeutralFeatureGroup",
    "NeutralFeatureReceipt",
    "NeutralInputIdentity",
    "NeutralPacketCommitment",
    "extract_neutral_features",
    "feature_group_catalog",
    "feature_group_catalog_digest",
    "feature_space_for_group",
    "feature_space_for_groups",
    "full_neutral_feature_space",
    "project_neutral_feature_extraction",
    "verify_neutral_feature_extraction",
)


def __getattr__(name: str) -> object:
    source_name = _NEUTRAL_FEATURE_PUBLIC_TO_SOURCE.get(name)
    if source_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".neutral_features", __name__), source_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
