"""Verified Bongard corpus, exposure, and benchmark infrastructure."""

from .runtime_source_snapshot import capture_loaded_source


_PACKAGE_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from importlib import import_module

from .corpus import (
    BongardTask,
    CorpusError,
    CorpusLayoutError,
    CorpusManifest,
    CorpusValidationError,
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_REGIME_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    PanelManifest,
    ShapeBongardCorpus,
    SplitAssignment,
    SplitIndex,
    TaskManifest,
    discover_corpus,
)
from .exposure import (
    ExposureError,
    ExposureIntegrityError,
    ExposureLedger,
    ExposureViolation,
    TaskPartition,
    deterministic_partition,
    import_historical_exposures,
    task_id_from_panel_id,
)
from .historical_exposure import (
    BasicFamilyPartition,
    HistoricalExposureError,
    HistoricalExposureSeed,
    load_historical_exposure,
    verify_historical_exposure,
)
from .cohorts import (
    CohortError,
    CohortReport,
    ParsedOfficialTaskId,
    TaskCohortRecord,
    build_cohort_report,
    classify_task,
    parse_official_task_id,
    select_tasks,
)
from .release import (
    OfficialReleaseDescriptor,
    ReleaseIdentityError,
    load_official_release,
)
from .image_audit import (
    ImageAnomaly,
    ImageAuditError,
    ImageAuditReport,
    ImageExpectationError,
    ImageExpectations,
    audit_corpus_images,
)
_LAZY_EXPORT_MODULES = {
    name: ".soft_predicates"
    for name in (
        "CalibrationBand",
        "CalibrationDesign",
        "CalibrationError",
        "CalibrationObservation",
        "CalibratedPredictiveSupport",
        "DevelopmentUnit",
        "FrozenVisualScore",
        "MonotoneCalibrationArtifact",
        "ObservationRole",
        "PreregisteredCalibrationPlan",
        "RegisteredSoftPredicate",
        "SoftPredicateClaim",
        "SoftPredicateError",
        "SoftPredicateIntegrityError",
        "fit_monotone_calibration",
        "register_soft_predicate",
    )
}
_LAZY_EXPORT_MODULES.update(
    {
        name: ".support_prototypes"
        for name in (
            "ContrastiveMargin",
            "FeatureDimension",
            "FeatureInterval",
            "FrozenFeatureSpace",
            "FrozenPanelFeatures",
            "FrozenSupportPrototypes",
            "PositivePrototypeFormula",
            "SUPPORT_PROTOTYPE_FEATURES",
            "SupportMember",
            "SupportPrototypeError",
            "SupportPrototypeIntegrityError",
            "SupportPrototypePlan",
            "contrastive_margin",
            "evaluate_frozen_support_member",
            "evaluate_support_prototype",
            "fit_support_prototypes",
            "panel_side_assignment_digest",
            "register_support_prototype_leg",
            "validate_prototype_formula",
            "verify_support_prototypes",
        )
    }
)
_LAZY_EXPORT_MODULES.update(
    {
        name: ".prototype_artifacts"
        for name in (
            "FeatureExtractionPreimage",
            "PrototypeFreezePolicy",
            "PrototypePreQueryFreeze",
            "PrototypeQueryArtifact",
            "PrototypeSupportReplayArtifact",
            "PrototypeTruthEvidence",
        )
    }
)
_LAZY_EXPORT_MODULES.update(
    {
        name: ".prototype_calibration"
        for name in (
            "PrototypeCalibrationError",
            "PrototypeCalibrationIntegrityError",
            "PrototypeCalibrationRecord",
            "calibrate_prototype_margins",
        )
    }
)
_LAZY_EXPORT_MODULES.update(
    {
        name: ".prototype_episode"
        for name in ("HeadlessPrototypeEpisode", "PrototypeEpisodeError")
    }
)

__all__ = [
    "BongardTask",
    "CalibrationBand",
    "CalibrationDesign",
    "CalibrationError",
    "CalibrationObservation",
    "CalibratedPredictiveSupport",
    "CorpusError",
    "CorpusLayoutError",
    "CorpusManifest",
    "CorpusValidationError",
    "ContrastiveMargin",
    "DevelopmentUnit",
    "CohortError",
    "CohortReport",
    "EXPECTED_FAMILY_COUNTS",
    "EXPECTED_REGIME_COUNTS",
    "EXPECTED_SPLIT_COUNTS",
    "ExposureError",
    "ExposureIntegrityError",
    "ExposureLedger",
    "ExposureViolation",
    "HistoricalExposureError",
    "HistoricalExposureSeed",
    "FrozenVisualScore",
    "FeatureDimension",
    "FeatureInterval",
    "FrozenFeatureSpace",
    "FrozenPanelFeatures",
    "FrozenSupportPrototypes",
    "FeatureExtractionPreimage",
    "HeadlessPrototypeEpisode",
    "ImageAnomaly",
    "ImageAuditError",
    "ImageAuditReport",
    "ImageExpectationError",
    "ImageExpectations",
    "BasicFamilyPartition",
    "OfficialReleaseDescriptor",
    "MonotoneCalibrationArtifact",
    "ObservationRole",
    "PanelManifest",
    "ParsedOfficialTaskId",
    "PreregisteredCalibrationPlan",
    "PositivePrototypeFormula",
    "PrototypeCalibrationError",
    "PrototypeCalibrationIntegrityError",
    "PrototypeCalibrationRecord",
    "PrototypeEpisodeError",
    "PrototypeFreezePolicy",
    "PrototypePreQueryFreeze",
    "PrototypeQueryArtifact",
    "PrototypeSupportReplayArtifact",
    "PrototypeTruthEvidence",
    "SUPPORT_PROTOTYPE_FEATURES",
    "RegisteredSoftPredicate",
    "ReleaseIdentityError",
    "ShapeBongardCorpus",
    "SplitAssignment",
    "SplitIndex",
    "SoftPredicateClaim",
    "SoftPredicateError",
    "SoftPredicateIntegrityError",
    "SupportMember",
    "SupportPrototypeError",
    "SupportPrototypeIntegrityError",
    "SupportPrototypePlan",
    "TaskManifest",
    "TaskCohortRecord",
    "TaskPartition",
    "audit_corpus_images",
    "build_cohort_report",
    "calibrate_prototype_margins",
    "classify_task",
    "contrastive_margin",
    "deterministic_partition",
    "discover_corpus",
    "fit_monotone_calibration",
    "fit_support_prototypes",
    "import_historical_exposures",
    "load_historical_exposure",
    "load_official_release",
    "parse_official_task_id",
    "panel_side_assignment_digest",
    "register_support_prototype_leg",
    "register_soft_predicate",
    "select_tasks",
    "task_id_from_panel_id",
    "evaluate_frozen_support_member",
    "evaluate_support_prototype",
    "validate_prototype_formula",
    "verify_support_prototypes",
    "verify_historical_exposure",
]


def __getattr__(name: str) -> object:
    """Load compatibility-only experiment exports on first explicit use.

    Importing ``bongard`` or an active ``bongard.*`` visual module therefore
    no longer initializes the superseded soft/prototype pipelines.  Their
    public names remain available until those pipelines are retired.
    """

    relative_module = _LAZY_EXPORT_MODULES.get(name)
    if relative_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(relative_module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
