"""OpenCV preprocessing, fixture, and fire-candidate detection components."""

from firesentinel.vision.anomalies import (
    DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS,
    ContextualAnomalyComponent,
    ContextualAnomalyParameters,
    ContextualAnomalyResult,
    extract_contextual_anomalies,
)
from firesentinel.vision.fixtures import (
    FIXTURE_SEED,
    FIXTURE_SHAPE,
    FIXTURE_VERSION,
    NUMERIC_TOLERANCES,
    OFFLINE_MANIFEST_PATH,
    ExpectedComponent,
    FixtureIntegrityError,
    FrameQualityExpectation,
    NumericTolerances,
    PersistenceExpectation,
    SyntheticFixtureBundle,
    SyntheticFixtureCase,
    fixture_bundle_digest,
    fixture_case_digests,
    generate_synthetic_fixture_bundle,
    load_offline_fixture_bundle,
    offline_fixture_manifest,
    verify_fixture_bundle,
)
from firesentinel.vision.persistence import (
    DEVELOPMENT_PERSISTENCE_PARAMETERS,
    AlignedObservation,
    GeospatialGrid,
    PersistenceParameters,
    PersistenceTrack,
    RegionMatch,
    TemporalObservation,
    TemporalPersistenceResult,
    measure_temporal_persistence,
)
from firesentinel.vision.quality import (
    DEVELOPMENT_QUALITY_THRESHOLDS,
    THRESHOLD_SELECTION_SCOPE,
    ObservationQuality,
    ObservationQualityThresholds,
    apply_quality_gate,
    measure_observation_quality,
    measure_prepared_tile_quality,
)
from firesentinel.vision.tiles import (
    PreparedTile,
    TilePreparationParameters,
    prepare_calibrated_tile,
    prepare_tile,
)

_ENGINE_EXPORTS = frozenset(
    {
        "EVIDENCE_SCHEMA_VERSION",
        "JOB_SCHEMA_VERSION",
        "EvidenceJob",
        "EvidenceJobCancelled",
        "EvidenceJobFailure",
        "EvidenceJobObservation",
        "EvidenceJobResult",
        "EvidenceJobSource",
        "EvidenceJobTimeout",
        "load_evidence_job",
        "run_evidence_job",
    }
)


def __getattr__(name: str) -> object:
    """Lazily expose the CLI module without preloading it for ``python -m``."""

    if name in _ENGINE_EXPORTS:
        from firesentinel.vision import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEVELOPMENT_CONTEXTUAL_ANOMALY_PARAMETERS",
    "ContextualAnomalyComponent",
    "ContextualAnomalyParameters",
    "ContextualAnomalyResult",
    "extract_contextual_anomalies",
    "EVIDENCE_SCHEMA_VERSION",
    "JOB_SCHEMA_VERSION",
    "EvidenceJob",
    "EvidenceJobCancelled",
    "EvidenceJobFailure",
    "EvidenceJobObservation",
    "EvidenceJobResult",
    "EvidenceJobSource",
    "EvidenceJobTimeout",
    "load_evidence_job",
    "run_evidence_job",
    "FIXTURE_SEED",
    "FIXTURE_SHAPE",
    "FIXTURE_VERSION",
    "NUMERIC_TOLERANCES",
    "OFFLINE_MANIFEST_PATH",
    "ExpectedComponent",
    "FixtureIntegrityError",
    "FrameQualityExpectation",
    "NumericTolerances",
    "PersistenceExpectation",
    "SyntheticFixtureBundle",
    "SyntheticFixtureCase",
    "fixture_bundle_digest",
    "fixture_case_digests",
    "generate_synthetic_fixture_bundle",
    "load_offline_fixture_bundle",
    "offline_fixture_manifest",
    "verify_fixture_bundle",
    "PreparedTile",
    "TilePreparationParameters",
    "prepare_calibrated_tile",
    "prepare_tile",
    "DEVELOPMENT_QUALITY_THRESHOLDS",
    "THRESHOLD_SELECTION_SCOPE",
    "ObservationQuality",
    "ObservationQualityThresholds",
    "apply_quality_gate",
    "measure_observation_quality",
    "measure_prepared_tile_quality",
    "DEVELOPMENT_PERSISTENCE_PARAMETERS",
    "AlignedObservation",
    "GeospatialGrid",
    "PersistenceParameters",
    "PersistenceTrack",
    "RegionMatch",
    "TemporalObservation",
    "TemporalPersistenceResult",
    "measure_temporal_persistence",
]
