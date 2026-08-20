"""Cautious, reviewer-facing calibration of accumulated thermal evidence.

This module intentionally produces only review, insufficient-evidence, or
no-persistent-evidence conclusions.  Its thresholds are fixed for development
cases and synthetic fixtures; they are not operational or incident labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from firesentinel.core.records import OutcomeState, ReasonCode

OUTCOME_THRESHOLD_SELECTION_SCOPE = "development_cases_and_synthetic_fixtures_only"

_POOR_QUALITY_REASONS = frozenset(
    (
        ReasonCode.COVERAGE_INSUFFICIENT,
        ReasonCode.FRAME_BLANK,
        ReasonCode.FRAME_SATURATED,
        ReasonCode.CONTRAST_LOW,
    )
)
_FAILURE_REASONS = frozenset(
    (
        ReasonCode.SOURCE_MISSING,
        ReasonCode.SOURCE_CORRUPT,
        ReasonCode.TIMEOUT,
        ReasonCode.CANCELLED,
        ReasonCode.ARTIFACT_WRITE_FAILED,
        ReasonCode.CONFIGURATION_INVALID,
    )
)


@dataclass(frozen=True, slots=True)
class OutcomeThresholds:
    """Pinned development-only limits used to terminate a review case safely."""

    minimum_review_persistence_count: int = 2
    minimum_review_persistence_confidence: float = 0.5
    minimum_observations_for_no_persistent_evidence: int = 2
    maximum_insufficient_evidence_confidence: float = 0.49

    def __post_init__(self) -> None:
        for field in (
            "minimum_review_persistence_count",
            "minimum_observations_for_no_persistent_evidence",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field} must be an integer of at least two")
        for field in (
            "minimum_review_persistence_confidence",
            "maximum_insufficient_evidence_confidence",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a finite confidence")
            value = float(value)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be within [0, 1]")
            object.__setattr__(self, field, value)
        if (
            self.maximum_insufficient_evidence_confidence
            >= self.minimum_review_persistence_confidence
        ):
            raise ValueError(
                "maximum_insufficient_evidence_confidence must be below the "
                "review-escalation confidence"
            )

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the complete, reviewer-visible threshold configuration."""

        return {
            "selection_scope": OUTCOME_THRESHOLD_SELECTION_SCOPE,
            "minimum_review_persistence_count": self.minimum_review_persistence_count,
            "minimum_review_persistence_confidence": (
                self.minimum_review_persistence_confidence
            ),
            "minimum_observations_for_no_persistent_evidence": (
                self.minimum_observations_for_no_persistent_evidence
            ),
            "maximum_insufficient_evidence_confidence": (
                self.maximum_insufficient_evidence_confidence
            ),
        }


DEVELOPMENT_OUTCOME_THRESHOLDS = OutcomeThresholds()


@dataclass(frozen=True, slots=True)
class OutcomeEvidence:
    """Only the explicit facts that outcome calibration may inspect."""

    observation_count: int
    usable_observation_count: int
    candidate_region_count: int
    persistence_count: int
    persistence_confidence: float
    reason_codes: tuple[ReasonCode, ...] = ()
    budget_exhausted: bool = False

    def __post_init__(self) -> None:
        for field in (
            "observation_count",
            "usable_observation_count",
            "candidate_region_count",
            "persistence_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.usable_observation_count > self.observation_count:
            raise ValueError("usable_observation_count cannot exceed observation_count")
        if self.persistence_count > self.usable_observation_count:
            raise ValueError("persistence_count cannot exceed usable_observation_count")
        if self.persistence_count and self.candidate_region_count == 0:
            raise ValueError("persistence_count requires at least one candidate region")
        confidence = self.persistence_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("persistence_confidence must be a finite confidence")
        confidence = float(confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("persistence_confidence must be within [0, 1]")
        reasons = tuple(ReasonCode(reason) for reason in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not repeat")
        if not isinstance(self.budget_exhausted, bool):
            raise ValueError("budget_exhausted must be a boolean")
        object.__setattr__(self, "persistence_confidence", confidence)
        object.__setattr__(self, "reason_codes", reasons)

    @classmethod
    def from_local_evidence(cls, payload: Mapping[str, object]) -> OutcomeEvidence:
        """Extract calibration facts from a completed local evidence packet."""

        observations = payload.get("observations")
        persistence = payload.get("persistence")
        if not isinstance(observations, list) or not observations:
            raise ValueError("completed evidence observations are invalid")
        if not isinstance(persistence, Mapping):
            raise ValueError("completed evidence persistence is invalid")

        reasons: list[ReasonCode] = []
        usable_observation_count = 0
        candidate_region_count = 0
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError("completed evidence observation is invalid")
            anomaly = observation.get("anomaly")
            if not isinstance(anomaly, Mapping):
                raise ValueError("completed anomaly evidence is invalid")
            candidate_count = anomaly.get("candidate_pixel_count")
            raw_reasons = anomaly.get("reason_codes")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < 0
            ):
                raise ValueError("completed candidate count is invalid")
            if not isinstance(raw_reasons, list):
                raise ValueError("completed anomaly reasons are invalid")
            try:
                observation_reasons = tuple(
                    ReasonCode(reason) for reason in raw_reasons
                )
            except ValueError as error:
                raise ValueError("completed anomaly reason is unknown") from error
            if not _POOR_QUALITY_REASONS.intersection(observation_reasons):
                usable_observation_count += 1
            candidate_region_count += candidate_count
            reasons.extend(observation_reasons)

        persistence_count = persistence.get("persistence_count")
        persistence_confidence = persistence.get("confidence")
        if (
            isinstance(persistence_count, bool)
            or not isinstance(persistence_count, int)
            or persistence_count < 0
        ):
            raise ValueError("completed persistence_count is invalid")
        if isinstance(persistence_confidence, bool) or not isinstance(
            persistence_confidence, (int, float)
        ):
            raise ValueError("completed persistence confidence is invalid")
        return cls(
            observation_count=len(observations),
            usable_observation_count=usable_observation_count,
            candidate_region_count=candidate_region_count,
            persistence_count=persistence_count,
            persistence_confidence=float(persistence_confidence),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


_REASON_TEMPLATES: dict[ReasonCode, str] = {
    ReasonCode.VALID: "The supplied observation passed the configured checks.",
    ReasonCode.SOURCE_MISSING: "A required cached source was unavailable.",
    ReasonCode.SOURCE_CORRUPT: (
        "A required source or evidence packet could not be verified."
    ),
    ReasonCode.OBSERVATION_NOT_ALLOWED: (
        "The requested observation was not allowlisted."
    ),
    ReasonCode.COVERAGE_INSUFFICIENT: (
        "Usable image coverage was below the development limit."
    ),
    ReasonCode.FRAME_BLANK: "The observation did not contain usable thermal variation.",
    ReasonCode.FRAME_SATURATED: (
        "The observation was too clipped or saturated for this check."
    ),
    ReasonCode.CONTRAST_LOW: "Thermal contrast was too weak for a reliable comparison.",
    ReasonCode.ALIGNMENT_FAILED: (
        "Observations could not be aligned reliably for a temporal comparison."
    ),
    ReasonCode.THERMAL_ANOMALY_WEAK: (
        "A thermal anomaly was present but did not meet the persistence threshold."
    ),
    ReasonCode.THERMAL_EVIDENCE_PERSISTENT: (
        "Thermal evidence persisted across the required aligned observations."
    ),
    ReasonCode.THERMAL_EVIDENCE_ABSENT: (
        "No contextual thermal evidence was measured in the usable observations."
    ),
    ReasonCode.BANDS_CONFLICT: (
        "The compared bands disagree and need reviewer interpretation."
    ),
    ReasonCode.BUDGET_EXHAUSTED: (
        "The configured observation, byte, or elapsed-time budget was exhausted."
    ),
    ReasonCode.TIMEOUT: "Processing reached its configured time limit.",
    ReasonCode.CANCELLED: (
        "Processing was cancelled before sufficient evidence was available."
    ),
    ReasonCode.ARTIFACT_WRITE_FAILED: (
        "The evidence artifact could not be written completely."
    ),
    ReasonCode.CONFIGURATION_INVALID: (
        "The configured evidence processing parameters were invalid."
    ),
    ReasonCode.HUMAN_REVIEW_REQUIRED: (
        "A qualified reviewer must assess this thermal evidence."
    ),
    ReasonCode.INSUFFICIENT_EVIDENCE: (
        "The available evidence is not sufficient for a stronger outcome."
    ),
    ReasonCode.NO_PERSISTENT_EVIDENCE: (
        "No thermal region persisted at the configured development threshold."
    ),
}

_OUTCOME_TEMPLATES: dict[OutcomeState, str] = {
    OutcomeState.REVIEW_ESCALATION: (
        "Review escalation: persistent thermal evidence met the development threshold."
    ),
    OutcomeState.HUMAN_REVIEW: (
        "Human review: the evidence contains an unresolved conflict."
    ),
    OutcomeState.NO_PERSISTENT_EVIDENCE: (
        "No persistent evidence: the completed usable comparison did not meet the "
        "development threshold."
    ),
    OutcomeState.INSUFFICIENT_EVIDENCE: (
        "Insufficient evidence: the case was safely abstained rather than strengthened."
    ),
    OutcomeState.FAILED: (
        "Evidence processing failed and did not produce a usable outcome."
    ),
}


@dataclass(frozen=True, slots=True)
class CalibratedOutcome:
    """A bounded outcome plus the fixed-language reviewer explanation."""

    state: OutcomeState
    reason_codes: tuple[ReasonCode, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, OutcomeState):
            raise TypeError("state must be OutcomeState")
        reasons = tuple(ReasonCode(reason) for reason in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not repeat")
        confidence = self.confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a finite confidence")
        confidence = float(confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "confidence", confidence)

    @property
    def explanation(self) -> str:
        """Return a deterministic plain-language explanation for reviewers."""

        return explain_outcome(self.state, self.reason_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


def calibrate_outcome(
    evidence: OutcomeEvidence,
    thresholds: OutcomeThresholds = DEVELOPMENT_OUTCOME_THRESHOLDS,
) -> CalibratedOutcome:
    """Safely turn accumulated facts into a non-operational reviewer outcome."""

    if not isinstance(evidence, OutcomeEvidence):
        raise TypeError("evidence must be OutcomeEvidence")
    if not isinstance(thresholds, OutcomeThresholds):
        raise TypeError("thresholds must be OutcomeThresholds")

    reasons = evidence.reason_codes
    has_poor_quality = bool(_POOR_QUALITY_REASONS.intersection(reasons))
    has_alignment_failure = ReasonCode.ALIGNMENT_FAILED in reasons
    has_budget_exhaustion = (
        evidence.budget_exhausted or ReasonCode.BUDGET_EXHAUSTED in reasons
    )

    if _FAILURE_REASONS.intersection(reasons):
        return _outcome(OutcomeState.FAILED, reasons, 0.0)
    if has_poor_quality or has_alignment_failure or has_budget_exhaustion:
        additions = [ReasonCode.INSUFFICIENT_EVIDENCE]
        if has_budget_exhaustion:
            additions.insert(0, ReasonCode.BUDGET_EXHAUSTED)
        return _outcome(
            OutcomeState.INSUFFICIENT_EVIDENCE,
            _with_reasons(reasons, *additions),
            min(
                evidence.persistence_confidence,
                thresholds.maximum_insufficient_evidence_confidence,
            ),
        )
    if ReasonCode.BANDS_CONFLICT in reasons:
        return _outcome(
            OutcomeState.HUMAN_REVIEW,
            _with_reasons(reasons, ReasonCode.HUMAN_REVIEW_REQUIRED),
            0.0,
        )
    if evidence.usable_observation_count < (
        thresholds.minimum_observations_for_no_persistent_evidence
    ):
        return _outcome(
            OutcomeState.INSUFFICIENT_EVIDENCE,
            _with_reasons(reasons, ReasonCode.INSUFFICIENT_EVIDENCE),
            min(
                evidence.persistence_confidence,
                thresholds.maximum_insufficient_evidence_confidence,
            ),
        )
    if (
        evidence.persistence_count >= thresholds.minimum_review_persistence_count
        and evidence.persistence_confidence
        >= thresholds.minimum_review_persistence_confidence
    ):
        return _outcome(
            OutcomeState.REVIEW_ESCALATION,
            _with_reasons(
                reasons,
                ReasonCode.THERMAL_EVIDENCE_PERSISTENT,
                ReasonCode.HUMAN_REVIEW_REQUIRED,
            ),
            evidence.persistence_confidence,
        )
    if evidence.candidate_region_count == 0:
        reason = ReasonCode.THERMAL_EVIDENCE_ABSENT
    else:
        reason = ReasonCode.THERMAL_ANOMALY_WEAK
    return _outcome(
        OutcomeState.NO_PERSISTENT_EVIDENCE,
        _with_reasons(reasons, reason, ReasonCode.NO_PERSISTENT_EVIDENCE),
        0.0,
    )


def explain_reason_codes(reason_codes: tuple[ReasonCode, ...]) -> tuple[str, ...]:
    """Return fixed plain-language text for each closed reason code."""

    reasons = tuple(ReasonCode(reason) for reason in reason_codes)
    if len(reasons) != len(set(reasons)):
        raise ValueError("reason_codes must not repeat")
    return tuple(_REASON_TEMPLATES[reason] for reason in reasons)


def explain_outcome(state: OutcomeState, reason_codes: tuple[ReasonCode, ...]) -> str:
    """Build one stable reviewer explanation from outcome and reason templates."""

    outcome_state = OutcomeState(state)
    return " ".join(
        (_OUTCOME_TEMPLATES[outcome_state], *explain_reason_codes(reason_codes))
    )


def _outcome(
    state: OutcomeState, reason_codes: tuple[ReasonCode, ...], confidence: float
) -> CalibratedOutcome:
    return CalibratedOutcome(state, reason_codes, confidence)


def _with_reasons(
    existing: tuple[ReasonCode, ...], *additions: ReasonCode
) -> tuple[ReasonCode, ...]:
    return tuple(dict.fromkeys((*existing, *additions)))


__all__ = [
    "CalibratedOutcome",
    "DEVELOPMENT_OUTCOME_THRESHOLDS",
    "OUTCOME_THRESHOLD_SELECTION_SCOPE",
    "OutcomeEvidence",
    "OutcomeThresholds",
    "calibrate_outcome",
    "explain_outcome",
    "explain_reason_codes",
]
