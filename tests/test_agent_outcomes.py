"""Safety and terminology contracts for calibrated reviewer outcomes."""

from __future__ import annotations

import pytest

from firesentinel.agent.outcomes import (
    DEVELOPMENT_OUTCOME_THRESHOLDS,
    OUTCOME_THRESHOLD_SELECTION_SCOPE,
    OutcomeEvidence,
    OutcomeThresholds,
    calibrate_outcome,
    explain_outcome,
    explain_reason_codes,
)
from firesentinel.core.records import OutcomeState, ReasonCode


def _evidence(
    *reason_codes: ReasonCode,
    observation_count: int = 2,
    usable_observation_count: int | None = None,
    candidate_region_count: int = 1,
    persistence_count: int = 0,
    persistence_confidence: float = 0.0,
    budget_exhausted: bool = False,
) -> OutcomeEvidence:
    return OutcomeEvidence(
        observation_count=observation_count,
        usable_observation_count=(
            observation_count
            if usable_observation_count is None
            else usable_observation_count
        ),
        candidate_region_count=candidate_region_count,
        persistence_count=persistence_count,
        persistence_confidence=persistence_confidence,
        reason_codes=reason_codes,
        budget_exhausted=budget_exhausted,
    )


def test_development_thresholds_are_visible_and_keep_a_confidence_gap() -> None:
    thresholds = DEVELOPMENT_OUTCOME_THRESHOLDS

    assert thresholds.to_dict()["selection_scope"] == OUTCOME_THRESHOLD_SELECTION_SCOPE
    assert thresholds.minimum_review_persistence_count == 2
    assert thresholds.minimum_observations_for_no_persistent_evidence == 2
    assert (
        thresholds.maximum_insufficient_evidence_confidence
        < thresholds.minimum_review_persistence_confidence
    )


@pytest.mark.parametrize(
    ("name", "evidence", "state", "required_reason"),
    [
        (
            "poor coverage",
            _evidence(ReasonCode.COVERAGE_INSUFFICIENT, persistence_count=2),
            OutcomeState.INSUFFICIENT_EVIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
        ),
        (
            "alignment failure",
            _evidence(ReasonCode.ALIGNMENT_FAILED, persistence_count=2),
            OutcomeState.INSUFFICIENT_EVIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
        ),
        (
            "conflicting bands",
            _evidence(ReasonCode.BANDS_CONFLICT, persistence_count=2),
            OutcomeState.HUMAN_REVIEW,
            ReasonCode.HUMAN_REVIEW_REQUIRED,
        ),
        (
            "budget exhausted",
            _evidence(persistence_count=2, budget_exhausted=True),
            OutcomeState.INSUFFICIENT_EVIDENCE,
            ReasonCode.BUDGET_EXHAUSTED,
        ),
        (
            "one usable observation",
            _evidence(observation_count=1, persistence_count=1),
            OutcomeState.INSUFFICIENT_EVIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
        ),
    ],
)
def test_low_confidence_or_conflicted_evidence_always_terminates_safely(
    name: str,
    evidence: OutcomeEvidence,
    state: OutcomeState,
    required_reason: ReasonCode,
) -> None:
    del name
    outcome = calibrate_outcome(evidence)

    assert outcome.state is state
    assert required_reason in outcome.reason_codes
    if state is OutcomeState.INSUFFICIENT_EVIDENCE:
        assert (
            outcome.confidence
            <= DEVELOPMENT_OUTCOME_THRESHOLDS.maximum_insufficient_evidence_confidence
        )


def test_persistent_aligned_thermal_evidence_escalates_only_at_threshold() -> None:
    insufficient = calibrate_outcome(
        _evidence(persistence_count=2, persistence_confidence=0.49)
    )
    review = calibrate_outcome(
        _evidence(persistence_count=2, persistence_confidence=0.5)
    )

    assert insufficient.state is OutcomeState.NO_PERSISTENT_EVIDENCE
    assert review.state is OutcomeState.REVIEW_ESCALATION
    assert review.reason_codes[-2:] == (
        ReasonCode.THERMAL_EVIDENCE_PERSISTENT,
        ReasonCode.HUMAN_REVIEW_REQUIRED,
    )


def test_no_candidates_need_the_minimum_usable_comparison_before_no_persistence() -> (
    None
):
    early = calibrate_outcome(_evidence(observation_count=1, candidate_region_count=0))
    complete = calibrate_outcome(_evidence(candidate_region_count=0))

    assert early.state is OutcomeState.INSUFFICIENT_EVIDENCE
    assert complete.state is OutcomeState.NO_PERSISTENT_EVIDENCE
    assert ReasonCode.THERMAL_EVIDENCE_ABSENT in complete.reason_codes


def test_packet_adapter_keeps_weak_anomalies_usable_but_rejects_bad_coverage() -> None:
    facts = OutcomeEvidence.from_local_evidence(
        {
            "observations": [
                {
                    "anomaly": {
                        "candidate_pixel_count": 3,
                        "reason_codes": ["thermal_anomaly_weak"],
                    }
                },
                {
                    "anomaly": {
                        "candidate_pixel_count": 0,
                        "reason_codes": ["coverage_insufficient"],
                    }
                },
            ],
            "persistence": {"persistence_count": 0, "confidence": 0.0},
        }
    )

    assert facts.usable_observation_count == 1
    assert calibrate_outcome(facts).state is OutcomeState.INSUFFICIENT_EVIDENCE


def test_explanations_are_fixed_reason_templates_and_never_name_a_confirmed_event() -> (
    None
):
    reason_codes = (
        ReasonCode.THERMAL_ANOMALY_WEAK,
        ReasonCode.INSUFFICIENT_EVIDENCE,
    )
    explanation = explain_outcome(OutcomeState.INSUFFICIENT_EVIDENCE, reason_codes)

    assert explain_reason_codes(reason_codes) == (
        "A thermal anomaly was present but did not meet the persistence threshold.",
        "The available evidence is not sufficient for a stronger outcome.",
    )
    assert "thermal anomaly" in explanation.lower()
    assert "confirmed wildfire" not in explanation.lower()
    assert "confirmed fire" not in explanation.lower()


def test_invalid_outcome_thresholds_fail_fast() -> None:
    with pytest.raises(ValueError, match="below"):
        OutcomeThresholds(
            minimum_review_persistence_confidence=0.5,
            maximum_insufficient_evidence_confidence=0.5,
        )
