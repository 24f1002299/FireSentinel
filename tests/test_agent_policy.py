"""Table-driven contracts for the transparent, state-free agent policy."""

from __future__ import annotations

import pytest

from firesentinel.agent.policy import (
    ConsiderationStatus,
    EvidenceSnapshot,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    TransparentAgentPolicy,
)
from firesentinel.agent.tools import ToolError, ToolErrorCode, ToolResult
from firesentinel.core.records import ActionType, Budget, ReasonCode


def _budget(
    *,
    used_observations: int = 1,
    max_observations: int = 3,
    used_bytes: int = 100,
    max_bytes: int = 1_000,
    used_elapsed_seconds: float = 1.0,
    max_elapsed_seconds: float = 30.0,
) -> Budget:
    return Budget(
        max_observations=max_observations,
        used_observations=used_observations,
        max_bytes=max_bytes,
        used_bytes=used_bytes,
        max_elapsed_seconds=max_elapsed_seconds,
        used_elapsed_seconds=used_elapsed_seconds,
        max_retries=0,
        used_retries=0,
    )


def _evidence(*reasons: ReasonCode, persistence_count: int = 0) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        evidence_ids=("evidence-a",),
        reason_codes=reasons,
        candidate_region_count=1,
        persistence_count=persistence_count,
        persistence_confidence=0.6 if persistence_count >= 2 else 0.0,
    )


AVAILABLE = (
    PolicyAction(ActionType.NEXT_TIMESTAMP, "later"),
    PolicyAction(ActionType.ALTERNATE_BAND, "alternate"),
    PolicyAction(ActionType.PRE_EVENT_BASELINE, "baseline"),
)


def _failed_tool_result() -> ToolResult:
    budget = _budget()
    return ToolResult(
        action_type=ActionType.NEXT_TIMESTAMP,
        observation_id="later",
        accepted=False,
        idempotent=False,
        evidence_ids=("evidence-a",),
        budget=budget,
        terminal_action=None,
        error=ToolError(
            ToolErrorCode.SOURCE_UNAVAILABLE,
            ReasonCode.SOURCE_MISSING,
            "pinned cached source is unavailable",
        ),
    )


@pytest.mark.parametrize(
    (
        "name,evidence,budget,available,last_result,expected_action,expected_rule,condition"
    ),
    [
        (
            "emerging",
            _evidence(ReasonCode.THERMAL_ANOMALY_WEAK),
            _budget(),
            AVAILABLE,
            None,
            PolicyAction(ActionType.NEXT_TIMESTAMP, "later"),
            PolicyRule.WEAK_CONTEXTUAL_FOLLOWUP,
            PolicyCondition.WEAK_CONTEXTUAL_CONTRAST,
        ),
        (
            "control",
            _evidence(ReasonCode.NO_PERSISTENT_EVIDENCE),
            _budget(used_observations=2),
            AVAILABLE,
            None,
            PolicyAction(ActionType.FINALIZE),
            PolicyRule.ABSENT_PERSISTENCE_FINALIZE,
            PolicyCondition.PERSISTENCE_ABSENT,
        ),
        (
            "abstention",
            _evidence(ReasonCode.FRAME_BLANK),
            _budget(),
            (),
            None,
            PolicyAction(ActionType.ABSTAIN),
            PolicyRule.NO_FOLLOWUP_ABSTAIN,
            PolicyCondition.POOR_QUALITY,
        ),
        (
            "tool_failure",
            _evidence(ReasonCode.THERMAL_ANOMALY_WEAK),
            _budget(),
            AVAILABLE,
            _failed_tool_result(),
            PolicyAction(ActionType.ABSTAIN),
            PolicyRule.TOOL_FAILURE_ABSTAIN,
            PolicyCondition.TOOL_FAILURE,
        ),
        (
            "conflict",
            _evidence(ReasonCode.BANDS_CONFLICT),
            _budget(),
            AVAILABLE,
            None,
            PolicyAction(ActionType.ALTERNATE_BAND, "alternate"),
            PolicyRule.BANDS_CONFLICT_ALTERNATE,
            PolicyCondition.BANDS_CONFLICT,
        ),
        (
            "success",
            _evidence(ReasonCode.THERMAL_EVIDENCE_PERSISTENT, persistence_count=2),
            _budget(used_observations=2),
            AVAILABLE,
            None,
            PolicyAction(ActionType.REQUEST_HUMAN_REVIEW),
            PolicyRule.PERSISTENT_EVIDENCE_REVIEW,
            PolicyCondition.PERSISTENT_EVIDENCE,
        ),
        (
            "conflict_blocks_apparent_success",
            _evidence(
                ReasonCode.BANDS_CONFLICT,
                ReasonCode.THERMAL_EVIDENCE_PERSISTENT,
                persistence_count=2,
            ),
            _budget(used_observations=2),
            AVAILABLE,
            None,
            PolicyAction(ActionType.ALTERNATE_BAND, "alternate"),
            PolicyRule.BANDS_CONFLICT_ALTERNATE,
            PolicyCondition.BANDS_CONFLICT,
        ),
        (
            "budget_exhausted",
            _evidence(ReasonCode.THERMAL_ANOMALY_WEAK),
            _budget(used_observations=3),
            AVAILABLE,
            None,
            PolicyAction(ActionType.ABSTAIN),
            PolicyRule.BUDGET_EXHAUSTED_ABSTAIN,
            PolicyCondition.BUDGET_EXHAUSTED,
        ),
    ],
)
def test_priority_table_selects_a_cautious_explicit_action(
    name: str,
    evidence: EvidenceSnapshot,
    budget: Budget,
    available: tuple[PolicyAction, ...],
    last_result: ToolResult | None,
    expected_action: PolicyAction,
    expected_rule: PolicyRule,
    condition: PolicyCondition,
) -> None:
    del name
    decision = TransparentAgentPolicy().decide(
        evidence,
        budget,
        available,
        last_tool_result=last_result,
    )

    assert decision.selected_action == expected_action
    assert decision.rule is expected_rule
    assert condition in decision.satisfied_conditions
    assert any(
        item.status is ConsiderationStatus.SELECTED and item.action == expected_action
        for item in decision.considered_actions
    )
    assert decision.rejected_actions
    assert all(
        item.status is ConsiderationStatus.REJECTED
        for item in decision.rejected_actions
    )


def test_same_inputs_always_select_the_same_action_and_full_log() -> None:
    policy = TransparentAgentPolicy()
    evidence = _evidence(ReasonCode.THERMAL_ANOMALY_WEAK)
    budget = _budget()

    first = policy.decide(evidence, budget, AVAILABLE)
    second = policy.decide(evidence, budget, AVAILABLE)

    assert first == second
    assert first.selected_action == PolicyAction(ActionType.NEXT_TIMESTAMP, "later")
    assert first.selection_reason == (
        "weak contextual contrast needs the next prescribed comparison"
    )
    assert len(first.considered_actions) == len(AVAILABLE) + 3
    assert len(first.rejected_actions) == len(AVAILABLE) + 2


def test_low_confidence_persistence_does_not_bypass_outcome_calibration() -> None:
    evidence = EvidenceSnapshot(
        evidence_ids=("evidence-low",),
        reason_codes=(ReasonCode.THERMAL_EVIDENCE_PERSISTENT,),
        candidate_region_count=2,
        persistence_count=2,
        persistence_confidence=0.49,
    )

    decision = TransparentAgentPolicy().decide(evidence, _budget(), ())

    assert decision.selected_action == PolicyAction(ActionType.ABSTAIN)
    assert PolicyCondition.PERSISTENT_EVIDENCE not in decision.satisfied_conditions


def test_controlled_evidence_change_changes_action_and_is_logged() -> None:
    policy = TransparentAgentPolicy()
    initial = _evidence(ReasonCode.THERMAL_ANOMALY_WEAK)
    persistent = EvidenceSnapshot(
        evidence_ids=("evidence-b",),
        reason_codes=(ReasonCode.THERMAL_EVIDENCE_PERSISTENT,),
        candidate_region_count=2,
        persistence_count=2,
        persistence_confidence=0.75,
    )
    absent = EvidenceSnapshot(
        evidence_ids=("evidence-c",),
        reason_codes=(ReasonCode.NO_PERSISTENT_EVIDENCE,),
        candidate_region_count=0,
        persistence_count=0,
        persistence_confidence=0.0,
    )

    emerging = policy.decide(initial, _budget(), AVAILABLE)
    escalated = policy.decide(
        persistent,
        _budget(used_observations=2),
        AVAILABLE,
        previous_evidence=initial,
    )
    control = policy.decide(
        absent,
        _budget(used_observations=2),
        AVAILABLE,
        previous_evidence=initial,
    )

    assert emerging.selected_action.action_type is ActionType.NEXT_TIMESTAMP
    assert escalated.selected_action.action_type is ActionType.REQUEST_HUMAN_REVIEW
    assert control.selected_action.action_type is ActionType.FINALIZE
    assert {change.field for change in escalated.evidence_changes} >= {
        "candidate_region_count",
        "persistence_count",
        "persistence_confidence",
        "reason_codes",
        "evidence_ids",
    }


def test_local_evidence_adapter_reads_only_explicit_packet_measurements() -> None:
    snapshot = EvidenceSnapshot.from_local_evidence(
        {
            "observations": [
                {
                    "anomaly": {
                        "reason_codes": ["thermal_anomaly_weak"],
                        "components": [{"label": 1}],
                    }
                },
                {"anomaly": {"reason_codes": [], "components": [{"label": 2}]}},
            ],
            "persistence": {"persistence_count": 2, "confidence": 0.5},
        },
        evidence_id="evidence-packet",
    )

    assert snapshot.candidate_region_count == 2
    assert snapshot.persistence_count == 2
    assert snapshot.reason_codes == (ReasonCode.THERMAL_ANOMALY_WEAK,)
