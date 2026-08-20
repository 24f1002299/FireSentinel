"""A deterministic, inspectable rule policy for bounded thermal evidence.

There is no language model, score optimization, learned state, or prediction of
future value in this module.  ``TransparentAgentPolicy.decide`` is a pure
function of explicit evidence, an explicit resource budget, explicit allowed
actions, and the prior tool reply.  Its complete reasoning is returned in the
decision record for reviewers and later trace construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from re import compile

from firesentinel.agent.tools import BoundedObservationTools, ToolResult
from firesentinel.core.records import ActionType, Budget, ReasonCode

_IDENTIFIER = compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_OBSERVATION_ACTIONS = frozenset(
    (
        ActionType.NEXT_TIMESTAMP,
        ActionType.ALTERNATE_BAND,
        ActionType.PRE_EVENT_BASELINE,
    )
)
_TERMINAL_ACTIONS = frozenset(
    (
        ActionType.FINALIZE,
        ActionType.ABSTAIN,
        ActionType.REQUEST_HUMAN_REVIEW,
    )
)
_POOR_QUALITY_REASONS = frozenset(
    (
        ReasonCode.COVERAGE_INSUFFICIENT,
        ReasonCode.FRAME_BLANK,
        ReasonCode.FRAME_SATURATED,
    )
)
_WEAK_CONTEXTUAL_REASONS = frozenset(
    (ReasonCode.CONTRAST_LOW, ReasonCode.THERMAL_ANOMALY_WEAK)
)
_ABSENT_PERSISTENCE_REASONS = frozenset(
    (ReasonCode.THERMAL_EVIDENCE_ABSENT, ReasonCode.NO_PERSISTENT_EVIDENCE)
)
_ACTION_ORDER = {
    ActionType.NEXT_TIMESTAMP: 0,
    ActionType.ALTERNATE_BAND: 1,
    ActionType.PRE_EVENT_BASELINE: 2,
    ActionType.FINALIZE: 3,
    ActionType.ABSTAIN: 4,
    ActionType.REQUEST_HUMAN_REVIEW: 5,
}


class PolicyCondition(StrEnum):
    """Closed conditions evaluated directly from supplied facts."""

    TOOL_FAILURE = "tool_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PERSISTENT_EVIDENCE = "persistent_evidence"
    POOR_QUALITY = "poor_quality"
    BANDS_CONFLICT = "bands_conflict"
    PERSISTENCE_ABSENT = "persistence_absent"
    WEAK_CONTEXTUAL_CONTRAST = "weak_contextual_contrast"
    NO_FOLLOWUP_AVAILABLE = "no_followup_available"
    NO_DECISIVE_EVIDENCE = "no_decisive_evidence"


class PolicyRule(StrEnum):
    """The transparent priority table; no numeric future-value formula exists."""

    TOOL_FAILURE_ABSTAIN = "tool_failure_abstain"
    BUDGET_EXHAUSTED_ABSTAIN = "budget_exhausted_abstain"
    PERSISTENT_EVIDENCE_REVIEW = "persistent_evidence_review"
    POOR_QUALITY_RETRY = "poor_quality_retry"
    BANDS_CONFLICT_ALTERNATE = "bands_conflict_alternate"
    ABSENT_PERSISTENCE_FINALIZE = "absent_persistence_finalize"
    WEAK_CONTEXTUAL_FOLLOWUP = "weak_contextual_followup"
    NO_DECISIVE_EVIDENCE_FOLLOWUP = "no_decisive_evidence_followup"
    NO_FOLLOWUP_ABSTAIN = "no_followup_abstain"


class ConsiderationStatus(StrEnum):
    """Whether a policy action was selected or why it was not selected."""

    SELECTED = "selected"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One possible bounded-tool call, without a path, URL, or mutable state."""

    action_type: ActionType
    observation_id: str | None = None

    def __post_init__(self) -> None:
        if self.action_type in _OBSERVATION_ACTIONS:
            if not isinstance(self.observation_id, str) or not _IDENTIFIER.fullmatch(
                self.observation_id
            ):
                raise ValueError("observation actions require a safe observation_id")
        elif self.action_type in _TERMINAL_ACTIONS:
            if self.observation_id is not None:
                raise ValueError("terminal actions must not have an observation_id")
        else:
            raise ValueError("policy action_type is unsupported")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "action_type": self.action_type.value,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """The bounded evidence facts that a policy is permitted to inspect."""

    evidence_ids: tuple[str, ...]
    reason_codes: tuple[ReasonCode, ...]
    candidate_region_count: int
    persistence_count: int
    persistence_confidence: float

    def __post_init__(self) -> None:
        identifiers = tuple(self.evidence_ids)
        if not all(
            isinstance(item, str) and _IDENTIFIER.fullmatch(item)
            for item in identifiers
        ):
            raise ValueError("evidence_ids must be safe identifiers")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence_ids must not repeat")
        reasons = tuple(ReasonCode(reason) for reason in self.reason_codes)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not repeat")
        if isinstance(self.candidate_region_count, bool) or not isinstance(
            self.candidate_region_count, int
        ):
            raise ValueError("candidate_region_count must be an integer")
        if self.candidate_region_count < 0:
            raise ValueError("candidate_region_count must be non-negative")
        if isinstance(self.persistence_count, bool) or not isinstance(
            self.persistence_count, int
        ):
            raise ValueError("persistence_count must be an integer")
        if self.persistence_count < 0:
            raise ValueError("persistence_count must be non-negative")
        if isinstance(self.persistence_confidence, bool) or not isinstance(
            self.persistence_confidence, (int, float)
        ):
            raise ValueError("persistence_confidence must be numeric")
        confidence = float(self.persistence_confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("persistence_confidence must be within [0, 1]")
        object.__setattr__(self, "evidence_ids", identifiers)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "persistence_confidence", confidence)

    @classmethod
    def from_local_evidence(
        cls, payload: Mapping[str, object], *, evidence_id: str
    ) -> EvidenceSnapshot:
        """Summarize a Day 17 evidence JSON payload without reopening any files."""

        observations = payload.get("observations")
        persistence = payload.get("persistence")
        if not isinstance(observations, list) or not isinstance(persistence, Mapping):
            raise ValueError("local evidence packet has an invalid shape")
        reasons: list[ReasonCode] = []
        candidate_regions = 0
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError("local evidence observation is invalid")
            anomaly = observation.get("anomaly")
            if not isinstance(anomaly, Mapping):
                raise ValueError("local anomaly evidence is invalid")
            raw_reasons = anomaly.get("reason_codes")
            components = anomaly.get("components")
            if not isinstance(raw_reasons, list) or not isinstance(components, list):
                raise ValueError("local anomaly evidence has invalid measurements")
            try:
                reasons.extend(ReasonCode(reason) for reason in raw_reasons)
            except ValueError as error:
                raise ValueError("local anomaly has an unknown reason code") from error
            candidate_regions += len(components)
        persistence_count = persistence.get("persistence_count")
        confidence = persistence.get("confidence")
        if isinstance(persistence_count, bool) or not isinstance(
            persistence_count, int
        ):
            raise ValueError("local persistence_count is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("local persistence confidence is invalid")
        return cls(
            evidence_ids=(evidence_id,),
            reason_codes=tuple(dict.fromkeys(reasons)),
            candidate_region_count=candidate_regions,
            persistence_count=persistence_count,
            persistence_confidence=float(confidence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": [reason.value for reason in self.reason_codes],
            "candidate_region_count": self.candidate_region_count,
            "persistence_count": self.persistence_count,
            "persistence_confidence": self.persistence_confidence,
        }


@dataclass(frozen=True, slots=True)
class EvidenceChange:
    """One reviewer-visible difference between two explicit evidence states."""

    field: str
    before: object
    after: object

    def to_dict(self) -> dict[str, object]:
        return {"field": self.field, "before": self.before, "after": self.after}


@dataclass(frozen=True, slots=True)
class ConsideredAction:
    """A candidate logged with the deterministic reason it lost or won."""

    action: PolicyAction
    status: ConsiderationStatus
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A complete, stateless decision table evaluation suitable for a trace."""

    selected_action: PolicyAction
    rule: PolicyRule
    satisfied_conditions: tuple[PolicyCondition, ...]
    selection_reason: str
    considered_actions: tuple[ConsideredAction, ...]
    rejected_actions: tuple[ConsideredAction, ...]
    evidence_changes: tuple[EvidenceChange, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selected_action, PolicyAction):
            raise TypeError("selected_action must be PolicyAction")
        if not isinstance(self.rule, PolicyRule):
            raise TypeError("rule must be PolicyRule")
        conditions = tuple(PolicyCondition(item) for item in self.satisfied_conditions)
        if len(conditions) != len(set(conditions)):
            raise ValueError("satisfied_conditions must not repeat")
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("selection_reason must be non-empty")
        if not self.considered_actions:
            raise ValueError("considered_actions must not be empty")
        selected = tuple(
            item
            for item in self.considered_actions
            if item.status is ConsiderationStatus.SELECTED
        )
        if selected != (
            ConsideredAction(
                self.selected_action,
                ConsiderationStatus.SELECTED,
                self.selection_reason,
            ),
        ):
            raise ValueError(
                "considered_actions must contain exactly the selected action"
            )
        rejected = tuple(
            item
            for item in self.considered_actions
            if item.status is ConsiderationStatus.REJECTED
        )
        if rejected != self.rejected_actions:
            raise ValueError("rejected_actions must match considered_actions")
        object.__setattr__(self, "satisfied_conditions", conditions)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_action": self.selected_action.to_dict(),
            "rule": self.rule.value,
            "satisfied_conditions": [item.value for item in self.satisfied_conditions],
            "selection_reason": self.selection_reason,
            "considered_actions": [item.to_dict() for item in self.considered_actions],
            "rejected_actions": [item.to_dict() for item in self.rejected_actions],
            "evidence_changes": [item.to_dict() for item in self.evidence_changes],
        }


class TransparentAgentPolicy:
    """A stateless priority table for choosing the next bounded action."""

    def decide(
        self,
        evidence: EvidenceSnapshot,
        budget: Budget,
        available_actions: Iterable[PolicyAction],
        *,
        previous_evidence: EvidenceSnapshot | None = None,
        last_tool_result: ToolResult | None = None,
    ) -> PolicyDecision:
        """Select one action and return every condition and rejected candidate.

        The arguments contain all decision inputs.  This method reads no files,
        calls no tools, changes no state, and has no random component.
        """

        if not isinstance(evidence, EvidenceSnapshot):
            raise TypeError("evidence must be EvidenceSnapshot")
        if not isinstance(budget, Budget):
            raise TypeError("budget must be Budget")
        if previous_evidence is not None and not isinstance(
            previous_evidence, EvidenceSnapshot
        ):
            raise TypeError("previous_evidence must be EvidenceSnapshot or None")
        if last_tool_result is not None and not isinstance(
            last_tool_result, ToolResult
        ):
            raise TypeError("last_tool_result must be ToolResult or None")
        available = _available_actions(available_actions)
        changes = evidence_changes(previous_evidence, evidence)
        conditions = _conditions(evidence, budget, last_tool_result)
        selection, rule, selection_reason = _select(available, conditions)
        considered = _considered_actions(available, selection, selection_reason)
        rejected = tuple(
            item for item in considered if item.status is ConsiderationStatus.REJECTED
        )
        return PolicyDecision(
            selected_action=selection,
            rule=rule,
            satisfied_conditions=conditions,
            selection_reason=selection_reason,
            considered_actions=considered,
            rejected_actions=rejected,
            evidence_changes=changes,
        )


def evidence_changes(
    previous: EvidenceSnapshot | None, current: EvidenceSnapshot
) -> tuple[EvidenceChange, ...]:
    """Return canonical evidence deltas without inferring anything unobserved."""

    if previous is None:
        return ()
    changes: list[EvidenceChange] = []
    for field in (
        "candidate_region_count",
        "persistence_count",
        "persistence_confidence",
    ):
        before = getattr(previous, field)
        after = getattr(current, field)
        if before != after:
            changes.append(EvidenceChange(field, before, after))
    previous_reasons = tuple(reason.value for reason in previous.reason_codes)
    current_reasons = tuple(reason.value for reason in current.reason_codes)
    if previous_reasons != current_reasons:
        changes.append(
            EvidenceChange("reason_codes", previous_reasons, current_reasons)
        )
    previous_ids = previous.evidence_ids
    current_ids = current.evidence_ids
    if previous_ids != current_ids:
        changes.append(EvidenceChange("evidence_ids", previous_ids, current_ids))
    return tuple(changes)


def apply_policy_decision(
    tools: BoundedObservationTools, decision: PolicyDecision
) -> ToolResult:
    """Dispatch a transparent selected action through the bounded tool surface."""

    if not isinstance(tools, BoundedObservationTools):
        raise TypeError("tools must be BoundedObservationTools")
    if not isinstance(decision, PolicyDecision):
        raise TypeError("decision must be PolicyDecision")
    action = decision.selected_action
    if action.action_type is ActionType.NEXT_TIMESTAMP:
        assert action.observation_id is not None
        return tools.next_timestamp(action.observation_id)
    if action.action_type is ActionType.ALTERNATE_BAND:
        assert action.observation_id is not None
        return tools.alternate_band(action.observation_id)
    if action.action_type is ActionType.PRE_EVENT_BASELINE:
        assert action.observation_id is not None
        return tools.pre_event_baseline(action.observation_id)
    if action.action_type is ActionType.FINALIZE:
        return tools.finalize()
    if action.action_type is ActionType.ABSTAIN:
        return tools.abstain()
    if action.action_type is ActionType.REQUEST_HUMAN_REVIEW:
        return tools.request_human_review()
    raise AssertionError(f"unsupported policy action {action.action_type!r}")


def _available_actions(actions: Iterable[PolicyAction]) -> tuple[PolicyAction, ...]:
    values = tuple(actions)
    if not all(isinstance(item, PolicyAction) for item in values):
        raise TypeError("available_actions must contain PolicyAction values")
    if any(item.action_type not in _OBSERVATION_ACTIONS for item in values):
        raise ValueError("available_actions may contain observation actions only")
    keys = tuple((item.action_type, item.observation_id) for item in values)
    if len(keys) != len(set(keys)):
        raise ValueError("available_actions must not repeat an action")
    return tuple(
        sorted(
            values,
            key=lambda item: (
                _ACTION_ORDER[item.action_type],
                item.observation_id or "",
            ),
        )
    )


def _conditions(
    evidence: EvidenceSnapshot, budget: Budget, last_tool_result: ToolResult | None
) -> tuple[PolicyCondition, ...]:
    conditions: list[PolicyCondition] = []
    if last_tool_result is not None and not last_tool_result.accepted:
        conditions.append(PolicyCondition.TOOL_FAILURE)
    if _budget_exhausted(budget):
        conditions.append(PolicyCondition.BUDGET_EXHAUSTED)
    if _POOR_QUALITY_REASONS.intersection(evidence.reason_codes):
        conditions.append(PolicyCondition.POOR_QUALITY)
    if ReasonCode.BANDS_CONFLICT in evidence.reason_codes:
        conditions.append(PolicyCondition.BANDS_CONFLICT)
    if (
        ReasonCode.THERMAL_EVIDENCE_PERSISTENT in evidence.reason_codes
        or evidence.persistence_count >= 2
        and evidence.persistence_confidence > 0.0
    ):
        conditions.append(PolicyCondition.PERSISTENT_EVIDENCE)
    if _ABSENT_PERSISTENCE_REASONS.intersection(evidence.reason_codes):
        conditions.append(PolicyCondition.PERSISTENCE_ABSENT)
    if _WEAK_CONTEXTUAL_REASONS.intersection(evidence.reason_codes):
        conditions.append(PolicyCondition.WEAK_CONTEXTUAL_CONTRAST)
    if not conditions:
        conditions.append(PolicyCondition.NO_DECISIVE_EVIDENCE)
    return tuple(conditions)


def _budget_exhausted(budget: Budget) -> bool:
    return (
        budget.used_observations >= budget.max_observations
        or budget.used_bytes >= budget.max_bytes
        or budget.used_elapsed_seconds >= budget.max_elapsed_seconds
    )


def _select(
    available: Sequence[PolicyAction], conditions: tuple[PolicyCondition, ...]
) -> tuple[PolicyAction, PolicyRule, str]:
    active = set(conditions)
    if PolicyCondition.TOOL_FAILURE in active:
        return _terminal_selection(
            ActionType.ABSTAIN,
            PolicyRule.TOOL_FAILURE_ABSTAIN,
            "the prior bounded tool failed; stop with insufficient evidence",
        )
    if PolicyCondition.BUDGET_EXHAUSTED in active:
        return _terminal_selection(
            ActionType.ABSTAIN,
            PolicyRule.BUDGET_EXHAUSTED_ABSTAIN,
            "a configured resource limit is exhausted",
        )
    if PolicyCondition.POOR_QUALITY in active:
        action = _first_available(
            available,
            (
                ActionType.NEXT_TIMESTAMP,
                ActionType.ALTERNATE_BAND,
                ActionType.PRE_EVENT_BASELINE,
            ),
        )
        if action is not None:
            return (
                action,
                PolicyRule.POOR_QUALITY_RETRY,
                "poor-quality evidence requires an allowlisted replacement observation",
            )
        return _terminal_selection(
            ActionType.ABSTAIN,
            PolicyRule.NO_FOLLOWUP_ABSTAIN,
            "poor-quality evidence has no allowlisted replacement observation",
        )
    if PolicyCondition.BANDS_CONFLICT in active:
        action = _first_available(
            available,
            (ActionType.ALTERNATE_BAND, ActionType.NEXT_TIMESTAMP),
        )
        if action is not None:
            return (
                action,
                PolicyRule.BANDS_CONFLICT_ALTERNATE,
                "conflicting bands require the prescribed alternate-band comparison",
            )
        return _terminal_selection(
            ActionType.ABSTAIN,
            PolicyRule.NO_FOLLOWUP_ABSTAIN,
            "band conflict has no allowlisted comparison observation",
        )
    if PolicyCondition.PERSISTENT_EVIDENCE in active:
        return _terminal_selection(
            ActionType.REQUEST_HUMAN_REVIEW,
            PolicyRule.PERSISTENT_EVIDENCE_REVIEW,
            "persistent thermal evidence requires human review",
        )
    if PolicyCondition.PERSISTENCE_ABSENT in active:
        return _terminal_selection(
            ActionType.FINALIZE,
            PolicyRule.ABSENT_PERSISTENCE_FINALIZE,
            "the contextual response lacks persistence",
        )
    if PolicyCondition.WEAK_CONTEXTUAL_CONTRAST in active:
        action = _first_available(
            available,
            (ActionType.NEXT_TIMESTAMP, ActionType.PRE_EVENT_BASELINE),
        )
        if action is not None:
            return (
                action,
                PolicyRule.WEAK_CONTEXTUAL_FOLLOWUP,
                "weak contextual contrast needs the next prescribed comparison",
            )
        return _terminal_selection(
            ActionType.ABSTAIN,
            PolicyRule.NO_FOLLOWUP_ABSTAIN,
            "weak contextual contrast has no allowlisted follow-up",
        )
    action = _first_available(
        available,
        (ActionType.NEXT_TIMESTAMP, ActionType.PRE_EVENT_BASELINE),
    )
    if action is not None:
        return (
            action,
            PolicyRule.NO_DECISIVE_EVIDENCE_FOLLOWUP,
            "no decisive evidence is present; use the first prescribed comparison",
        )
    return _terminal_selection(
        ActionType.ABSTAIN,
        PolicyRule.NO_FOLLOWUP_ABSTAIN,
        "no allowlisted follow-up observation is available",
    )


def _first_available(
    available: Sequence[PolicyAction], priorities: Sequence[ActionType]
) -> PolicyAction | None:
    for action_type in priorities:
        for candidate in available:
            if candidate.action_type is action_type:
                return candidate
    return None


def _terminal_selection(
    action_type: ActionType, rule: PolicyRule, reason: str
) -> tuple[PolicyAction, PolicyRule, str]:
    return PolicyAction(action_type), rule, reason


def _considered_actions(
    available: Sequence[PolicyAction], selected: PolicyAction, reason: str
) -> tuple[ConsideredAction, ...]:
    candidates = (*available, *(PolicyAction(action) for action in _TERMINAL_ACTIONS))
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                _ACTION_ORDER[item.action_type],
                item.observation_id or "",
            ),
        )
    )
    considered: list[ConsideredAction] = []
    for candidate in ordered:
        if candidate == selected:
            considered.append(
                ConsideredAction(candidate, ConsiderationStatus.SELECTED, reason)
            )
        else:
            considered.append(
                ConsideredAction(
                    candidate,
                    ConsiderationStatus.REJECTED,
                    "lower priority than the selected rule",
                )
            )
    return tuple(considered)


__all__ = [
    "ConsiderationStatus",
    "ConsideredAction",
    "EvidenceChange",
    "EvidenceSnapshot",
    "PolicyAction",
    "PolicyCondition",
    "PolicyDecision",
    "PolicyRule",
    "TransparentAgentPolicy",
    "apply_policy_decision",
    "evidence_changes",
]
