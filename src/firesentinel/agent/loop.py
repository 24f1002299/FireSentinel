"""Deterministic, checkpointed local replay for the bounded agent workflow.

The journal is deliberately a sequence of complete state snapshots instead of
an in-memory controller transcript.  A resume starts from the last complete
JSONL record, restores the bounded tool budget, and either continues at the
next state or returns an already terminal result.  Replayed observations reuse
the content-addressed evidence artifacts produced by the bounded tools.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from re import compile
from typing import Self, cast

from firesentinel.agent.outcomes import (
    CalibratedOutcome,
    OutcomeEvidence,
    calibrate_outcome,
)
from firesentinel.agent.policy import (
    EvidenceSnapshot,
    PolicyAction,
    PolicyDecision,
    PolicyRule,
    TransparentAgentPolicy,
    apply_policy_decision,
)
from firesentinel.agent.tools import (
    BoundedObservationTools,
    ToolError,
    ToolErrorCode,
    ToolManifest,
    ToolResult,
    load_tool_manifest,
)
from firesentinel.config import load_settings
from firesentinel.core.records import ActionType, Budget, OutcomeState, ReasonCode

LOOP_TRACE_SCHEMA_VERSION = 1
LOOP_TRACE_RECORD_TYPE = "bounded_agent_loop_transition"
_IDENTIFIER = compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_OBSERVATION_ACTIONS = frozenset(
    (
        ActionType.NEXT_TIMESTAMP,
        ActionType.ALTERNATE_BAND,
        ActionType.PRE_EVENT_BASELINE,
    )
)
_TERMINAL_STATES = frozenset(
    (
        "finalize",
        "abstain",
        "review",
        "failure",
    )
)
_CHECKPOINT_KEYS = frozenset(
    (
        "schema_version",
        "record_type",
        "trace_id",
        "case_id",
        "sequence",
        "from_state",
        "to_state",
        "event",
        "selected_observation_ids",
        "evidence_ids",
        "budget",
        "analysis",
        "decision",
        "pending_action",
        "last_tool_result",
        "outcome",
        "failure_reason",
    )
)


class AgentLoopState(StrEnum):
    """The explicit replay states, including every safe terminal state."""

    OBSERVE = "observe"
    ANALYZE = "analyze"
    DECIDE = "decide"
    ACT = "act"
    FINALIZE = "finalize"
    ABSTAIN = "abstain"
    REVIEW = "review"
    FAILURE = "failure"

    @property
    def is_terminal(self) -> bool:
        """Whether no further tool action is allowed from this state."""

        return self.value in _TERMINAL_STATES


@dataclass(frozen=True, slots=True)
class EvidenceAnalysis:
    """The two existing bounded views of one completed evidence packet."""

    evidence_id: str
    snapshot: EvidenceSnapshot
    outcome_evidence: OutcomeEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not _IDENTIFIER.fullmatch(
            self.evidence_id
        ):
            raise ValueError("evidence_id must be a safe identifier")
        if not isinstance(self.snapshot, EvidenceSnapshot):
            raise TypeError("snapshot must be EvidenceSnapshot")
        if not isinstance(self.outcome_evidence, OutcomeEvidence):
            raise TypeError("outcome_evidence must be OutcomeEvidence")
        if self.snapshot.evidence_ids != (self.evidence_id,):
            raise ValueError("snapshot must link only its analyzed evidence_id")

    def to_dict(self) -> dict[str, object]:
        facts = self.outcome_evidence
        return {
            "evidence_id": self.evidence_id,
            "snapshot": self.snapshot.to_dict(),
            "outcome_evidence": {
                "observation_count": facts.observation_count,
                "usable_observation_count": facts.usable_observation_count,
                "candidate_region_count": facts.candidate_region_count,
                "persistence_count": facts.persistence_count,
                "persistence_confidence": facts.persistence_confidence,
                "reason_codes": [reason.value for reason in facts.reason_codes],
                "budget_exhausted": facts.budget_exhausted,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise ValueError("analysis must be an object")
        if set(value) != {"evidence_id", "snapshot", "outcome_evidence"}:
            raise ValueError("analysis has an invalid shape")
        snapshot = value["snapshot"]
        facts = value["outcome_evidence"]
        if not isinstance(snapshot, Mapping) or not isinstance(facts, Mapping):
            raise ValueError("analysis has invalid nested data")
        required_snapshot = {
            "evidence_ids",
            "reason_codes",
            "candidate_region_count",
            "persistence_count",
            "persistence_confidence",
        }
        required_facts = {
            "observation_count",
            "usable_observation_count",
            "candidate_region_count",
            "persistence_count",
            "persistence_confidence",
            "reason_codes",
            "budget_exhausted",
        }
        if set(snapshot) != required_snapshot or set(facts) != required_facts:
            raise ValueError("analysis fields are invalid")
        return cls(
            evidence_id=cast(str, value["evidence_id"]),
            snapshot=EvidenceSnapshot(
                evidence_ids=tuple(cast(list[str], snapshot["evidence_ids"])),
                reason_codes=tuple(
                    ReasonCode(item)
                    for item in cast(list[str], snapshot["reason_codes"])
                ),
                candidate_region_count=cast(int, snapshot["candidate_region_count"]),
                persistence_count=cast(int, snapshot["persistence_count"]),
                persistence_confidence=cast(float, snapshot["persistence_confidence"]),
            ),
            outcome_evidence=OutcomeEvidence(
                observation_count=cast(int, facts["observation_count"]),
                usable_observation_count=cast(int, facts["usable_observation_count"]),
                candidate_region_count=cast(int, facts["candidate_region_count"]),
                persistence_count=cast(int, facts["persistence_count"]),
                persistence_confidence=cast(float, facts["persistence_confidence"]),
                reason_codes=tuple(
                    ReasonCode(item) for item in cast(list[str], facts["reason_codes"])
                ),
                budget_exhausted=cast(bool, facts["budget_exhausted"]),
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """The current persisted state after a complete or intentionally paused run."""

    trace_id: str
    state: AgentLoopState
    trace_path: Path
    evidence_ids: tuple[str, ...]
    budget: Budget
    outcome: CalibratedOutcome | None

    @property
    def is_terminal(self) -> bool:
        """Whether the journal contains a safe terminal state."""

        return self.state.is_terminal


class BoundedAgentLoop:
    """Join bounded perception, policy, action, and terminal calibration."""

    def __init__(
        self,
        tools: BoundedObservationTools,
        trace_path: Path,
        *,
        policy: TransparentAgentPolicy | None = None,
        trace_id: str | None = None,
        checkpoint: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(tools, BoundedObservationTools):
            raise TypeError("tools must be BoundedObservationTools")
        if policy is not None and not isinstance(policy, TransparentAgentPolicy):
            raise TypeError("policy must be TransparentAgentPolicy or None")
        self._tools = tools
        self._trace_path = Path(trace_path)
        self._policy = TransparentAgentPolicy() if policy is None else policy
        self._trace_id = (
            f"{tools.case.case_id}-agent-loop" if trace_id is None else trace_id
        )
        if not _IDENTIFIER.fullmatch(self._trace_id):
            raise ValueError("trace_id must be a safe identifier")
        self._state = AgentLoopState.OBSERVE
        self._sequence = 0
        self._analysis: EvidenceAnalysis | None = None
        self._decision: PolicyDecision | None = None
        self._pending_action: PolicyAction | None = None
        self._last_tool_result: ToolResult | None = None
        self._outcome: CalibratedOutcome | None = None
        self._failure_reason: ReasonCode | None = None
        if checkpoint is None:
            if self._trace_path.exists() and self._trace_path.stat().st_size:
                raise ValueError("trace_path already contains a checkpoint; use open")
            self._transition(AgentLoopState.OBSERVE, "started")
        else:
            self._restore(checkpoint)

    @classmethod
    def open(
        cls,
        manifest: ToolManifest,
        *,
        source_cache_root: Path,
        artifacts_root: Path,
        project_root: Path,
        trace_path: Path,
        maximum_bytes: int,
        maximum_elapsed_seconds: float,
        maximum_observations: int = 3,
        maximum_retries: int = 1,
        policy: TransparentAgentPolicy | None = None,
        trace_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> Self:
        """Create or resume a loop using the last complete JSONL checkpoint."""

        if not isinstance(manifest, ToolManifest):
            raise TypeError("manifest must be ToolManifest")
        trace = Path(trace_path)
        checkpoint = load_last_complete_transition(trace)
        if checkpoint is None and trace.exists() and trace.stat().st_size:
            _discard_torn_final_line(trace)
        selected_ids: tuple[str, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        initial_used_retries = 0
        initial_elapsed_seconds = 0.0
        if checkpoint is not None:
            _validate_checkpoint(checkpoint)
            if checkpoint["case_id"] != manifest.case.case_id:
                raise ValueError("checkpoint case_id does not match tool manifest")
            stored_trace_id = cast(str, checkpoint["trace_id"])
            if trace_id is not None and trace_id != stored_trace_id:
                raise ValueError("trace_id does not match checkpoint")
            trace_id = stored_trace_id
            selected_ids = tuple(
                cast(list[str], checkpoint["selected_observation_ids"])
            )
            evidence_ids = tuple(cast(list[str], checkpoint["evidence_ids"]))
            budget = Budget.from_dict(checkpoint["budget"])
            if (
                budget.max_observations != maximum_observations
                or budget.max_bytes != maximum_bytes
                or budget.max_elapsed_seconds != maximum_elapsed_seconds
                or budget.max_retries != maximum_retries
            ):
                raise ValueError("resume limits must match the persisted budget")
            initial_used_retries = budget.used_retries
            initial_elapsed_seconds = budget.used_elapsed_seconds
        tools = BoundedObservationTools(
            manifest,
            source_cache_root=source_cache_root,
            artifacts_root=artifacts_root,
            project_root=project_root,
            maximum_bytes=maximum_bytes,
            maximum_elapsed_seconds=maximum_elapsed_seconds,
            maximum_observations=maximum_observations,
            maximum_retries=maximum_retries,
            initial_selected_observation_ids=selected_ids,
            initial_evidence_ids=evidence_ids,
            initial_used_retries=initial_used_retries,
            initial_elapsed_seconds=initial_elapsed_seconds,
            clock=clock,
        )
        return cls(
            tools,
            trace,
            policy=policy,
            trace_id=trace_id,
            checkpoint=checkpoint,
        )

    @property
    def state(self) -> AgentLoopState:
        """Return the current, already persisted loop state."""

        return self._state

    def run(self, *, transition_limit: int | None = None) -> AgentLoopResult:
        """Run until terminal, or pause after a requested persisted transition count."""

        if transition_limit is not None and (
            isinstance(transition_limit, bool)
            or not isinstance(transition_limit, int)
            or transition_limit < 0
        ):
            raise ValueError("transition_limit must be a non-negative integer or None")
        transitions = 0
        maximum_transitions = 8 + 4 * (
            self._tools.budget.max_observations + self._tools.budget.max_retries
        )
        while not self._state.is_terminal:
            if transition_limit is not None and transitions >= transition_limit:
                return self._result()
            if transitions >= maximum_transitions:
                self._fail(ReasonCode.CONFIGURATION_INVALID, "transition guard reached")
            elif self._state is AgentLoopState.OBSERVE:
                self._observe()
            elif self._state is AgentLoopState.ANALYZE:
                self._analyze()
            elif self._state is AgentLoopState.DECIDE:
                self._decide()
            elif self._state is AgentLoopState.ACT:
                self._act()
            else:
                raise AssertionError(f"unhandled loop state {self._state!r}")
            transitions += 1
        return self._result()

    def _observe(self) -> None:
        if self._tools.evidence_ids:
            self._transition(AgentLoopState.ANALYZE, "evidence_available")
        else:
            self._transition(AgentLoopState.DECIDE, "no_evidence_yet")

    def _analyze(self) -> None:
        try:
            evidence_id = self._tools.evidence_ids[-1]
            path = (
                self._tools.evidence_artifact_directory(evidence_id) / "evidence.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("completed evidence packet must be an object")
            self._analysis = EvidenceAnalysis(
                evidence_id=evidence_id,
                snapshot=EvidenceSnapshot.from_local_evidence(
                    payload, evidence_id=evidence_id
                ),
                outcome_evidence=OutcomeEvidence.from_local_evidence(payload),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            self._fail(
                ReasonCode.SOURCE_CORRUPT, "completed evidence packet is unreadable"
            )
            return
        self._transition(AgentLoopState.DECIDE, "evidence_analyzed")

    def _decide(self) -> None:
        evidence = (
            _empty_evidence_snapshot()
            if self._analysis is None
            else self._analysis.snapshot
        )
        available = tuple(
            PolicyAction(item.action_type, item.observation_id)
            for item in self._tools.available_observations
        )
        self._decision = self._policy.decide(
            evidence,
            self._tools.budget,
            available,
            last_tool_result=self._last_tool_result,
        )
        self._pending_action = self._decision.selected_action
        self._transition(AgentLoopState.ACT, "action_selected")

    def _act(self) -> None:
        if self._pending_action is None:
            self._fail(
                ReasonCode.CONFIGURATION_INVALID, "act state has no pending action"
            )
            return
        is_recovery_retry = (
            self._decision is not None
            and self._decision.rule is PolicyRule.POOR_QUALITY_RETRY
            and self._pending_action.action_type in _OBSERVATION_ACTIONS
        )
        if is_recovery_retry and not self._tools.consume_recovery_retry():
            self._pending_action = None
            self._transition(AgentLoopState.DECIDE, "recovery_retry_exhausted")
            return
        try:
            result = (
                _apply_pending_action(self._tools, self._pending_action)
                if self._decision is None
                else apply_policy_decision(self._tools, self._decision)
            )
        except (OSError, ValueError, TypeError):
            self._fail(ReasonCode.CONFIGURATION_INVALID, "bounded action could not run")
            return
        self._last_tool_result = result
        self._pending_action = None
        if not result.accepted:
            self._transition(
                AgentLoopState.DECIDE,
                "recovery_retry_rejected"
                if is_recovery_retry
                else "bounded_action_rejected",
            )
            return
        if result.action_type in _OBSERVATION_ACTIONS:
            self._transition(
                AgentLoopState.OBSERVE,
                "recovery_retry_completed"
                if is_recovery_retry
                else "observation_completed",
            )
            return
        self._finish_terminal_action(result.action_type)

    def _finish_terminal_action(self, action_type: ActionType) -> None:
        facts = self._analysis.outcome_evidence if self._analysis is not None else None
        if action_type is ActionType.ABSTAIN:
            reasons = () if facts is None else facts.reason_codes
            if (
                self._tools.budget.used_observations
                >= self._tools.budget.max_observations
            ):
                reasons = _unique_reasons((*reasons, ReasonCode.BUDGET_EXHAUSTED))
            self._outcome = CalibratedOutcome(
                OutcomeState.INSUFFICIENT_EVIDENCE,
                _unique_reasons((*reasons, ReasonCode.INSUFFICIENT_EVIDENCE)),
                0.0,
            )
        elif facts is None:
            self._outcome = CalibratedOutcome(
                OutcomeState.INSUFFICIENT_EVIDENCE,
                (ReasonCode.INSUFFICIENT_EVIDENCE,),
                0.0,
            )
        else:
            self._outcome = calibrate_outcome(facts)

        if (
            action_type is ActionType.REQUEST_HUMAN_REVIEW
            and self._outcome.state
            not in {
                OutcomeState.REVIEW_ESCALATION,
                OutcomeState.HUMAN_REVIEW,
            }
        ):
            self._outcome = CalibratedOutcome(
                OutcomeState.HUMAN_REVIEW,
                _unique_reasons(
                    (*self._outcome.reason_codes, ReasonCode.HUMAN_REVIEW_REQUIRED)
                ),
                self._outcome.confidence,
            )
        target = _terminal_state_for_outcome(self._outcome.state)
        self._transition(target, f"{action_type.value}_completed")

    def _fail(self, reason: ReasonCode, detail: str) -> None:
        del detail
        self._failure_reason = reason
        self._outcome = CalibratedOutcome(OutcomeState.FAILED, (reason,), 0.0)
        self._pending_action = None
        self._transition(AgentLoopState.FAILURE, "failure")

    def _transition(self, target: AgentLoopState, event: str) -> None:
        if not isinstance(target, AgentLoopState):
            raise TypeError("target must be AgentLoopState")
        if not isinstance(event, str) or not event:
            raise ValueError("event must be non-empty")
        previous = self._state
        self._state = target
        self._sequence += 1
        payload = {
            "schema_version": LOOP_TRACE_SCHEMA_VERSION,
            "record_type": LOOP_TRACE_RECORD_TYPE,
            "trace_id": self._trace_id,
            "case_id": self._tools.case.case_id,
            "sequence": self._sequence,
            "from_state": None if self._sequence == 1 else previous.value,
            "to_state": target.value,
            "event": event,
            "selected_observation_ids": list(self._tools.selected_observation_ids),
            "evidence_ids": list(self._tools.evidence_ids),
            "budget": self._tools.budget.to_dict(),
            "analysis": None if self._analysis is None else self._analysis.to_dict(),
            "decision": None if self._decision is None else self._decision.to_dict(),
            "pending_action": (
                None if self._pending_action is None else self._pending_action.to_dict()
            ),
            "last_tool_result": (
                None
                if self._last_tool_result is None
                else self._last_tool_result.to_dict()
            ),
            "outcome": None if self._outcome is None else self._outcome.to_dict(),
            "failure_reason": (
                None if self._failure_reason is None else self._failure_reason.value
            ),
        }
        _append_transition(self._trace_path, payload)

    def _restore(self, checkpoint: Mapping[str, object]) -> None:
        _validate_checkpoint(checkpoint)
        if checkpoint["trace_id"] != self._trace_id:
            raise ValueError("checkpoint trace_id does not match loop")
        if checkpoint["case_id"] != self._tools.case.case_id:
            raise ValueError("checkpoint case_id does not match loop tools")
        self._sequence = cast(int, checkpoint["sequence"])
        self._state = AgentLoopState(cast(str, checkpoint["to_state"]))
        raw_analysis = checkpoint["analysis"]
        raw_pending = checkpoint["pending_action"]
        raw_result = checkpoint["last_tool_result"]
        raw_outcome = checkpoint["outcome"]
        raw_reason = checkpoint["failure_reason"]
        self._analysis = (
            None if raw_analysis is None else EvidenceAnalysis.from_dict(raw_analysis)
        )
        self._pending_action = (
            None if raw_pending is None else _policy_action_from_dict(raw_pending)
        )
        self._last_tool_result = (
            None if raw_result is None else _tool_result_from_dict(raw_result)
        )
        self._outcome = None if raw_outcome is None else _outcome_from_dict(raw_outcome)
        self._failure_reason = (
            None if raw_reason is None else ReasonCode(cast(str, raw_reason))
        )
        if self._state is AgentLoopState.ACT and self._pending_action is None:
            raise ValueError("act checkpoint must contain a pending_action")
        if self._state.is_terminal and self._outcome is None:
            raise ValueError("terminal checkpoint must contain an outcome")

    def _result(self) -> AgentLoopResult:
        return AgentLoopResult(
            trace_id=self._trace_id,
            state=self._state,
            trace_path=self._trace_path,
            evidence_ids=self._tools.evidence_ids,
            budget=self._tools.budget,
            outcome=self._outcome,
        )


def load_last_complete_transition(path: Path) -> dict[str, object] | None:
    """Return the final complete journal record, ignoring only a torn last line."""

    trace_path = Path(path)
    if not trace_path.exists():
        return None
    try:
        lines = [
            line for line in trace_path.read_text(encoding="utf-8").splitlines() if line
        ]
    except OSError as error:
        raise ValueError(f"cannot read loop trace: {trace_path}") from error
    latest: dict[str, object] | None = None
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                break
            raise ValueError("loop trace has a corrupt non-final transition") from error
        if not isinstance(value, dict):
            raise ValueError("loop trace transition must be an object")
        _validate_checkpoint(value)
        if latest is not None and (
            value["trace_id"] != latest["trace_id"]
            or value["case_id"] != latest["case_id"]
            or value["sequence"] != cast(int, latest["sequence"]) + 1
        ):
            raise ValueError("loop trace transitions are not contiguous")
        latest = value
    return latest


def _append_transition(path: Path, payload: Mapping[str, object]) -> None:
    _validate_checkpoint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _discard_torn_final_line(destination)
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _discard_torn_final_line(path: Path) -> None:
    """Remove only an unterminated final write before appending a checkpoint."""

    if not path.exists() or path.stat().st_size == 0:
        return
    contents = path.read_bytes()
    if contents.endswith(b"\n"):
        return
    final_newline = contents.rfind(b"\n")
    with path.open("r+b") as handle:
        handle.truncate(final_newline + 1)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_checkpoint(value: Mapping[str, object]) -> None:
    if set(value) != _CHECKPOINT_KEYS:
        raise ValueError("loop trace checkpoint has an invalid shape")
    if (
        value["schema_version"] != LOOP_TRACE_SCHEMA_VERSION
        or value["record_type"] != LOOP_TRACE_RECORD_TYPE
    ):
        raise ValueError("loop trace checkpoint has an unsupported schema")
    for field in ("trace_id", "case_id"):
        item = value[field]
        if not isinstance(item, str) or not _IDENTIFIER.fullmatch(item):
            raise ValueError(f"loop trace {field} is invalid")
    sequence = value["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("loop trace sequence is invalid")
    from_state = value["from_state"]
    if sequence == 1 and from_state is not None:
        raise ValueError("first loop transition must not have from_state")
    if sequence > 1 and from_state is None:
        raise ValueError("later loop transitions require from_state")
    if from_state is not None:
        AgentLoopState(cast(str, from_state))
    AgentLoopState(cast(str, value["to_state"]))
    if not isinstance(value["event"], str) or not value["event"]:
        raise ValueError("loop trace event is invalid")
    for field in ("selected_observation_ids", "evidence_ids"):
        items = value[field]
        if not isinstance(items, list) or not all(
            isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in items
        ):
            raise ValueError(f"loop trace {field} is invalid")
        if len(items) != len(set(items)):
            raise ValueError(f"loop trace {field} repeats an identifier")
    Budget.from_dict(value["budget"])
    for field in (
        "analysis",
        "decision",
        "pending_action",
        "last_tool_result",
        "outcome",
    ):
        if value[field] is not None and not isinstance(value[field], Mapping):
            raise ValueError(f"loop trace {field} is invalid")
    if value["failure_reason"] is not None:
        ReasonCode(cast(str, value["failure_reason"]))


def _empty_evidence_snapshot() -> EvidenceSnapshot:
    return EvidenceSnapshot((), (), 0, 0, 0.0)


def _policy_action_from_dict(value: object) -> PolicyAction:
    if not isinstance(value, Mapping) or set(value) != {
        "action_type",
        "observation_id",
    }:
        raise ValueError("pending_action is invalid")
    return PolicyAction(
        ActionType(cast(str, value["action_type"])),
        cast(str | None, value["observation_id"]),
    )


def _apply_pending_action(
    tools: BoundedObservationTools, action: PolicyAction
) -> ToolResult:
    """Dispatch a persisted pending action without recreating a policy decision."""

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


def _tool_result_from_dict(value: object) -> ToolResult:
    if not isinstance(value, Mapping):
        raise ValueError("last_tool_result is invalid")
    required = {
        "action_type",
        "observation_id",
        "accepted",
        "idempotent",
        "evidence_ids",
        "resource_use",
        "terminal_action",
        "error",
    }
    if set(value) != required:
        raise ValueError("last_tool_result has an invalid shape")
    raw_error = value["error"]
    error: ToolError | None
    if raw_error is None:
        error = None
    elif isinstance(raw_error, Mapping) and frozenset(raw_error) in {
        frozenset(("code", "reason_code", "detail")),
        frozenset(("code", "reason_code", "detail", "recovery_action")),
    }:
        error = ToolError(
            ToolErrorCode(cast(str, raw_error["code"])),
            ReasonCode(cast(str, raw_error["reason_code"])),
            cast(str, raw_error["detail"]),
        )
        recovery_action = raw_error.get("recovery_action")
        if recovery_action is not None and recovery_action != error.recovery_action:
            raise ValueError("last_tool_result recovery_action is invalid")
    else:
        raise ValueError("last_tool_result error is invalid")
    return ToolResult(
        action_type=ActionType(cast(str, value["action_type"])),
        observation_id=cast(str | None, value["observation_id"]),
        accepted=cast(bool, value["accepted"]),
        idempotent=cast(bool, value["idempotent"]),
        evidence_ids=tuple(cast(list[str], value["evidence_ids"])),
        budget=Budget.from_dict(value["resource_use"]),
        terminal_action=(
            None
            if value["terminal_action"] is None
            else ActionType(cast(str, value["terminal_action"]))
        ),
        error=error,
    )


def _outcome_from_dict(value: object) -> CalibratedOutcome:
    if not isinstance(value, Mapping) or set(value) != {
        "state",
        "reason_codes",
        "confidence",
        "explanation",
    }:
        raise ValueError("outcome is invalid")
    outcome = CalibratedOutcome(
        OutcomeState(cast(str, value["state"])),
        tuple(ReasonCode(item) for item in cast(list[str], value["reason_codes"])),
        cast(float, value["confidence"]),
    )
    if value["explanation"] != outcome.explanation:
        raise ValueError("outcome explanation does not match its fixed templates")
    return outcome


def _terminal_state_for_outcome(outcome: OutcomeState) -> AgentLoopState:
    if outcome in {OutcomeState.REVIEW_ESCALATION, OutcomeState.HUMAN_REVIEW}:
        return AgentLoopState.REVIEW
    if outcome is OutcomeState.NO_PERSISTENT_EVIDENCE:
        return AgentLoopState.FINALIZE
    if outcome is OutcomeState.INSUFFICIENT_EVIDENCE:
        return AgentLoopState.ABSTAIN
    return AgentLoopState.FAILURE


def _unique_reasons(reasons: tuple[ReasonCode, ...]) -> tuple[ReasonCode, ...]:
    return tuple(dict.fromkeys(reasons))


def main(argv: list[str] | None = None) -> int:
    """Run or resume one manifest-bounded local investigation."""

    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--source-cache", type=Path, default=settings.source_cache_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--maximum-bytes", type=int, required=True)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=120.0)
    parser.add_argument("--maximum-observations", type=int, default=3)
    parser.add_argument("--maximum-retries", type=int, default=1)
    arguments = parser.parse_args(argv)
    manifest = load_tool_manifest(
        arguments.tool_manifest, project_root=settings.root_dir
    )
    trace_path = arguments.trace
    if trace_path is None:
        trace_path = (
            Path(arguments.artifacts_dir) / manifest.case.case_id / "agent-loop.jsonl"
        )
    loop = BoundedAgentLoop.open(
        manifest,
        source_cache_root=arguments.source_cache,
        artifacts_root=arguments.artifacts_dir,
        project_root=settings.root_dir,
        trace_path=trace_path,
        maximum_bytes=arguments.maximum_bytes,
        maximum_elapsed_seconds=arguments.maximum_elapsed_seconds,
        maximum_observations=arguments.maximum_observations,
        maximum_retries=arguments.maximum_retries,
    )
    result = loop.run()
    print(
        json.dumps(
            {
                "trace_id": result.trace_id,
                "state": result.state.value,
                "terminal": result.is_terminal,
                "trace_path": str(result.trace_path),
                "evidence_ids": list(result.evidence_ids),
                "budget": result.budget.to_dict(),
                "outcome": None if result.outcome is None else result.outcome.to_dict(),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 1 if result.state is AgentLoopState.FAILURE else 0


__all__ = [
    "AgentLoopResult",
    "AgentLoopState",
    "BoundedAgentLoop",
    "EvidenceAnalysis",
    "LOOP_TRACE_RECORD_TYPE",
    "LOOP_TRACE_SCHEMA_VERSION",
    "load_last_complete_transition",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
