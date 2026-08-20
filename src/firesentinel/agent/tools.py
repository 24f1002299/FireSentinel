"""Bounded, manifest-backed observation tools for the local agent.

Tool callers can name only an allowlisted observation identifier.  They never
provide a URL, filesystem path, band, or processing parameter.  Every accepted
request replays the cumulative Day 17 C07/C14 evidence job from hash-verified
cache files, so an additional observation materially changes the OpenCV
anomaly and persistence inputs available to a later policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from firesentinel.agent.label_boundary import runtime_input_path
from firesentinel.core.records import (
    ActionType,
    Budget,
    Channel,
    ManifestCase,
    ReasonCode,
)
from firesentinel.vision.engine import (
    EvidenceJob,
    EvidenceJobFailure,
    EvidenceJobObservation,
    EvidenceJobSource,
    run_evidence_job,
)

TOOL_MANIFEST_SCHEMA_VERSION = 1
MAXIMUM_OBSERVATIONS = 3
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
_EXPECTED_CHANNELS = {
    ActionType.NEXT_TIMESTAMP: Channel.C07,
    ActionType.ALTERNATE_BAND: Channel.C14,
    ActionType.PRE_EVENT_BASELINE: Channel.C07,
}


class ToolErrorCode(StrEnum):
    """Stable machine-readable categories for rejected bounded-tool calls."""

    OBSERVATION_NOT_ALLOWED = "observation_not_allowed"
    ACTION_NOT_ALLOWED = "action_not_allowed"
    OBSERVATION_BUDGET_EXHAUSTED = "observation_budget_exhausted"
    BYTE_BUDGET_EXHAUSTED = "byte_budget_exhausted"
    ELAPSED_TIME_EXHAUSTED = "elapsed_time_exhausted"
    TERMINAL = "terminal"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CORRUPT = "source_corrupt"
    EVIDENCE_FAILED = "evidence_failed"


@dataclass(frozen=True, slots=True)
class ToolError:
    """A structured, non-throwing request failure returned to a policy."""

    code: ToolErrorCode
    reason_code: ReasonCode
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "reason_code": self.reason_code.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ToolSource:
    """One immutable, locally cached source that may feed an observation."""

    source_id: str
    catalog_key: str
    source_path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _IDENTIFIER.fullmatch(
            self.source_id
        ):
            raise ValueError("tool source_id must be a lowercase identifier")
        if not isinstance(self.catalog_key, str) or not self.catalog_key:
            raise ValueError("tool catalog_key must be non-empty")
        object.__setattr__(self, "source_path", Path(self.source_path))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("tool source size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("tool source size_bytes must be positive")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("tool source sha256 must be a lowercase digest")

    def to_dict(self, *, include_path: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "source_id": self.source_id,
            "catalog_key": self.catalog_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if include_path:
            result["source_path"] = str(self.source_path)
        return result


@dataclass(frozen=True, slots=True)
class AllowedObservation:
    """A single permitted action and its complete C07/C14 evidence pairing."""

    observation_id: str
    action_type: ActionType
    observation_time: datetime
    requested_channel: Channel
    channel7: ToolSource
    channel14: ToolSource

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not _IDENTIFIER.fullmatch(
            self.observation_id
        ):
            raise ValueError("tool observation_id must be a lowercase identifier")
        if self.action_type not in _OBSERVATION_ACTIONS:
            raise ValueError("tool observation action_type is unsupported")
        if not isinstance(self.observation_time, datetime):
            raise ValueError("tool observation_time must be a UTC datetime")
        if (
            self.observation_time.tzinfo is None
            or self.observation_time.utcoffset() != UTC.utcoffset(self.observation_time)
        ):
            raise ValueError("tool observation_time must be UTC")
        object.__setattr__(
            self, "observation_time", self.observation_time.astimezone(UTC)
        )
        if self.requested_channel != _EXPECTED_CHANNELS[self.action_type]:
            raise ValueError("tool requested_channel does not match action_type")
        if not isinstance(self.channel7, ToolSource) or not isinstance(
            self.channel14, ToolSource
        ):
            raise TypeError("tool channel sources must be ToolSource values")

    def to_dict(self, *, include_paths: bool) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "action_type": self.action_type.value,
            "observation_time_utc": _timestamp(self.observation_time),
            "requested_channel": self.requested_channel.value,
            "channel7": self.channel7.to_dict(include_path=include_paths),
            "channel14": self.channel14.to_dict(include_path=include_paths),
        }


@dataclass(frozen=True, slots=True)
class ToolManifest:
    """Complete static scope for one agent session and no other files."""

    case: ManifestCase
    evidence_template: EvidenceJob
    observations: tuple[AllowedObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, ManifestCase):
            raise TypeError("tool manifest case must be ManifestCase")
        if not isinstance(self.evidence_template, EvidenceJob):
            raise TypeError("tool manifest evidence_template must be EvidenceJob")
        if not self.observations or not all(
            isinstance(item, AllowedObservation) for item in self.observations
        ):
            raise ValueError("tool manifest requires allowed observations")
        identifiers = tuple(item.observation_id for item in self.observations)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("tool manifest repeats observation_id")
        if set(identifiers) != set(self.case.allowed_observation_ids):
            raise ValueError(
                "tool manifest observations must exactly match the case allowlist"
            )
        by_source_id: dict[str, ToolSource] = {}
        for observation in self.observations:
            for source in (observation.channel7, observation.channel14):
                existing = by_source_id.setdefault(source.source_id, source)
                if existing != source:
                    raise ValueError("tool source_id must name one immutable object")

    @property
    def observations_by_id(self) -> dict[str, AllowedObservation]:
        return {item.observation_id: item for item in self.observations}

    def to_dict(self, *, include_paths: bool) -> dict[str, object]:
        return {
            "schema_version": TOOL_MANIFEST_SCHEMA_VERSION,
            "record_type": "bounded_observation_tool_manifest",
            "case": self.case.to_dict(),
            "evidence_template": self.evidence_template.to_dict(
                include_paths=include_paths
            ),
            "observations": [
                item.to_dict(include_paths=include_paths) for item in self.observations
            ],
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool reply with cumulative evidence IDs, resources, and any error."""

    action_type: ActionType
    observation_id: str | None
    accepted: bool
    idempotent: bool
    evidence_ids: tuple[str, ...]
    budget: Budget
    terminal_action: ActionType | None
    error: ToolError | None = None

    def __post_init__(self) -> None:
        if self.action_type not in _OBSERVATION_ACTIONS | _TERMINAL_ACTIONS:
            raise ValueError("tool result action_type is unsupported")
        if self.observation_id is not None and (
            not isinstance(self.observation_id, str)
            or not _IDENTIFIER.fullmatch(self.observation_id)
        ):
            raise ValueError("tool result observation_id is invalid")
        if not isinstance(self.accepted, bool) or not isinstance(self.idempotent, bool):
            raise ValueError("tool result acceptance flags must be booleans")
        if not isinstance(self.budget, Budget):
            raise TypeError("tool result budget must be Budget")
        if (
            self.terminal_action is not None
            and self.terminal_action not in _TERMINAL_ACTIONS
        ):
            raise ValueError("tool result terminal_action is invalid")
        if self.error is not None and not isinstance(self.error, ToolError):
            raise TypeError("tool result error must be ToolError or None")
        if self.accepted == (self.error is not None):
            raise ValueError("accepted tool results must agree with their error")

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type.value,
            "observation_id": self.observation_id,
            "accepted": self.accepted,
            "idempotent": self.idempotent,
            "evidence_ids": list(self.evidence_ids),
            "resource_use": self.budget.to_dict(),
            "terminal_action": (
                None if self.terminal_action is None else self.terminal_action.value
            ),
            "error": None if self.error is None else self.error.to_dict(),
        }


class BoundedObservationTools:
    """A small, stateful action surface with no arbitrary-file capability."""

    def __init__(
        self,
        manifest: ToolManifest,
        *,
        source_cache_root: Path,
        artifacts_root: Path,
        project_root: Path,
        maximum_bytes: int,
        maximum_elapsed_seconds: float,
        maximum_observations: int = MAXIMUM_OBSERVATIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(manifest, ToolManifest):
            raise TypeError("manifest must be ToolManifest")
        if isinstance(maximum_observations, bool) or not isinstance(
            maximum_observations, int
        ):
            raise ValueError("maximum_observations must be an integer")
        if not 1 <= maximum_observations <= MAXIMUM_OBSERVATIONS:
            raise ValueError(
                f"maximum_observations must be within [1, {MAXIMUM_OBSERVATIONS}]"
            )
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
            raise ValueError("maximum_bytes must be an integer")
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes must be non-negative")
        if isinstance(maximum_elapsed_seconds, bool) or not isinstance(
            maximum_elapsed_seconds, (int, float)
        ):
            raise ValueError("maximum_elapsed_seconds must be a finite positive number")
        if (
            not math.isfinite(float(maximum_elapsed_seconds))
            or maximum_elapsed_seconds <= 0
        ):
            raise ValueError("maximum_elapsed_seconds must be a finite positive number")
        self._manifest = manifest
        self._project_root = Path(project_root).resolve()
        self._source_cache_root = Path(source_cache_root).resolve()
        self._artifacts_root = Path(artifacts_root).resolve()
        self._labels_root = (self._project_root / "evaluation-data").resolve()
        self._assert_non_label_path(self._source_cache_root, "source_cache_root")
        self._assert_non_label_path(self._artifacts_root, "artifacts_root")
        self._maximum_observations = maximum_observations
        self._maximum_bytes = maximum_bytes
        self._maximum_elapsed_seconds = float(maximum_elapsed_seconds)
        self._clock = clock
        self._started = self._clock()
        self._selected_ids: list[str] = []
        self._evidence_ids: list[str] = []
        self._used_source_ids: set[str] = set()
        self._terminal_action: ActionType | None = None
        self._terminal_results: dict[ActionType, ToolResult] = {}
        self._validate_source_scope()

    @property
    def case(self) -> ManifestCase:
        """Return the sole case visible to this session."""

        return self._manifest.case

    def next_timestamp(self, observation_id: str) -> ToolResult:
        """Request one manifest-pinned later C07 observation."""

        return self._observe(ActionType.NEXT_TIMESTAMP, observation_id)

    def alternate_band(self, observation_id: str) -> ToolResult:
        """Request one manifest-pinned C14 contextual observation."""

        return self._observe(ActionType.ALTERNATE_BAND, observation_id)

    def pre_event_baseline(self, observation_id: str) -> ToolResult:
        """Request one manifest-pinned pre-event C07 baseline."""

        return self._observe(ActionType.PRE_EVENT_BASELINE, observation_id)

    def finalize(self) -> ToolResult:
        """Safely stop requesting observations without asserting an outcome."""

        return self._terminal(ActionType.FINALIZE)

    def abstain(self) -> ToolResult:
        """Safely stop when available evidence is insufficient."""

        return self._terminal(ActionType.ABSTAIN)

    def request_human_review(self) -> ToolResult:
        """Safely stop and escalate the accumulated evidence for review."""

        return self._terminal(ActionType.REQUEST_HUMAN_REVIEW)

    def human_review(self) -> ToolResult:
        """Alias for :meth:`request_human_review` used by simple tool callers."""

        return self.request_human_review()

    def _observe(self, action_type: ActionType, observation_id: str) -> ToolResult:
        if not isinstance(observation_id, str) or not _IDENTIFIER.fullmatch(
            observation_id
        ):
            return self._rejected(
                action_type,
                None,
                ToolError(
                    ToolErrorCode.OBSERVATION_NOT_ALLOWED,
                    ReasonCode.OBSERVATION_NOT_ALLOWED,
                    "observation_id must be a manifest identifier",
                ),
            )
        definition = self._manifest.observations_by_id.get(observation_id)
        if definition is None:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.OBSERVATION_NOT_ALLOWED,
                    ReasonCode.OBSERVATION_NOT_ALLOWED,
                    "observation_id is not allowlisted for this case",
                ),
            )
        if definition.action_type != action_type:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.ACTION_NOT_ALLOWED,
                    ReasonCode.OBSERVATION_NOT_ALLOWED,
                    "observation_id is not available through this tool",
                ),
            )
        if observation_id in self._selected_ids:
            return self._accepted(action_type, observation_id, idempotent=True)
        if self._terminal_action is not None:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.TERMINAL,
                    ReasonCode.INSUFFICIENT_EVIDENCE,
                    f"session terminated by {self._terminal_action.value}",
                ),
            )
        if len(self._selected_ids) >= self._maximum_observations:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.OBSERVATION_BUDGET_EXHAUSTED,
                    ReasonCode.BUDGET_EXHAUSTED,
                    "maximum observation count has been reached",
                ),
            )
        if self._elapsed_seconds() >= self._maximum_elapsed_seconds:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.ELAPSED_TIME_EXHAUSTED,
                    ReasonCode.TIMEOUT,
                    "elapsed-time limit has been reached",
                ),
            )
        new_sources = self._new_sources(definition)
        projected_bytes = self._used_bytes() + sum(
            source.size_bytes for source in new_sources
        )
        if projected_bytes > self._maximum_bytes:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.BYTE_BUDGET_EXHAUSTED,
                    ReasonCode.BUDGET_EXHAUSTED,
                    "request would exceed the byte budget",
                ),
            )
        try:
            self._verify_sources(
                self._sources_for_observations((*self._selected_ids, observation_id))
            )
            remaining = self._maximum_elapsed_seconds - self._elapsed_seconds()
            if remaining <= 0.0:
                raise _ElapsedLimitError
            job = self._evidence_job((*self._selected_ids, observation_id))
            result = run_evidence_job(
                job,
                self._artifacts_root,
                timeout_seconds=remaining,
                clock=self._clock,
            )
            if self._elapsed_seconds() > self._maximum_elapsed_seconds:
                raise _ElapsedLimitError
        except _ElapsedLimitError:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.ELAPSED_TIME_EXHAUSTED,
                    ReasonCode.TIMEOUT,
                    "elapsed-time limit was exhausted while processing evidence",
                ),
            )
        except EvidenceJobFailure as error:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    (
                        ToolErrorCode.ELAPSED_TIME_EXHAUSTED
                        if error.reason_code is ReasonCode.TIMEOUT
                        else ToolErrorCode.EVIDENCE_FAILED
                    ),
                    error.reason_code,
                    error.detail,
                ),
            )
        except FileNotFoundError as error:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.SOURCE_UNAVAILABLE,
                    ReasonCode.SOURCE_MISSING,
                    str(error),
                ),
            )
        except OSError as error:
            return self._rejected(
                action_type,
                observation_id,
                ToolError(
                    ToolErrorCode.SOURCE_CORRUPT, ReasonCode.SOURCE_CORRUPT, str(error)
                ),
            )
        self._selected_ids.append(observation_id)
        self._evidence_ids.append(result.content_hash)
        self._used_source_ids.update(source.source_id for source in new_sources)
        return self._accepted(action_type, observation_id, idempotent=False)

    def _terminal(self, action_type: ActionType) -> ToolResult:
        existing = self._terminal_results.get(action_type)
        if existing is not None:
            return ToolResult(
                action_type=existing.action_type,
                observation_id=None,
                accepted=True,
                idempotent=True,
                evidence_ids=tuple(self._evidence_ids),
                budget=self._budget(),
                terminal_action=existing.terminal_action,
            )
        if self._terminal_action is not None:
            return self._rejected(
                action_type,
                None,
                ToolError(
                    ToolErrorCode.TERMINAL,
                    ReasonCode.INSUFFICIENT_EVIDENCE,
                    f"session already terminated by {self._terminal_action.value}",
                ),
            )
        self._terminal_action = action_type
        result = self._accepted(action_type, None, idempotent=False)
        self._terminal_results[action_type] = result
        return result

    def _accepted(
        self, action_type: ActionType, observation_id: str | None, *, idempotent: bool
    ) -> ToolResult:
        return ToolResult(
            action_type=action_type,
            observation_id=observation_id,
            accepted=True,
            idempotent=idempotent,
            evidence_ids=tuple(self._evidence_ids),
            budget=self._budget(),
            terminal_action=self._terminal_action,
        )

    def _rejected(
        self, action_type: ActionType, observation_id: str | None, error: ToolError
    ) -> ToolResult:
        return ToolResult(
            action_type=action_type,
            observation_id=observation_id,
            accepted=False,
            idempotent=False,
            evidence_ids=tuple(self._evidence_ids),
            budget=self._budget(),
            terminal_action=self._terminal_action,
            error=error,
        )

    def _evidence_job(self, observation_ids: tuple[str, ...]) -> EvidenceJob:
        definitions = self._manifest.observations_by_id
        selected = sorted(
            (definitions[identifier] for identifier in observation_ids),
            key=lambda item: (item.observation_time, item.observation_id),
        )
        template = self._manifest.evidence_template
        return EvidenceJob(
            case_id=self._manifest.case.case_id,
            crop_parameters=template.crop_parameters,
            tile_parameters=template.tile_parameters,
            observations=tuple(
                EvidenceJobObservation(
                    item.observation_id,
                    EvidenceJobSource(
                        item.channel7.catalog_key,
                        self._validated_source_path(item.channel7),
                    ),
                    EvidenceJobSource(
                        item.channel14.catalog_key,
                        self._validated_source_path(item.channel14),
                    ),
                )
                for item in selected
            ),
            allow_single_observation=len(selected) == 1,
            anomaly_parameters=template.anomaly_parameters,
            quality_thresholds=template.quality_thresholds,
            persistence_parameters=template.persistence_parameters,
        )

    def _new_sources(self, definition: AllowedObservation) -> tuple[ToolSource, ...]:
        sources: list[ToolSource] = []
        seen: set[str] = set()
        for source in (definition.channel7, definition.channel14):
            if (
                source.source_id not in self._used_source_ids
                and source.source_id not in seen
            ):
                sources.append(source)
                seen.add(source.source_id)
        return tuple(sources)

    def _sources_for_observations(
        self, observation_ids: tuple[str, ...]
    ) -> tuple[ToolSource, ...]:
        definitions = self._manifest.observations_by_id
        sources: dict[str, ToolSource] = {}
        for observation_id in observation_ids:
            observation = definitions[observation_id]
            sources.setdefault(observation.channel7.source_id, observation.channel7)
            sources.setdefault(observation.channel14.source_id, observation.channel14)
        return tuple(sources.values())

    def _verify_sources(self, sources: tuple[ToolSource, ...]) -> None:
        for source in sources:
            path = self._validated_source_path(source)
            if not path.is_file():
                raise FileNotFoundError(f"allowlisted cached source is missing: {path}")
            if path.stat().st_size != source.size_bytes:
                raise OSError(f"allowlisted cached source has the wrong size: {path}")
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != source.sha256:
                raise OSError(f"allowlisted cached source hash differs: {path}")

    def _validate_source_scope(self) -> None:
        for observation in self._manifest.observations:
            self._validated_source_path(observation.channel7)
            self._validated_source_path(observation.channel14)

    def _validated_source_path(self, source: ToolSource) -> Path:
        path = source.source_path.resolve()
        if not path.is_relative_to(self._source_cache_root):
            raise ValueError("tool source_path must remain within source_cache_root")
        self._assert_non_label_path(path, "tool source_path")
        return path

    def _assert_non_label_path(self, path: Path, field: str) -> None:
        if path.is_relative_to(self._labels_root):
            raise ValueError(f"{field} cannot access evaluation-only labels")

    def _used_bytes(self) -> int:
        by_id = {
            source.source_id: source
            for observation in self._manifest.observations
            for source in (observation.channel7, observation.channel14)
        }
        return sum(by_id[source_id].size_bytes for source_id in self._used_source_ids)

    def _elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock() - self._started))

    def _budget(self) -> Budget:
        return Budget(
            max_observations=self._maximum_observations,
            used_observations=len(self._selected_ids),
            max_bytes=self._maximum_bytes,
            used_bytes=self._used_bytes(),
            max_elapsed_seconds=self._maximum_elapsed_seconds,
            used_elapsed_seconds=min(
                self._elapsed_seconds(), self._maximum_elapsed_seconds
            ),
            max_retries=0,
            used_retries=0,
        )


class _ElapsedLimitError(RuntimeError):
    """Private sentinel preventing a partial observation-state update."""


def load_tool_manifest(path: Path, *, project_root: Path) -> ToolManifest:
    """Load a tool manifest while refusing evaluation-only label paths."""

    safe_path = runtime_input_path(Path(path), project_root=Path(project_root))
    try:
        payload = json.loads(safe_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read tool manifest: {safe_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tool manifest JSON: {safe_path}") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "record_type",
        "case",
        "evidence_template",
        "observations",
    }:
        raise ValueError("tool manifest has an invalid shape")
    if (
        payload["schema_version"] != TOOL_MANIFEST_SCHEMA_VERSION
        or payload["record_type"] != "bounded_observation_tool_manifest"
    ):
        raise ValueError("tool manifest has an unsupported schema")
    raw_observations = payload["observations"]
    if not isinstance(raw_observations, list):
        raise ValueError("tool manifest observations must be an array")
    return ToolManifest(
        case=ManifestCase.from_dict(payload["case"]),
        evidence_template=EvidenceJob.from_dict(
            payload["evidence_template"], base_directory=safe_path.parent
        ),
        observations=tuple(
            _observation_from_dict(item, base_directory=safe_path.parent)
            for item in raw_observations
        ),
    )


def _observation_from_dict(
    value: object, *, base_directory: Path
) -> AllowedObservation:
    if not isinstance(value, Mapping) or set(value) != {
        "observation_id",
        "action_type",
        "observation_time_utc",
        "requested_channel",
        "channel7",
        "channel14",
    }:
        raise ValueError("tool observation has an invalid shape")
    action = value["action_type"]
    channel = value["requested_channel"]
    timestamp = value["observation_time_utc"]
    if not isinstance(action, str) or not isinstance(channel, str):
        raise ValueError("tool observation action_type and channel must be strings")
    return AllowedObservation(
        observation_id=cast(str, value["observation_id"]),
        action_type=ActionType(action),
        observation_time=_parse_timestamp(timestamp),
        requested_channel=Channel(channel),
        channel7=_source_from_dict(value["channel7"], base_directory=base_directory),
        channel14=_source_from_dict(value["channel14"], base_directory=base_directory),
    )


def _source_from_dict(value: object, *, base_directory: Path) -> ToolSource:
    if not isinstance(value, Mapping) or set(value) != {
        "source_id",
        "catalog_key",
        "source_path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("tool source has an invalid shape")
    source_path = value["source_path"]
    if not isinstance(source_path, str):
        raise ValueError("tool source_path must be a string")
    path = Path(source_path)
    if not path.is_absolute():
        path = base_directory / path
    return ToolSource(
        source_id=cast(str, value["source_id"]),
        catalog_key=cast(str, value["catalog_key"]),
        source_path=path,
        size_bytes=cast(int, value["size_bytes"]),
        sha256=cast(str, value["sha256"]),
    )


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("tool observation_time_utc must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("tool observation_time_utc is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("tool observation_time_utc must be UTC")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


__all__ = [
    "MAXIMUM_OBSERVATIONS",
    "TOOL_MANIFEST_SCHEMA_VERSION",
    "AllowedObservation",
    "BoundedObservationTools",
    "ToolError",
    "ToolErrorCode",
    "ToolManifest",
    "ToolResult",
    "ToolSource",
    "load_tool_manifest",
]
