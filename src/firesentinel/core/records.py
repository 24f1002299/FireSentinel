"""Small, strict JSON contracts for local FireSentinel evidence packets.

These records are deliberately plain dataclasses.  They make the local replay
contract inspectable without introducing a database, a schema service, or a
runtime validation dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self, cast

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_MEASUREMENT_NAME = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}\Z")


class RecordValidationError(ValueError):
    """Raised when a record cannot represent a complete, normalized fact."""


class Channel(StrEnum):
    """The only ABI channels in the frozen local product scope."""

    C07 = "C07"
    C14 = "C14"


class Unit(StrEnum):
    """Canonical units used in JSON measurement values."""

    KELVIN = "K"
    SQUARE_KILOMETRES = "km2"
    PIXELS = "px"
    SECONDS = "s"
    BYTES = "B"
    DEGREES = "deg"
    DIMENSIONLESS = "1"


class ReasonCode(StrEnum):
    """Closed reviewer-facing reason codes; free text stays outside facts."""

    VALID = "valid"
    SOURCE_MISSING = "source_missing"
    SOURCE_CORRUPT = "source_corrupt"
    OBSERVATION_NOT_ALLOWED = "observation_not_allowed"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    FRAME_BLANK = "frame_blank"
    FRAME_SATURATED = "frame_saturated"
    CONTRAST_LOW = "contrast_low"
    ALIGNMENT_FAILED = "alignment_failed"
    THERMAL_ANOMALY_WEAK = "thermal_anomaly_weak"
    THERMAL_EVIDENCE_PERSISTENT = "thermal_evidence_persistent"
    THERMAL_EVIDENCE_ABSENT = "thermal_evidence_absent"
    BANDS_CONFLICT = "bands_conflict"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    CONFIGURATION_INVALID = "configuration_invalid"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_PERSISTENT_EVIDENCE = "no_persistent_evidence"


class ActionType(StrEnum):
    """The bounded action space from the product contract."""

    NEXT_TIMESTAMP = "next_timestamp"
    ALTERNATE_BAND = "alternate_band"
    PRE_EVENT_BASELINE = "pre_event_baseline"
    FINALIZE = "finalize"
    ABSTAIN = "abstain"
    REQUEST_HUMAN_REVIEW = "request_human_review"


class OutcomeState(StrEnum):
    """Cautious terminal states; none assert that a wildfire is confirmed."""

    REVIEW_ESCALATION = "review_escalation"
    NO_PERSISTENT_EVIDENCE = "no_persistent_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


def _fail(field_name: str, message: str) -> NoReturn:
    raise RecordValidationError(f"{field_name}: {message}")


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        _fail(
            field_name,
            "must use lowercase letters, digits, hyphens, or underscores "
            "and start with a letter or digit",
        )
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(field_name, "must be a non-empty string")
    return value


def _content_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _CONTENT_HASH.fullmatch(value):
        _fail(field_name, "must be a lowercase 64-character SHA-256 hex digest")
    return value


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(field_name, f"must be an integer greater than or equal to {minimum}")
    return value


def _number(value: object, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field_name, "must be a finite number")
    number = float(value)
    if not isfinite(number):
        _fail(field_name, "must be a finite number")
    if minimum is not None and number < minimum:
        _fail(field_name, f"must be greater than or equal to {minimum}")
    return number


def _confidence(value: object, field_name: str) -> float:
    confidence = _number(value, field_name)
    if not 0.0 <= confidence <= 1.0:
        _fail(field_name, "must be in the inclusive range [0.0, 1.0]")
    return confidence


def _utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(field_name, "must be a timezone-aware UTC datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        _fail(field_name, "must use UTC")
    return value.astimezone(UTC)


def _timestamp_json(value: datetime) -> str:
    timestamp = _utc_timestamp(value, "timestamp")
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp_from_json(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(field_name, "must be an RFC 3339 UTC timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RecordValidationError(f"{field_name}: invalid timestamp") from error
    return _utc_timestamp(parsed, field_name)


def _enum(value: object, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        _fail(field_name, "must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise RecordValidationError(
            f"{field_name}: must be one of: {choices}"
        ) from error


def _tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(field_name, "must be a JSON array")
    return tuple(value)


def _unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        _fail(field_name, "must not contain duplicates")
    return values


def _reason_codes(value: object, field_name: str) -> tuple[ReasonCode, ...]:
    codes = tuple(
        _enum(item, ReasonCode, f"{field_name}[{index}]")
        for index, item in enumerate(_tuple(value, field_name))
    )
    if not codes:
        _fail(field_name, "must contain at least one reason code")
    if len(set(codes)) != len(codes):
        _fail(field_name, "must not contain duplicates")
    return tuple(ReasonCode(code) for code in codes)


def _record_payload(
    value: object, record_type: str, keys: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(record_type, "must be a JSON object")
    actual = set(value)
    expected = {"record_type", "schema_version", *keys}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        _fail(record_type, "; ".join(details))
    if value["record_type"] != record_type:
        _fail("record_type", f"must be {record_type!r}")
    if value["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"must be {SCHEMA_VERSION}")
    return cast(Mapping[str, Any], value)


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return _timestamp_json(value)
    if isinstance(value, JsonRecord):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (Coordinates, Measurement, ConfigurationReference)):
        return value.to_dict()
    return value


@dataclass(frozen=True, slots=True)
class Coordinates:
    """WGS 84 latitude/longitude in decimal degrees, serialized as ``lat``/``lon``."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = _number(self.latitude, "coordinates.latitude")
        longitude = _number(self.longitude, "coordinates.longitude")
        if not -90.0 <= latitude <= 90.0:
            _fail("coordinates.latitude", "must be in the inclusive range [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            _fail("coordinates.longitude", "must be in the inclusive range [-180, 180]")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.latitude, "lon": self.longitude}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping) or set(value) != {"lat", "lon"}:
            _fail("coordinates", "must contain exactly 'lat' and 'lon'")
        return cls(latitude=value["lat"], longitude=value["lon"])


@dataclass(frozen=True, slots=True)
class Measurement:
    """One numeric value with a canonical unit or an explicit null/missing reason."""

    name: str
    value: float | None
    unit: Unit
    missing_reason: ReasonCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _MEASUREMENT_NAME.fullmatch(self.name):
            _fail(
                "measurement.name",
                "must use lowercase letters, digits, and underscores "
                "after the first letter",
            )
        unit = _enum(self.unit, Unit, "measurement.unit")
        missing_reason = (
            None
            if self.missing_reason is None
            else _enum(self.missing_reason, ReasonCode, "measurement.missing_reason")
        )
        if self.value is None:
            if missing_reason is None:
                _fail("measurement.missing_reason", "is required when value is null")
        else:
            numeric_value = _number(self.value, "measurement.value")
            if missing_reason is not None:
                _fail(
                    "measurement.missing_reason", "must be null when value is present"
                )
            object.__setattr__(self, "value", numeric_value)
        object.__setattr__(self, "unit", Unit(unit))
        object.__setattr__(
            self,
            "missing_reason",
            None if missing_reason is None else ReasonCode(missing_reason),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value,
            "missing_reason": (
                None if self.missing_reason is None else self.missing_reason.value
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "value",
            "unit",
            "missing_reason",
        }:
            _fail(
                "measurement",
                "must contain exactly name, value, unit, and missing_reason",
            )
        return cls(
            name=value["name"],
            value=value["value"],
            unit=value["unit"],
            missing_reason=value["missing_reason"],
        )


@dataclass(frozen=True, slots=True)
class ConfigurationReference:
    """An immutable configuration identity and its canonical content hash."""

    configuration_id: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_id",
            _identifier(self.configuration_id, "configuration_id"),
        )
        object.__setattr__(
            self,
            "content_hash",
            _content_hash(self.content_hash, "configuration.content_hash"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "configuration_id": self.configuration_id,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping) or set(value) != {
            "configuration_id",
            "content_hash",
        }:
            _fail(
                "configuration",
                "must contain exactly configuration_id and content_hash",
            )
        return cls(
            configuration_id=value["configuration_id"],
            content_hash=value["content_hash"],
        )


@dataclass(frozen=True, slots=True)
class JsonRecord:
    """Common canonical JSON codec for top-level evidence records."""

    RECORD_TYPE: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            field.name: _json_value(getattr(self, field.name)) for field in fields(self)
        }
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": SCHEMA_VERSION,
            **payload,
        }

    def to_json(self) -> str:
        """Return deterministic, UTF-8-safe canonical JSON without whitespace."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Build a typed record from a decoded JSON object."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, value: str) -> Self:
        if not isinstance(value, str):
            _fail("json", "must be a string")
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise RecordValidationError("json: invalid JSON") from error
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class ManifestCase(JsonRecord):
    """Pinned historical case and the allowlisted observations it may request."""

    case_id: str
    title: str
    location: Coordinates
    created_at: datetime
    content_hash: str
    allowed_observation_ids: tuple[str, ...]

    RECORD_TYPE: ClassVar[str] = "manifest_case"

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(self, "title", _text(self.title, "title"))
        if not isinstance(self.location, Coordinates):
            _fail("location", "must be Coordinates")
        object.__setattr__(
            self, "created_at", _utc_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "content_hash", _content_hash(self.content_hash, "content_hash")
        )
        allowed_ids = tuple(
            _identifier(item, f"allowed_observation_ids[{index}]")
            for index, item in enumerate(
                _tuple(self.allowed_observation_ids, "allowed_observation_ids")
            )
        )
        if not allowed_ids:
            _fail("allowed_observation_ids", "must contain at least one observation id")
        object.__setattr__(
            self,
            "allowed_observation_ids",
            _unique(allowed_ids, "allowed_observation_ids"),
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "case_id",
                "title",
                "location",
                "created_at",
                "content_hash",
                "allowed_observation_ids",
            },
        )
        return cls(
            case_id=payload["case_id"],
            title=payload["title"],
            location=Coordinates.from_dict(payload["location"]),
            created_at=_timestamp_from_json(payload["created_at"], "created_at"),
            content_hash=payload["content_hash"],
            allowed_observation_ids=_tuple(
                payload["allowed_observation_ids"], "allowed_observation_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationRequest(JsonRecord):
    """One permitted image observation in a manifest case."""

    observation_id: str
    case_id: str
    requested_at: datetime
    observation_time: datetime
    channel: Channel
    coordinates: Coordinates

    RECORD_TYPE: ClassVar[str] = "observation_request"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        requested_at = _utc_timestamp(self.requested_at, "requested_at")
        observation_time = _utc_timestamp(self.observation_time, "observation_time")
        if observation_time < requested_at:
            _fail("observation_time", "must not be before requested_at")
        if not isinstance(self.coordinates, Coordinates):
            _fail("coordinates", "must be Coordinates")
        object.__setattr__(self, "requested_at", requested_at)
        object.__setattr__(self, "observation_time", observation_time)
        object.__setattr__(
            self, "channel", Channel(_enum(self.channel, Channel, "channel"))
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "observation_id",
                "case_id",
                "requested_at",
                "observation_time",
                "channel",
                "coordinates",
            },
        )
        return cls(
            observation_id=payload["observation_id"],
            case_id=payload["case_id"],
            requested_at=_timestamp_from_json(payload["requested_at"], "requested_at"),
            observation_time=_timestamp_from_json(
                payload["observation_time"], "observation_time"
            ),
            channel=payload["channel"],
            coordinates=Coordinates.from_dict(payload["coordinates"]),
        )


@dataclass(frozen=True, slots=True)
class SourceObject(JsonRecord):
    """Immutable provenance for the source object used by one observation."""

    source_id: str
    observation_id: str
    provider: str
    bucket: str
    object_key: str
    content_hash: str
    size_bytes: int
    scan_time: datetime
    discovered_at: datetime

    RECORD_TYPE: ClassVar[str] = "source_object"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "bucket", _text(self.bucket, "bucket"))
        object.__setattr__(self, "object_key", _text(self.object_key, "object_key"))
        object.__setattr__(
            self, "content_hash", _content_hash(self.content_hash, "content_hash")
        )
        object.__setattr__(self, "size_bytes", _integer(self.size_bytes, "size_bytes"))
        scan_time = _utc_timestamp(self.scan_time, "scan_time")
        discovered_at = _utc_timestamp(self.discovered_at, "discovered_at")
        if discovered_at < scan_time:
            _fail("discovered_at", "must not be before scan_time")
        object.__setattr__(self, "scan_time", scan_time)
        object.__setattr__(self, "discovered_at", discovered_at)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "source_id",
                "observation_id",
                "provider",
                "bucket",
                "object_key",
                "content_hash",
                "size_bytes",
                "scan_time",
                "discovered_at",
            },
        )
        return cls(
            source_id=payload["source_id"],
            observation_id=payload["observation_id"],
            provider=payload["provider"],
            bucket=payload["bucket"],
            object_key=payload["object_key"],
            content_hash=payload["content_hash"],
            size_bytes=payload["size_bytes"],
            scan_time=_timestamp_from_json(payload["scan_time"], "scan_time"),
            discovered_at=_timestamp_from_json(
                payload["discovered_at"], "discovered_at"
            ),
        )


@dataclass(frozen=True, slots=True)
class VisionEvidence(JsonRecord):
    """Calibrated, reviewer-inspectable evidence produced for one source object."""

    evidence_id: str
    case_id: str
    observation_id: str
    source_id: str
    configuration: ConfigurationReference
    created_at: datetime
    coordinates: Coordinates
    measurements: tuple[Measurement, ...]
    confidence: float
    reason_codes: tuple[ReasonCode, ...]
    content_hash: str

    RECORD_TYPE: ClassVar[str] = "vision_evidence"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if not isinstance(self.configuration, ConfigurationReference):
            _fail("configuration", "must be ConfigurationReference")
        object.__setattr__(
            self, "created_at", _utc_timestamp(self.created_at, "created_at")
        )
        if not isinstance(self.coordinates, Coordinates):
            _fail("coordinates", "must be Coordinates")
        measurements = tuple(
            item
            for item in _tuple(self.measurements, "measurements")
            if isinstance(item, Measurement)
        )
        if len(measurements) != len(self.measurements):
            _fail("measurements", "must contain Measurement values")
        if not measurements:
            _fail("measurements", "must contain at least one measurement")
        names = tuple(measurement.name for measurement in measurements)
        _unique(names, "measurements")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(
            self, "confidence", _confidence(self.confidence, "confidence")
        )
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "content_hash", _content_hash(self.content_hash, "content_hash")
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "evidence_id",
                "case_id",
                "observation_id",
                "source_id",
                "configuration",
                "created_at",
                "coordinates",
                "measurements",
                "confidence",
                "reason_codes",
                "content_hash",
            },
        )
        return cls(
            evidence_id=payload["evidence_id"],
            case_id=payload["case_id"],
            observation_id=payload["observation_id"],
            source_id=payload["source_id"],
            configuration=ConfigurationReference.from_dict(payload["configuration"]),
            created_at=_timestamp_from_json(payload["created_at"], "created_at"),
            coordinates=Coordinates.from_dict(payload["coordinates"]),
            measurements=tuple(
                Measurement.from_dict(item)
                for item in _tuple(payload["measurements"], "measurements")
            ),
            confidence=payload["confidence"],
            reason_codes=_tuple(payload["reason_codes"], "reason_codes"),
            content_hash=payload["content_hash"],
        )


@dataclass(frozen=True, slots=True)
class Action(JsonRecord):
    """A considered or selected bounded action, with the evidence it used."""

    action_id: str
    case_id: str
    action_type: ActionType
    created_at: datetime
    reason_codes: tuple[ReasonCode, ...]
    evidence_ids: tuple[str, ...]
    selected: bool

    RECORD_TYPE: ClassVar[str] = "action"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _identifier(self.action_id, "action_id"))
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "action_type",
            ActionType(_enum(self.action_type, ActionType, "action_type")),
        )
        object.__setattr__(
            self, "created_at", _utc_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, "reason_codes")
        )
        evidence_ids = tuple(
            _identifier(item, f"evidence_ids[{index}]")
            for index, item in enumerate(_tuple(self.evidence_ids, "evidence_ids"))
        )
        object.__setattr__(self, "evidence_ids", _unique(evidence_ids, "evidence_ids"))
        if not isinstance(self.selected, bool):
            _fail("selected", "must be a boolean")

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "action_id",
                "case_id",
                "action_type",
                "created_at",
                "reason_codes",
                "evidence_ids",
                "selected",
            },
        )
        return cls(
            action_id=payload["action_id"],
            case_id=payload["case_id"],
            action_type=payload["action_type"],
            created_at=_timestamp_from_json(payload["created_at"], "created_at"),
            reason_codes=_tuple(payload["reason_codes"], "reason_codes"),
            evidence_ids=_tuple(payload["evidence_ids"], "evidence_ids"),
            selected=payload["selected"],
        )


@dataclass(frozen=True, slots=True)
class Budget(JsonRecord):
    """Limits and consumed resource units at a point in an investigation."""

    max_observations: int
    used_observations: int
    max_bytes: int
    used_bytes: int
    max_elapsed_seconds: float
    used_elapsed_seconds: float
    max_retries: int
    used_retries: int

    RECORD_TYPE: ClassVar[str] = "budget"

    def __post_init__(self) -> None:
        for limit, used in (
            ("observations", "observations"),
            ("bytes", "bytes"),
            ("retries", "retries"),
        ):
            maximum = _integer(getattr(self, f"max_{limit}"), f"max_{limit}")
            consumed = _integer(getattr(self, f"used_{used}"), f"used_{used}")
            if consumed > maximum:
                _fail(f"used_{used}", f"must not exceed max_{limit}")
            object.__setattr__(self, f"max_{limit}", maximum)
            object.__setattr__(self, f"used_{used}", consumed)
        max_elapsed = _number(
            self.max_elapsed_seconds, "max_elapsed_seconds", minimum=0.0
        )
        used_elapsed = _number(
            self.used_elapsed_seconds, "used_elapsed_seconds", minimum=0.0
        )
        if used_elapsed > max_elapsed:
            _fail("used_elapsed_seconds", "must not exceed max_elapsed_seconds")
        object.__setattr__(self, "max_elapsed_seconds", max_elapsed)
        object.__setattr__(self, "used_elapsed_seconds", used_elapsed)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "max_observations",
                "used_observations",
                "max_bytes",
                "used_bytes",
                "max_elapsed_seconds",
                "used_elapsed_seconds",
                "max_retries",
                "used_retries",
            },
        )
        return cls(
            max_observations=payload["max_observations"],
            used_observations=payload["used_observations"],
            max_bytes=payload["max_bytes"],
            used_bytes=payload["used_bytes"],
            max_elapsed_seconds=payload["max_elapsed_seconds"],
            used_elapsed_seconds=payload["used_elapsed_seconds"],
            max_retries=payload["max_retries"],
            used_retries=payload["used_retries"],
        )


@dataclass(frozen=True, slots=True)
class Outcome(JsonRecord):
    """A terminal reviewer-facing conclusion linked to evidence and configuration."""

    outcome_id: str
    trace_id: str
    case_id: str
    state: OutcomeState
    created_at: datetime
    evidence_ids: tuple[str, ...]
    configuration: ConfigurationReference
    confidence: float
    reason_codes: tuple[ReasonCode, ...]

    RECORD_TYPE: ClassVar[str] = "outcome"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcome_id", _identifier(self.outcome_id, "outcome_id")
        )
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self, "state", OutcomeState(_enum(self.state, OutcomeState, "state"))
        )
        object.__setattr__(
            self, "created_at", _utc_timestamp(self.created_at, "created_at")
        )
        evidence_ids = tuple(
            _identifier(item, f"evidence_ids[{index}]")
            for index, item in enumerate(_tuple(self.evidence_ids, "evidence_ids"))
        )
        if not evidence_ids:
            _fail("evidence_ids", "must link at least one evidence record")
        object.__setattr__(self, "evidence_ids", _unique(evidence_ids, "evidence_ids"))
        if not isinstance(self.configuration, ConfigurationReference):
            _fail("configuration", "must be ConfigurationReference")
        object.__setattr__(
            self, "confidence", _confidence(self.confidence, "confidence")
        )
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, "reason_codes")
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "outcome_id",
                "trace_id",
                "case_id",
                "state",
                "created_at",
                "evidence_ids",
                "configuration",
                "confidence",
                "reason_codes",
            },
        )
        return cls(
            outcome_id=payload["outcome_id"],
            trace_id=payload["trace_id"],
            case_id=payload["case_id"],
            state=payload["state"],
            created_at=_timestamp_from_json(payload["created_at"], "created_at"),
            evidence_ids=_tuple(payload["evidence_ids"], "evidence_ids"),
            configuration=ConfigurationReference.from_dict(payload["configuration"]),
            confidence=payload["confidence"],
            reason_codes=_tuple(payload["reason_codes"], "reason_codes"),
        )


@dataclass(frozen=True, slots=True)
class Trace(JsonRecord):
    """Complete local investigation trace with all cross-record links checked."""

    trace_id: str
    case: ManifestCase
    configuration: ConfigurationReference
    started_at: datetime
    completed_at: datetime
    observation_requests: tuple[ObservationRequest, ...]
    sources: tuple[SourceObject, ...]
    evidence: tuple[VisionEvidence, ...]
    actions: tuple[Action, ...]
    budget: Budget
    outcome: Outcome

    RECORD_TYPE: ClassVar[str] = "trace"

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        if not isinstance(self.case, ManifestCase):
            _fail("case", "must be ManifestCase")
        if not isinstance(self.configuration, ConfigurationReference):
            _fail("configuration", "must be ConfigurationReference")
        started_at = _utc_timestamp(self.started_at, "started_at")
        completed_at = _utc_timestamp(self.completed_at, "completed_at")
        if completed_at < started_at:
            _fail("completed_at", "must not be before started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        requests = _typed_records(
            self.observation_requests, ObservationRequest, "observation_requests"
        )
        sources = _typed_records(self.sources, SourceObject, "sources")
        evidence = _typed_records(self.evidence, VisionEvidence, "evidence")
        actions = _typed_records(self.actions, Action, "actions")
        if not isinstance(self.budget, Budget):
            _fail("budget", "must be Budget")
        if not isinstance(self.outcome, Outcome):
            _fail("outcome", "must be Outcome")
        object.__setattr__(self, "observation_requests", requests)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "actions", actions)
        self._validate_links()

    def _validate_links(self) -> None:
        case_id = self.case.case_id
        request_ids = _unique(
            tuple(request.observation_id for request in self.observation_requests),
            "observation_requests",
        )
        if not request_ids:
            _fail("observation_requests", "must contain at least one request")
        if not set(request_ids).issubset(self.case.allowed_observation_ids):
            _fail(
                "observation_requests",
                "must be allowlisted by case.allowed_observation_ids",
            )
        if any(request.case_id != case_id for request in self.observation_requests):
            _fail("observation_requests", "all requests must link to trace.case_id")

        source_ids = _unique(
            tuple(source.source_id for source in self.sources), "sources"
        )
        if any(source.observation_id not in request_ids for source in self.sources):
            _fail("sources", "every source must link to a trace observation request")

        evidence_ids = _unique(
            tuple(item.evidence_id for item in self.evidence), "evidence"
        )
        if not evidence_ids:
            _fail("evidence", "must contain at least one evidence record")
        for item in self.evidence:
            if item.case_id != case_id:
                _fail("evidence", "every item must link to trace.case_id")
            if item.observation_id not in request_ids:
                _fail("evidence", "every item must link to a trace observation request")
            if item.source_id not in source_ids:
                _fail("evidence", "every item must link to a trace source")
            if item.configuration != self.configuration:
                _fail("evidence", "every item must use trace.configuration")

        _unique(tuple(action.action_id for action in self.actions), "actions")
        for action in self.actions:
            if action.case_id != case_id:
                _fail("actions", "every action must link to trace.case_id")
            if not set(action.evidence_ids).issubset(evidence_ids):
                _fail("actions", "evidence_ids must link to trace evidence")

        if self.outcome.trace_id != self.trace_id:
            _fail("outcome.trace_id", "must link to trace_id")
        if self.outcome.case_id != case_id:
            _fail("outcome.case_id", "must link to trace.case_id")
        if not set(self.outcome.evidence_ids).issubset(evidence_ids):
            _fail("outcome.evidence_ids", "must link to trace evidence")
        if self.outcome.configuration != self.configuration:
            _fail("outcome.configuration", "must link to trace.configuration")

    @classmethod
    def from_dict(cls, value: object) -> Self:
        payload = _record_payload(
            value,
            cls.RECORD_TYPE,
            {
                "trace_id",
                "case",
                "configuration",
                "started_at",
                "completed_at",
                "observation_requests",
                "sources",
                "evidence",
                "actions",
                "budget",
                "outcome",
            },
        )
        return cls(
            trace_id=payload["trace_id"],
            case=ManifestCase.from_dict(payload["case"]),
            configuration=ConfigurationReference.from_dict(payload["configuration"]),
            started_at=_timestamp_from_json(payload["started_at"], "started_at"),
            completed_at=_timestamp_from_json(payload["completed_at"], "completed_at"),
            observation_requests=tuple(
                ObservationRequest.from_dict(item)
                for item in _tuple(
                    payload["observation_requests"], "observation_requests"
                )
            ),
            sources=tuple(
                SourceObject.from_dict(item)
                for item in _tuple(payload["sources"], "sources")
            ),
            evidence=tuple(
                VisionEvidence.from_dict(item)
                for item in _tuple(payload["evidence"], "evidence")
            ),
            actions=tuple(
                Action.from_dict(item) for item in _tuple(payload["actions"], "actions")
            ),
            budget=Budget.from_dict(payload["budget"]),
            outcome=Outcome.from_dict(payload["outcome"]),
        )


def _typed_records(
    value: object, record_type: type[JsonRecord], field_name: str
) -> tuple[Any, ...]:
    records = _tuple(value, field_name)
    if not all(isinstance(item, record_type) for item in records):
        _fail(field_name, f"must contain {record_type.__name__} values")
    return records


RECORD_TYPES: dict[str, type[JsonRecord]] = {
    ManifestCase.RECORD_TYPE: ManifestCase,
    ObservationRequest.RECORD_TYPE: ObservationRequest,
    SourceObject.RECORD_TYPE: SourceObject,
    VisionEvidence.RECORD_TYPE: VisionEvidence,
    Action.RECORD_TYPE: Action,
    Budget.RECORD_TYPE: Budget,
    Trace.RECORD_TYPE: Trace,
    Outcome.RECORD_TYPE: Outcome,
}


def record_from_json(value: str) -> JsonRecord:
    """Decode one tagged top-level record and validate its declared record type."""
    if not isinstance(value, str):
        _fail("json", "must be a string")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise RecordValidationError("json: invalid JSON") from error
    if not isinstance(payload, Mapping):
        _fail("json", "must contain a JSON object")
    record_type = payload.get("record_type")
    if not isinstance(record_type, str) or record_type not in RECORD_TYPES:
        _fail("record_type", "must name a supported FireSentinel record")
    return RECORD_TYPES[record_type].from_dict(payload)


def canonical_content_hash(value: JsonRecord | Mapping[str, Any] | str | bytes) -> str:
    """Return the SHA-256 digest of canonical record JSON or supplied bytes."""
    if isinstance(value, JsonRecord):
        payload = value.to_json().encode("utf-8")
    elif isinstance(value, Mapping):
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, bytes):
        payload = value
    else:
        _fail("content", "must be a record, JSON mapping, text, or bytes")
    return sha256(payload).hexdigest()


ArtifactDirectory = Path


def artifact_directory(
    artifacts_root: Path, case_id: str, content_hash: str
) -> ArtifactDirectory:
    """Return ``{root}/{case_id}/{sha256}`` without creating or writing anything."""
    root = Path(artifacts_root).resolve()
    validated_case_id = _identifier(case_id, "case_id")
    validated_hash = _content_hash(content_hash, "content_hash")
    destination = (root / validated_case_id / validated_hash).resolve()
    if not destination.is_relative_to(root):
        _fail("artifact_directory", "must stay within artifacts_root")
    return destination
