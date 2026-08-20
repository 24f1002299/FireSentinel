"""Deterministic, offline thermal fixtures for vision and policy development.

The arrays intentionally model only small, controlled situations.  They are
not calibrated GOES observations and must never be used as wildfire evidence.
Their role is to make the expected behaviour of later vision stages explicit
without downloading a source product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from firesentinel.core.records import ReasonCode
from firesentinel.vision.quality import measure_observation_quality

FIXTURE_VERSION = 1
FIXTURE_SEED = 20_260_819
FIXTURE_SHAPE = (12, 12)
SATURATION_VALUE = np.float32(350.0)
OFFLINE_MANIFEST_PATH = Path(__file__).with_name("synthetic_fixture_manifest.json")

FloatArray = NDArray[np.float32]
MaskArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class NumericTolerances:
    """Fixed comparison bounds for all synthetic fixture assertions."""

    absolute: float = 1e-6
    relative: float = 0.0


NUMERIC_TOLERANCES = NumericTolerances()


@dataclass(frozen=True, slots=True)
class ExpectedComponent:
    """One expected connected component in a binary heat mask."""

    label: int
    area_pixels: int
    bounding_box_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FrameQualityExpectation:
    """Mask-aware quality values expected for one synthetic thermal frame."""

    coverage_fraction: float
    saturated_fraction: float
    contrast_span: float
    missing_fraction: float = 0.0
    texture_standard_deviation: float = 0.0
    mean_absolute_neighbor_difference: float = 0.0


@dataclass(frozen=True, slots=True)
class PersistenceExpectation:
    """Expected overlap before and after applying a known image translation."""

    unaligned_iou: float
    aligned_iou: float
    translation_xy: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SyntheticFixtureCase:
    """Immutable inputs and expected outputs for one small thermal condition."""

    name: str
    thermal_frames: tuple[FloatArray, ...]
    valid_masks: tuple[MaskArray, ...]
    expected_heat_masks: tuple[MaskArray, ...]
    expected_components: tuple[tuple[ExpectedComponent, ...], ...]
    expected_quality: tuple[FrameQualityExpectation, ...]
    expected_persistence: PersistenceExpectation | None
    expected_reason_codes: tuple[ReasonCode, ...]

    def __post_init__(self) -> None:
        count = len(self.thermal_frames)
        if not self.name or count == 0:
            raise ValueError("fixture case requires a name and at least one frame")
        if not all(
            len(values) == count
            for values in (
                self.valid_masks,
                self.expected_heat_masks,
                self.expected_components,
                self.expected_quality,
            )
        ):
            raise ValueError(
                f"fixture case {self.name!r} has inconsistent frame counts"
            )
        if not self.expected_reason_codes:
            raise ValueError(f"fixture case {self.name!r} requires reason codes")
        frames = tuple(_immutable_float_array(frame) for frame in self.thermal_frames)
        valid_masks = tuple(_immutable_mask_array(mask) for mask in self.valid_masks)
        heat_masks = tuple(
            _immutable_mask_array(mask) for mask in self.expected_heat_masks
        )
        for frame, valid_mask, heat_mask in zip(
            frames, valid_masks, heat_masks, strict=True
        ):
            if frame.shape != FIXTURE_SHAPE:
                raise ValueError(f"fixture frame must have shape {FIXTURE_SHAPE}")
            if valid_mask.shape != frame.shape or heat_mask.shape != frame.shape:
                raise ValueError("fixture masks must match the thermal frame shape")
            if np.any(np.logical_and(heat_mask, np.logical_not(valid_mask))):
                raise ValueError("expected heat masks cannot include invalid pixels")
        object.__setattr__(self, "thermal_frames", frames)
        object.__setattr__(self, "valid_masks", valid_masks)
        object.__setattr__(self, "expected_heat_masks", heat_masks)
        object.__setattr__(
            self,
            "expected_reason_codes",
            tuple(ReasonCode(code) for code in self.expected_reason_codes),
        )


@dataclass(frozen=True, slots=True)
class SyntheticFixtureBundle:
    """The versioned, local-only fixture set and its expected outputs."""

    version: int
    seed: int
    shape: tuple[int, int]
    tolerances: NumericTolerances
    cases: tuple[SyntheticFixtureCase, ...]

    def __post_init__(self) -> None:
        if self.version != FIXTURE_VERSION:
            raise ValueError(f"unsupported fixture version: {self.version}")
        if self.seed != FIXTURE_SEED or self.shape != FIXTURE_SHAPE:
            raise ValueError("fixture seed and shape are fixed for bundle version 1")
        names = tuple(case.name for case in self.cases)
        if len(names) != len(set(names)):
            raise ValueError("fixture case names must be unique")

    def case(self, name: str) -> SyntheticFixtureCase:
        """Return a named fixture case or fail with an actionable error."""
        for fixture_case in self.cases:
            if fixture_case.name == name:
                return fixture_case
        raise KeyError(f"unknown synthetic fixture case: {name}")


class FixtureIntegrityError(ValueError):
    """An offline fixture no longer matches its checked-in source fingerprint."""

    def __init__(self, detail: str, reason_code: ReasonCode) -> None:
        super().__init__(f"{reason_code.value}: {detail}")
        self.reason_code = reason_code


def generate_synthetic_fixture_bundle() -> SyntheticFixtureBundle:
    """Generate the complete offline fixture set from a pinned seed.

    Every fixture uses a private NumPy generator, so calls are independent of
    global random state and repeat exactly with the locked NumPy version.
    """
    generator = np.random.default_rng(FIXTURE_SEED)
    valid = np.ones(FIXTURE_SHAPE, dtype=np.bool_)

    persistent_first = _background(generator)
    persistent_second = _background(generator)
    _heat(persistent_first, 3, 4)
    _heat(persistent_second, 3, 4)
    persistent_mask = _heat_mask(3, 4)

    transient_first = _background(generator)
    transient_second = _background(generator)
    _heat(transient_first, 3, 4)

    shifted_first = _background(generator)
    shifted_second = _background(generator)
    _heat(shifted_first, 3, 4)
    _heat(shifted_second, 4, 2)
    shifted_first_mask = _heat_mask(3, 4)
    shifted_second_mask = _heat_mask(4, 2)

    missing = _background(generator)
    _heat(missing, 2, 3)
    missing_valid = valid.copy()
    missing_valid[:7, :] = False
    missing_heat_mask = np.zeros(FIXTURE_SHAPE, dtype=np.bool_)

    saturated = _background(generator)
    saturated[6:8, 6:8] = SATURATION_VALUE

    low_contrast = np.full(FIXTURE_SHAPE, 300.0, dtype=np.float32)
    low_contrast += generator.uniform(-0.004, 0.004, FIXTURE_SHAPE).astype(np.float32)

    empty = np.zeros(FIXTURE_SHAPE, dtype=np.float32)
    no_heat = np.zeros(FIXTURE_SHAPE, dtype=np.bool_)
    component = _component(4, 3)
    shifted_component = _component(2, 4)

    cases = (
        SyntheticFixtureCase(
            name="persistent_heat",
            thermal_frames=(persistent_first, persistent_second),
            valid_masks=(valid, valid),
            expected_heat_masks=(persistent_mask, persistent_mask),
            expected_components=((component,), (component,)),
            expected_quality=(
                _quality(persistent_first, valid),
                _quality(persistent_second, valid),
            ),
            expected_persistence=PersistenceExpectation(1.0, 1.0, (0, 0)),
            expected_reason_codes=(ReasonCode.THERMAL_EVIDENCE_PERSISTENT,),
        ),
        SyntheticFixtureCase(
            name="transient_heat",
            thermal_frames=(transient_first, transient_second),
            valid_masks=(valid, valid),
            expected_heat_masks=(persistent_mask, no_heat),
            expected_components=((component,), ()),
            expected_quality=(
                _quality(transient_first, valid),
                _quality(transient_second, valid),
            ),
            expected_persistence=PersistenceExpectation(0.0, 0.0, (0, 0)),
            expected_reason_codes=(ReasonCode.NO_PERSISTENT_EVIDENCE,),
        ),
        SyntheticFixtureCase(
            name="image_shift",
            thermal_frames=(shifted_first, shifted_second),
            valid_masks=(valid, valid),
            expected_heat_masks=(shifted_first_mask, shifted_second_mask),
            expected_components=((component,), (shifted_component,)),
            expected_quality=(
                _quality(shifted_first, valid),
                _quality(shifted_second, valid),
            ),
            expected_persistence=PersistenceExpectation(0.0, 1.0, (-2, 1)),
            expected_reason_codes=(ReasonCode.VALID,),
        ),
        SyntheticFixtureCase(
            name="missing_pixels",
            thermal_frames=(missing,),
            valid_masks=(missing_valid,),
            expected_heat_masks=(missing_heat_mask,),
            expected_components=((),),
            expected_quality=(_quality(missing, missing_valid),),
            expected_persistence=None,
            expected_reason_codes=(ReasonCode.COVERAGE_INSUFFICIENT,),
        ),
        SyntheticFixtureCase(
            name="saturated_pixels",
            thermal_frames=(saturated,),
            valid_masks=(valid,),
            expected_heat_masks=(no_heat,),
            expected_components=((),),
            expected_quality=(_quality(saturated, valid),),
            expected_persistence=None,
            expected_reason_codes=(ReasonCode.FRAME_SATURATED,),
        ),
        SyntheticFixtureCase(
            name="low_contrast",
            thermal_frames=(low_contrast,),
            valid_masks=(valid,),
            expected_heat_masks=(no_heat,),
            expected_components=((),),
            expected_quality=(_quality(low_contrast, valid),),
            expected_persistence=None,
            expected_reason_codes=(ReasonCode.CONTRAST_LOW,),
        ),
        SyntheticFixtureCase(
            name="empty_frame",
            thermal_frames=(empty,),
            valid_masks=(valid,),
            expected_heat_masks=(no_heat,),
            expected_components=((),),
            expected_quality=(_quality(empty, valid),),
            expected_persistence=None,
            expected_reason_codes=(ReasonCode.FRAME_BLANK,),
        ),
    )
    return SyntheticFixtureBundle(
        version=FIXTURE_VERSION,
        seed=FIXTURE_SEED,
        shape=FIXTURE_SHAPE,
        tolerances=NUMERIC_TOLERANCES,
        cases=cases,
    )


def fixture_case_digests(bundle: SyntheticFixtureBundle) -> dict[str, str]:
    """Return stable SHA-256 digests for each fixture's raw arrays and masks."""
    return {
        fixture_case.name: _case_digest(fixture_case) for fixture_case in bundle.cases
    }


def fixture_bundle_digest(bundle: SyntheticFixtureBundle) -> str:
    """Return a stable fingerprint for all named fixture payloads."""
    digest = hashlib.sha256()
    for name, case_digest in fixture_case_digests(bundle).items():
        digest.update(name.encode("utf-8"))
        digest.update(b":")
        digest.update(case_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_offline_fixture_bundle(
    manifest_path: Path = OFFLINE_MANIFEST_PATH,
) -> SyntheticFixtureBundle:
    """Load and verify the checked-in fixture manifest without network access."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureIntegrityError(
            f"could not read offline fixture manifest {manifest_path}",
            ReasonCode.SOURCE_CORRUPT,
        ) from error
    if not isinstance(manifest, dict):
        raise FixtureIntegrityError(
            "offline fixture manifest is not an object", ReasonCode.SOURCE_CORRUPT
        )
    bundle = generate_synthetic_fixture_bundle()
    _verify_manifest_configuration(bundle, manifest)
    verify_fixture_bundle(bundle, manifest)
    return bundle


def verify_fixture_bundle(
    bundle: SyntheticFixtureBundle, manifest: dict[str, object]
) -> None:
    """Reject fixture corruption with a reviewer-facing source-corrupt reason."""
    expected_case_digests = manifest.get("case_digests")
    expected_bundle_digest = manifest.get("bundle_digest")
    if not isinstance(expected_case_digests, dict) or not isinstance(
        expected_bundle_digest, str
    ):
        raise FixtureIntegrityError(
            "offline fixture manifest is missing digest fields",
            ReasonCode.SOURCE_CORRUPT,
        )
    actual_case_digests = fixture_case_digests(bundle)
    if actual_case_digests != expected_case_digests:
        raise FixtureIntegrityError(
            "fixture arrays or masks do not match their pinned digest",
            ReasonCode.SOURCE_CORRUPT,
        )
    if fixture_bundle_digest(bundle) != expected_bundle_digest:
        raise FixtureIntegrityError(
            "fixture bundle does not match its pinned digest", ReasonCode.SOURCE_CORRUPT
        )


def offline_fixture_manifest() -> dict[str, object]:
    """Return the checked-in manifest for test inspection and bundle tooling."""
    payload = json.loads(OFFLINE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixtureIntegrityError(
            "offline fixture manifest is not an object", ReasonCode.SOURCE_CORRUPT
        )
    return payload


def _immutable_float_array(array: FloatArray) -> FloatArray:
    copied = np.array(array, dtype=np.float32, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _immutable_mask_array(array: MaskArray) -> MaskArray:
    copied = np.array(array, dtype=np.bool_, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _background(generator: np.random.Generator) -> FloatArray:
    return generator.normal(290.0, 0.25, FIXTURE_SHAPE).astype(np.float32)


def _heat(frame: FloatArray, row: int, column: int) -> None:
    frame[row : row + 2, column : column + 2] = np.float32(325.0)


def _heat_mask(row: int, column: int) -> MaskArray:
    mask = np.zeros(FIXTURE_SHAPE, dtype=np.bool_)
    mask[row : row + 2, column : column + 2] = True
    return mask


def _component(column: int, row: int) -> ExpectedComponent:
    return ExpectedComponent(
        label=1,
        area_pixels=4,
        bounding_box_xywh=(column, row, 2, 2),
        centroid_xy=(column + 0.5, row + 0.5),
    )


def _quality(frame: FloatArray, valid_mask: MaskArray) -> FrameQualityExpectation:
    quality = measure_observation_quality(frame, ~valid_mask)
    return FrameQualityExpectation(
        coverage_fraction=quality.usable_coverage_fraction,
        saturated_fraction=quality.saturated_pixel_fraction,
        contrast_span=quality.contrast_span_kelvin,
        missing_fraction=quality.missing_pixel_fraction,
        texture_standard_deviation=quality.texture_standard_deviation_kelvin,
        mean_absolute_neighbor_difference=(
            quality.mean_absolute_neighbor_difference_kelvin
        ),
    )


def _case_digest(fixture_case: SyntheticFixtureCase) -> str:
    digest = hashlib.sha256()
    digest.update(fixture_case.name.encode("utf-8"))
    for collection in (
        fixture_case.thermal_frames,
        fixture_case.valid_masks,
        fixture_case.expected_heat_masks,
    ):
        for array in collection:
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_manifest_configuration(
    bundle: SyntheticFixtureBundle, manifest: dict[str, object]
) -> None:
    expected = {
        "fixture_version": bundle.version,
        "seed": bundle.seed,
        "shape": list(bundle.shape),
        "numeric_tolerances": {
            "absolute": bundle.tolerances.absolute,
            "relative": bundle.tolerances.relative,
        },
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FixtureIntegrityError(
                f"offline fixture manifest has an unexpected {key}",
                ReasonCode.SOURCE_CORRUPT,
            )
