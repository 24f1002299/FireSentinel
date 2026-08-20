# Mask-aware tile preparation

`firesentinel.vision.tiles` is the boundary between a calibrated GOES crop and
OpenCV-ready inputs. It deliberately has two products:

- `calibrated` preserves the original float32 crop and source invalid mask.
- `resized_calibrated` is a derived physically clipped, mask-aware-resized
  analysis array. Evidence stages must consume it together with `valid_mask`.

`display` is a robust quantile-scaled `uint8` rendering only. It is not a
measurement array. If configured, `clahe_display` is a separate optional
review rendering; it never replaces `display` or `resized_calibrated`.

```python
from firesentinel.vision.tiles import TilePreparationParameters, prepare_tile

parameters = TilePreparationParameters(
    physical_minimum_kelvin=250.0,
    physical_maximum_kelvin=400.0,
    target_shape=(256, 256),
    minimum_valid_coverage=1.0,
    clahe_clip_limit=None,
)
tile = prepare_tile(calibrated_crop, parameters)
metadata = tile.metadata()
```

The default coverage threshold is `1.0`: every contributor to a resized output
pixel must be valid. A lower explicit threshold may retain a pixel built from
valid contributors only; invalid values are excluded from the weighted sum and
the output mask is still mandatory for evidence. In all cases, invalid output
samples are `NaN`, black in display images, and false in `valid_mask`.

`metadata()` emits the source crop checksum/timing, all processing parameters,
input/effective/resized mask counts, physical and robust display ranges,
per-stage timings, array hashes, and OpenCV version/build-information hash.
The deterministic content checksum excludes performance timings and build
metadata so golden numerical outputs remain comparable across runs; array
hashes expose any implementation difference.
