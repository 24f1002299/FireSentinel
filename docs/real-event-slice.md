# Park Fire Channel 7 vertical slice

This is one manually reviewed, non-operational historical example. It uses the
Park Fire near Bidwell Park, Chico, California, which CAL FIRE records as
starting on 2024-07-24. The audit freezes two GOES-18 ABI-L2-CMIPF Channel 7
full-disk scans on 2024-07-25 at 22:30:20.8Z and 22:50:20.8Z. The fire context
and incident location are retained in the manifest's `manual_audit` field.

The image is thermal evidence only. Channel 7 hot regions can also reflect
sunlit land, cloud, or other non-fire sources; this single-band slice neither
confirms a wildfire nor supports alerting, dispatch, or emergency decisions.
A human reviewer must interpret it with the source provenance, timestamps,
quality mask, and later multi-band work.

## Reproduction

The first command obtains the two bytestrings named by the manifest and
publishes them through the verified content-addressed cache. It is the only
networked operation.

```powershell
.\.venv\Scripts\python -m firesentinel.data.download --manifest manifests\park-fire-20240725.json
```

Once cached, this one command is offline and deterministic:

```powershell
.\.venv\Scripts\python -m scripts.tasks slice
```

It performs mask-aware fixed-range display scaling, an OpenCV binary thermal
threshold, 3-by-3 elliptical opening then closing, connected components, and
external contours. It writes:

```text
artifacts/park-fire-20240725/{configuration_sha256}/
  evidence.json
  before-after.png
```

`evidence.json` contains source/crop/display/mask/contour SHA-256 values, exact
contour points, components, measurements, and the reviewer image digest. The
task invokes `--verify`, which fails unless the evidence content hash and PNG
hash equal the values manually pinned in the manifest.
