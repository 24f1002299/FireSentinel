# Data manifests

Put dataset declarations here. `datasets.json` is intentionally empty so the
download command is safe in a fresh checkout.

The legacy `datasets` shape remains supported for simple named files:

```json
{
  "name": "example.nc",
  "source_url": "https://example.org/example.nc",
  "sha256": "optional lowercase SHA-256"
}
```

Pinned source objects use the Day 7 `cases` shape. A source may specify its
full HTTPS URL, or an anonymous S3 `bucket` plus `object_key` (as produced by
GOES-18 discovery). `size_bytes` and a lowercase SHA-256 are mandatory for a
repeatable verified download.

```json
{
  "cases": [
    {
      "case_id": "pine-creek",
      "sources": [
        {
          "source_id": "c07-001",
          "bucket": "noaa-goes18",
          "object_key": "ABI-L2-CMIPF/.../object.nc",
          "size_bytes": 123456,
          "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        }
      ]
    }
  ]
}
```

Run `python -m firesentinel.data.download inspect` to summarize the verified
cache. `python -m firesentinel.data.download clean-case --case-id pine-creek`
removes that case's references and only reclaims content no other cached case
references.

## Manually audited real-event slice

`park-fire-20240725.json` is also a valid pinned-source manifest. Its single
case adds the Day 9 audit fields: a documented human review, exact initial and
later Channel 7 observations, calibrated crop policy, OpenCV parameters, and
the expected evidence/PNG SHA-256 values. The downloader uses its ordinary
`cases[].sources[]` fields; the replay reads the additional fields and never
contacts a catalog or source URL.
