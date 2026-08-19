# Data manifests

Put dataset declarations here. `datasets.json` is intentionally empty so the
download command is safe in a fresh checkout.

Each dataset entry uses this shape:

```json
{
  "name": "example.nc",
  "source_url": "https://example.org/example.nc",
  "sha256": "optional lowercase SHA-256"
}
```
