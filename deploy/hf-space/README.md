---
title: eval-card-registry
emoji: 🗂️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# eval-card-registry

Query-only disambiguation API for AI evaluation entity names. Resolves raw benchmark / model / metric / harness strings (e.g. `"MATH Level 5"`) to stable canonical IDs (`math`).

This Space runs in **read-only mode** — it serves lookups against pre-built entity data. Write operations (entity creation, alias edits) happen in a separate pipeline.

## Base URL

```
https://evaleval-entity-registry.hf.space/api/v1
```

## Resolve

```bash
curl -X POST https://evaleval-entity-registry.hf.space/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "MATH Level 5", "entity_type": "benchmark"}'
```

Response (a type-agnostic core + `ancestry` + a typed `resolution_detail`):

```json
{
  "raw_value": "MATH Level 5",
  "entity_type": "benchmark",
  "canonical_id": "math-level-5",
  "strategy": "exact",
  "confidence": 1.0,
  "created_new": false,
  "resolution_source": null,
  "review_status": "reviewed",
  "ancestry": [],
  "resolution_detail": {"level": "benchmark", "matched_subset": "MATH Level 5"}
}
```

`resolution_detail` is a typed sub-object keyed by `entity_type`: `model` →
`{granularity, hf_repo_id}`, `benchmark` → `{level: composite|family|benchmark|slice, matched_subset}`,
others → `{}`. The 10 top-level fields are always present (null when nothing to
report); only `resolution_detail`'s inner keys vary by type, and it is `{}` on a
no-match. If nothing matches, `canonical_id` is `null` and `strategy` is
`"no_match"`. In read-only mode, no draft entity is created.

`entity_type` is one of: `benchmark`, `model`, `metric`, `harness`, `org`, `composite`, `family`. Optional `source_config` scopes the lookup to a specific source. A model's `canonical_id` is the real HF repo id (e.g. `meta-llama/Llama-3.1-8B-Instruct`); `ancestry` carries its group/family membership.

**Batch resolve:**

```bash
curl -X POST https://evaleval-entity-registry.hf.space/api/v1/resolve/batch \
  -H 'Content-Type: application/json' \
  -d '[
    {"raw_value": "MATH Level 5", "entity_type": "benchmark"},
    {"raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"}
  ]'
```

## Browse entities

```
GET /api/v1/benchmarks?search=math
GET /api/v1/benchmarks/{id}
GET /api/v1/models
GET /api/v1/metrics
GET /api/v1/harnesses
GET /api/v1/families/{id}        # canonical_families (a benchmark ancestry target)
GET /api/v1/composites/{id}      # canonical_composites
GET /api/v1/aliases?status=uncertain&entity_type=benchmark
```

## Health

```
GET /api/v1/health
GET /api/v1/stats
```

## Write endpoints

Disabled in this Space. `POST`/`PATCH` on entities and aliases return `405 Method Not Allowed`. Mutations happen in the data pipeline (separate from this Space).

## Interactive docs

OpenAPI docs at `/docs`.

## Data sources

- Entity data: HF Dataset repo `evaleval/entity-registry-data` (read at startup)
- Resolve logs: HF Storage Bucket `evaleval/entity-registry-storage` (written asynchronously for resolver improvement)
