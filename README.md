# eval-card-registry

Entity resolution registry for AI evaluation data. Maps raw model, benchmark, metric, and harness names from the EEE datastore to stable canonical IDs, and stores resolved evaluation results in a flat mapping table (`eval_results`).

---

## Quickstart

Resolve a raw string against the hosted registry:

```bash
curl -X POST https://evaleval-entity-registry.hf.space/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "MATH Level 5", "entity_type": "benchmark"}'
```

```json
{"canonical_id": "math-level-5", "strategy": "exact", "confidence": 1.0, "created_new": false, "review_status": "reviewed"}
```

`entity_type` is one of `benchmark`, `model`, `metric`, `harness`. See the [API section](#api) for batch resolve, entity browsing, and the full endpoint list.

---

## Local development

```bash
git clone <repo>
cd eval-card-registry
uv sync
cp .env.example .env          # defaults work for local dev
```

**1. Seed the registry with known entities:**

```bash
uv run eval-card-registry seed --local
```

This loads benchmarks, metrics, and harnesses from `seed/*.yaml` into `fixtures/*.parquet`. You should see counts printed for each entity type. Note that these are automatically generated placeholders for internal development and will likely be changed in the future.

**2. Check what's in the registry:**

```bash
uv run eval-card-registry stats --local
```

Expected output:

```
  models      total=0  draft=0
  benchmarks  total=61  draft=0
  metrics     total=22  draft=0
  harnesses   total=11  draft=0

  aliases        total=403  uncertain=0
  eval_results   total=0
  resolution_log total=0
  sync_runs      total=0
```

**3. Sync an EEE config — resolve entities and populate the mapping table:**

```bash
uv run eval-card-registry sync --config hfopenllm_v2 --local
```

This downloads the EEE dataset config from HuggingFace (first run will take a few minutes), resolves every raw string to a canonical entity, and writes results to `fixtures/eval_results.parquet` — the mapping table (one row per model × benchmark × metric result).

**4. Verify results:**

```bash
uv run eval-card-registry stats --local
```

You should now see `eval_results`, `aliases`, and entity counts populated. Each row in `eval_results` looks like:

```json
{
  "evaluation_id": "hfopenllm_v2/...",
  "result_index": 0,
  "source_config": "hfopenllm_v2",
  "model_id": "meta-llama/Llama-3.1-8B",
  "harness_id": "lm-evaluation-harness",
  "benchmark_id": "ifeval",
  "parent_benchmark_id": null,
  "metric_id": "accuracy",
  "benchmark_card_id": null,
  "score": 0.42,
  "score_details": "{\"score\": 0.42}"
}
```

---

## How it works

Raw strings from EEE (e.g. `"MATH Level 5"`, `"lm-evaluation-harness"`) are resolved to canonical IDs (`math`, `lm-evaluation-harness`) through a strategy chain: exact alias match → normalized match (collapses case + all separators — spaces, hyphens, underscores, and slashes) → fuzzy stem match (strips known suffixes like `-fc`/`-prompt`, normalizes org prefixes) → auto-create draft. Every resolution is logged with its strategy and confidence score.

Canonical entities start as `draft` and can be promoted to `reviewed`. Aliases that fall below the confidence threshold are flagged `uncertain` for human review.

---

## Project layout

```
eval-card-registry/
├── packages/eval-entity-resolver/   # Standalone resolver package (uv workspace member)
├── src/eval_card_registry/          # FastAPI service + CLI
│   ├── api/                         # Route handlers
│   ├── services/                    # resolution_service, ingestion pipeline
│   └── store/                       # In-memory store backed by HF Dataset parquet
├── seed/                            # Known benchmarks, metrics, harnesses (YAML)
├── fixtures/                        # Local parquet files for offline dev/tests
└── tests/
```

The service and the resolver package are separate. The resolver can be imported directly by other pipelines (e.g. AutoBenchmarkCard) without pulling in the full service.

---

## CLI reference

All commands require `uv run` prefix (or install the package first with `uv pip install -e .`).

```bash
# Seed known entities from seed/ YAML files
uv run eval-card-registry seed --local

# Print entity counts, draft counts, uncertain aliases
uv run eval-card-registry stats --local

# Sync one EEE config — resolves entities, writes to eval_results table
uv run eval-card-registry sync --config hfopenllm_v2 --local

# Sync all configs
uv run eval-card-registry sync --all --local

# Re-resolve everything (after updating seed data or fuzzy matching logic)
uv run eval-card-registry sync --config hfopenllm_v2 --rerun --local
```

Drop `--local` and configure `.env` with HF credentials to read/write from HF Hub instead of `fixtures/`.

---

## API

Start the server:

```bash
LOCAL_MODE=true uv run uvicorn eval_card_registry.main:app --reload
```

Base path: `http://localhost:8000/api/v1`

**Resolve a raw string:**

```bash
curl -X POST http://localhost:8000/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value": "MATH Level 5", "entity_type": "benchmark", "source_config": "hfopenllm_v2"}'
```

```json
{
  "canonical_id": "math-level-5",
  "strategy": "exact",
  "confidence": 1.0,
  "created_new": false,
  "review_status": "reviewed"
}
```

**Batch resolve:**

```bash
POST /api/v1/resolve/batch
Body: [{ "raw_value": "...", "entity_type": "..." }, ...]
```

**Entity CRUD** (models, benchmarks, metrics, harnesses):

```
GET    /api/v1/benchmarks?search=math&review_status=draft
GET    /api/v1/benchmarks/{id}
POST   /api/v1/benchmarks
PATCH  /api/v1/benchmarks/{id}
```

Model IDs containing `/` (e.g. `meta-llama/Llama-3.1-8B`) work in path params directly.

**Aliases:**

```
GET    /api/v1/aliases?status=uncertain&entity_type=benchmark
PATCH  /api/v1/aliases/{id}   # confirm, reject, or correct an alias
```

**Health and stats:**

```
GET  /api/v1/health
GET  /api/v1/stats
```

Interactive docs at `http://localhost:8000/docs`.

---

## Using the resolver standalone

The `eval-entity-resolver` package can be used independently — no service required:

```python
from eval_entity_resolver import AliasStore, Resolver, ResolverConfig

store = AliasStore.from_hf("org/eval-card-registry")
# or locally:
store = AliasStore.from_parquet("./fixtures/")

resolver = Resolver(store, ResolverConfig(threshold=0.85))

result = resolver.resolve(
    raw_value="MATH",
    entity_type="benchmark",
    source_config="hfopenllm_v2",
)
# result.canonical_id  — None if no match
# result.strategy      — "exact" | "normalized" | "fuzzy" | "no_match"
# result.confidence    — 0.0–1.0
```

Install from this workspace:

```bash
uv add eval-entity-resolver --workspace
```

---

## Tests

```bash
uv run pytest
```

Tests use the in-memory fixture store — no HF credentials or network needed.

---

## Resolution behaviour

| Alias status | Meaning |
|---|---|
| `auto` | Resolved above confidence threshold — no review needed |
| `uncertain` | Below threshold or no match — auto-created draft, flagged for review |
| `confirmed` | Manually verified |
| `rejected` | Wrong match, excluded from future resolution |

Resolution order for a given `(entity_type, raw_value)`:

1. Config-scoped alias (`source_config` matches)
2. Global alias (`source_config` is null)
3. Resolver chain (exact → normalized → fuzzy → auto-create draft)

Resolving the same raw string twice returns the same canonical ID. Re-running with `--rerun` re-evaluates existing aliases — prior resolution log entries are preserved.

---

## ID conventions

| Entity | Format | Example |
|---|---|---|
| Model (HF) | `{org}/{model}` | `meta-llama/Llama-3.1-8B` |
| Model (non-HF) | `{org}:{slug}` | `anthropic:claude-opus-4-5` |
| Benchmark / Metric / Harness | lowercase slug | `math`, `lm-evaluation-harness` |
| `eval_results` row ID | `sha256(evaluation_id:result_index)[:16]` | `a3f2b1c9d4e5f678` |

Entity IDs use human-readable slugs (not hashes) because they appear in seed files, API responses, and are referenced during manual curation. Internal row IDs (like `eval_results.id`) use deterministic hashes for uniform length and collision resistance.

---

## HF Hub deployment

For production, configure `.env`:

```
LOCAL_MODE=false
HF_TOKEN=hf_...
HF_DATASET_REPO=org/eval-card-registry
```

Then run the same commands without `--local`:

```bash
uv run eval-card-registry seed
uv run eval-card-registry sync --config hfopenllm_v2
```

Data is stored as one parquet config per table in the HF Dataset repo.

---

## HF Space deployment (query-only API)

The service can be deployed to a HuggingFace Space as a **query-only** disambiguation API — read-only resolve + entity/alias GETs, no writes.

**Architecture:**
- **Space** (`evaleval/entity-registry`) — Docker SDK, runs FastAPI on port 7860
- **Dataset repo** (`evaleval/entity-registry-data`) — entity parquet tables, read at startup
- **Storage Bucket** (`evaleval/entity-registry-storage`) — async resolve logs, written periodically

**Read-only mode behaviour:**
- `POST /resolve` runs the full resolver chain but does NOT auto-create draft entities or write aliases on no_match — `canonical_id` is `null`
- `POST`/`PATCH` entity + alias endpoints return `405 Method Not Allowed`
- Only 5 tables (models, benchmarks, metrics, harnesses, aliases) are loaded — `eval_results`, `resolution_log`, `sync_runs` are skipped
- Every resolve request is logged asynchronously to the Storage Bucket (buffered in memory, flushed every 5 min as partitioned parquet)

**Deploy:**

```bash
# Prerequisites: create the Space, Dataset repo, and Storage Bucket on HF;
# seed + sync the Dataset repo with entity data locally first.

bash deploy/push-to-space.sh
```

Configure the Space in HF Space Settings:

| Variable | Type | Value |
|---|---|---|
| `HF_TOKEN` | Secret | Token with read access to dataset + write access to log bucket |
| `HF_DATASET_REPO` | Variable | `evaleval/entity-registry-data` |
| `HF_LOG_BUCKET` | Variable | `evaleval/entity-registry-storage` |

`READ_ONLY=true` and `LOCAL_MODE=false` are set in the Dockerfile ENV.

**Local test of read-only mode:**

```bash
READ_ONLY=true LOCAL_MODE=true uv run uvicorn eval_card_registry.main:app --reload
```

See `deploy/END_TO_END.md` for a step-by-step verification guide (local smoke
test, Docker test, Space deploy + checks).

---

## TODO
- Combine logic with EEE codebase's model registry and evalcard backend metric registry
- Verify metric extraction logic — although likely partially addressed with future schema versions and fixes.
- Clean up how we implement registry updates + check against regression
- Populate `benchmark_card_id` once an auto-benchmarkcard has been generated and linked for each benchmark.