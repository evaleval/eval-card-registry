# End-to-end verification

How to verify the query-only deployment works, from local smoke test to live HF Space.

---

## 1. Local read-only smoke test

No HF credentials or network needed. Uses `fixtures/` as the data source.

```bash
# Seed the fixtures with known entities
uv run eval-card-registry seed --local

# Start the API in read-only mode
LOCAL_MODE=true READ_ONLY=true \
  uv run uvicorn eval_card_registry.main:app --host 127.0.0.1 --port 7860
```

In a second terminal, run the curl checks below. These were verified against a
freshly seeded store (51 benchmarks, 22 metrics, 11 harnesses, 227 aliases).

### Health + stats

```bash
curl -s localhost:7860/api/v1/health | jq
# → {"status":"ok","loaded":true,"counts":{"models":0,"benchmarks":51,"metrics":22,"harnesses":11,"aliases":227}}

curl -s localhost:7860/api/v1/stats | jq
```

### Resolve (the core API)

```bash
# Exact match
curl -s -X POST localhost:7860/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value":"MATH","entity_type":"benchmark"}' | jq
# → canonical_id="math", strategy="exact", confidence=1.0, created_new=false

# Normalized match (case/punctuation collapsed) — resolves "math" via normalized
curl -s -X POST localhost:7860/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value":"MATH  ","entity_type":"benchmark"}' | jq
# → canonical_id="math", strategy="normalized" or "exact" depending on aliases

# Exact alias (display_name match) — seeded as its own benchmark
curl -s -X POST localhost:7860/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value":"MATH Level 5","entity_type":"benchmark"}' | jq
# → canonical_id="math-level-5", strategy="exact"

# No-match — in read-only mode this MUST return canonical_id=null
# (drafts are NOT auto-created, unlike the pipeline)
curl -s -X POST localhost:7860/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value":"TotallyFakeBenchmark","entity_type":"benchmark"}' | jq
# → canonical_id=null, strategy="no_match", created_new=false

# Batch
curl -s -X POST localhost:7860/api/v1/resolve/batch \
  -H 'Content-Type: application/json' \
  -d '[{"raw_value":"MATH","entity_type":"benchmark"},
       {"raw_value":"Accuracy","entity_type":"metric"}]' | jq
```

### Writes must be blocked (405)

```bash
curl -s -o /dev/null -w "POST /benchmarks -> %{http_code}\n" \
  -X POST localhost:7860/api/v1/benchmarks \
  -H 'Content-Type: application/json' \
  -d '{"id":"x","display_name":"x"}'
# → 405

curl -s -o /dev/null -w "PATCH /aliases -> %{http_code}\n" \
  -X PATCH localhost:7860/api/v1/aliases/some-id \
  -H 'Content-Type: application/json' \
  -d '{"status":"confirmed"}'
# → 405
```

### Reads

```bash
curl -s localhost:7860/api/v1/benchmarks/math | jq
curl -s "localhost:7860/api/v1/benchmarks?search=math" | jq
curl -s "localhost:7860/api/v1/aliases?entity_type=benchmark" | jq '. | length'
```

### Confirm the draft counter didn't move

```bash
uv run eval-card-registry stats --local
# benchmarks.draft should still be 0 — the no_match call above must NOT have
# auto-created a draft entity (read-only mode blocks that path).
```

---

## 2. Docker test (mirrors the Space runtime)

```bash
docker build -t evalcard-registry -f Dockerfile .

docker run --rm -p 7860:7860 \
  -e LOCAL_MODE=true \
  -e READ_ONLY=true \
  -v "$PWD/fixtures:/app/fixtures" \
  evalcard-registry
```

Re-run the curl checks from section 1 against `localhost:7860`. Same behaviour.

---

## 3. Deploy to HuggingFace Space

### Prerequisites

1. An HF account with `huggingface-cli login` completed.
2. A Space created on HF (SDK: **Docker**). Example: `your-user/evalcard-registry`.
3. A Dataset repo populated with the 5 query tables
   (`canonical_models`, `canonical_benchmarks`, `canonical_metrics`,
   `eval_harnesses`, `aliases`). The `seed` command pushes them directly
   from the local seed YAMLs — no sync required for the query-only API:

   ```bash
   # In .env:
   # LOCAL_MODE=false
   # HF_TOKEN=hf_...
   # HF_DATASET_REPO=your-user/evalcard-registry-data
   uv run eval-card-registry seed
   ```

   Running `sync --config <eee_config>` later populates `eval_results` for
   analytical use, but that table is not loaded by the query-only Space and
   is not required for deployment.

4. (Optional) An HF Storage Bucket for request logs.

### Push the Space

```bash
export HF_SPACE_REPO=your-user/evalcard-registry
./deploy/push-to-space.sh
```

### Configure Space secrets/variables

In the Space UI → **Settings → Variables and secrets**:

| Name | Type | Value |
|---|---|---|
| `HF_DATASET_REPO` | Variable | `your-user/evalcard-registry-data` |
| `HF_TOKEN` | Secret | Token with read access to the dataset (and write to the log bucket, if used) |
| `HF_LOG_BUCKET` | Secret | Bucket id, e.g. `your-user/evalcard-registry-logs` (optional) |

`READ_ONLY=true` and `LOCAL_MODE=false` are baked into the Dockerfile.

### Verify the live Space

```bash
SPACE=https://your-user-evalcard-registry.hf.space

curl -s $SPACE/api/v1/health | jq
# → loaded=true, non-zero entity counts

curl -s -X POST $SPACE/api/v1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"raw_value":"MATH","entity_type":"benchmark"}' | jq

curl -s -o /dev/null -w "%{http_code}\n" -X POST $SPACE/api/v1/benchmarks \
  -H 'Content-Type: application/json' -d '{"id":"x","display_name":"x"}'
# → 405
```

If you configured `HF_LOG_BUCKET`, wait ~5 minutes (the default
`LOG_FLUSH_INTERVAL_SECONDS`) and check the bucket for a new parquet file at
`api_resolve_log/part-YYYYMMDDTHHMMSS.parquet`.

---

## What "it works" means

All of these must be true:

- [ ] `/api/v1/health` reports `loaded=true` with non-zero counts.
- [ ] Resolve returns the expected canonical id for a seeded value.
- [ ] Resolve returns `canonical_id=null` for an unknown value (no draft created).
- [ ] `POST`/`PATCH` on entity + alias routes return `405`.
- [ ] Entity GETs (`/benchmarks`, `/benchmarks/{id}`, `/benchmarks?search=...`)
      return `200` and serialize null columns as JSON `null`.
- [ ] `stats --local` shows no change in draft counts after a no-match resolve.
- [ ] (Optional) Resolve logs land in the HF Storage Bucket within
      `LOG_FLUSH_INTERVAL_SECONDS`.
