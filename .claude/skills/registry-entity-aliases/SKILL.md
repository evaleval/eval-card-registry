---
name: registry-entity-aliases
description: >-
  Add or fix model / benchmark / metric / harness entities in the
  eval-card-registry so a raw slug resolves to the right canonical id. Use when a
  model or benchmark name lands on `no_match` or an auto-created `draft`, when
  adding aliases or a new canonical to the seed, or when an EEE adapter's ids
  won't resolve.
license: MIT
metadata:
  version: 0.1.0
---

# Registry entities & aliases — agent procedure

The registry maps raw model/benchmark/metric/harness strings to stable **canonical
ids**; the common task is teaching it a slug it doesn't resolve.

## Procedure
1. **Search the seed first** — `grep -ri <slug> seed/` for an existing canonical.
2. If one exists, **add your raw form to its `aliases`** ("upstream an alias") — the common case.
3. **New canonical only if genuinely absent — and ask the operator first.** Minting a
   canonical is a lasting namespace decision; don't do it silently.
4. **Verify or flag ids** — resolve against the registry (`POST /resolve`), or record
   each id as "unverified — maintainer confirm"; never present an assumed id as canonical.

## Where things go
| Adding… | File |
|---|---|
| A **model** alias | `seed/models/enrichments/aliases.yaml` (`- id: <canonical HF repo id>` + `aliases: [...]`) |
| A **model** canonical / override | `seed/models/core.yaml` |
| A **benchmark** alias or new canonical | `seed/benchmarks.yaml` (inline `aliases:`; new entry needs `id`, `display_name`, `dataset_repo`, `tags`, `review_status`, `aliases`) |
| A **metric** / **harness** alias or canonical | `seed/metrics.yaml` / `seed/harnesses.yaml` |

## Traps
- **Don't add mechanical variants** — the `normalized` matcher already collapses case +
  all separators + dots-between-digits (one alias covers `DeepSeek-3.1` / `deepseek_3.1`
  / `deepseek-3-1`). Add an explicit alias only for forms it *can't* reach: separator
  removed (`deepseek3.1`), token reshape (`olmo-2-7-b`), semantic (`-it` / `-v2`), date
  suffixes, marketing names.
- **Look-alikes** — `arc` (AI2 Reasoning Challenge, `allenai/ai2_arc`) ≠ `arc-agi`
  (Chollet). Confirm from the paper before aliasing.

## Reference (pre-existing human docs — don't restate)
- **Id formats / casing / three-tier source of truth** → `README.md` → "## ID
  conventions" and "## How it works".
- **The reseed→gate workflow, org-split handling, pre-PR checklist** → `CONTRIBUTING.md`.

## Verify — prune stale fixtures FIRST (a stale `fixtures/` gives phantom pass/fail)
```bash
rm -f fixtures/*.parquet && uv run eval-card-registry seed --local
```
then assert every slug resolves:
```python
from eval_entity_resolver import Resolver
r = Resolver.from_parquet("fixtures/")
assert r.resolve("your-raw-slug", entity_type="benchmark").canonical_id == "expected-id"
```
Add `tests/test_<source>_aliases.py` and run `pytest`. A PR should state which slugs
were `no_match` before and the canonical each now resolves to.
