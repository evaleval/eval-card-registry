---
name: registry-entity-aliases
description: >-
  Add or fix model / benchmark / metric / harness entities in the
  eval-card-registry so a raw slug resolves to the right canonical id. Use when a
  model or benchmark name lands on `no_match` or an auto-created `draft`, when
  adding aliases or a new canonical to the seed, or when an EEE adapter's ids
  won't resolve.
metadata:
  version: 0.1.0
---

# Registry entities & aliases — agent procedure

The registry maps raw model/benchmark/metric/harness strings to stable **canonical
ids**; the common task is teaching it a slug it doesn't resolve.

> **How this runs.** A person (the **operator**) runs you and can answer mid-run —
> you're not fully autonomous. Minting a new canonical id is a lasting namespace
> call: **ask the operator** first (step 3), and log any id you couldn't verify.

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

> **NEVER hand-edit `seed/models/sources/*.generated.yaml`.** Your grep in step 1
> WILL surface matches in those files — they are machine-generated and the daily
> refresh cron rewrites them, so a hand edit is silently lost. Route edits to
> `seed/models/enrichments/aliases.yaml` or `seed/models/core.yaml` per the table
> above.

**Ambiguous surface forms → `scoped_aliases`.** A form that means different
things in different data sources (e.g. `"Overall"`) must NOT be a global alias —
scope it to its source under the entry's `scoped_aliases:` map, keyed by
`source_config` (see the `scoped_aliases:` blocks in `seed/benchmarks.yaml`).
It then resolves only for that config and can't leak across sources.

## Traps
- **Don't add mechanical variants** — the `normalized` matcher already collapses case +
  all separators + dots-between-digits (one alias covers `DeepSeek-3.1` / `deepseek_3.1`
  / `deepseek-3-1`). Add an explicit alias only for forms it *can't* reach: separator
  removed (`deepseek3.1`), token reshape (`olmo-2-7-b`), semantic (`-it` / `-v2`), date
  suffixes, marketing names.
- **Normalized collisions — check BEFORE adding an alias.** The normalized index
  is last-write-wins by seed row order: if your new alias normalizes to a form
  that ALREADY resolves to a *different* canonical, one of the two silently
  breaks — either the existing entity's resolution flips to your target, or
  your new alias silently loses. Resolve the form first (`Resolver.from_parquet`
  on fresh fixtures, see Verify below); only add it if it's a `no_match` or
  already points at your target.
- **Look-alikes** — `ai2-reasoning-challenge-arc` (AI2 Reasoning Challenge, which
  carries `dataset_repo: allenai/ai2_arc`) ≠ `arc-agi` (Chollet). Confirm from the
  paper before aliasing. Note bare `arc`, `arc-c`, and `arc-e` canonicals also
  exist, pending reconciliation — don't assume `arc` is the AI2 one.

## Reference (pre-existing human docs — don't restate)
- **Id formats / casing / three-tier source of truth** → `README.md` → "## ID
  conventions" and "## How it works".
- **The reseed→gate workflow, org-split handling, pre-PR checklist** → `CONTRIBUTING.md`.

## Verify — prune stale fixtures FIRST (a stale `fixtures/` gives phantom pass/fail)
```bash
find fixtures -name '*.parquet' -delete 2>/dev/null || true; uv run eval-card-registry seed --local
```
then, BEFORE adding an alias, check it doesn't already resolve elsewhere
(last-write-wins collision — see Traps), and after adding it assert it resolves:
```python
from eval_entity_resolver import Resolver
r = Resolver.from_parquet("fixtures/")
# pre-check (on fixtures built WITHOUT your change): anything other than
# no_match / your target canonical means adding the alias would flip an
# existing resolution — stop and ask the operator.
print(r.resolve("your-raw-slug", entity_type="benchmark"))
# post-check (on fixtures rebuilt WITH your change):
assert r.resolve("your-raw-slug", entity_type="benchmark").canonical_id == "expected-id"
```
Add `tests/test_<source>_aliases.py` and run `pytest`. A PR should state which slugs
were `no_match` before and the canonical each now resolves to.
