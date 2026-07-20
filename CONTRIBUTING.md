# Contributing entities & aliases

This registry maps raw model / benchmark / metric / harness strings (as they
appear in the EEE datastore) to stable **canonical ids**. The most common
outside contribution is teaching it a raw slug it doesn't yet resolve — usually
while building an EEE adapter and finding your model or benchmark names land on
`no_match` or an auto-created `draft`. This guide is the how-to for that.

For *how resolution works* (the exact strategy chain and the resolve response
shape) see [README.md](README.md) → **How it works** and **Using the resolver
standalone**. This doc is about **what to add, where, and how to verify it**.

## The one rule: search first, alias to the existing canonical

A raw slug resolves through: exact alias → **normalized** (collapses case + all
separators) → fuzzy stem → auto-create `draft`. Before adding anything:

1. **Search** the seed for an existing canonical that already means your entity
   (`grep -i` the slug and its obvious variants in `seed/`).
2. If one exists, **add your raw form to that canonical's `aliases`** ("upstream
   an alias"). This is the common case.
3. **Create a new canonical only if the entity is genuinely absent.** A spurious
   new canonical fragments the data (two ids for one thing) — worse than a
   missing alias.

Aliases are **gap-filling only**: each form you add must be *unclaimed* by any
generator canonical (it resolved to `no_match`/`draft` before). The loader
UNIONs aliases onto the matching canonical (case-insensitive dedup), so an alias
that collides with an existing canonical is a bug, not a merge.

## Where things go

| You're adding… | File | Shape |
|---|---|---|
| A **model** alias | `seed/models/enrichments/aliases.yaml` | `- id: <canonical HF repo id>` then `aliases: [<raw form>]` |
| A **model** canonical / override | `seed/models/core.yaml` | curated override floor (see S6) |
| A **benchmark** alias **or** new canonical | `seed/benchmarks.yaml` | inline `aliases:` on the entry; new entry gets `id`, `display_name`, `dataset_repo`, `tags`, `review_status`, `aliases` |
| A benchmark's source dataset | `seed/benchmarks.yaml` → `dataset_repo` | the HF repo the benchmark's data lives in (verify HTTP 200) |

Model canonical ids and benchmark ids use **different schemes** — see S1 and S7.

## Don't enumerate mechanical variants

The `normalized` strategy already collapses **case + all separators**
(`-` / `_` / space / `/`) **+ dots-between-digits**, so a single alias covers a
whole family of typographic variants. Verified: from the one alias
`deepseek-3.1`, all of `DeepSeek-3.1`, `deepseek_3.1`, `deepseek 3.1`, and
`deepseek-3-1` resolve.

Add an explicit alias **only** for forms normalization *cannot* reach:
- **separator removed** (`deepseek3.1`), **token reshape** (`olmo-2-7-b`),
- **semantic** (`-instructed` → `-it`, `-v2`),
- **date suffix** (`-0905`, `-2026-03-05`) — deliberately **not** stripped (S5),
- **marketing / display names** (`Claude 3.5 Sonnet (2024-06-20)`).

## Disambiguate look-alikes

Similar names can be *different datasets*. `arc` (AI2 Reasoning Challenge,
`allenai/ai2_arc`) is **not** `arc-agi` (Chollet's Abstraction & Reasoning
Corpus). Confirm the identity from the paper / dataset card before aliasing —
an alias onto the wrong canonical silently mismerges results.

## Standards (S1–S7)

Extracted from the README + seed loader; a contribution should provably fit them.

| # | Standard |
|---|---|
| **S1** | **Model `canonical_id` = the real HF repo id, HF-true casing** (e.g. `allenai/OLMo-2-0325-32B`). Do not lowercase or reshape it. |
| **S2** | **Closed / off-HF models** → `{org}/{name}` with a **lowercase org**, `hf_repo_id: null` (e.g. `anthropic/claude-opus-4.6`). Keep these in curated seed, not generator output. |
| **S3** | **Three-tier source of truth:** HuggingFace (authoritative) > models.dev > name-based tier3. **HF casing wins**; use `skip_source_ids` to drop a lower-tier id that fights an HF-cased curated entry. |
| **S4** | **Aliases are gap-filling only** — each form must be UNCLAIMED by a generator canonical; the loader UNIONs them (case-insensitive dedup). |
| **S4a** | **Don't add mechanical variants** — `normalized` already collapses case + all separators + dots-between-digits. Add only forms it can't reach (see above). |
| **S5** | **Date suffixes are deliberately NOT stripped** (they preserve a snapshot's `release_date`), so dated slugs need an explicit alias. |
| **S6** | `seed/models/core.yaml` is the **minimal override floor**; `skip_source_ids` drops a bad source id "when models.dev ships bad data for a model core curates correctly". |
| **S7** | **Benchmarks / metrics / harnesses use lowercase-hyphenated slugs** (`math-level-5`, `lm-evaluation-harness`) — a different scheme from model ids (S1). |

## Verify before opening a PR

Use the repo's own toolchain (Python 3.11 works for dev):

```bash
eval-card-registry seed --local           # -> "Seed complete"
```

Then confirm **every** slug you care about resolves to the intended canonical:

```python
from eval_entity_resolver import Resolver          # standalone package
r = Resolver.from_parquet("fixtures/")             # after seed --local
assert r.resolve("your-raw-slug", entity_type="benchmark").canonical_id == "expected-id"
```

Finally run the suite (it's fast) and add a small test asserting your slugs
resolve (e.g. `tests/test_<source>_aliases.py`, mirroring the existing
alias-resolution tests):

```bash
pytest
```

A PR that adds aliases should state: which slugs were `no_match` before, the
canonical each now resolves to, and any data-quality fix (e.g. an HF-casing
correction) made along the way.
