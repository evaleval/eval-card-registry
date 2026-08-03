# Contributing

Thanks for helping keep the registry accurate. There are two kinds of
contributions:

- **Seed data (the common case).** This registry is a curated set of entities —
  orgs, models, benchmarks, metrics, harnesses — plus the aliases that resolve raw
  strings to them. Most contributions edit the YAML under `seed/`: adding a missing
  entity, correcting one, or adding an alias so a slug some tool emits resolves to
  the right canonical entity. If a name isn't resolving, or resolves to the wrong
  thing, that's a seed-data contribution. See **Contributing a seed change** below.
- **Code (generators, the resolver, the CLI/API, bug fixes).** Normal open-source
  flow: open an issue to discuss non-trivial changes, then a PR. Run `uv run pytest`
  before pushing — the same gate suite guards code changes too.

## Contributing a seed change

On PRs, the publish workflow (`.github/workflows/publish-registry-data.yml`)
runs a **dry-run job**: it reseeds from your branch and runs the full invariant
gate suite, so a change that resolves cleanly in your editor can still fail CI.
(Note the gate suite runs only on the PR dry-run — the push-triggered publish
job that fires after merge publishes without running tests, so the PR check is
the last gate your change passes.) Run the same steps locally before opening
the PR:

```bash
# 1. Reseed from a CLEAN fixture state.
#    The seed UPSERTS by id and does NOT prune rows you renamed or removed, so a
#    stale fixtures/ directory produces phantom pass/fail results. Always prune
#    first. (A bare `rm -f fixtures/*.parquet` aborts under zsh on a fresh clone —
#    the glob fails to match before rm runs — so use a portable form or
#    `--prune-stale`.)
find fixtures -name '*.parquet' -delete 2>/dev/null || true
uv run eval-card-registry seed --local

# 2. Run the FULL test suite — this is what CI runs, not just the resolver tests.
#    The gate suite is where seed regressions surface.
uv run pytest
```

Running only a subset (e.g. the resolver tests) and calling the change green is the
most common way a seed PR passes locally but fails the publish dry-run.

### What publishing actually does

Merging to `main` publishes the **flat dataset layout** of
`evaleval/entity-registry-data` — the copy consumed by the downstream producer
pipeline. The **live resolve API** (the deployed Space) reads a different,
per-table-subdirectory layout that this publish never touches: it updates only
when a maintainer runs a manual sync (`seed --prune-stale` + push). So a merged
alias fix will NOT show up on the live API until a maintainer does that — don't
re-open a PR because the hosted `/resolve` endpoint still misses your alias.

### Nuances that trip people up

**1. Attach an alias to the EXISTING canonical. Never introduce a competing
canonical, and never rename one.**

Adding a second, differently-cased canonical for the same model — or renaming an
existing canonical "to fix its casing" — orphans the oracle typed-parent edges
that point at the old id, and the gate suite fails on it
(`test_oracle_org_aware_match` and the other oracle gates). If a slug or variant
should resolve to an entity that already exists, add it as an **alias** and leave
the canonical id untouched. (What a canonical id and its aliases must look like —
HF-true casing, and when an alias is even needed — is in
[README → ID conventions](README.md#id-conventions).)

**2. Two org spellings that are the same uploader → merge; genuinely distinct
uploaders → allowlist.**

`test_no_separator_split_orgs` flags two org ids that differ only by a
separator/case (e.g. `arc-prize` vs `arcprize`). If they are the **same uploader**,
merge them by adding the other spelling as the curated org's `hf_org` and/or
`aliases` in `seed/orgs.yaml` — the generated twin then folds into the curated
row at seed time. Only add a pair to `seed/orgs_distinct_allowlist.yaml` when they
really are **two different HF uploaders**: the allowlist asserts distinctness, so
using it to silence a same-uploader split records false data.

**3. Put a model change in the right seed layer.**

Model **aliases** go in `seed/models/enrichments/aliases.yaml` (`{id, aliases}`
entries — attach the raw form to the existing canonical id). Never hand-edit
`seed/models/sources/*.generated.yaml`: those files are machine-generated and
the daily refresh cron rewrites them, so a hand edit is silently lost on the
next regeneration. Curated model entries and overrides belong in
`seed/models/core.yaml`.

### Pre-PR checklist

- [ ] `uv run eval-card-registry seed --local --prune-stale` succeeds (from a clean fixture state).
- [ ] `uv run pytest` is green (gate suite included), run from a clean tree on your branch.
- [ ] New aliases target an existing canonical; no canonical was renamed or duplicated.
- [ ] Any org-split resolution is a merge (same uploader) or a justified allowlist entry (distinct uploaders).
