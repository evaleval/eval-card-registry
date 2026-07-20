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

The publish workflow (`.github/workflows/publish-registry-data.yml`) reseeds from
your branch and runs the **full invariant gate suite** before it will publish, so
a change that resolves cleanly in your editor can still fail CI. Run the same steps
locally before opening the PR:

```bash
# 1. Reseed from a CLEAN fixture state.
#    The seed UPSERTS by id and does NOT prune rows you renamed or removed, so a
#    stale fixtures/ directory produces phantom pass/fail results. Always prune
#    first (or pass --prune-stale).
rm -f fixtures/*.parquet
uv run eval-card-registry seed --local

# 2. Run the FULL test suite — this is what CI runs, not just the resolver tests.
#    The gate suite is where seed regressions surface.
uv run pytest
```

Running only a subset (e.g. the resolver tests) and calling the change green is the
most common way a seed PR passes locally but fails the publish dry-run.

### The two rules that trip people up

**1. Attach an alias to the EXISTING canonical. Never introduce a competing
canonical, and never rename one.**

Canonical model ids are the real HF repo id, in HF-true casing. Adding a second,
differently-cased canonical for the same model — or renaming an existing canonical
"to fix its casing" — orphans the oracle typed-parent edges that point at the old
id. The gate suite fails on this (`test_oracle_org_aware_match` and the other
oracle gates). If a slug or variant should resolve to an entity that already
exists, add it as an **alias** and leave the canonical id untouched. Aliases are
gap-filling only: each form must be unclaimed by a generator canonical, and the
`normalized` strategy already collapses case and all separators, so you only need
an alias for forms normalization can't reach (a removed separator, a token
reshape, a semantic variant, a date-suffixed snapshot).

**2. Two org spellings that are the same uploader → merge; genuinely distinct
uploaders → allowlist.**

`test_no_separator_split_orgs` flags two org ids that differ only by a
separator/case (e.g. `arc-prize` vs `arcprize`). If they are the **same uploader**,
merge them by adding the other spelling as the curated org's `hf_org` and/or
`aliases` in `seed/orgs.yaml` — the generated twin then folds into the curated
row at seed time. Only add a pair to `seed/orgs_distinct_allowlist.yaml` when they
really are **two different HF uploaders**: the allowlist asserts distinctness, so
using it to silence a same-uploader split records false data.

### Pre-PR checklist

- [ ] `rm -f fixtures/*.parquet && uv run eval-card-registry seed --local` succeeds.
- [ ] `uv run pytest` is green (gate suite included), run from a clean tree on your branch.
- [ ] New aliases target an existing canonical; no canonical was renamed or duplicated.
- [ ] Any org-split resolution is a merge (same uploader) or a justified allowlist entry (distinct uploaders).
