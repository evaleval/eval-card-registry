# Third-party data attributions

This project includes data derived from the following third-party sources.
Their license terms apply to that derived data; the rest of this repository
is governed by its own license.

## cfahlgren1/hub-stats

Model lineage and metadata in
`seed/models/sources/hub_stats.generated.yaml` are derived from the
[cfahlgren1/hub-stats](https://huggingface.co/datasets/cfahlgren1/hub-stats)
dataset, a continuously-refreshed snapshot of HuggingFace Hub model
metadata.

- **Source:** https://huggingface.co/datasets/cfahlgren1/hub-stats
- **License:** Apache-2.0

The generator script (`scripts/refresh_from_hub_stats.py`) issues a
DuckDB query over the dataset's parquet files for HF model rows whose
`id` is already aliased to one of our canonicals, and emits enrichment
entries that backfill `release_date`, `params_billions`, `license`,
and useful tags via the seed loader's field-level merge. The
lineage-descendant pre-load (community quants/finetunes via
`baseModels` chain) is deferred — on-demand enrichment at draft
creation will handle that case without bulk pre-loading.

## models.dev

Model seed data in `seed/models/sources/models_dev.generated.yaml` and
`seed/models/sources/models_dev_catalog.generated.yaml` is generated from
[models.dev](https://models.dev), an open-source community-maintained
database of AI model specifications.

- **Source:** https://models.dev (https://github.com/anomalyco/models.dev)
- **License:** MIT
- **Copyright:** © 2025 models.dev

The generator script (`scripts/refresh_from_modelsdev.py`) fetches
`https://models.dev/api.json`, filters to known model-author providers,
and emits one canonical per snapshot/variant with typed `parents` edges
back to the family root (multi-level chains for compound suffixes like
`-instruct-v0.3`). It writes two derived files from the same fetch:
`models_dev.generated.yaml` (the re-cased model-author output) and, via
`--catalog`, `models_dev_catalog.generated.yaml` (the full-catalog split:
fresh mints for models.dev-only / not-on-HF models plus alias-only
enrichments for models that already exist as HF canonicals). Both runs
dedup against the full canonical universe — including `seed/models/core.yaml`
— so they only add coverage and never clobber a curated id. Hand-curated
additions and overrides live in `seed/models/core.yaml` (canonical entities)
and `seed/models/enrichments/aliases.yaml` (alias-only additions); both
win over the generated sources via the seed loader's field-level merge.

```
MIT License

Copyright (c) 2025 models.dev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## HuggingFace Hub (model-id oracle)

`seed/models/sources/hf_oracle.generated.yaml` (Tier-1 canonical models) and the
resolution oracle `hf_model_id_resolution.json` are derived from public model
metadata on the HuggingFace Hub.

- **Source:** https://huggingface.co/ (Hub model repositories + metadata)
- **Use:** only repository ids / metadata are referenced; each model remains
  under its own repository license.

## AIR-Bench 2024 (safety taxonomy)

`seed/benchmarks_generated/air_bench.yaml` encodes the AIR-Bench 2024 AI-risk
taxonomy (category names/ids).

- **Source:** AIR-Bench 2024 — Stanford CRFM (`stanford-crfm/air-bench-2024`)
- **Use:** taxonomy category labels only; refer to the upstream project for its
  license terms.
