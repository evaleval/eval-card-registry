"""Empty DataFrame schemas for each table."""
import pandas as pd


_SCHEMAS: dict[str, dict] = {
    "canonical_orgs": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "parent_org_id": pd.StringDtype(),
        "website": pd.StringDtype(),
        "hf_org": pd.StringDtype(),
        # Org category — `lab` (curated first-party), `community` (multi-model
        # HF org), `individual` (single-person account), `unknown` (auto-
        # created, undecided). Consumers wanting "real labs only" filter on
        # `lab` or on `review_status = reviewed`.
        "kind": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_models": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "developer": pd.StringDtype(),
        "org_id": pd.StringDtype(),
        "family": pd.StringDtype(),
        "architecture": pd.StringDtype(),
        "params_billions": "float64",
        # JSON-encoded list of parent edges:
        #   [{"id": "...", "relationship": "variant|finetune|quantized|merge|adapter",
        #     "axis": "size|mode|modality|domain|version|training_stage|tier"}]
        # `relationship: quantized` covers all inference-precision variants
        # (-turbo, -fp8, -int4, -awq, -gguf, etc.) — matches hub-stats's
        # baseModels.relation values and keeps the source/registry encoding
        # consistent. Quants are NOT a sub-axis of `variant`.
        # `axis` is optional and only meaningful for `relationship: variant`.
        # Multi-element lists support merges (multiple parents share the
        # `merge` relationship) and the rare case of a model that's both a
        # family variant AND a finetune of an external base.
        "parents": pd.StringDtype(),
        # Identity-group root: walk `parents` up following identity-
        # preserving edges (`quantized` + `variant axis=version`). NULL when
        # self has no such ancestor (i.e. self IS the group root). Resolver
        # default-returns this when set; callers wanting the leaf get the
        # un-collapsed canonical id in `resolved_leaf_id`.
        "model_group_id": pd.StringDtype(),
        # Family-release root: fold the versioned release line (version,
        # quantized, mode, training_stage, size, tier), stopping at
        # finetune/merge and major-version boundaries. NULL when self IS the
        # family root. Populated by `derive_model_lineage_fields`.
        "model_family_id": pd.StringDtype(),
        # Deepest non-`variant` ancestor's id (what this model was built
        # from). NULL when self IS the origin. Populated by
        # `derive_model_lineage_fields`.
        "lineage_origin_model_id": pd.StringDtype(),
        # Denormalized: `org_id` of the deepest non-`variant` ancestor in
        # `parents`. For Meta-originated models = self.org_id. For
        # finetunes/quants of someone else's weights = the upstream lab's
        # org_id. Recomputed on every refresh; treat as a cache.
        "lineage_origin_model_org_id": pd.StringDtype(),
        # Provenance enum {hf|models_dev|curated|inferred|none} of how this
        # canonical was minted. Null by default.
        "resolution_source": pd.StringDtype(),
        # Granularity enum {variant|group|family} this canonical resolves
        # at. Null by default.
        "resolution_granularity": pd.StringDtype(),
        # Open vs closed weights. NULL when unknown. Populated:
        #   - models.dev refresh   → from `open_weights` field directly
        #   - hub-stats refresh    → True iff safetensors/gguf data present
        #   - live hub-stats lookup → same inference as bulk refresh
        # Closed-API models (Anthropic / OpenAI / Google) get False from
        # models.dev. Hand-curated core.yaml entries may set explicitly.
        "open_weights": pd.BooleanDtype(),
        # ISO date (YYYY-MM-DD or YYYY-MM) of the family's earliest known
        # snapshot. Populated automatically by scripts/refresh_from_modelsdev.py
        # from models.dev. Per-snapshot dates remain in metadata.release_dates.
        "release_date": pd.StringDtype(),
        # JSON-encoded list of input modality strings (e.g. ["text", "image"]).
        # NULL when unknown. Populated by scripts/refresh_from_modelsdev.py
        # from models.dev's `modalities.input` field. Hand-curated core.yaml
        # entries may set explicitly. Frontend's models_view + the
        # eval_results_view.model_info.modalities STRUCT both consume this.
        "input_modalities": pd.StringDtype(),
        # JSON-encoded list of output modality strings (e.g. ["text"]).
        # Same population path as input_modalities.
        "output_modalities": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_benchmarks": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "description": pd.StringDtype(),
        "dataset_repo": pd.StringDtype(),
        "parent_benchmark_id": pd.StringDtype(),
        "tags": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Multi-benchmark families — curated groupings where multiple
    # canonical benchmarks share an object-of-measurement or
    # methodological lineage (e.g. mmlu family contains mmlu + mmlu-pro).
    # Singletons default to family.id == benchmark.id at the producer
    # layer and don't appear here.
    "canonical_families": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        # Single-valued category from a curated enum (general, agentic,
        # reasoning, knowledge, multimodal, tool-use, math, security,
        # factuality, reward-modelling, safety, code, instruction-
        # following, other). Optional; null defaults to "other" downstream.
        "category": pd.StringDtype(),
        # JSON-encoded list of canonical benchmark ids that belong to
        # this family. Validated at seed time: each benchmark may belong
        # to at most one curated family.
        "benchmark_ids": pd.StringDtype(),
        # Optional: the benchmark id whose primary metric is the
        # family-level rollup (e.g. artificial-analysis →
        # artificial-analysis-intelligence-index). Null when none is
        # designated.
        "primary_benchmark_key": pd.StringDtype(),
        # JSON-encoded list of EEE folder names (source_config values)
        # that resolve to this family. Ports the reference script's
        # EXPLICIT_FAMILY_MAP into the registry. The producer joins on
        # this when classifying raw EEE folders.
        "folder_aliases": pd.StringDtype(),
        # JSON-encoded list of composite slugs nested under this family.
        # Mirrors composites.yaml's structure from the family side; lets
        # callers walk family → composites without scanning composites.
        "composite_keys": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Composites — named leaderboard surfaces that aggregate one or
    # more EEE source_configs under a unified presentation (e.g. HELM
    # Classic, HAL Leaderboard, Open LLM Leaderboard v2). Identity is
    # the slug; one-to-one mappings (slug == single source_config)
    # don't strictly require an entry here, but curated entries can
    # override display name and group multiple configs.
    "canonical_composites": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "category": pd.StringDtype(),
        # JSON-encoded list of EEE source_config values that belong to
        # this composite.
        "source_configs": pd.StringDtype(),
        # Optional: family this composite is nested under in the tree.
        # Null when the composite IS its own top-level (most cases today).
        "family_id": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_metrics": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "score_type": pd.StringDtype(),
        "lower_is_better": "bool",
        "min_score": "float64",
        "max_score": "float64",
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "eval_harnesses": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "version": pd.StringDtype(),
        "fork_url": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    # Inference platforms — author labs, inference hosts, aggregator
    # gateways, regional variants, and coding plans. The host-token
    # spellings (`fireworks/`, `-bedrock`, `azure/`, …) live in the JSON-
    # encoded `aliases` list and are inverted into the single-sourced
    # host-token → platform map by lib/inference_platforms_map.py.
    "canonical_inference_platforms": {
        "id": pd.StringDtype(),            # PK; = models.dev provider slug where one exists
        "display_name": pd.StringDtype(),
        # Closed enum: author_lab | inference_platform |
        # aggregator_gateway | regional_variant | coding_plan
        "kind": pd.StringDtype(),
        "aliases": pd.StringDtype(),       # JSON-encoded list of host-token spellings
        "canonical_org": pd.StringDtype(), # FK→canonical_orgs.id when kind=author_lab; nullable
        "variant_of": pd.StringDtype(),    # base platform id for regional_variant/coding_plan; nullable
        "homepage": pd.StringDtype(),      # provenance (models.dev / provider doc URL); nullable
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "aliases": {
        "id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "canonical_id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "source_field": pd.StringDtype(),
        "status": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "notes": pd.StringDtype(),
        # FK→canonical_inference_platforms.id; provenance of this spelling's
        # serving platform. Populated from fuzzy host capture / models.dev
        # provider; null by default.
        "inference_platform": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "resolution_log": {
        "id": pd.StringDtype(),
        "sync_run_id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "canonical_id": pd.StringDtype(),
        "created_new": "bool",
        "timestamp": pd.StringDtype(),
    },
    "eval_results": {
        "id": pd.StringDtype(),            # hash(evaluation_id + result_index)
        "evaluation_id": pd.StringDtype(), # from EEE record
        "result_index": "Int64",           # position in evaluation_results array
        "source_config": pd.StringDtype(),
        "model_id": pd.StringDtype(),      # resolved canonical ID
        "harness_id": pd.StringDtype(),    # resolved canonical ID (nullable)
        "benchmark_id": pd.StringDtype(),  # resolved canonical ID
        "parent_benchmark_id": pd.StringDtype(),  # nullable
        "metric_id": pd.StringDtype(),     # resolved canonical ID (nullable)
        "benchmark_card_id": pd.StringDtype(),  # FK to auto-benchmarkcard output (nullable)
        "score": "float64",
        "score_details": pd.StringDtype(), # JSON-encoded dict
        # Per-run serving platform (FK→canonical_inference_platforms.id).
        # Determined per resolution: matched provider spelling, else a raw-id
        # host token, else null. Null by default.
        "inference_platform": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "sync_runs": {
        "id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "started_at": pd.StringDtype(),
        "completed_at": pd.StringDtype(),
        "status": pd.StringDtype(),
        "rerun": "bool",
        "entities_created": "Int64",
        "entities_updated": "Int64",
        "aliases_created": "Int64",
        "aliases_updated": "Int64",
        "errors": pd.StringDtype(),  # JSON-encoded list
    },
    # Periodically-refreshed local index of HF model ids (from
    # cfahlgren1/hub-stats, filtered to repos with downloadable weights).
    # Consulted by the read-only resolve path to CONFIRM an exact HF model
    # id that was never minted into the registry — a confirmation, not an
    # entity. Built by scripts/build_hub_stats_index.py + refreshed by the
    # refresh-hub-stats-index cron; never written by seed/sync.
    "hub_stats_index": {
        # HF-true repo id (e.g. `meta-llama/Llama-3.1-8B`).
        "id": pd.StringDtype(),
        # Normalized form (lower + separator-collapse) — mirrors
        # services.hub_stats.normalize so case/separator-variant inputs hit.
        "id_norm": pd.StringDtype(),
        "release_date": pd.StringDtype(),
        "pipeline_tag": pd.StringDtype(),
        "params_billions": "float64",
        "downloads": "Int64",
        # Always True in the index (it's filtered to safetensors/gguf repos).
        "open_weights": pd.BooleanDtype(),
    },
}


def empty(table: str) -> pd.DataFrame:
    schema = _SCHEMAS[table]
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
