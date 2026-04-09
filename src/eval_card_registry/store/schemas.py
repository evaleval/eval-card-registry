"""Empty DataFrame schemas for each table."""
import pandas as pd


_SCHEMAS: dict[str, dict] = {
    "canonical_models": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "developer": pd.StringDtype(),
        "family": pd.StringDtype(),
        "architecture": pd.StringDtype(),
        "params_billions": "float64",
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
}


def empty(table: str) -> pd.DataFrame:
    schema = _SCHEMAS[table]
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
