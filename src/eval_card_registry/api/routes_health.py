from fastapi import APIRouter, Depends

from eval_card_registry.store.hf_store import get_store, RegistryStore

router = APIRouter()


def _safe_count(store: RegistryStore, table: str) -> int:
    return len(store.table(table)) if store.has_table(table) else 0


@router.get("/health")
def health(store: RegistryStore = Depends(get_store)):
    return {
        "status": "ok",
        "store": "loaded" if store.loaded else "not_loaded",
        "entities": {
            "models": _safe_count(store, "canonical_models"),
            "benchmarks": _safe_count(store, "canonical_benchmarks"),
            "metrics": _safe_count(store, "canonical_metrics"),
            "harnesses": _safe_count(store, "eval_harnesses"),
        },
    }


@router.get("/stats")
def stats(store: RegistryStore = Depends(get_store)):
    def _counts(table: str) -> dict:
        if not store.has_table(table):
            return {"total": 0, "draft": 0, "reviewed": 0}
        df = store.table(table)
        total = len(df)
        draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
        return {"total": total, "draft": draft, "reviewed": total - draft}

    if store.has_table("aliases"):
        aliases_df = store.table("aliases")
        uncertain = int((aliases_df["status"] == "uncertain").sum()) if "status" in aliases_df.columns else 0
        aliases_stats = {"total": len(aliases_df), "uncertain": uncertain}
    else:
        aliases_stats = {"total": 0, "uncertain": 0}

    return {
        "models": _counts("canonical_models"),
        "benchmarks": _counts("canonical_benchmarks"),
        "metrics": _counts("canonical_metrics"),
        "harnesses": _counts("eval_harnesses"),
        "aliases": aliases_stats,
        "resolution_log": {"total": _safe_count(store, "resolution_log")},
        "sync_runs": {"total": _safe_count(store, "sync_runs")},
    }
