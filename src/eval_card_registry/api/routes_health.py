from fastapi import APIRouter, Depends

from eval_card_registry.store.hf_store import get_store, RegistryStore

router = APIRouter()


@router.get("/health")
def health(store: RegistryStore = Depends(get_store)):
    return {
        "status": "ok",
        "store": "loaded" if store.loaded else "not_loaded",
        "entities": {
            "models": len(store.table("canonical_models")),
            "benchmarks": len(store.table("canonical_benchmarks")),
            "metrics": len(store.table("canonical_metrics")),
            "harnesses": len(store.table("eval_harnesses")),
        },
    }


@router.get("/stats")
def stats(store: RegistryStore = Depends(get_store)):
    def _counts(table: str) -> dict:
        df = store.table(table)
        total = len(df)
        draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
        return {"total": total, "draft": draft, "reviewed": total - draft}

    aliases_df = store.table("aliases")
    uncertain = int((aliases_df["status"] == "uncertain").sum()) if "status" in aliases_df.columns else 0

    return {
        "models": _counts("canonical_models"),
        "benchmarks": _counts("canonical_benchmarks"),
        "metrics": _counts("canonical_metrics"),
        "harnesses": _counts("eval_harnesses"),
        "aliases": {
            "total": len(aliases_df),
            "uncertain": uncertain,
        },
        "resolution_log": {"total": len(store.table("resolution_log"))},
        "sync_runs": {"total": len(store.table("sync_runs"))},
    }
