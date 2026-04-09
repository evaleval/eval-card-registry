"""Alias management routes — v0-defer (read + patch only; no review UI yet)."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from eval_card_registry.api.schemas import AliasPatch
from eval_card_registry.store.hf_store import get_store, RegistryStore
from eval_card_registry.store import queries

router = APIRouter()


@router.get("/aliases")
def list_aliases(
    status: Optional[str] = None,
    entity_type: Optional[str] = None,
    source_config: Optional[str] = None,
    store: RegistryStore = Depends(get_store),
):
    return queries.list_entities(
        store,
        "aliases",
        review_status=None,
        **{k: v for k, v in {"status": status, "entity_type": entity_type, "source_config": source_config}.items() if v is not None},
    )


@router.patch("/aliases/{alias_id}")
def patch_alias(alias_id: str, body: AliasPatch, store: RegistryStore = Depends(get_store)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result = queries.update_alias(store, alias_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Alias '{alias_id}' not found")
    return result
