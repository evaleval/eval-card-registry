import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from eval_card_registry.api.deps import writable as _writable
from eval_card_registry.api.schemas import OrgCreate, OrgPatch, ReviewStatus
from eval_card_registry.store.hf_store import get_store, RegistryStore
from eval_card_registry.store import queries

router = APIRouter()


def _get_or_404(store, entity_id):
    entity = queries.get_entity(store, "canonical_orgs", entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"canonical_orgs '{entity_id}' not found")
    return entity


_JSON_FIELDS = {"tags", "metadata"}


def _encode(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if k in _JSON_FIELDS and isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _decode(entity: dict) -> dict:
    out = {}
    for k, v in entity.items():
        if k in _JSON_FIELDS and isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        else:
            out[k] = v
    return out


@router.get("/orgs")
def list_orgs(
    search: Optional[str] = None,
    review_status: Optional[ReviewStatus] = None,
    store: RegistryStore = Depends(get_store),
):
    return [_decode(e) for e in queries.list_entities(store, "canonical_orgs", search=search, review_status=review_status)]


@router.get("/orgs/{org_id}")
def get_org(org_id: str, store: RegistryStore = Depends(get_store)):
    return _decode(_get_or_404(store, org_id))


@router.post("/orgs", status_code=201, dependencies=_writable)
def create_org(body: OrgCreate, store: RegistryStore = Depends(get_store)):
    return _decode(queries.upsert_entity(store, "canonical_orgs", _encode(body.model_dump())))


@router.patch("/orgs/{org_id}", dependencies=_writable)
def patch_org(org_id: str, body: OrgPatch, store: RegistryStore = Depends(get_store)):
    _get_or_404(store, org_id)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data["id"] = org_id
    return _decode(queries.upsert_entity(store, "canonical_orgs", _encode(data)))
