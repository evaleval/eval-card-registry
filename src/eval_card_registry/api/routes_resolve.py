from fastapi import APIRouter, Depends

from eval_card_registry.api.schemas import ResolveRequest, ResolveResponse
from eval_card_registry.store.hf_store import get_store, RegistryStore
from eval_card_registry.services.resolution_service import ResolutionService

router = APIRouter()


def _svc(store: RegistryStore = Depends(get_store)) -> ResolutionService:
    return ResolutionService(store)


@router.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest, svc: ResolutionService = Depends(_svc)):
    result = svc.resolve(
        raw_value=req.raw_value,
        entity_type=req.entity_type,
        source_config=req.source_config,
        source_field=req.source_field,
    )
    return ResolveResponse(**result)


@router.post("/resolve/batch", response_model=list[ResolveResponse])
def resolve_batch(reqs: list[ResolveRequest], svc: ResolutionService = Depends(_svc)):
    return [
        ResolveResponse(
            **svc.resolve(
                raw_value=r.raw_value,
                entity_type=r.entity_type,
                source_config=r.source_config,
                source_field=r.source_field,
            )
        )
        for r in reqs
    ]
