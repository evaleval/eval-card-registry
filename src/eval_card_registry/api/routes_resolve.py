import uuid

from fastapi import APIRouter, Depends, Request
from datetime import datetime, timezone

from eval_card_registry.api.schemas import ResolveRequest, ResolveResponse
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter

router = APIRouter()


def _svc(request: Request) -> ResolutionService:
    return request.app.state.resolution_service


def _log_writer(request: Request) -> ResolveLogWriter:
    return request.app.state.log_writer


def _log_resolve(
    log_writer: ResolveLogWriter,
    request_id: str,
    req: ResolveRequest,
    result: dict,
) -> None:
    log_writer.append({
        "request_id": request_id,
        "raw_value": req.raw_value,
        "entity_type": req.entity_type,
        "source_config": req.source_config,
        "canonical_id": result.get("canonical_id"),
        "strategy": result.get("strategy"),
        "confidence": result.get("confidence"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@router.post("/resolve", response_model=ResolveResponse)
def resolve(
    req: ResolveRequest,
    svc: ResolutionService = Depends(_svc),
    log_writer: ResolveLogWriter = Depends(_log_writer),
):
    result = svc.resolve(
        raw_value=req.raw_value,
        entity_type=req.entity_type,
        source_config=req.source_config,
        source_field=req.source_field,
    )
    _log_resolve(log_writer, str(uuid.uuid4()), req, result)
    return ResolveResponse(**result)


@router.post("/resolve/batch", response_model=list[ResolveResponse])
def resolve_batch(
    reqs: list[ResolveRequest],
    svc: ResolutionService = Depends(_svc),
    log_writer: ResolveLogWriter = Depends(_log_writer),
):
    request_id = str(uuid.uuid4())
    responses = []
    for r in reqs:
        result = svc.resolve(
            raw_value=r.raw_value,
            entity_type=r.entity_type,
            source_config=r.source_config,
            source_field=r.source_field,
        )
        _log_resolve(log_writer, request_id, r, result)
        responses.append(ResolveResponse(**result))
    return responses
