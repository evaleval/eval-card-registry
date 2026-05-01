from contextlib import asynccontextmanager

from fastapi import FastAPI

from eval_card_registry.config import settings
from eval_card_registry.store.hf_store import get_store, QUERY_TABLE_NAMES
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter
from eval_card_registry.api.routes_resolve import router as resolve_router
from eval_card_registry.api.routes_entities import router as entities_router
from eval_card_registry.api.routes_aliases import router as aliases_router
from eval_card_registry.api.routes_orgs import router as orgs_router
from eval_card_registry.api.routes_health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    if settings.read_only:
        store.load(tables=QUERY_TABLE_NAMES)
    else:
        store.load()

    # Singleton ResolutionService — avoids rebuilding AliasStore per request
    app.state.resolution_service = ResolutionService(store)

    # Resolve log writer
    log_writer = ResolveLogWriter(settings.hf_log_bucket)
    app.state.log_writer = log_writer
    log_writer.start(settings.log_flush_interval_seconds)

    yield

    await log_writer.stop()


app = FastAPI(
    title="eval-card-registry",
    description="Entity resolution registry for EEE evaluation data.",
    version="0.1.0",
    lifespan=lifespan,
)

PREFIX = "/api/v1"

app.include_router(resolve_router, prefix=PREFIX)
app.include_router(entities_router, prefix=PREFIX)
app.include_router(orgs_router, prefix=PREFIX)
app.include_router(aliases_router, prefix=PREFIX)
app.include_router(health_router, prefix=PREFIX)
