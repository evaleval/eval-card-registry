from contextlib import asynccontextmanager

from fastapi import FastAPI

from eval_card_registry.store.hf_store import get_store
from eval_card_registry.api.routes_resolve import router as resolve_router
from eval_card_registry.api.routes_entities import router as entities_router
from eval_card_registry.api.routes_aliases import router as aliases_router
from eval_card_registry.api.routes_health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    store.load()
    yield


app = FastAPI(
    title="eval-card-registry",
    description="Entity resolution registry for EEE evaluation data.",
    version="0.1.0",
    lifespan=lifespan,
)

PREFIX = "/api/v1"

app.include_router(resolve_router, prefix=PREFIX)
app.include_router(entities_router, prefix=PREFIX)
app.include_router(aliases_router, prefix=PREFIX)
app.include_router(health_router, prefix=PREFIX)
