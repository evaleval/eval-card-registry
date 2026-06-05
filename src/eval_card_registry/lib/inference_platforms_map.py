"""Single-sourced host-token → inference_platform mapping.

There is exactly ONE authority for the host-token → platform mapping:
``seed/inference_platforms.yaml``. This module lazy-loads that file at import
time, inverts each row's ``aliases`` list into ``{host_token_lower: platform_id}``,
and exports the accessors that downstream consumers (fuzzy.py, the models.dev
refresh) use. Do NOT hand-copy the map into the strategy files — import it here.

Imports ONLY stdlib + yaml (no fuzzy, no schemas) to avoid an import cycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

# Sentinel host token that means "missing developer field" → no platform.
_UNKNOWN_SENTINEL = "unknown"

# seed/inference_platforms.yaml lives at the repo root's seed/ dir.
# This file is src/eval_card_registry/lib/inference_platforms_map.py, so the
# repo root is four parents up.
_SEED_PATH = (
    Path(__file__).resolve().parents[3] / "seed" / "inference_platforms.yaml"
)

_HOST_TOKEN_TO_PLATFORM: dict[str, Optional[str]] = {}
_LOADED = False


def _coerce_aliases(raw) -> list[str]:
    """Accept either a YAML list or a JSON-encoded list string (the seed CLI
    JSON-encodes the column, but the YAML on disk holds native lists)."""
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw else []
        return list(decoded) if isinstance(decoded, list) else []
    return list(raw or [])


def _load() -> None:
    global _LOADED
    _HOST_TOKEN_TO_PLATFORM.clear()
    if _SEED_PATH.exists():
        with open(_SEED_PATH) as f:
            platforms = yaml.safe_load(f) or []
        for plat in platforms:
            platform_id = plat.get("id")
            for alias in _coerce_aliases(plat.get("aliases")):
                if alias:
                    _HOST_TOKEN_TO_PLATFORM[alias.lower()] = platform_id
    # The missing-developer sentinel maps to None.
    _HOST_TOKEN_TO_PLATFORM[_UNKNOWN_SENTINEL] = None
    _LOADED = True


def _ensure_loaded() -> None:
    if not _LOADED:
        _load()


def get_host_token_platform(token: str) -> Optional[str]:
    """Return the inference_platforms.id for a host token (e.g. 'fireworks/',
    '-bedrock', 'azure/'), or None if the token is unknown / the `unknown`
    sentinel. Case-insensitive."""
    if not token:
        return None
    _ensure_loaded()
    return _HOST_TOKEN_TO_PLATFORM.get(token.lower())


def all_host_tokens() -> set[str]:
    """Return the set of known host tokens (lowercased), including the
    `unknown` sentinel."""
    _ensure_loaded()
    return set(_HOST_TOKEN_TO_PLATFORM.keys())


# Load eagerly on import; safe (graceful no-op if the seed file is absent).
_load()
