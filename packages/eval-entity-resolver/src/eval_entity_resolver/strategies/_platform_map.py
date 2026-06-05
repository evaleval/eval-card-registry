"""Single-sourced host-token → inference_platform map for the fuzzy strategy.

The resolver is a STANDALONE package — it must NOT hard-depend on
``eval_card_registry``. But the host-token → ``inference_platforms.id``
mapping has exactly ONE authority: ``seed/inference_platforms.yaml``. The
registry exposes that data via
``eval_card_registry.lib.inference_platforms_map``.

To keep the resolver standalone while still single-sourcing the DATA (no
hand-copied literal of the host→platform pairs), this loader reads the same
authored YAML through TWO fallback channels, in order:

1. ``eval_card_registry.lib.inference_platforms_map`` — the registry's own
   accessor. Available whenever the registry package is importable (the
   workspace dev env, and the producer's path-dep env when the registry is
   installed alongside the resolver).
2. The co-located ``seed/inference_platforms.yaml`` — the resolver package
   lives inside the registry repo (``packages/eval-entity-resolver/…``), so
   the authored seed file is reachable by walking up to the repo root. This
   covers the case where only the resolver tree is on the path.

Both channels read the SAME authored YAML and invert each row's ``aliases``
list into ``{host_token_lower: platform_id}``. If neither channel is available (e.g. a bare wheel install with no seed and
no registry pkg), the map is empty and every host token captures ``None``
(strip-for-matching still works; only the platform side-value is absent).

Stdlib + optional yaml only — no import of ``fuzzy`` (avoids a cycle).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Sentinel host token meaning "missing developer field" → no platform.
_UNKNOWN_SENTINEL = "unknown"

_HOST_TOKEN_TO_PLATFORM: dict[str, Optional[str]] = {}
_LOADED = False


def _coerce_aliases(raw: Any) -> list[str]:
    """Accept either a YAML/native list or a JSON-encoded list string."""
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw else []
        return list(decoded) if isinstance(decoded, list) else []
    return list(raw or [])


def _load_via_registry_lib() -> Optional[dict[str, Optional[str]]]:
    """Channel 1: the registry's own single-source accessor, if importable."""
    try:
        from eval_card_registry.lib.inference_platforms_map import (  # type: ignore
            all_host_tokens,
            get_host_token_platform,
        )
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None
    mapping: dict[str, Optional[str]] = {}
    for token in all_host_tokens():
        mapping[token] = get_host_token_platform(token)
    return mapping or None


def _find_seed_yaml() -> Optional[Path]:
    """Channel 2: locate the co-located seed/inference_platforms.yaml by
    walking up from this module to the registry repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "seed" / "inference_platforms.yaml"
        if candidate.exists():
            return candidate
    return None


def _load_via_seed_yaml() -> Optional[dict[str, Optional[str]]]:
    seed_path = _find_seed_yaml()
    if seed_path is None:
        return None
    try:
        import yaml  # optional dep; present in the workspace env
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        with open(seed_path) as f:
            platforms = yaml.safe_load(f) or []
    except (OSError, ValueError):
        return None
    mapping: dict[str, Optional[str]] = {}
    for plat in platforms:
        if not isinstance(plat, dict):
            continue
        platform_id = plat.get("id")
        for alias in _coerce_aliases(plat.get("aliases")):
            if alias:
                mapping[alias.lower()] = platform_id
    mapping[_UNKNOWN_SENTINEL] = None
    return mapping or None


def _load() -> None:
    global _LOADED
    _HOST_TOKEN_TO_PLATFORM.clear()
    mapping = _load_via_registry_lib()
    if mapping is None:
        mapping = _load_via_seed_yaml()
    if mapping:
        _HOST_TOKEN_TO_PLATFORM.update(mapping)
    # Always honour the missing-developer sentinel.
    _HOST_TOKEN_TO_PLATFORM[_UNKNOWN_SENTINEL] = None
    _LOADED = True


def _ensure_loaded() -> None:
    if not _LOADED:
        _load()


def get_host_token_platform(token: str) -> Optional[str]:
    """Return the inference_platforms.id for a host token spelling
    (e.g. ``'fireworks/'``, ``'-bedrock'``, ``'together/'``) or None when the
    token is unknown / the ``unknown`` sentinel. Case-insensitive."""
    if not token:
        return None
    _ensure_loaded()
    return _HOST_TOKEN_TO_PLATFORM.get(token.lower())


# Load eagerly on import (graceful no-op if no source is reachable).
_load()
