from typing import Optional

from eval_entity_resolver.normalization import normalize


def normalized_match(
    raw_value: str,
    entity_type: str,
    alias_store,
    source_config: Optional[str] = None,
) -> Optional[str]:
    """Normalize input and look up against normalized alias index.

    When ``source_config`` is given, scoped aliases for that config are
    considered in addition to global aliases; otherwise scoped aliases are
    excluded.
    """
    norm = normalize(raw_value)
    lookup = alias_store.get_normalized_lookup(entity_type, source_config)
    return lookup.get(norm)
