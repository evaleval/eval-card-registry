from typing import Optional

from eval_entity_resolver.normalization import normalize


def normalized_match(
    raw_value: str,
    entity_type: str,
    alias_store,
) -> Optional[str]:
    """Normalize input and look up against normalized alias index."""
    norm = normalize(raw_value)
    lookup = alias_store.get_normalized_lookup(entity_type)
    return lookup.get(norm)
