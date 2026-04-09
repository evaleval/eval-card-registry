from typing import Optional


def exact_match(
    raw_value: str,
    entity_type: str,
    source_config: Optional[str],
    alias_store,
) -> Optional[str]:
    """Direct lookup in alias store — config-scoped then global."""
    return alias_store.lookup(raw_value, entity_type, source_config)
