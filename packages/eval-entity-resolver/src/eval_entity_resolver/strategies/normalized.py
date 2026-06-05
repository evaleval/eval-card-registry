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

    No org-agreement guard here (unlike fuzzy): normalized matching is org-safe
    BY CONSTRUCTION. It matches the WHOLE normalized id/alias, so an `org/name`
    raw only hits an `org/name` alias whose org+name collapse to the same string
    — the same developer modulo case/separator (`ai21labs/x` -> `ai21-labs/x`),
    never a genuinely different developer (whose org token would change the
    normalized string). An org-LESS raw (bare leaf) has no developer to disagree
    on, and a leaf shared by two developers is already deduped to one owner by the
    seed's alias-collision detector.
    """
    norm = normalize(raw_value)
    lookup = alias_store.get_normalized_lookup(entity_type, source_config)
    return lookup.get(norm)
