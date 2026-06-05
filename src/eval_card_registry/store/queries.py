"""
Read/write helpers for entity tables, aliases, and logs.
All operate on in-memory DataFrames from RegistryStore.

Performance note: write helpers that are called in tight loops (add_alias,
upsert_eval_result, append_resolution_log) accumulate rows in a pending
buffer on the store.  Call flush_pending(store) once at the end of a sync
to apply all buffered writes in a single pd.concat per table.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import schemas


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Date-suffix patterns used by `_derive_release_date_from_id`. The id
# encodes the snapshot/release date in three common shapes:
#   `-YYYY-MM-DD`  — OpenAI / Google daily snapshot (gpt-4o-2024-08-06)
#   `-YYYYMMDD`    — Anthropic / xAI snapshot (claude-sonnet-4-20250514)
#   `-YYYY-MM`     — OpenAI monthly pointer (gpt-5-2025-08); day defaults
#                    to "01" since the snapshot is month-grained
_DATE_ISO_FULL_RE = re.compile(r"-(\d{4})-(\d{2})-(\d{2})$")
_DATE_PACKED_RE = re.compile(r"-(\d{4})(\d{2})(\d{2})$")
_DATE_ISO_MONTH_RE = re.compile(r"-(\d{4})-(\d{2})$")


def _derive_release_date_from_id(canonical_id: str) -> Optional[str]:
    """Best-effort: parse a date suffix off the canonical id and return
    ISO-8601 YYYY-MM-DD, or None when no recognisable suffix is present.

    Year-range guard (2015-2035) keeps non-year 4-digit tails (parameter
    counts, batch numbers, etc.) from being mis-interpreted as a release
    year. The day/month components are validated to plausible ranges.
    Returns None on guard failure rather than a malformed date.
    """
    if not canonical_id:
        return None

    def _ok_year(s: str) -> bool:
        try:
            return 2015 <= int(s) <= 2035
        except ValueError:
            return False

    m = _DATE_ISO_FULL_RE.search(canonical_id)
    if m:
        y, mo, d = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
        return None

    m = _DATE_PACKED_RE.search(canonical_id)
    if m:
        y, mo, d = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
        return None

    m = _DATE_ISO_MONTH_RE.search(canonical_id)
    if m:
        y, mo = m.groups()
        if _ok_year(y) and 1 <= int(mo) <= 12:
            # Monthly pointer: day defaults to 01. Consumers wanting more
            # precision should rely on hand-curated or hub-stats sourced
            # release_dates, which always win over this derivation.
            return f"{y}-{mo}-01"
        return None

    return None


def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _source_config_key(value) -> Optional[str]:
    """Normalize nullable source_config values for alias-index keys."""
    return None if _is_na(value) else value


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a Series to dict, coercing pandas NA/NaN/NaT to None for JSON.
    Uses Series.to_dict() so numpy scalars are unboxed to Python types."""
    return {k: (None if _is_na(v) else v) for k, v in row.to_dict().items()}


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to list-of-dicts, coercing NA/NaN to None for JSON."""
    if df.empty:
        return []
    return df.astype(object).mask(df.isna(), None).to_dict(orient="records")


# ------------------------------------------------------------------
# Pending-row buffer  (avoids O(n²) pd.concat-per-row)
# ------------------------------------------------------------------

def _get_pending(store: RegistryStore, table: str) -> list[dict]:
    """Return the pending-row list for *table*, creating it if needed."""
    if not hasattr(store, "_pending"):
        store._pending = {}
    return store._pending.setdefault(table, [])


def decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON-encoded list-of-edges) to a
    Python list. Tolerant of NA/NaN, None, empty strings, and pre-decoded lists."""
    if _is_na(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        try:
            decoded = json.loads(s)
            return list(decoded) if isinstance(decoded, list) else []
        except (ValueError, TypeError):
            return []
    return []


def derive_model_lineage_fields(store: RegistryStore) -> dict[str, int]:
    """Walk `canonical_models.parents` and populate the denormalized
    `model_group_id`, `model_family_id`, `lineage_origin_model_id`,
    `lineage_origin_model_org_id`, and inherited `open_weights` columns.

    Three materialised id-hierarchy walks:
    - `model_group_id` (renamed from `root_model_id`): walk parents up
      through edges that preserve API identity — `quantized` (different
      precision, same model) and `variant axis=version` (dated snapshot of
      the same release, e.g. `gpt-4o-2024-05-13` -> `gpt-4o`). GROUP
      MEMBERSHIP is a total partition — ALWAYS set, equal to SELF when self
      has no such ancestor (self IS the identity-group root; a singleton is a
      group of one whose id is itself). NOT null at the root.
      Other variant axes (size, mode, training_stage, tier, modality,
      domain) keep separate identity at the leaf.

      DEFERRED — mode folding into the GROUP walk: the group ultimately folds
      `{version, quantized, mode}`. We intentionally do NOT
      add `mode` here yet. The current generated data MISLABELS chat/instruct
      variants as `axis=mode` (~307 edges); folding `mode` now would collapse
      those into their group and balloon the canonical_id flip blast radius.
      Once the chat/instruct edges are reclassified to `axis=training_stage`
      (different trained weights, kept separate), it becomes
      safe to add `mode` to `_is_identity_edge`. Until then the group walk
      stays `{quantized, variant·version}`, keeping the canonical_id flip
      blast radius small.
    - `model_family_id` (NEW): walk up the versioned release line, folding
      `quantized` + `variant` axes `{version, mode, training_stage, size,
      tier}`. Does NOT fold `modality`/`domain` (a vision/coder variant is a
      distinct artifact kept at the leaf/group level). Stops at
      finetune/merge/adapter. The "full labeled-version boundary" (3.5 ≠ 3.7
      ≠ 4; 4 ≠ 4o) is respected naturally: there are no cross-version parent
      edges in the graph — `variant·version` edges are snapshot->pointer
      within ONE labeled version, so the walk never crosses a version line.
      FAMILY MEMBERSHIP is a total partition — ALWAYS set, equal to SELF when
      self is the family root. NOT null at the root.
    - `lineage_origin_model_id` (NEW): deepest ancestor reached by walking
      non-`variant` edges (quantized / finetune / merge / adapter); the id of
      that ancestor. NULL when self is the origin — NO self-fallback (unlike
      `lineage_origin_model_org_id` below, which keeps its self-fallback).
    - `lineage_origin_model_org_id` (renamed from `lineage_origin_org_id`):
      walk through any non-`variant` edge (quantized / finetune / merge /
      adapter) to the deepest ancestor, then read its `org_id`. For
      Meta-originated models = self.org_id; for finetunes/quants of someone
      else's weights = upstream lab. KEEPS its self-fallback.
    - `open_weights`: if the row has an explicit value, keep it. Otherwise
      walk parents through `variant` + `quantized` edges (identity-
      preserving — a mode/size variant or a quant of an open-weight base
      is also open-weight) until we find a parent with an explicit value.
      Stops at finetune/merge/adapter edges since those produce new
      releases whose openness is independent of the base.

    All are caches recomputed on every seed/refresh. Returns counts
    dict for logging.
    """
    df = store.table("canonical_models")
    if df.empty:
        return {
            "group_set": 0,
            "family_set": 0,
            "lineage_model_set": 0,
            "lineage_org_set": 0,
            "open_weights_inherited": 0,
            "release_date_derived_from_id": 0,
        }

    parents_by_id: dict[str, list[dict]] = {}
    org_by_id: dict[str, Optional[str]] = {}
    open_by_id: dict[str, Optional[bool]] = {}
    release_by_id: dict[str, Optional[str]] = {}
    for _, row in df.iterrows():
        cid = row["id"]
        # Defensive: drop self-referencing parent edges (a malformed seed row
        # can point a model at itself, e.g. a finetune/quant edge id == cid).
        # A self-edge is never meaningful and would otherwise be a degenerate
        # cycle in every walk; stripping it here keeps group/family/lineage
        # walks honest. Genuinely-wrong upstream bases (e.g. tulu-3 pointing at
        # self instead of its Llama base) are a data fix for curation, not here.
        parents_by_id[cid] = [
            p
            for p in decode_parents(row.get("parents"))
            if not (isinstance(p, dict) and p.get("id") == cid)
        ]
        org = row.get("org_id")
        org_by_id[cid] = None if _is_na(org) else org
        ow = row.get("open_weights")
        open_by_id[cid] = None if _is_na(ow) else bool(ow)
        rd = row.get("release_date")
        release_by_id[cid] = None if _is_na(rd) else str(rd)

    # Org-from-prefix RULE (registry-proper — NOT a hotfix). For an HF-shaped
    # `org/name` id whose org_id was never set, derive the developer the SAME
    # way the resolver / auto-create does: the curated HF-org map
    # (canonical_orgs.hf_org + strategies/fuzzy._ORG_ALIASES), exactly like
    # `hf_id_to_canonical_cased`. So `openai/gpt-5.1`->openai, `xai/grok-2`->xai
    # fall out of the registry's OWN vendor catalog — no string hardcoding.
    # Guarantees:
    #   - NEVER overrides an existing org_id (curated / source values
    #     win — e.g. a finetune re-attributed to its real maker keeps it);
    #   - NEVER invents an org for a BARE id (no `/` -> left unresolved): if the
    #     source doesn't carry the developer namespace, we don't guess it from
    #     the model name.
    from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

    hf_to_dev = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for _, orow in store.table("canonical_orgs").iterrows():
        ho, oi = orow.get("hf_org"), orow.get("id")
        if isinstance(ho, str) and ho.strip() and isinstance(oi, str):
            hf_to_dev[ho.lower()] = oi
    org_prefix_updates: dict[str, str] = {}
    for cid in parents_by_id:
        if org_by_id.get(cid) is None and isinstance(cid, str) and "/" in cid:
            prefix = cid.split("/", 1)[0]
            dev = hf_to_dev.get(prefix.lower(), prefix)
            if dev:
                org_by_id[cid] = dev  # feed the lineage_origin_org walk below
                org_prefix_updates[cid] = dev

    def _walk(start: str, edge_ok) -> str:
        """Walk parents through edges where `edge_ok(edge)` is True.
        Returns the deepest reachable id; stops on no-match or cycle.

        When a node has multiple matching edges (e.g. a MergeKit model with
        several `merge` parents), pick deterministically by min-id rather
        than first-by-YAML-order, so lineage_origin_model_org_id doesn't depend on
        edge insertion order.
        """
        visited = {start}
        current = start
        while True:
            edges = parents_by_id.get(current, []) or []
            candidate_ids = [
                p["id"] for p in edges
                if isinstance(p, dict) and edge_ok(p) and p.get("id")
            ]
            next_id = min(candidate_ids) if candidate_ids else None
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return current
            visited.add(next_id)
            current = next_id

    def _is_identity_edge(p: dict) -> bool:
        rel = p.get("relationship")
        if rel == "quantized":
            return True
        # `version` = dated snapshot of the moving pointer; `mode` = a runtime
        # reasoning/thinking toggle of the same weights. Both are the same model at
        # the API level, so both fold into the identity group. (`mode` folds only
        # because the mislabelled chat/instruct edges were reclassified to
        # `training_stage`, which stays OUT of the group.)
        if rel == "variant" and p.get("axis") in ("version", "mode"):
            return True
        return False

    def _walk_group(start: str) -> str:
        """Identity-group walk. Like `_walk(_is_identity_edge)`, but an identity
        edge (quantized / variant·version / variant·mode) folds into the parent's
        group ONLY when child and parent share the SAME developer org
        (`org_by_id`). A third-party / community quant (e.g. `unsloth/...-bnb-4bit`
        quantizing `microsoft/phi-4`) therefore keeps its OWN group — its
        `quantized` edge still feeds `lineage_origin_model_id` (the link is
        preserved), but its scores never merge into the base lab's model. The same
        guard drops a spurious cross-org version edge (a snapshot pointer to a
        different lab). A first-party precision variant (same org) still folds."""
        visited = {start}
        current = start
        while True:
            cur_org = org_by_id.get(current)
            edges = parents_by_id.get(current, []) or []
            candidate_ids = [
                p["id"] for p in edges
                if isinstance(p, dict) and _is_identity_edge(p) and p.get("id")
                and cur_org is not None
                and org_by_id.get(p["id"]) == cur_org
            ]
            next_id = min(candidate_ids) if candidate_ids else None
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return current
            visited.add(next_id)
            current = next_id

    def _is_lineage_edge(p: dict) -> bool:
        return p.get("relationship") in {"quantized", "finetune", "merge", "adapter"}

    def _is_family_fold_edge(p: dict) -> bool:
        """Family walk: fold the versioned release line. `quantized`
        plus `variant` axes {version, mode, training_stage, size, tier}.
        Does NOT fold modality/domain (kept at the leaf — a vision/coder
        sibling is a distinct artifact). Stops at finetune/merge/adapter."""
        rel = p.get("relationship")
        if rel == "quantized":
            return True
        if rel == "variant":
            return p.get("axis") in {"version", "mode", "training_stage", "size", "tier"}
        return False

    def _inherit_open_from_ancestors(start: str) -> Optional[bool]:
        """Walk ONLY ancestors (skip self) through `variant` + `quantized`
        edges and return the first explicit `open_weights` value found.
        Returns None when no identity-preserving ancestor has it set.
        Caller is responsible for preferring self's explicit value over
        anything this returns."""
        visited = {start}
        current = start
        while True:
            edges = parents_by_id.get(current, []) or []
            next_id: Optional[str] = None
            for p in edges:
                if not isinstance(p, dict):
                    continue
                if p.get("relationship") in {"variant", "quantized"} and p.get("id"):
                    next_id = p["id"]
                    break
            if not next_id or next_id in visited or next_id not in parents_by_id:
                return None
            visited.add(next_id)
            current = next_id
            v = open_by_id.get(current)
            if v is not None:
                return v

    group_updates: dict[str, Optional[str]] = {}
    family_updates: dict[str, Optional[str]] = {}
    lineage_model_updates: dict[str, Optional[str]] = {}
    lineage_updates: dict[str, Optional[str]] = {}
    open_updates: dict[str, Optional[bool]] = {}
    release_updates: dict[str, Optional[str]] = {}
    inherited_count = 0
    release_derived_count = 0
    for cid in parents_by_id:
        # Identity-group root via quantized + variant·(version|mode) walk — each
        # treats the parent as the same model at the API level. Org-conditional:
        # an identity edge folds into the group ONLY when child and
        # parent are the same developer org — a third-party quant keeps its own
        # group (the link survives via lineage_origin).
        group = _walk_group(cid)
        # GROUP MEMBERSHIP is a total partition: every model is in exactly one
        # group, and a singleton is a group of one whose id is itself. So
        # model_group_id is ALWAYS set (self at the root — _walk returns self
        # when there is no identity-preserving ancestor, which IS the group id).
        group_updates[cid] = group
        # Family root via the versioned-release-line fold (NEW).
        family = _walk(cid, _is_family_fold_edge)
        # FAMILY MEMBERSHIP is likewise a total partition — ALWAYS set (self at
        # the family root). NOT null-at-root.
        family_updates[cid] = family
        # Lineage origin via any non-variant edge.
        # `ancestor` is reused for BOTH the model_id walk (NO self-fallback)
        # and the org_id walk (KEEPS self-fallback) — easy to confuse, so:
        ancestor = _walk(cid, _is_lineage_edge)
        # model_id: deepest non-variant ancestor; None when self is the origin.
        lineage_model_updates[cid] = ancestor if ancestor != cid else None
        # org_id: org of deepest ancestor, WITH self-fallback. Use explicit
        # None check, not `or` — an upstream lab whose org_id is an empty
        # string or missing should keep lineage as None, not flip to
        # self.org_id (which would mis-attribute a finetune as same-org).
        ancestor_org = org_by_id.get(ancestor)
        lineage_updates[cid] = ancestor_org if ancestor_org is not None else org_by_id.get(cid)
        # Open weights — explicit self value WINS; only fall back to
        # ancestor inheritance when self has no value set. Never overwrite
        # an explicit True/False with an inherited value.
        explicit = open_by_id.get(cid)
        if explicit is not None:
            open_updates[cid] = explicit
        else:
            inherited = _inherit_open_from_ancestors(cid)
            open_updates[cid] = inherited
            if inherited is not None:
                inherited_count += 1

        # Release date — explicit value (hand-curated, hub-stats createdAt,
        # or models.dev release_dates) WINS. Fall back to parsing the date
        # off the id when the canonical name encodes it (`-YYYY-MM-DD`,
        # `-YYYYMMDD`, `-YYYY-MM`). Avoids the silly "id literally says
        # 2025-04-14, registry says <NA>" gap on dated openai snapshots.
        explicit_release = release_by_id.get(cid)
        if explicit_release is not None and explicit_release.strip():
            release_updates[cid] = explicit_release
        else:
            derived = _derive_release_date_from_id(cid)
            release_updates[cid] = derived
            if derived is not None:
                release_derived_count += 1

    df = df.copy()
    # `model_group_id` (formerly `root_model_id`) and
    # `lineage_origin_model_org_id` (formerly `lineage_origin_org_id`) plus the
    # family + lineage-model walks are all written here.
    df["model_group_id"] = df["id"].map(group_updates).astype(pd.StringDtype())
    df["model_family_id"] = df["id"].map(family_updates).astype(pd.StringDtype())
    df["lineage_origin_model_id"] = df["id"].map(lineage_model_updates).astype(pd.StringDtype())
    df["lineage_origin_model_org_id"] = df["id"].map(lineage_updates).astype(pd.StringDtype())
    df["open_weights"] = df["id"].map(open_updates).astype(pd.BooleanDtype())
    df["release_date"] = df["id"].map(release_updates).astype(pd.StringDtype())
    if org_prefix_updates:
        df["org_id"] = df.apply(
            lambda r: org_prefix_updates.get(r["id"], r.get("org_id")), axis=1
        ).astype(pd.StringDtype())

    # Org-less honesty: a row whose developer is the `unknown` sentinel org has no
    # extractable developer, so it carries the `org-unknown` tag. The FK and the
    # tag move together — the org-less bucket is surfaced for review either way.
    def _add_org_unknown_tag(row) -> Any:
        raw = row.get("tags")
        if row.get("org_id") != "unknown":
            return raw
        try:
            cur = json.loads(raw) if isinstance(raw, str) else []
        except (ValueError, TypeError):
            cur = []
        if not isinstance(cur, list):
            cur = []
        if "org-unknown" not in cur:
            cur = [*cur, "org-unknown"]
        return json.dumps(cur)

    df["tags"] = df.apply(_add_org_unknown_tag, axis=1).astype(pd.StringDtype())
    store.set_table("canonical_models", df)

    # No dangling FK: every org the prefix rule derived must exist in
    # canonical_orgs. Most are curated (openai, xai, …) and already present;
    # an uploader prefix not in the catalog gets a community row (mirrors the
    # resolver's `_ensure_hf_org`).
    if org_prefix_updates:
        odf = store.table("canonical_orgs")
        have = set(odf["id"])
        missing = sorted({o for o in org_prefix_updates.values() if o not in have})
        if missing:
            cols = list(odf.columns)
            new_rows = []
            for o in missing:
                r = {c: None for c in cols}
                r.update({"id": o, "display_name": o, "hf_org": o, "kind": "community"})
                for c, v in (("tags", "[]"), ("metadata", "{}"), ("review_status", "reviewed")):
                    if c in r:
                        r[c] = v
                new_rows.append(r)
            store.set_table(
                "canonical_orgs", pd.concat([odf, pd.DataFrame(new_rows)], ignore_index=True)
            )

    return {
        "group_set": int(df["model_group_id"].notna().sum()),
        "family_set": int(df["model_family_id"].notna().sum()),
        "lineage_model_set": int(df["lineage_origin_model_id"].notna().sum()),
        "lineage_org_set": int(df["lineage_origin_model_org_id"].notna().sum()),
        "open_weights_inherited": inherited_count,
        "release_date_derived_from_id": release_derived_count,
    }


def flush_pending(store: RegistryStore) -> None:
    """Concat all buffered rows into their respective tables in one shot."""
    pending = getattr(store, "_pending", {})
    for table, rows in pending.items():
        if not rows:
            continue
        df = store.table(table)
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)
        store.set_table(table, df)
    store._pending = {}


# ------------------------------------------------------------------
# Generic entity helpers
# ------------------------------------------------------------------

def get_entity(store: RegistryStore, table: str, entity_id: str) -> Optional[dict]:
    df = store.table(table)
    row = df[df["id"] == entity_id]
    if row.empty:
        # Check pending rows too
        for pending_row in _get_pending(store, table):
            if pending_row.get("id") == entity_id:
                return pending_row
        return None
    return _row_to_dict(row.iloc[0])


def list_entities(
    store: RegistryStore,
    table: str,
    search: Optional[str] = None,
    review_status: Optional[str] = None,
    **filters: Any,
) -> list[dict]:
    df = store.table(table)
    if search:
        mask = df["id"].str.contains(search, case=False, na=False)
        if "display_name" in df.columns:
            mask = mask | df["display_name"].str.contains(search, case=False, na=False)
        df = df[mask]
    if review_status:
        df = df[df["review_status"] == review_status]
    for col, val in filters.items():
        if col in df.columns and val is not None:
            df = df[df[col] == val]
    return _records(df)


def upsert_entity(store: RegistryStore, table: str, data: dict, buffered: bool = False) -> dict:
    """Insert or update an entity row. `data` must contain `id`.
    If buffered=True, new rows go to the pending buffer (flushed by flush_pending).
    """
    df = store.table(table)
    entity_id = data["id"]
    now = _now()
    existing = df[df["id"] == entity_id]

    if existing.empty:
        # Check pending too
        pending = _get_pending(store, table)
        for p in pending:
            if p.get("id") == entity_id:
                p.update({k: v for k, v in data.items() if k != "id"})
                p["updated_at"] = now
                return p
        row = {**data, "created_at": now, "updated_at": now}
        if buffered:
            pending.append(row)
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            store.set_table(table, df)
        return row
    else:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == entity_id, col] = val
        df.loc[df["id"] == entity_id, "updated_at"] = now
        store.set_table(table, df)
        return _row_to_dict(df[df["id"] == entity_id].iloc[0])


# ------------------------------------------------------------------
# Alias helpers
# ------------------------------------------------------------------

# In-memory index for fast alias lookups during sync.
# Key: (entity_type, raw_value, source_config_or_None) → dict
_alias_index: dict[tuple, dict] = {}


def _rebuild_alias_index(store: RegistryStore) -> None:
    """Rebuild the in-memory alias index from the aliases table."""
    global _alias_index
    _alias_index = {}
    df = store.table("aliases")
    for _, row in df.iterrows():
        if row.get("status") != "rejected":
            row_dict = _row_to_dict(row)
            key = (
                row_dict["entity_type"],
                row_dict["raw_value"],
                _source_config_key(row_dict.get("source_config")),
            )
            _alias_index[key] = row_dict
    # Also index pending aliases
    for pending_row in _get_pending(store, "aliases"):
        if pending_row.get("status") != "rejected":
            key = (
                pending_row["entity_type"],
                pending_row["raw_value"],
                _source_config_key(pending_row.get("source_config")),
            )
            _alias_index[key] = pending_row


def get_alias(
    store: RegistryStore,
    raw_value: str,
    entity_type: str,
    source_config: Optional[str],
) -> Optional[dict]:
    source_config = _source_config_key(source_config)
    # Fast path: use index if available
    if _alias_index:
        if source_config:
            scoped = _alias_index.get((entity_type, raw_value, source_config))
            if scoped:
                return scoped
        global_ = _alias_index.get((entity_type, raw_value, None))
        if global_:
            return global_
        return None

    # Slow path: scan DataFrame
    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config:
        scoped = df[mask & (df["source_config"] == source_config)]
        if not scoped.empty:
            return _row_to_dict(scoped.iloc[0])
    global_ = df[mask & df["source_config"].isna()]
    if not global_.empty:
        return _row_to_dict(global_.iloc[0])
    return None


def add_alias(store: RegistryStore, data: dict, buffered: bool = False) -> dict:
    """
    Insert a new alias row. Enforces uniqueness on (entity_type, raw_value, source_config).
    Raises ValueError if a non-rejected alias already exists for that key.

    If buffered=True, the row is added to the pending buffer (flushed by flush_pending).
    Otherwise it is written immediately to the DataFrame.
    """
    raw_value = data["raw_value"]
    entity_type = data["entity_type"]
    source_config = _source_config_key(data.get("source_config"))
    key = (entity_type, raw_value, source_config)

    # Check uniqueness via index if available
    if _alias_index and key in _alias_index:
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    # Check DataFrame
    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config is not None:
        mask = mask & (df["source_config"] == source_config)
    else:
        mask = mask & df["source_config"].isna()
    if mask.any():
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    # Check pending buffer
    for p in _get_pending(store, "aliases"):
        if (p["entity_type"] == entity_type and p["raw_value"] == raw_value
                and _source_config_key(p.get("source_config")) == source_config
                and p.get("status") != "rejected"):
            raise ValueError(
                f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
                "Use update_alias() to modify an existing alias."
            )

    now = _now()
    row = {
        **data,
        "source_config": source_config,
        "id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
    }

    if buffered:
        _get_pending(store, "aliases").append(row)
    else:
        df = store.table("aliases")  # re-read in case it changed
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        store.set_table("aliases", df)

    # Update index only if it has already been built. If it is empty, get_alias
    # should keep using the DataFrame/pending slow path instead of a partial index.
    if _alias_index and row.get("status") != "rejected":
        _alias_index[key] = row
    return row


def update_alias(store: RegistryStore, alias_id: str, updates: dict) -> Optional[dict]:
    df = store.table("aliases")
    if not (df["id"] == alias_id).any():
        return None
    for col, val in updates.items():
        if col in df.columns:
            df.loc[df["id"] == alias_id, col] = val
    df.loc[df["id"] == alias_id, "updated_at"] = _now()
    store.set_table("aliases", df)
    updated = _row_to_dict(df[df["id"] == alias_id].iloc[0])
    # Keep the in-memory index in sync if it was built — otherwise a follow-up
    # add_alias() / get_alias() would see stale canonical data for this key.
    if _alias_index:
        key = (
            updated["entity_type"],
            updated["raw_value"],
            _source_config_key(updated.get("source_config")),
        )
        if updated.get("status") != "rejected":
            _alias_index[key] = updated
        else:
            _alias_index.pop(key, None)
    return updated


# ------------------------------------------------------------------
# Eval results (mapping table: one row per EEE evaluation result)
# ------------------------------------------------------------------

def _eval_result_id(evaluation_id: str, result_index: int) -> str:
    """Deterministic ID from evaluation_id + result_index."""
    key = f"{evaluation_id}:{result_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Track IDs already in pending buffer to detect upsert-vs-insert
_pending_result_ids: set[str] = set()


def upsert_eval_result(store: RegistryStore, data: dict) -> dict:
    """Insert or update an eval_results row. Uses deterministic ID from evaluation_id + result_index."""
    row_id = _eval_result_id(data["evaluation_id"], data["result_index"])
    now = _now()

    # Check if already in pending buffer
    if row_id in _pending_result_ids:
        pending = _get_pending(store, "eval_results")
        for p in pending:
            if p["id"] == row_id:
                for col, val in data.items():
                    if col != "id":
                        p[col] = val
                p["updated_at"] = now
                return p

    # Check committed table
    df = store.table("eval_results")
    existing = df[df["id"] == row_id]
    if not existing.empty:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == row_id, col] = val
        df.loc[df["id"] == row_id, "updated_at"] = now
        store.set_table("eval_results", df)
        return _row_to_dict(df[df["id"] == row_id].iloc[0])

    # New row — buffer it
    row = {**data, "id": row_id, "created_at": now, "updated_at": now}
    _get_pending(store, "eval_results").append(row)
    _pending_result_ids.add(row_id)
    return row


def get_eval_results(
    store: RegistryStore,
    model_id: Optional[str] = None,
    benchmark_id: Optional[str] = None,
    source_config: Optional[str] = None,
) -> list[dict]:
    """Query eval_results with optional filters."""
    df = store.table("eval_results")
    if model_id:
        df = df[df["model_id"] == model_id]
    if benchmark_id:
        df = df[df["benchmark_id"] == benchmark_id]
    if source_config:
        df = df[df["source_config"] == source_config]
    return _records(df)


# ------------------------------------------------------------------
# Resolution log
# ------------------------------------------------------------------

def append_resolution_log(store: RegistryStore, entry: dict) -> None:
    row = {**entry, "id": str(uuid.uuid4()), "timestamp": _now()}
    _get_pending(store, "resolution_log").append(row)


# ------------------------------------------------------------------
# Sync runs
# ------------------------------------------------------------------

def start_sync_run(
    store: RegistryStore, source_config: str, rerun: bool
) -> str:
    run_id = str(uuid.uuid4())
    df = store.table("sync_runs")
    row = {
        "id": run_id,
        "source_config": source_config,
        "started_at": _now(),
        "completed_at": None,
        "status": "running",
        "rerun": rerun,
        "entities_created": 0,
        "entities_updated": 0,
        "aliases_created": 0,
        "aliases_updated": 0,
        "errors": json.dumps([]),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    store.set_table("sync_runs", df)
    return run_id


def finish_sync_run(store: RegistryStore, run_id: str, counts: dict, errors: list) -> None:
    df = store.table("sync_runs")
    df.loc[df["id"] == run_id, "completed_at"] = _now()
    df.loc[df["id"] == run_id, "status"] = "failed" if errors else "completed"
    for col, val in counts.items():
        if col in df.columns:
            df.loc[df["id"] == run_id, col] = val
    df.loc[df["id"] == run_id, "errors"] = json.dumps(errors)
    store.set_table("sync_runs", df)
