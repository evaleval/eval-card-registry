"""
eval-card-registry CLI.

Commands:
  seed      Load known entities from seed/ YAML files
  stats     Print registry summary
  sync      Batch sync one or all EEE configs → eval_results table
"""
import json
from pathlib import Path
from typing import Optional

import typer
import yaml


def _json_encode_if_needed(value):
    """Encode lists/dicts as JSON strings; pass through anything else.

    seed/models.yaml uses YAML-native lists for `tags` (e.g. `["open-weight"]`)
    while seed/benchmarks.yaml stores them pre-encoded as strings (e.g.
    `'["instruction-following"]'`). The canonical_* parquet columns are all
    VARCHAR, so we coerce on the way in to keep both formats supported.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _legacy_parent_model_id_to_parents(entry: dict) -> None:
    """Translate a legacy `parent_model_id: X` field to the typed `parents`
    list shape. Mutates the entry in place.

    Legacy core.yaml / sources/*.generated.yaml use a single scalar
    `parent_model_id` to express a family/variant relationship (e.g.
    Llama-3-8B → Llama-3). The new schema replaces this with a typed list
    of parent edges. This shim converts on load so existing YAML keeps
    working until each file is migrated to emit `parents` natively.

    No-op when `parents` is already present (new shape wins) or when neither
    field is set.
    """
    if "parents" in entry and entry["parents"] is not None:
        entry.pop("parent_model_id", None)
        return
    legacy = entry.pop("parent_model_id", None)
    if legacy:
        entry["parents"] = [{"id": legacy, "relationship": "variant", "axis": "size"}]

from eval_card_registry.store.hf_store import get_store
from eval_card_registry.store import queries, schemas
from eval_card_registry.store.queries import _is_na

app = typer.Typer(help="eval-card-registry CLI")


def _load_store():
    store = get_store()
    if not store.loaded:
        store.load()
    return store


# ------------------------------------------------------------------
# seed
# ------------------------------------------------------------------

@app.command()
def seed(
    local: bool = typer.Option(False, "--local", help="Write to fixtures/ instead of HF Hub"),
    seed_dir: str = typer.Option("./seed", "--seed-dir"),
    prune_stale: bool = typer.Option(
        False,
        "--prune-stale/--no-prune-stale",
        help="Remove reviewed seed entities and seed aliases absent from the current YAML snapshot.",
    ),
):
    """Load known canonical entities from seed YAML files."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()
    seed_path = Path(seed_dir)

    # ------------------------------------------------------------------
    # Models — three-layer load from seed/models/:
    #   sources/*.generated.yaml  → external catalog data (e.g. models.dev),
    #                               flat lists, never hand-edited
    #   core.yaml                 → curated canonicals (the source of truth),
    #                               flat list OR {skip_ids, entries} dict
    #   enrichments/aliases.yaml  → optional alias-only entries ({id, aliases})
    #                               that union onto whatever exists
    #
    # Merge order: sources → core → enrichments. Field-level merge per entry
    # (aliases / tags UNION; other scalars prefer non-empty, last-write-wins).
    # `skip_ids` from core drops generated entries we don't want.
    # ------------------------------------------------------------------
    def _load_models_merged() -> list[dict]:
        models_dir = seed_path / "models"
        sources_dir = models_dir / "sources"
        core_file = models_dir / "core.yaml"
        enrichments_file = models_dir / "enrichments" / "aliases.yaml"

        source_entries: list[dict] = []
        core_entries: list[dict] = []
        enrichment_entries: list[dict] = []
        skip_ids: set[str] = set()

        if sources_dir.is_dir():
            for src_path in sorted(sources_dir.glob("*.generated.yaml")):
                with open(src_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{src_path} must be a flat list")
                source_entries.extend(loaded)

        skip_source_ids: set[str] = set()
        if core_file.exists():
            with open(core_file) as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, list):
                core_entries = loaded
            elif isinstance(loaded, dict):
                core_entries = loaded.get("entries", []) or []
                skip_ids = set(loaded.get("skip_ids", []) or [])
                # `skip_source_ids` drops these ids from sources/enrichments only,
                # leaving core entries authoritative. Used when models.dev (or any
                # auto-generated source) ships bad aliases for a model that core.yaml
                # curates correctly — otherwise the loader's UNION-merge would
                # re-introduce the bad aliases on every refresh.
                skip_source_ids = set(loaded.get("skip_source_ids", []) or [])
            else:
                raise typer.BadParameter(f"{core_file} unexpected shape {type(loaded)}")

        if enrichments_file.exists():
            with open(enrichments_file) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{enrichments_file} must be a flat list")
            enrichment_entries = loaded

        def _merge_into(target: dict, src: dict) -> dict:
            """Merge two entries with the same canonical_id.

            Field-level merge policy:
            - `aliases`: UNION (case-insensitive dedup).
            - `tags`: UNION (case-insensitive dedup). Both YAML-list and
              JSON-encoded-string forms supported. Protects against session
              additions overwriting `[open-weight, moe]` with `[open-weight]`.
            - Other scalars: prefer non-empty across the pair; when both
              sides have a non-empty value, last-write-wins. Protects against
              session-batch entries that omit `architecture` /
              `params_billions` from silently overwriting earlier rich entries.

            "Empty" means: None, "", [], {}, or default-looking '{}' / '[]'.
            """
            import json as _json

            existing_aliases = list(target.get("aliases") or [])
            existing_lc = {a.lower() for a in existing_aliases if a}
            new_aliases = list(src.get("aliases") or [])
            for a in new_aliases:
                if a and a.lower() not in existing_lc:
                    existing_aliases.append(a)
                    existing_lc.add(a.lower())

            def _decode_list_field(v):
                """tags / metadata may be either YAML-list or JSON-encoded
                string. Return a list (best-effort) and a boolean indicating
                whether to re-encode on write."""
                if v is None:
                    return [], False
                if isinstance(v, list):
                    return list(v), False
                if isinstance(v, str):
                    s = v.strip()
                    if not s or s in ("[]", "null"):
                        return [], True
                    try:
                        d = _json.loads(s)
                        if isinstance(d, list):
                            return list(d), True
                    except (ValueError, TypeError):
                        pass
                return [v], False

            # Union tags (handles both list and JSON-string formats)
            tgt_tags, tgt_was_json = _decode_list_field(target.get("tags"))
            src_tags, src_was_json = _decode_list_field(src.get("tags"))
            seen_tags_lc = {str(t).lower() for t in tgt_tags}
            for t in src_tags:
                if t is not None and str(t).lower() not in seen_tags_lc:
                    tgt_tags.append(t)
                    seen_tags_lc.add(str(t).lower())
            # Re-encode if either source was a JSON string (the parquet column
            # is VARCHAR; _json_encode_if_needed downstream handles either).
            tags_merged = _json.dumps(tgt_tags) if (tgt_was_json or src_was_json) else tgt_tags

            def _is_empty(v) -> bool:
                if v is None:
                    return True
                if isinstance(v, (list, dict)) and len(v) == 0:
                    return True
                if isinstance(v, str) and v.strip() in ("", "[]", "{}"):
                    return True
                return False

            # Union `parents` by id. For an edge present in both, field-merge
            # within the edge so a later source can fill in `axis` (or correct
            # `relationship`) without duplicating the edge. Edges from the
            # target preserve their order; new edges from src are appended.
            tgt_parents, tgt_p_was_json = _decode_list_field(target.get("parents"))
            src_parents, src_p_was_json = _decode_list_field(src.get("parents"))
            parents_by_id: dict[str, dict] = {}
            parents_order: list[str] = []
            for p in tgt_parents:
                if isinstance(p, dict) and p.get("id"):
                    pid = p["id"]
                    if pid not in parents_by_id:
                        parents_order.append(pid)
                        parents_by_id[pid] = dict(p)
            for p in src_parents:
                if not isinstance(p, dict) or not p.get("id"):
                    continue
                pid = p["id"]
                if pid in parents_by_id:
                    merged_edge = dict(parents_by_id[pid])
                    for k, v in p.items():
                        if _is_empty(v):
                            continue
                        merged_edge[k] = v
                    parents_by_id[pid] = merged_edge
                else:
                    parents_order.append(pid)
                    parents_by_id[pid] = dict(p)
            parents_list = [parents_by_id[pid] for pid in parents_order]
            parents_merged = (
                _json.dumps(parents_list)
                if (tgt_p_was_json or src_p_was_json)
                else parents_list
            )

            merged = dict(target)
            for k, v in src.items():
                if k in ("aliases", "tags", "parents"):
                    continue  # handled separately
                if _is_empty(v):
                    continue
                merged[k] = v
            merged["aliases"] = existing_aliases
            merged["tags"] = tags_merged
            # Only emit `parents` if at least one side had any (avoids creating
            # a spurious empty list on entries that never had a parents field).
            if tgt_parents or src_parents:
                merged["parents"] = parents_merged
            return merged

        by_id: dict[str, dict] = {}

        def _absorb(entries: list[dict], extra_skip: set[str] = frozenset()) -> None:
            drop = skip_ids | extra_skip
            for e in entries:
                if "id" not in e:
                    raise typer.BadParameter(f"models seed entry missing id: {e!r}")
                if e["id"] in drop:
                    continue
                # Translate legacy `parent_model_id` scalar to the typed
                # `parents` list before any merge / column-filter step.
                _legacy_parent_model_id_to_parents(e)
                if e["id"] in by_id:
                    by_id[e["id"]] = _merge_into(by_id[e["id"]], e)
                else:
                    by_id[e["id"]] = e

        # Sources/enrichments respect both skip_ids and skip_source_ids;
        # core entries respect only skip_ids so curated overrides always apply.
        _absorb(source_entries, extra_skip=skip_source_ids)
        _absorb(core_entries)
        _absorb(enrichment_entries, extra_skip=skip_source_ids)
        return list(by_id.values())

    # ------------------------------------------------------------------
    # Orgs — two-file load:
    #   seed/orgs.yaml            → curated first-party labs (the source
    #                               of truth, hand-edited)
    #   seed/orgs.generated.yaml  → auto-created orgs from hub-stats refresh
    #                               (HF authors that aren't curated labs)
    #
    # Curated wins on id collision. Unlike the models merge (field-level),
    # orgs use a simple "drop generated entry if id is in curated" policy:
    # curated entries are deliberate and richer; auto-created entries are
    # thin (just id, display_name, kind=unknown), so a partial overlay
    # would never improve the curated record.
    # ------------------------------------------------------------------
    def _load_orgs_merged() -> list[dict]:
        curated_path = seed_path / "orgs.yaml"
        generated_path = seed_path / "orgs.generated.yaml"

        curated: list[dict] = []
        if curated_path.exists():
            with open(curated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{curated_path} must be a flat list")
            curated = loaded

        generated: list[dict] = []
        if generated_path.exists():
            with open(generated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{generated_path} must be a flat list")
            generated = loaded

        curated_ids = {e["id"] for e in curated if "id" in e}
        out = list(curated)
        for e in generated:
            if "id" not in e:
                raise typer.BadParameter(f"orgs.generated.yaml entry missing id: {e!r}")
            if e["id"] not in curated_ids:
                out.append(e)
        return out

    # table name, yaml file, label, entity_type (for alias creation)
    seed_specs = [
        # Orgs: load via merge helper to combine curated + auto-generated.
        ("canonical_orgs", "__merged_orgs__", "orgs", "org"),
        ("canonical_benchmarks", seed_path / "benchmarks.yaml", "benchmarks", "benchmark"),
        ("canonical_metrics", seed_path / "metrics.yaml", "metrics", "metric"),
        ("eval_harnesses", seed_path / "harnesses.yaml", "harnesses", "harness"),
        # Models: load via the merge helper; pass a sentinel path that
        # signals the loop below to invoke _load_models_merged() instead of
        # reading a single YAML file.
        ("canonical_models", "__merged_models__", "models", "model"),
    ]

    alias_count = 0
    # Track all seed entity IDs and alias keys so we can remove stale ones.
    # Alias key: (raw_value, entity_type, canonical_id, source_config)
    seed_snapshot: list[tuple[str, str, set[str], set[tuple[str, str, str, Optional[str]]]]] = []

    # Build the alias index once so add_alias collision checks are O(1) instead
    # of O(N) DataFrame mask scans. Combined with buffered=True below, this
    # avoids the O(N²) pd.concat-per-row cost on ~1k entities + ~13k aliases.
    queries._rebuild_alias_index(store)

    for table, yaml_file, label, entity_type in seed_specs:
        table_columns = set(schemas.empty(table).columns)
        if yaml_file == "__merged_models__":
            items = _load_models_merged()
            if not items:
                typer.echo(f"  [skip] no model entries found in seed/models.yaml or _overrides/")
                continue
        elif yaml_file == "__merged_orgs__":
            items = _load_orgs_merged()
            if not items:
                typer.echo(f"  [skip] no org entries found in seed/orgs.yaml or seed/orgs.generated.yaml")
                continue
        else:
            if not yaml_file.exists():
                typer.echo(f"  [skip] {yaml_file} not found")
                continue
            with open(yaml_file) as f:
                items = yaml.safe_load(f) or []

        yaml_ids: set[str] = set()
        yaml_alias_keys: set[tuple[str, str, str, Optional[str]]] = set()

        for original_item in items:
            item = dict(original_item)
            # Pop 'aliases' / 'scoped_aliases' before upserting — not table columns.
            extra_aliases = item.pop("aliases", []) or []
            scoped_aliases = item.pop("scoped_aliases", {}) or {}
            # Normalize list/dict columns: YAML may have native lists/dicts,
            # but the canonical_* parquet columns are VARCHAR, so encode if
            # needed. `parents` is a list-of-edges on canonical_models.
            for col in ("tags", "metadata", "parents"):
                if col in item:
                    item[col] = _json_encode_if_needed(item[col])
            entity_item = {k: v for k, v in item.items() if k in table_columns}
            if "id" not in entity_item:
                raise typer.BadParameter(f"{label} seed entry is missing required id: {original_item!r}")
            queries.upsert_entity(store, table, entity_item, buffered=True)
            canonical_id = entity_item["id"]
            display_name = entity_item.get("display_name", "")
            yaml_ids.add(canonical_id)

            # Global aliases (source_config=None): matched regardless of caller's source_config.
            # Scoped aliases (source_config=<name>): matched only when the caller passes that
            # source_config — lets short tokens ("Overall", "Arabic") map to different
            # benchmarks depending on which EEE config they came from.
            global_aliases = {canonical_id, display_name} | set(extra_aliases)

            alias_specs: list[tuple[str, Optional[str]]] = [
                (raw, None) for raw in global_aliases if raw
            ]
            for source_cfg, raw_values in scoped_aliases.items():
                for raw in raw_values or []:
                    if raw:
                        alias_specs.append((raw, source_cfg))

            for raw_value, source_cfg in alias_specs:
                # Index stale-removal by (raw_value, entity_type, canonical_id, source_config)
                yaml_alias_keys.add((raw_value, entity_type, canonical_id, source_cfg))
                try:
                    queries.add_alias(store, {
                        "raw_value": raw_value,
                        "entity_type": entity_type,
                        "canonical_id": canonical_id,
                        "source_config": source_cfg,
                        "source_field": "seed",
                        "status": "confirmed",
                        "strategy": "seed",
                        "confidence": 1.0,
                        "notes": None,
                    }, buffered=True)
                    alias_count += 1
                except ValueError:
                    # add_alias raises on uniqueness collision: an alias row
                    # already exists for (entity_type, raw_value, source_config).
                    # YAML is the source of truth, so if the existing row points
                    # at a different canonical_id, this is a YAML rename and we
                    # must REPOINT the existing row — NOT silently swallow it.
                    # Without this, stale-removal at the end of seed would then
                    # delete the row (its old key is no longer in
                    # yaml_alias_keys), causing total alias loss.
                    aliases_df = store.table("aliases")
                    mask = (
                        (aliases_df["raw_value"] == raw_value)
                        & (aliases_df["entity_type"] == entity_type)
                        & (aliases_df["status"] != "rejected")
                    )
                    if source_cfg is not None:
                        mask = mask & (aliases_df["source_config"] == source_cfg)
                    else:
                        mask = mask & aliases_df["source_config"].isna()
                    existing = aliases_df[mask]
                    if existing.empty:
                        # Collision came from the pending buffer (this run added
                        # the same key earlier). For same-canonical re-adds this
                        # is a no-op; for different-canonical we must mutate the
                        # pending dict in place so the rename isn't lost on
                        # flush. _alias_index points at the same dict, so
                        # updating it here keeps the index consistent.
                        for p in queries._get_pending(store, "aliases"):
                            if (p.get("entity_type") == entity_type
                                    and p.get("raw_value") == raw_value
                                    and queries._source_config_key(p.get("source_config")) == queries._source_config_key(source_cfg)
                                    and p.get("status") != "rejected"):
                                if p["canonical_id"] != canonical_id:
                                    prev = p["canonical_id"]
                                    p["canonical_id"] = canonical_id
                                    p["source_field"] = "seed"
                                    p["status"] = "confirmed"
                                    p["strategy"] = "seed"
                                    p["confidence"] = 1.0
                                    typer.echo(
                                        f"  [rename] alias {raw_value!r} ({entity_type}) "
                                        f"moved {prev!r} -> {canonical_id!r} (pending)"
                                    )
                                    alias_count += 1
                                break
                        continue
                    row = existing.iloc[0]
                    if row["canonical_id"] != canonical_id:
                        # Rename: repoint the existing row at the new canonical.
                        queries.update_alias(store, row["id"], {
                            "canonical_id": canonical_id,
                            "source_field": "seed",
                            "status": "confirmed",
                            "strategy": "seed",
                            "confidence": 1.0,
                        })
                        typer.echo(
                            f"  [rename] alias {raw_value!r} ({entity_type}) "
                            f"moved {row['canonical_id']!r} -> {canonical_id!r}"
                        )
                        alias_count += 1
                    # else: identical re-seed of an existing alias — no-op.

        seed_snapshot.append((table, entity_type, yaml_ids, yaml_alias_keys))
        typer.echo(f"  {label}: {len(items)}")

    # Flush all buffered upserts (entities + aliases) into their tables in a
    # single pd.concat per table. prune_stale below reads store.table(...)
    # directly, so this must happen before that block.
    queries.flush_pending(store)

    # Derive denormalized parent-walk caches now that all canonical_models
    # rows are present. `root_model_id` and `lineage_origin_org_id` are
    # computed from `parents` and need the full graph to be in place.
    lineage_counts = queries.derive_model_lineage_fields(store)
    typer.echo(
        f"  derived: root_model_id={lineage_counts['root_set']}, "
        f"lineage_origin_org_id={lineage_counts['lineage_set']}, "
        f"open_weights_inherited={lineage_counts['open_weights_inherited']}, "
        f"release_date_from_id={lineage_counts['release_date_derived_from_id']}"
    )

    removed_entities = 0
    removed_aliases = 0
    if prune_stale:
        # Remove seed-originated entities and aliases that are no longer in the YAML.
        # Only touches rows that were created by seed (strategy == "seed"), never
        # sync-created aliases or auto-draft entities.
        for table, entity_type, yaml_ids, yaml_alias_keys in seed_snapshot:
            # Remove stale seed aliases for this entity type.
            aliases_df = store.table("aliases")
            seed_mask = (aliases_df["strategy"] == "seed") & (aliases_df["entity_type"] == entity_type)
            if seed_mask.any():
                seed_aliases = aliases_df[seed_mask]
                stale_alias_mask = seed_mask.copy()
                for idx in seed_aliases.index:
                    row = seed_aliases.loc[idx]
                    sc = row.get("source_config")
                    if _is_na(sc):
                        sc = None
                    key = (row["raw_value"], row["entity_type"], row["canonical_id"], sc)
                    if key in yaml_alias_keys:
                        stale_alias_mask[idx] = False
                n_stale = stale_alias_mask.sum()
                if n_stale > 0:
                    store.set_table("aliases", aliases_df[~stale_alias_mask].reset_index(drop=True))
                    removed_aliases += int(n_stale)

            # Remove stale seed entities — only those with review_status "reviewed"
            # that came from seed and are no longer in the YAML.
            entity_df = store.table(table)
            if len(entity_df) > 0:
                stale = entity_df["id"].isin(yaml_ids)
                stale_entities = entity_df[~stale & (entity_df["review_status"] == "reviewed")]
                # Only remove if every alias for this entity is also seed-originated,
                # meaning it wasn't referenced by sync data.
                current_aliases = store.table("aliases")
                for eid in stale_entities["id"]:
                    entity_aliases = current_aliases[
                        (current_aliases["canonical_id"] == eid)
                        & (current_aliases["entity_type"] == entity_type)
                    ]
                    if len(entity_aliases) == 0 or (entity_aliases["strategy"] == "seed").all():
                        entity_df = entity_df[entity_df["id"] != eid]
                        # Also remove any remaining aliases pointing to it.
                        current_aliases = current_aliases[
                            ~((current_aliases["canonical_id"] == eid)
                              & (current_aliases["entity_type"] == entity_type))
                        ]
                        removed_entities += 1
                store.set_table(table, entity_df.reset_index(drop=True))
                store.set_table("aliases", current_aliases.reset_index(drop=True))

    typer.echo(f"  aliases: {alias_count} added, {removed_aliases} removed")
    if removed_entities:
        typer.echo(f"  stale entities removed: {removed_entities}")

    store.push_to_hub()
    typer.echo("Seed complete.")


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@app.command()
def stats(
    local: bool = typer.Option(False, "--local", help="Read from fixtures/ instead of HF Hub"),
):
    """Print registry entity counts and pending review summary."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()

    def _row(table):
        df = store.table(table)
        total = len(df)
        draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
        return total, draft

    for label, table in [
        ("models    ", "canonical_models"),
        ("benchmarks", "canonical_benchmarks"),
        ("metrics   ", "canonical_metrics"),
        ("harnesses ", "eval_harnesses"),
    ]:
        total, draft = _row(table)
        typer.echo(f"  {label}  total={total}  draft={draft}")

    aliases_df = store.table("aliases")
    uncertain = int((aliases_df["status"] == "uncertain").sum()) if "status" in aliases_df.columns else 0
    typer.echo(f"\n  aliases        total={len(aliases_df)}  uncertain={uncertain}")
    typer.echo(f"  eval_results   total={len(store.table('eval_results'))}")
    typer.echo(f"  resolution_log total={len(store.table('resolution_log'))}")
    typer.echo(f"  sync_runs      total={len(store.table('sync_runs'))}")


# ------------------------------------------------------------------
# sync
# ------------------------------------------------------------------

@app.command()
def sync(
    config: Optional[str] = typer.Option(None, "--config", help="EEE config name"),
    all_configs: bool = typer.Option(False, "--all", help="Sync all EEE configs"),
    rerun: bool = typer.Option(False, "--rerun", help="Re-resolve all raw strings even if already aliased"),
    local: bool = typer.Option(False, "--local"),
):
    """
    Batch sync EEE config(s) → writes resolved results to eval_results table.
    Each result row is one (model × benchmark × metric) combination with resolved canonical IDs.
    """
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    if not config and not all_configs:
        typer.echo("Specify --config <name> or --all", err=True)
        raise typer.Exit(1)

    from eval_card_registry.services.ingestion import run_sync
    import datasets as ds_lib

    store = _load_store()

    configs_to_run: list[str] = []
    if all_configs:
        configs_to_run = ds_lib.get_dataset_config_names("evaleval/EEE_datastore")
    else:
        configs_to_run = [config]

    failed = []
    for cfg in configs_to_run:
        typer.echo(f"Syncing {cfg}...")
        try:
            counts = run_sync(cfg, store, rerun=rerun)
            typer.echo(f"  {cfg}: {counts}")
        except Exception as e:
            typer.echo(f"  {cfg}: FAILED — {e}", err=True)
            failed.append(cfg)

    typer.echo("Persisting tables...")
    store.push_to_hub()

    if failed:
        typer.echo(f"Done with {len(failed)} failed config(s): {', '.join(failed)}")
    else:
        typer.echo("Done.")
