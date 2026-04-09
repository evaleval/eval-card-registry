"""
eval-card-registry CLI.

Commands:
  seed      Load known entities from seed/ YAML files
  stats     Print registry summary
  sync      Batch sync one or all EEE configs → eval_results table
"""
from pathlib import Path
from typing import Optional

import typer
import yaml

from eval_card_registry.store.hf_store import get_store
from eval_card_registry.store import queries

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
):
    """Load known canonical entities from seed YAML files."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()
    seed_path = Path(seed_dir)

    # table name, yaml file, label, entity_type (for alias creation)
    seed_specs = [
        ("canonical_benchmarks", seed_path / "benchmarks.yaml", "benchmarks", "benchmark"),
        ("canonical_metrics", seed_path / "metrics.yaml", "metrics", "metric"),
        ("eval_harnesses", seed_path / "harnesses.yaml", "harnesses", "harness"),
    ]

    alias_count = 0
    # Track all seed entity IDs and alias keys so we can remove stale ones.
    # Each entry: (table, entity_type, set_of_ids, set_of (raw_value, entity_type, canonical_id))
    seed_snapshot: list[tuple[str, str, set[str], set[tuple[str, str, str]]]] = []

    for table, yaml_file, label, entity_type in seed_specs:
        if not yaml_file.exists():
            typer.echo(f"  [skip] {yaml_file} not found")
            continue
        with open(yaml_file) as f:
            items = yaml.safe_load(f) or []

        yaml_ids: set[str] = set()
        yaml_alias_keys: set[tuple[str, str, str]] = set()  # (raw_value, entity_type, canonical_id)

        for item in items:
            # Pop 'aliases' before upserting — it's not a table column.
            extra_aliases = item.pop("aliases", []) or []
            queries.upsert_entity(store, table, item)
            canonical_id = item["id"]
            display_name = item.get("display_name", "")
            yaml_ids.add(canonical_id)

            # Create global aliases so the resolver can match against seed entities.
            # At minimum: display_name → id, id → id, plus any extra aliases from YAML.
            alias_strings = {canonical_id, display_name} | set(extra_aliases)
            for raw_value in alias_strings:
                if not raw_value:
                    continue
                yaml_alias_keys.add((raw_value, entity_type, canonical_id))
                try:
                    queries.add_alias(store, {
                        "raw_value": raw_value,
                        "entity_type": entity_type,
                        "canonical_id": canonical_id,
                        "source_config": None,
                        "source_field": "seed",
                        "status": "confirmed",
                        "strategy": "seed",
                        "confidence": 1.0,
                        "notes": None,
                    })
                    alias_count += 1
                except ValueError:
                    pass  # alias already exists (e.g. re-seeding)

        seed_snapshot.append((table, entity_type, yaml_ids, yaml_alias_keys))
        typer.echo(f"  {label}: {len(items)}")

    # Remove seed-originated entities and aliases that are no longer in the YAML.
    # Only touches rows that were created by seed (strategy == "seed"), never
    # sync-created aliases or auto-draft entities.
    removed_entities = 0
    removed_aliases = 0
    for table, entity_type, yaml_ids, yaml_alias_keys in seed_snapshot:
        # Remove stale seed aliases for this entity type
        aliases_df = store.table("aliases")
        seed_mask = (aliases_df["strategy"] == "seed") & (aliases_df["entity_type"] == entity_type)
        if seed_mask.any():
            seed_aliases = aliases_df[seed_mask]
            stale_alias_mask = seed_mask.copy()
            for idx in seed_aliases.index:
                row = seed_aliases.loc[idx]
                key = (row["raw_value"], row["entity_type"], row["canonical_id"])
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
                    # Also remove any remaining aliases pointing to it
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
