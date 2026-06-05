# One-shot migration scripts (archived)

Spent, already-applied data migrations from the model-resolution rework. Kept
for provenance — **not** run by CI and not imported by any live code. Several
read gitignored/`curation/` inputs or are no-ops against the current seed; re-run
only with a fresh checkpoint (back up `fixtures/` + `seed/models/*` first).

The live, recurring scripts stay in `scripts/` (generators `generate_*`,
refreshers `refresh_*`, `fold_modelsdev_dupes.py` which the gate test imports,
`publish_registry_data.py`, `verify_sync.py`, `scan_eee_*`).

## What each did

- `canonicalize_orgs_and_unfold.py` — one-time org canonicalization + id un-fold.
- `integrate_modelsdev_catalog.py` — superseded; folded into
  `refresh_from_modelsdev.py --catalog`.
- `m10_b3_01_multisize_split.py` — split size-conflated nodes (gpt-oss, gemma-3).
- `m10_b3_02_merges_orgfix.py` / `m10_fix_labels_orgs.py` — org/label fixes.
  **Baichuan:** these two made OPPOSITE baichuan↔baichuan-inc decisions; the
  resolved state is **`baichuan` as a curated lab** (the
  `m10_fix_labels_orgs.py` baichuan→baichuan-inc repoint was superseded). Do not
  re-run either against the current seed.
- `m10_b3_03_apply_alias_audit.py` / `m10_fin_medium_alias.py` — alias-audit
  fixes (read the gitignored audit doc; will crash without it).
- `m10_b3_04_deferred_own.py`, `m10_fin_*` (casing/mode-reclass/serving/bareids)
  — assorted alias/id passes.
- `scan_suspicious_aliases.py` → `ground_sweep_cases.py` / `ground_mint_candidates.py`
  — the alias-sweep candidate scan + grounding (read `curation/`).
- `m10_sweep_*`, `m10_unsure_*`, `m10_link_session_mints.py`, `m10_rehost_repoint.py`,
  `m10_display_form_cleanup.py`, `m10_t4_id_fixes.py`, `m10_t5_quant_edges.py`,
  `m10_t6_slugify_agents.py` — the review-driven id/identity fix passes.
