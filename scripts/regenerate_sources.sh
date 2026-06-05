#!/usr/bin/env bash
# Reproducible, deterministic regeneration of ALL model seed sources from FROZEN
# inputs. The model seed sources (seed/models/sources/*.generated.yaml) are
# rebuilt from committed snapshots, so a clean checkout reproduces them exactly.
#
# Inputs (all committed / frozen, so this is reproducible):
#   - ../hf_model_id_resolution.json        (frozen HF oracle; hf_oracle + tier3)
#   - tests/fixtures/modelsdev_api.snapshot.json  (pinned models.dev pull; models_dev)
#   - curation/hub_stats_frozen.parquet     (frozen hub-stats subset; hub_stats)
#
# This is the reproducible baseline + the verification entrypoint; the daily
# cron (.github/workflows/refresh-models.yml) refreshes from LIVE sources on top
# of it additively. Run order follows the tier dependency: hf_oracle (HF truth,
# edits core/models_dev/hub_stats in place) -> models_dev (core-aware) ->
# hub_stats (enrich, offline cache) -> models_dev --catalog (additive split) ->
# seed -> tier3 (residual tail) -> seed.
#
# Flags:
#   --reset-core   empty seed/models/core.yaml entries first (keeps skip lists),
#                  forcing a full regeneration from the generators alone. The
#                  ongoing/cron path leaves core intact (generators are
#                  core-aware and won't clobber curated entries).
#
# Usage:
#   bash scripts/regenerate_sources.sh [--reset-core]
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCAL_MODE=true
RESET_CORE=0
[ "${1:-}" = "--reset-core" ] && RESET_CORE=1

log(){ echo "[regen] === $* ==="; }

if [ "$RESET_CORE" = "1" ]; then
  log "STEP 0: reset core.yaml entries (keep skip lists) + EMPTY generated model"
  log "        sources (so the regen starts clean & is reproducible — otherwise"
  log "        hf_oracle re-keys stale prior output and counts drift)."
  uv run python - <<'PY'
import yaml
from pathlib import Path
core = Path('seed/models/core.yaml')
d = yaml.safe_load(core.read_text())
out = {
    'skip_ids': (d.get('skip_ids') or []) if isinstance(d, dict) else [],
    'skip_source_ids': (d.get('skip_source_ids') or []) if isinstance(d, dict) else [],
    'entries': [],
}
core.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
# Truncate the generated MODEL sources hf_oracle edits/reads in place, so a
# re-run regenerates from scratch rather than re-keying prior output.
for name in ('hf_oracle', 'models_dev', 'hub_stats', 'models_dev_catalog', 'tier3_inferred'):
    p = Path(f'seed/models/sources/{name}.generated.yaml')
    p.write_text('# reset by regenerate_sources.sh --reset-core\n[]\n')
print(f"reset: core skip_ids={len(out['skip_ids'])}, 5 generated model sources emptied")
PY
fi

log "STEP 1: pin models.dev snapshot (frozen) for reproducible models_dev regen"
cp tests/fixtures/modelsdev_api.snapshot.json /tmp/modelsdev_api.json

log "STEP 2: hf_oracle (frozen oracle; mints HF repos, re-keys in place)"
uv run python scripts/generate_hf_oracle_seed.py 2>&1 | tail -4

log "STEP 3: models_dev non-catalog (core-aware, --no-fetch pinned snapshot)"
uv run python scripts/refresh_from_modelsdev.py --no-fetch 2>&1 | tail -4

log "STEP 4: hub_stats (offline frozen cache)"
uv run python scripts/refresh_from_hub_stats.py 2>&1 | grep -vE 'WARN' | tail -4

log "STEP 5: models_dev --catalog (additive split)"
uv run python scripts/refresh_from_modelsdev.py --no-fetch --catalog 2>&1 | tail -4

log "STEP 5b: cross-source alias reconciliation (must precede the seed)"
uv run python scripts/dedup_cross_source_aliases.py 2>&1 | tail -4

log "STEP 6: seed (build fixtures for tier3)"
rm -f fixtures/*.parquet
uv run eval-card-registry seed --local 2>&1 | tail -3

log "STEP 7: tier3 inferred (residual no_match tail)"
uv run python scripts/generate_tier3_inferred_seed.py 2>&1 | tail -5

log "STEP 7a: universal org reconcile (union of ALL sources incl. tier3)"
uv run python scripts/refresh_from_modelsdev.py --reconcile-orgs 2>&1 | tail -3

log "STEP 7b: cross-source alias reconciliation (deterministic finalize)"
uv run python scripts/dedup_cross_source_aliases.py 2>&1 | tail -6

log "STEP 8: final seed"
rm -f fixtures/*.parquet
uv run eval-card-registry seed --local 2>&1 | tail -4

log "DONE: source id counts"
grep -c '^- id:' seed/models/sources/*.generated.yaml 2>/dev/null || true
