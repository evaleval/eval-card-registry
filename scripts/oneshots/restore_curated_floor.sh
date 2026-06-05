#!/usr/bin/env bash
# ONE-SHOT (model-resolution-rework): rebuild seed/models/core.yaml as the
# minimal CURATED FLOOR over the generator sources. Reproducible:
#   1. snapshot the pre-un-consolidation core (73557fa~1) as the oracle of
#      hand-curated entities
#   2. EMPTY core (preserving skip lists) and seed generator-only fixtures
#   3. build the floor = old-core entries the generators don't reproduce
#      (resolver exact/normalized) + alias hygiene
#   4. reseed (core + sources) and report duplicate-claim collisions
#
# PRECONDITION — run from the COMMITTED generator sources. This script edits
# sources in place (org canonicalization + parent repointing) and writes core +
# enrichments, so re-running on its OWN output is not idempotent. To reproduce
# from a clean base:
#   git checkout <baseline-commit> -- seed/models/ seed/orgs.generated.yaml
#   rm -f seed/models/enrichments/aliases.yaml
#   bash scripts/oneshots/restore_curated_floor.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

OLD_CORE=/tmp/old_core.yaml
CORE=seed/models/core.yaml

echo "==> [1/4] snapshot pre-un-consolidation core (73557fa~1)"
git show 73557fa~1:seed/models/core.yaml > "$OLD_CORE"

echo "==> [2/4] empty core (keep skip lists) + seed generator-only fixtures"
LOCAL_MODE=true uv run python - <<'PY'
import yaml
from pathlib import Path
import glob
p = Path("seed/models/core.yaml")
doc = yaml.safe_load(p.read_text()) or {}
skip_ids = doc.get("skip_ids", []) if isinstance(doc, dict) else []
skip_src = set(doc.get("skip_source_ids", []) if isinstance(doc, dict) else [])

# A curated upstream-correction (enrichments/upstream_corrections.yaml) that
# re-aliases a WRONG source id onto the correct canonical implies SKIPPING that
# source id — otherwise the correction alias and the bogus source canonical both
# claim the id and the bootstrap seed aborts. Derive those skips from the
# corrections file + the source canonical ids so they are active at EVERY seed
# (durable + reproducible, not a hand-maintained skip list).
src_ids = set()
for f in glob.glob("seed/models/sources/*.generated.yaml"):
    d = yaml.safe_load(Path(f).read_text())
    for e in (d.get("entries") if isinstance(d, dict) else d) or []:
        if isinstance(e, dict) and e.get("id"):
            src_ids.add(str(e["id"]))
corr = Path("seed/models/enrichments/upstream_corrections.yaml")
if corr.exists():
    for e in yaml.safe_load(corr.read_text()) or []:
        if isinstance(e, dict):
            for a in e.get("aliases") or []:
                if isinstance(a, str) and a in src_ids and a != e.get("id"):
                    skip_src.add(a)

out = {"skip_ids": skip_ids, "skip_source_ids": sorted(skip_src), "entries": []}
header = "# (transient: emptied for generator-only fixture seed during floor rebuild)\n"
p.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
print("core emptied; skip_ids=%d skip_source_ids=%d" % (len(skip_ids), len(skip_src)))

# Drop any GENERATED org row whose id is now claimed by a curated org
# (id / hf_org / alias in seed/orgs.yaml) — e.g. a stale `EnnoAi` row after
# `EnnoAi` became a curated alias of `Enno-Ai`. generate_hf_oracle_seed applies
# the same exclusion; pruning it here keeps the bootstrap seed collision-free
# (a curated alias and a generated id can't both claim the org name).
gp = Path("seed/orgs.generated.yaml")
if gp.exists():
    claims = set()
    for e in yaml.safe_load(Path("seed/orgs.yaml").read_text()) or []:
        if isinstance(e, dict):
            for form in (e.get("id"), e.get("hf_org"), *(e.get("aliases") or [])):
                if isinstance(form, str) and form:
                    claims.add(form.lower())
    gtext = gp.read_text()
    ghdr = "\n".join(ln for ln in gtext.splitlines() if ln.startswith("#"))
    gens = yaml.safe_load(gtext) or []
    kept = [e for e in gens if not (isinstance(e, dict) and str(e.get("id", "")).lower() in claims)]
    if len(kept) != len(gens):
        gp.write_text((ghdr + "\n" if ghdr else "") + yaml.safe_dump(kept, sort_keys=False, allow_unicode=True, width=200))
        print("pruned %d generated org row(s) now claimed by a curated org" % (len(gens) - len(kept)))
PY
rm -f fixtures/*.parquet
LOCAL_MODE=true uv run eval-card-registry seed --local >/dev/null 2>&1
echo "    generator-only fixtures seeded ($(LOCAL_MODE=true uv run python -c "import pandas as pd; print(len(pd.read_parquet('fixtures/canonical_models.parquet')))") canonical_models)"

echo "==> [3/5] build curated floor + alias bridges"
LOCAL_MODE=true uv run python scripts/oneshots/build_curated_core_floor.py "$OLD_CORE"

echo "==> [4/6] universal org reconcile (canonicalize + mint missing core orgs)"
LOCAL_MODE=true uv run python scripts/refresh_from_modelsdev.py --reconcile-orgs 2>&1 | sed 's/^/    /'

echo "==> [5/6] deconflict floor vs sources (source wins; drop stale floor dups)"
LOCAL_MODE=true uv run python scripts/oneshots/deconflict_floor.py 2>&1 | sed 's/^/    /'

echo "==> [6/6] reseed (core + sources + enrichments) and check duplicate-claim collisions"
rm -f fixtures/*.parquet
LOCAL_MODE=true uv run eval-card-registry seed --local 2>&1 \
  | grep -c "declared by both" \
  | xargs -I{} echo "    duplicate-claim collisions: {}"
echo "done."
