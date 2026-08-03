#!/usr/bin/env bash
# Rebuild the local seed so you can resolve slugs against your edits.
# PRUNE FIRST: the seed UPSERTS by id and does NOT drop rows you renamed/removed,
# so a stale fixtures/ produces phantom pass/fail results (see CONTRIBUTING.md).
# Usage: scripts/verify.sh
#   then: python -c "from eval_entity_resolver import Resolver; \
#         r=Resolver.from_parquet('fixtures/'); \
#         print(r.resolve('your-slug', entity_type='benchmark').canonical_id)"
set -euo pipefail
rm -f fixtures/*.parquet
uv run eval-card-registry seed --local
echo "seed rebuilt -> fixtures/  (now run Resolver.from_parquet('fixtures/') + pytest)"
