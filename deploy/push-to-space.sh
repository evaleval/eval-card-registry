#!/usr/bin/env bash
# Push the eval-card-registry to its HF Space.
#
# Prerequisites:
#   - huggingface-cli logged in (or HF_TOKEN set)
#   - The target HF Space exists (Docker SDK)
#
# Usage:
#   bash deploy/push-to-space.sh
#   HF_SPACE_REPO=some-user/some-space bash deploy/push-to-space.sh  # override target

set -euo pipefail

SPACE_REPO="${HF_SPACE_REPO:-evaleval/entity-registry}"
if [[ -z "$SPACE_REPO" ]]; then
    echo "error: HF_SPACE_REPO is empty." >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create a temp dir for the Space contents
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Preparing Space contents in $TMPDIR ..."

# Copy Dockerfile
cp "$REPO_ROOT/Dockerfile" "$TMPDIR/"

# Copy workspace files
cp "$REPO_ROOT/pyproject.toml" "$TMPDIR/"
cp "$REPO_ROOT/uv.lock" "$TMPDIR/"

# Copy source packages
mkdir -p "$TMPDIR/packages/eval-entity-resolver"
cp "$REPO_ROOT/packages/eval-entity-resolver/pyproject.toml" "$TMPDIR/packages/eval-entity-resolver/"
cp -r "$REPO_ROOT/packages/eval-entity-resolver/src" "$TMPDIR/packages/eval-entity-resolver/src"
cp -r "$REPO_ROOT/src" "$TMPDIR/src"

# Copy Space README (required frontmatter)
cp "$SCRIPT_DIR/hf-space/README.md" "$TMPDIR/README.md"

echo "Uploading to HF Space $SPACE_REPO ..."
# Upload via upload_folder, NOT `hf upload`. `hf upload` always calls
# create_repo(exist_ok=True) first, and the repos/create endpoint is rate-limited
# far more aggressively than commits — that redundant create is what 429s a deploy
# when the token's API budget is hot. Instead: a cheap existence check (GET), then
# create ONLY if the Space is genuinely absent, then commit. Retries a transient
# 429 with backoff.
uv run python - "$SPACE_REPO" "$TMPDIR" <<'PY'
import sys, time
from huggingface_hub import HfApi

repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()

# Existence check (repo_info GET) — does NOT hit the throttled repos/create.
if not api.repo_exists(repo_id=repo, repo_type="space"):
    print(f"  Space {repo} does not exist — creating it…", file=sys.stderr)
    api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker", exist_ok=True)

for attempt in range(5):
    try:
        api.upload_folder(
            folder_path=folder, repo_id=repo, repo_type="space",
            commit_message="Deploy eval-card-registry service",
        )
        break
    except Exception as e:  # noqa: BLE001
        if "429" in str(e) and attempt < 4:
            wait = 30 * (attempt + 1)
            print(f"  429 rate-limited; retrying in {wait}s ({attempt + 1}/4)…", file=sys.stderr)
            time.sleep(wait)
            continue
        raise
PY

echo "Done. Space will rebuild automatically."
