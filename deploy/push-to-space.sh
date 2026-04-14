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
uv run hf upload "$SPACE_REPO" "$TMPDIR" . --repo-type space

echo "Done. Space will rebuild automatically."
