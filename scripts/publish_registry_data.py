"""Publish the registry's canonical_* parquets to evaleval/entity-registry-data.

Replaces the inline Python in `.github/workflows/seed.yml` with a
dedicated, testable publish path:

  1. Run the seed CLI end-to-end against `seed/`, materialising every
     canonical_* table to `fixtures/`.
  2. Compute a content hash over the parquet bytes — used as the
     idempotency key. If the previous publish on HF carries the same
     hash, skip the upload (a no-op merge to main shouldn't churn the
     dataset history).
  3. Write a `manifest.json` sidecar carrying:
       - schema_version (registry.<MAJOR>.<MINOR>; major bump on
         removed/renamed columns, minor on additions).
       - content_hash.
       - seed_git_sha — the registry repo's HEAD SHA at publish time.
       - generated_at — ISO timestamp.
       - per-table row counts (sanity surface for consumers).
  4. Upload `fixtures/` + `manifest.json` to the HF Dataset repo as a
     single revision via HfApi.upload_folder.

Usage:
    uv run python scripts/publish_registry_data.py [--dry-run]

`--dry-run` runs steps 1–3 (writes manifest locally, no push) and
exits 0 if everything is consistent. CI uses this on PRs to catch
schema breakage before merge.

Backend consumers (eval_card_backend) should assert
`manifest.json.schema_version`'s major matches their expected major.
That guards against a registry breaking change that ships before the
producer is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Bump when removing/renaming columns in any canonical_* table.
# Bump minor when ADDING columns (backward-compatible). The producer
# asserts the major matches what it expects.
SCHEMA_VERSION = "registry.3.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"
FIXTURES_DIR = REPO_ROOT / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
HF_DATASET_REPO = "evaleval/entity-registry-data"

# YAML overrides that ship alongside the parquets. The producer reads
# these when present in the registry data cache (slice_overrides drives
# slice→benchmark promotion; display_overrides feeds the display name
# polish layer). Storing them in the published dataset removes the
# producer's dependency on a sibling registry checkout.
SHIPPED_YAML = ("slice_overrides.yaml", "display_overrides.yaml")


def _git_sha() -> Optional[str]:
    """HEAD SHA of the registry repo at publish time. None when not in
    a git checkout (fresh CI runner with shallow clone? handled
    gracefully)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _content_hash(fixtures_dir: Path) -> str:
    """SHA-256 over the deterministically-sorted file bytes for every
    artifact we publish.

    Includes parquets and YAML overrides; excludes manifest.json (would
    create a circular dependency: hash depends on manifest, manifest
    contains hash). Hashes file PATH + file BYTES so a missing/added
    file changes the hash and the next publish doesn't get skipped.
    """
    h = hashlib.sha256()
    files = sorted(
        list(fixtures_dir.glob("*.parquet"))
        + [p for p in fixtures_dir.glob("*.yaml") if p.name != "manifest.json"]
    )
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _row_counts(fixtures_dir: Path) -> dict[str, int]:
    """Per-table row counts for the manifest sanity surface. Uses
    pyarrow's metadata-only read so we don't materialise the whole
    parquet for a count."""
    import pyarrow.parquet as pq

    out: dict[str, int] = {}
    for p in sorted(fixtures_dir.glob("*.parquet")):
        try:
            md = pq.read_metadata(p)
            out[p.stem] = md.num_rows
        except Exception as exc:
            # Don't let a broken parquet block publish — log and skip.
            print(f"  [warn] couldn't read row count for {p.name}: {exc}", file=sys.stderr)
    return out


def _run_seed() -> int:
    """Invoke the seed CLI in LOCAL_MODE so it writes to fixtures/
    without pushing. This script handles the upload to HF itself
    (see upload step in main)."""
    env = dict(os.environ)
    env["LOCAL_MODE"] = "true"
    proc = subprocess.run(
        ["uv", "run", "eval-card-registry", "seed", "--local"],
        cwd=REPO_ROOT,
        env=env,
    )
    return proc.returncode


def _read_remote_manifest() -> Optional[dict]:
    """Pull the existing manifest.json from HF (if any) so we can
    compare content_hash and skip no-op publishes. Returns None when
    the dataset hasn't been published before, the manifest doesn't
    exist yet (legacy revisions), or any HF call fails."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None

    try:
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="manifest.json",
            repo_type="dataset",
        )
        with open(local) as f:
            return json.load(f)
    except (
        RepositoryNotFoundError,
        EntryNotFoundError,
        HfHubHTTPError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"  [info] no remote manifest available ({type(exc).__name__}); will publish",
              file=sys.stderr)
        return None


def _push(fixtures_dir: Path, manifest: dict) -> None:
    """Upload fixtures/ + manifest.json to HF as a single dataset
    revision. Commit message includes the seed git SHA so consumers
    can pin to a specific registry source state."""
    from huggingface_hub import HfApi

    api = HfApi()
    sha = manifest.get("seed_git_sha") or "unknown"
    commit_msg = f"Publish registry data (seed @ {sha[:8] if sha != 'unknown' else 'unknown'})"
    api.upload_folder(
        folder_path=str(fixtures_dir),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
        # Ship parquets + manifest + the YAML overrides that the
        # producer reads alongside (slice_overrides.yaml,
        # display_overrides.yaml). Excluding everything else keeps
        # the dataset small and the artifact list reviewable.
        allow_patterns=["*.parquet", "manifest.json", "*.yaml"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run seed + write manifest locally; skip HF push. Used in PR CI.",
    )
    args = parser.parse_args()

    print("[1/4] Running seed CLI…", file=sys.stderr)
    rc = _run_seed()
    if rc != 0:
        print(f"seed failed with exit code {rc}", file=sys.stderr)
        return rc

    if not FIXTURES_DIR.is_dir():
        print(f"fixtures/ missing after seed; expected {FIXTURES_DIR}", file=sys.stderr)
        return 1

    print("[2/4] Staging YAML overrides into fixtures/…", file=sys.stderr)
    import shutil as _shutil
    for name in SHIPPED_YAML:
        src = SEED_DIR / name
        if src.exists():
            _shutil.copy2(src, FIXTURES_DIR / name)
            print(f"  staged: {name}", file=sys.stderr)
        else:
            print(f"  [warn] {src} missing; consumers will fall back to defaults",
                  file=sys.stderr)

    print("[2/4] Computing content hash…", file=sys.stderr)
    content_hash = _content_hash(FIXTURES_DIR)
    print(f"  content_hash: {content_hash[:16]}…", file=sys.stderr)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "content_hash": content_hash,
        "seed_git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_counts": _row_counts(FIXTURES_DIR),
    }

    print("[3/4] Writing manifest.json…", file=sys.stderr)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  schema_version: {SCHEMA_VERSION}", file=sys.stderr)
    print(f"  row_counts: {sum(manifest['row_counts'].values())} rows across "
          f"{len(manifest['row_counts'])} tables", file=sys.stderr)

    if args.dry_run:
        print("[4/4] --dry-run: skipping HF upload.", file=sys.stderr)
        return 0

    print("[4/4] Checking remote manifest for idempotency…", file=sys.stderr)
    remote = _read_remote_manifest()
    if remote and remote.get("content_hash") == content_hash:
        print(f"  remote content_hash matches local; skipping push (no-op).",
              file=sys.stderr)
        return 0

    print(f"  pushing to {HF_DATASET_REPO}…", file=sys.stderr)
    _push(FIXTURES_DIR, manifest)
    print(f"  done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
