import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    local_mode: bool = False
    fixtures_path: str = "./fixtures"
    hf_dataset_repo: str = ""
    hf_token: str = ""
    resolver_auto_merge_threshold: float = 0.85
    read_only: bool = False
    hf_log_bucket: str = ""
    log_flush_interval_seconds: int = 300
    # Live hub-stats enrichment at draft creation. When True (default for
    # CLI sync runs), the resolver's auto-create path looks up unmatched
    # HF-shaped raw values in cfahlgren1/hub-stats and pre-populates the
    # draft entity with release_date / params / parents / lineage_origin.
    # Disable for tests and offline dev — every lookup hits HF.
    hub_stats_lookup_enabled: bool = True
    # Live HF repo-id existence check (the `hf_live` tier of HfIdVerifier).
    # When True (default, incl. the deployed Space), an HF-shaped model raw
    # value that misses the local hub_stats_index is confirmed against the
    # Hub API (one small GET per unknown id, cached, budgeted, and behind a
    # circuit breaker). With it off the checker answers from the index and
    # cache only. Auto-disabled in tests via conftest.
    hf_live_id_check_enabled: bool = True
    # Timeout (seconds) for a single live Hub API id check.
    hf_live_id_check_timeout: float = 5.0


settings = Settings()

# Export HF_TOKEN to the environment so that libraries that read it directly
# (e.g. `datasets.load_dataset`) pick it up, not just code that uses
# `settings.hf_token`.
if settings.hf_token and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = settings.hf_token
