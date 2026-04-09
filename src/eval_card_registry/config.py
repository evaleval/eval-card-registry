import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    local_mode: bool = False
    fixtures_path: str = "./fixtures"
    hf_dataset_repo: str = ""
    hf_token: str = ""
    resolver_auto_merge_threshold: float = 0.85


settings = Settings()

# Export HF_TOKEN to the environment so that libraries that read it directly
# (e.g. `datasets.load_dataset`) pick it up, not just code that uses
# `settings.hf_token`.
if settings.hf_token and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = settings.hf_token
