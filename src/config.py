from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://mortality_user:mortality_pass@localhost:5432/mortality_db"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    model_dir: Path = Path("artifacts")
    data_dir: Path = Path("data/physionet")
    default_model: str = "gradient_boosting"


settings = Settings()
