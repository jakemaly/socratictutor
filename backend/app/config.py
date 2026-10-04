"""Backend-only settings for an OpenAI-compatible chat endpoint."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read model connection details from the environment or repository .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None

    @property
    def model_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)
