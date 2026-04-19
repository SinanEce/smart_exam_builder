from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "SmartExam Builder"
    environment: str = "development"

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SMARTEXAM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    use_mock_llm: bool = True

    embedding_dim: int = 384
    max_chunk_chars: int = 900
    chunk_overlap_chars: int = 160
    default_top_k: int = 5

    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    samples_dir: Path | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="SMARTEXAM_",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def derive_paths(self) -> "Settings":
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        if self.samples_dir is None:
            self.samples_dir = self.data_dir / "samples"
        return self

    def ensure_directories(self) -> None:
        for path in [self.data_dir, self.raw_data_dir, self.processed_data_dir, self.samples_dir]:
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

