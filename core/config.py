"""One configuration source for the supported Sentinel application."""

from pathlib import Path
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    camera_index: int = Field(0, ge=0, le=20)
    camera_width: int = Field(640, ge=160, le=1920)
    camera_height: int = Field(480, ge=120, le=1080)
    camera_fps: int = Field(30, ge=5, le=60)
    flip_horizontal: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = Field(8000, ge=1024, le=65535)
    data_dir: Path = ROOT / "data" / "sentinel"
    model_path: Path = ROOT / "face_landmarker.task"
    knowledge_dir: Path = ROOT / "docs" / "knowledge"
    embedding_path: Path = ROOT / "models" / "all-MiniLM-L6-v2"
    whisper_path: Path = ROOT / "models" / "whisper-base-en"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    gemini_daily_cap: int = Field(20, ge=0, le=20)
    gemini_interval_s: float = Field(10, ge=10)
    fatigue_warn_threshold: float = Field(55, ge=1, le=95)
    fatigue_crit_threshold: float = Field(72, ge=6, le=100)
    microsleep_ear_threshold: float = Field(0.18, ge=0.05, le=0.4)
    eye_closure_s: float = Field(1.5, ge=0.5, le=5)
    alert_cooldown_s: float = Field(8, ge=8)
    retention_days: int = Field(30, ge=1, le=365)
    audio_enabled: bool = True
    max_frame_age_s: float = Field(2, ge=0.2, le=5)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.fatigue_crit_threshold < self.fatigue_warn_threshold + 5:
            raise ValueError("Critical threshold must be at least five points above warning")
        if self.web_host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This local application must bind to loopback")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sentinel.db"


settings = Settings()
