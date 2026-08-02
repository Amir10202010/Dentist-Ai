"""Typed application configuration.

Every knob lives here and is validated at import time, so a misconfigured
deployment fails at boot with a precise error.
"""

from __future__ import annotations

import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_RATE_LIMIT_RE = re.compile(r"^(?P<count>\d+)/(?P<value>\d+)(?P<unit>[smhd])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

#: Below this a signing key is brute-forceable; refuse to boot in production.
MIN_SECRET_KEY_LENGTH = 32


class RateLimitRule(BaseModel):
    """A parsed ``<count>/<window>`` rule, e.g. ``10/5m``."""

    limit: int = Field(gt=0)
    window_seconds: int = Field(gt=0)

    @classmethod
    def parse(cls, raw: str) -> RateLimitRule:
        match = _RATE_LIMIT_RE.match(raw.strip())
        if match is None:
            msg = f"Invalid rate limit {raw!r}; expected e.g. '10/5m' (count/window)."
            raise ValueError(msg)
        window = int(match["value"]) * _UNIT_SECONDS[match["unit"]]
        return cls(limit=int(match["count"]), window_seconds=window)


class DatabaseSettings(BaseModel):
    url: str = "sqlite+aiosqlite:///./var/dentist_ai.sqlite3"
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_recycle_seconds: int = Field(default=1800, ge=60)
    echo: bool = False

    @field_validator("url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if not value.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            msg = (
                "DATABASE__URL must use an async driver: "
                "'postgresql+asyncpg://…' or 'sqlite+aiosqlite://…'"
            )
            raise ValueError(msg)
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class MLSettings(BaseModel):
    backend: Literal["yolo", "stub"] = "stub"
    weights_path: Path = Path("models/best.pt")
    device: str = "cpu"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, gt=0)
    #: Inference is CPU-bound and releases the GIL only partially; a bounded
    #: worker pool keeps the event loop responsive without thrashing cores.
    worker_threads: int = Field(default=2, ge=1, le=16)
    warm_up_on_startup: bool = True

    @property
    def resolved_weights_path(self) -> Path:
        if self.weights_path.is_absolute():
            return self.weights_path
        return PROJECT_ROOT / self.weights_path


class StorageSettings(BaseModel):
    root: Path = Path("var/storage")
    max_upload_bytes: int = Field(default=24 * 1024 * 1024, gt=0)
    #: Meshes get their own, larger ceiling: a full-arch intraoral scan is
    #: routinely 30-60 MB before decimation.
    max_mesh_bytes: int = Field(default=96 * 1024 * 1024, gt=0)
    #: CBCT studies are larger again: a 400-slice 16-bit series is ~200 MB
    #: uncompressed, and a zip of one compresses poorly because the payload is
    #: already high-entropy reconstruction noise.
    max_volume_bytes: int = Field(default=768 * 1024 * 1024, gt=0)
    #: Longest edge, in voxels, kept after ingest decimation. 256 fits a
    #: WebGL2 3D texture on integrated graphics; raising it past 320 starts
    #: failing texture allocation on the low end.
    max_volume_dimension: int = Field(default=256, ge=64, le=512)
    #: Longest edge, in pixels, kept for the stored master image.
    max_image_dimension: int = Field(default=4096, gt=0)
    thumbnail_dimension: int = Field(default=512, gt=0)

    @property
    def resolved_root(self) -> Path:
        return self.root if self.root.is_absolute() else PROJECT_ROOT / self.root


class SecuritySettings(BaseModel):
    session_cookie_name: str = "dentist_ai_session"
    session_max_age_seconds: int = Field(default=14 * 24 * 3600, gt=0)
    csrf_cookie_name: str = "dentist_ai_csrf"
    csrf_header_name: str = "X-CSRF-Token"
    login_rate_limit: str = "10/5m"
    register_rate_limit: str = "5/1h"
    upload_rate_limit: str = "60/1h"
    api_rate_limit: str = "600/1m"
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    #: Argon2id parameters. Defaults follow OWASP's 2024 guidance for
    #: interactive logins (19 MiB, 2 iterations, 1 lane).
    argon2_memory_kib: int = Field(default=19 * 1024, ge=8 * 1024)
    argon2_time_cost: int = Field(default=2, ge=1)
    argon2_parallelism: int = Field(default=1, ge=1)

    @property
    def login_rule(self) -> RateLimitRule:
        return RateLimitRule.parse(self.login_rate_limit)

    @property
    def register_rule(self) -> RateLimitRule:
        return RateLimitRule.parse(self.register_rate_limit)

    @property
    def upload_rule(self) -> RateLimitRule:
        return RateLimitRule.parse(self.upload_rate_limit)

    @property
    def api_rule(self) -> RateLimitRule:
        return RateLimitRule.parse(self.api_rate_limit)

    @model_validator(mode="after")
    def _validate_rules(self) -> SecuritySettings:
        for raw in (
            self.login_rate_limit,
            self.register_rate_limit,
            self.upload_rate_limit,
            self.api_rate_limit,
        ):
            RateLimitRule.parse(raw)
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DENTIST_AI__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "local"
    secret_key: SecretStr = SecretStr("")
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "production")

    @property
    def debug(self) -> bool:
        return self.environment == "local"

    @model_validator(mode="after")
    def _validate_secret(self) -> Settings:
        raw = self.secret_key.get_secret_value()
        if self.is_production:
            if len(raw) < MIN_SECRET_KEY_LENGTH:
                msg = (
                    "SECRET_KEY must be at least 32 characters in staging/production. "
                    "Generate one with: python -c "
                    '"import secrets; print(secrets.token_urlsafe(48))"'
                )
                raise ValueError(msg)
            if "change-me" in raw:
                msg = "SECRET_KEY still holds the placeholder value from .env.example."
                raise ValueError(msg)
        elif not raw:
            # Ephemeral key for local runs: sessions reset on restart.
            object.__setattr__(self, "secret_key", SecretStr(secrets.token_urlsafe(48)))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
