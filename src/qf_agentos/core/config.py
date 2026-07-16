"""Runtime configuration for QF-AgentOS.

Settings are read from the environment (prefix ``QF_``) with sane production
defaults, and can be overridden programmatically. Backend credentials are held
as :class:`~pydantic.SecretStr` so they are never accidentally logged or
serialised into an evidence bundle.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="QF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    tracing_enabled: bool = False  # OpenTelemetry spans per agent step

    # Determinism / numerics
    default_seed: int = 7
    numeric_rel_tol: float = 1e-6

    # Backend simulation budgets (guard-rails against runaway resource use)
    statevector_qubit_limit: int = Field(default=22, gt=0)
    bruteforce_qubit_limit: int = Field(default=20, gt=0)

    # Governance / experiment registry
    evidence_dir: Path = Path("evidence")
    # Where solved runs are persisted: "file" (default, dependency-free) or
    # "mlflow" (requires the 'mlflow' extra) for a shared experiment registry.
    registry_backend: Literal["file", "mlflow"] = "file"
    mlflow_tracking_uri: str | None = None  # None -> MLflow's default (./mlruns)
    mlflow_experiment: str = "qf-agentos"

    # API safety: reject specs larger than this over the (synchronous) REST path,
    # so a single unauthenticated request cannot drive a huge classical solve.
    api_max_inventory: int = Field(default=2000, gt=0)

    # API authentication: a comma-separated set of accepted X-API-Key values.
    # If empty, the API is OPEN (development mode) — a startup warning is logged.
    api_keys: str = Field(default="", description="Comma-separated API keys; empty = open.")
    api_rate_limit_per_minute: int = Field(default=60, gt=0)

    # Async job queue behind POST /jobs (in-process thread pool; single instance).
    api_job_workers: int = Field(default=2, gt=0, description="Concurrent solve workers.")
    api_max_jobs: int = Field(default=256, gt=0, description="Retained job records (LRU).")

    # Credentials (never logged; loaded only when the matching backend runs)
    ibm_token: SecretStr | None = None
    ibm_instance: str | None = None
    ibm_backend: str | None = None
    dwave_token: SecretStr | None = None

    def has_ibm_credentials(self) -> bool:
        return self.ibm_token is not None

    def has_dwave_credentials(self) -> bool:
        return self.dwave_token is not None

    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    def auth_required(self) -> bool:
        return bool(self.api_key_set())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (used in tests that mutate the environment)."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
