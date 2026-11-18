"""Configuração da aplicação via variáveis de ambiente (12-factor app)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("models")
    anomaly_threshold: float = 0.5
    max_batch_size: int = 1_000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            model_dir=Path(os.getenv("MODEL_DIR", str(cls.model_dir))),
            anomaly_threshold=float(os.getenv("ANOMALY_THRESHOLD", cls.anomaly_threshold)),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", cls.max_batch_size)),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )
        if not 0.0 < settings.anomaly_threshold < 1.0:
            raise ValueError("ANOMALY_THRESHOLD deve estar entre 0 e 1 (exclusivo).")
        return settings
