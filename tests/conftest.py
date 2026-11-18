from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from training.train import TrainingConfig, train

# Configuração reduzida para os testes rodarem em poucos segundos.
FAST_CONFIG = TrainingConfig(n_samples=600, n_estimators=10)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    train(FAST_CONFIG, path)
    return path


@pytest.fixture
def client(model_dir: Path) -> Iterator[TestClient]:
    app = create_app(Settings(model_dir=model_dir, max_batch_size=5))
    with TestClient(app) as test_client:  # o "with" dispara o lifespan (carga do modelo)
        yield test_client


@pytest.fixture
def client_sem_modelo(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(Settings(model_dir=tmp_path / "inexistente"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def leitura() -> dict[str, float]:
    return {
        "volume_consumo": 1.25,
        "variacao_pressao": -0.40,
        "vazao_noturna": 2.10,
        "fator_sazonal": 0.30,
    }
