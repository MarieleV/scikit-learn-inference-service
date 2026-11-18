from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.model import ModelLoadError, ModelService
from app.schemas import LeituraConsumo
from training.train import FEATURE_NAMES, UNLABELED, TrainingConfig, generate_dataset, hide_labels


def test_dataset_desbalanceado() -> None:
    X, y = generate_dataset(TrainingConfig(n_samples=2_000))
    assert X.shape == (2_000, len(FEATURE_NAMES))
    assert 0.10 < y.mean() < 0.20  # ~15% de anomalias


def test_hide_labels_respeita_proporcao_e_nao_altera_original() -> None:
    y = np.ones(10_000, dtype=int)
    y_partial = hide_labels(y, ratio=0.8, random_state=0)
    assert (y == 1).all()
    assert (y_partial == UNLABELED).mean() == pytest.approx(0.8, abs=0.02)


def test_artefatos_gerados(model_dir: Path) -> None:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (model_dir / "model.joblib").exists()
    assert metadata["features"] == FEATURE_NAMES
    assert metadata["dataset"]["train_unlabeled"] > 0
    for metrics in metadata["metrics"].values():
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_modelo_tem_desempenho_minimo(model_dir: Path) -> None:
    """Teste de sanidade: o modelo precisa ser melhor que o acaso."""
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["metrics"]["self_training"]["roc_auc"] > 0.75


def test_features_do_treino_batem_com_schema_da_api() -> None:
    assert set(FEATURE_NAMES) == set(LeituraConsumo.model_fields)


def test_load_falha_sem_artefatos(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        ModelService.load(tmp_path)


def test_load_falha_com_features_incompativeis(model_dir: Path, tmp_path: Path) -> None:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["features"] = ["outra_feature"]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "model.joblib").write_bytes((model_dir / "model.joblib").read_bytes())
    with pytest.raises(ModelLoadError, match="diferem do schema"):
        ModelService.load(tmp_path)


def test_threshold_altera_a_classe(model_dir: Path) -> None:
    leitura = LeituraConsumo(volume_consumo=0.5, variacao_pressao=0.1, vazao_noturna=0.2, fator_sazonal=0.0)
    sempre_anomalia = ModelService.load(model_dir, threshold=1e-9).predict([leitura])[0]
    nunca_anomalia = ModelService.load(model_dir, threshold=1 - 1e-9).predict([leitura])[0]
    assert sempre_anomalia.classe == 1
    assert nunca_anomalia.classe == 0 or nunca_anomalia.probabilidade_anomalia == 1.0
