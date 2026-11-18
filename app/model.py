"""Camada de serviço do modelo: carregamento do artefato e inferência.

Mantida separada das rotas para que a API não conheça detalhes do scikit-learn
(e para facilitar testes e uma futura troca de modelo).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from app.schemas import LeituraConsumo, Predicao

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
ROTULOS = {0: "normal", 1: "anomalia"}


class ModelLoadError(RuntimeError):
    """Artefato ausente, corrompido ou incompatível com o contrato da API."""


@dataclass
class ModelService:
    model: Any
    metadata: dict[str, Any]
    threshold: float = 0.5

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def feature_names(self) -> list[str]:
        return self.metadata["features"]

    @classmethod
    def load(cls, model_dir: Path, threshold: float = 0.5) -> ModelService:
        model_path = model_dir / MODEL_FILENAME
        metadata_path = model_dir / METADATA_FILENAME
        if not model_path.exists() or not metadata_path.exists():
            raise ModelLoadError(
                f"Artefatos não encontrados em '{model_dir}'. Rode 'python -m training.train' primeiro."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)

        # O contrato do modelo (features) precisa bater exatamente com o schema da API.
        expected = set(LeituraConsumo.model_fields)
        if set(metadata["features"]) != expected:
            raise ModelLoadError(
                f"Features do modelo {metadata['features']} diferem do schema da API {sorted(expected)}."
            )

        # Pickles do scikit-learn não são garantidos entre versões diferentes.
        if metadata.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Modelo treinado com scikit-learn %s, mas o runtime usa %s. Considere retreinar.",
                metadata.get("sklearn_version"),
                sklearn.__version__,
            )

        logger.info("Modelo versão %s carregado de '%s'", metadata["model_version"], model_dir)
        return cls(model=model, metadata=metadata, threshold=threshold)

    def _to_array(self, leituras: list[LeituraConsumo]) -> np.ndarray:
        # Monta o array na mesma ordem de colunas usada no treino.
        return np.array(
            [[getattr(leitura, name) for name in self.feature_names] for leitura in leituras],
            dtype=float,
        )

    def predict(self, leituras: list[LeituraConsumo]) -> list[Predicao]:
        """Inferência vetorizada: uma única chamada ao modelo para todo o lote."""
        probabilidades = self.model.predict_proba(self._to_array(leituras))[:, 1]
        predicoes = []
        for prob in probabilidades:
            classe = int(prob >= self.threshold)
            predicoes.append(
                Predicao(
                    classe=classe,
                    rotulo=ROTULOS[classe],
                    probabilidade_anomalia=round(float(prob), 4),
                )
            )
        return predicoes
