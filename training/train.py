"""Pipeline de treinamento do modelo de detecção de anomalias de consumo.

Etapas:
    1. Gera um dataset sintético e desbalanceado (anomalias são raras).
    2. Separa treino/teste de forma estratificada.
    3. Oculta parte dos rótulos de treino (cenário semi-supervisionado real:
       rotular uma anomalia exige inspeção em campo, então poucos casos são rotulados).
    4. Treina um baseline supervisionado (só dados rotulados) e o SelfTrainingClassifier.
    5. Avalia ambos no conjunto de teste e salva o modelo + metadados (versão, métricas).

Uso:
    python -m training.train
    python -m training.train --unlabeled-ratio 0.9 --output-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier

logger = logging.getLogger(__name__)

# Contrato de entrada do modelo: a ORDEM importa, pois o scikit-learn recebe um array 2D.
# A API lê esta lista dos metadados para montar o array na ordem correta.
FEATURE_NAMES: list[str] = [
    "volume_consumo",
    "variacao_pressao",
    "vazao_noturna",
    "fator_sazonal",
]

UNLABELED = -1  # convenção do scikit-learn para amostras sem rótulo
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class TrainingConfig:
    n_samples: int = 5_000
    anomaly_ratio: float = 0.15
    unlabeled_ratio: float = 0.8
    test_size: float = 0.2
    n_estimators: int = 100
    self_training_threshold: float = 0.9
    random_state: int = 42


def generate_dataset(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Gera dados sintéticos com classe minoritária (1 = anomalia)."""
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=3,
        n_redundant=1,
        weights=[1 - config.anomaly_ratio, config.anomaly_ratio],
        flip_y=0.02,
        class_sep=1.0,
        random_state=config.random_state,
    )
    return X, y


def hide_labels(y: np.ndarray, ratio: float, random_state: int) -> np.ndarray:
    """Marca uma fração dos rótulos como desconhecidos (-1), sem alterar o array original."""
    rng = np.random.default_rng(random_state)
    y_partial = y.copy()
    mask = rng.random(len(y)) < ratio
    y_partial[mask] = UNLABELED
    return y_partial


def evaluate(model: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Métricas focadas na classe positiva (anomalia), pois o dataset é desbalanceado."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
    }


def train(config: TrainingConfig, output_dir: Path) -> dict:
    """Executa o pipeline completo e retorna os metadados salvos."""
    X, y = generate_dataset(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    y_train_partial = hide_labels(y_train, config.unlabeled_ratio, config.random_state)
    labeled = y_train_partial != UNLABELED

    logger.info(
        "Treino: %d amostras (%d rotuladas, %d sem rótulo) | Teste: %d amostras",
        len(y_train),
        labeled.sum(),
        (~labeled).sum(),
        len(y_test),
    )

    def make_forest() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )

    # Baseline: o que conseguiríamos usando apenas os dados rotulados.
    baseline = make_forest().fit(X_train[labeled], y_train_partial[labeled])
    baseline_metrics = evaluate(baseline, X_test, y_test)

    # Self-training: aproveita as amostras sem rótulo via pseudo-rotulagem iterativa.
    model = SelfTrainingClassifier(
        estimator=make_forest(),
        threshold=config.self_training_threshold,
    ).fit(X_train, y_train_partial)
    model_metrics = evaluate(model, X_test, y_test)

    pseudo_labeled = int((model.labeled_iter_ > 0).sum())
    logger.info("Pseudo-rótulos adicionados pelo self-training: %d", pseudo_labeled)
    logger.info("Baseline (supervisionado): %s", baseline_metrics)
    logger.info("Self-training:            %s", model_metrics)

    trained_at = datetime.now(UTC)
    metadata = {
        "model_version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(),
        "algorithm": "SelfTrainingClassifier(RandomForestClassifier)",
        "features": FEATURE_NAMES,
        "classes": {"0": "normal", "1": "anomalia"},
        "sklearn_version": sklearn.__version__,
        "training_config": asdict(config),
        "dataset": {
            "train_samples": len(y_train),
            "train_labeled": int(labeled.sum()),
            "train_unlabeled": int((~labeled).sum()),
            "pseudo_labeled": pseudo_labeled,
            "test_samples": len(y_test),
        },
        "metrics": {"self_training": model_metrics, "baseline_supervised": baseline_metrics},
    }

    # Paralelismo ajuda no treino, mas em inferência de poucas linhas o custo de
    # orquestrar threads supera o ganho. A API já escala atendendo requisições em paralelo.
    model.estimator_.set_params(n_jobs=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / MODEL_FILENAME)
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Artefatos salvos em '%s'", output_dir.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Treina o modelo de detecção de anomalias.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-samples", type=int, default=defaults.n_samples)
    parser.add_argument("--unlabeled-ratio", type=float, default=defaults.unlabeled_ratio)
    parser.add_argument("--n-estimators", type=int, default=defaults.n_estimators)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = TrainingConfig(
        n_samples=args.n_samples,
        unlabeled_ratio=args.unlabeled_ratio,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
