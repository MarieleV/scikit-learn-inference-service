"""Experimento: o self-training compensa? Compara com o baseline supervisionado
variando a fração de rótulos ocultos, com média sobre várias seeds.

Uso:
    python -m training.experiment
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from statistics import mean

from training.train import TrainingConfig, train

UNLABELED_RATIOS = [0.5, 0.8, 0.9, 0.95, 0.98]
SEEDS = [0, 1, 2]


def main() -> None:
    logging.disable(logging.INFO)
    print("| % sem rótulo | F1 baseline | F1 self-training | ROC-AUC baseline | ROC-AUC self-training |")
    print("|---|---|---|---|---|")
    for ratio in UNLABELED_RATIOS:
        runs = []
        for seed in SEEDS:
            with tempfile.TemporaryDirectory() as tmp:
                config = TrainingConfig(unlabeled_ratio=ratio, random_state=seed)
                runs.append(train(config, Path(tmp))["metrics"])

        def avg(model: str, metric: str, runs: list = runs) -> str:
            return f"{mean(r[model][metric] for r in runs):.3f}"

        print(
            f"| {ratio:.0%} | {avg('baseline_supervised', 'f1')} | {avg('self_training', 'f1')} "
            f"| {avg('baseline_supervised', 'roc_auc')} | {avg('self_training', 'roc_auc')} |"
        )


if __name__ == "__main__":
    main()
