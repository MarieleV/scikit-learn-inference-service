import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
import joblib

print("Gerando dados sintéticos (Simulação de métricas de consumo/anomalias)...")
# Criando 1000 amostras com 4 características (features)
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

# Simulação do cenário semi-supervisionado: ocultando 30% das labels (marcando como -1)
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y)) < 0.3
y[random_unlabeled_points] = -1

print("Iniciando o treinamento do modelo Self-Training...")
# O algoritmo base será uma Floresta Aleatória
base_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_semi_supervisionado = SelfTrainingClassifier(base_classifier)

# Treinamento
modelo_semi_supervisionado.fit(X, y)
print("Treinamento concluído com sucesso!")

# Exportando o modelo treinado para um arquivo binário
nome_arquivo = 'modelo_anomalia.joblib'
joblib.dump(modelo_semi_supervisionado, nome_arquivo)
print(f"Modelo salvo e pronto para produção no arquivo: '{nome_arquivo}'")