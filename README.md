<div align="center">
  <img src="https://img.icons8.com/fluency/96/000000/brain.png" alt="Ícone de IA" width="80"/>

  <h1>Serviço de Inferência de ML com FastAPI</h1>
  <h3>Self-Training Classifier (aprendizado semi-supervisionado)</h3>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-API_REST-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/Joblib-Model_Export-8A2BE2?style=for-the-badge&logo=python&logoColor=white" alt="Joblib">
  </p>

  <p><em>Microsserviço que separa o treinamento da inferência e serve um modelo semi-supervisionado via API REST.</em></p>
</div>

---

## Sobre o Projeto

Este projeto demonstra como levar um modelo de Machine Learning para uma API, servindo como ponte entre Ciência de Dados e Engenharia de Software (deploy local).

O algoritmo utilizado é o **`SelfTrainingClassifier`** (semi-supervisionado) do scikit-learn, treinado sobre uma base de dados **sintética** de validação. O ciclo de vida é dividido em duas etapas independentes:

1. **Treinamento:** o script `train_model.py` gera os dados, treina o modelo e o serializa em arquivo binário com `joblib`.
2. **Inferência:** a API em **FastAPI** carrega o modelo em memória no *startup* e responde a requisições JSON na rota `/predict`.

> O projeto tem finalidade demonstrativa: os dados são sintéticos e o foco está na arquitetura de serviço, não na performance do modelo.

---

## Principais Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| **Treinamento isolado** | O treino ocorre em script separado (`train_model.py`) e não consome recursos da aplicação web, persistindo o resultado em um artefato binário. |
| **Inferência em memória** | No *startup*, a API carrega o modelo já treinado na RAM, respondendo sem retreinar e sem acessar banco de dados. |
| **Validação com Pydantic** | Garante que o payload JSON tenha a tipagem correta (`float`) exigida pelo array 2D do scikit-learn, evitando erros em tempo de execução. |
| **Documentação automática** | Swagger/OpenAPI gerado pelo FastAPI, com teste interativo da rota `/predict`. |

---

## Arquitetura

```
train_model.py ──► modelo_anomalia.joblib ──► main.py (FastAPI)
 (treino offline)     (artefato binário)       │
                                               ├─ startup: carrega o modelo na memória
                                               └─ POST /predict: valida (Pydantic) → inferência → JSON
```

<!-- AJUSTE: o artefato se chama "modelo_anomalia.joblib", mas o modelo é um SelfTrainingClassifier.
     Sugestão: renomear para "modelo_self_training.joblib" (ajustando train_model.py e main.py)
     ou, se o objetivo for detecção de anomalias, atualizar o título e a descrição deste README. -->

---

## Estrutura do Repositório

```
.
├── main.py                  # API FastAPI (carrega o modelo e expõe /predict)
├── train_model.py           # Geração dos dados, treino e serialização do modelo
├── modelo_anomalia.joblib   # Modelo treinado (artefato binário)
├── requirements.txt         # Dependências
└── README.md
```

---

## Como Executar

**1. Clone o repositório e entre na pasta:**

```bash
git clone https://github.com/MarieleV/scikit-learn-inference-service.git
cd scikit-learn-inference-service
```

**2. Crie e ative o ambiente virtual:**

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. (Opcional) Retreine o modelo:**

O repositório já inclui o modelo treinado. Para gerar um novo artefato:

```bash
python train_model.py
```

**5. Inicie a API:**

```bash
uvicorn main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`.

---

## Documentação da API (Swagger)

Com a aplicação rodando, acesse a documentação interativa para testar a rota `/predict` direto no navegador:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Exemplo de Uso

<!-- AJUSTE: confira em main.py o nome do campo do payload (aqui "features"), a quantidade de valores
     esperada e o formato real da resposta. Substitua o exemplo abaixo pelos valores corretos. -->

**Requisição:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.52, -1.30, 0.87, 0.14]}'
```

**Resposta:**

```json
{
  "prediction": 1
}
```

---

## Próximos Passos

- Adicionar testes automatizados da rota `/predict` com `pytest` e `TestClient`.
- Criar um `Dockerfile` para padronizar a execução do serviço.
- Incluir uma rota de *health check* e versionamento do modelo.
- Registrar métricas de avaliação do modelo (acurácia, F1) no `train_model.py`.
