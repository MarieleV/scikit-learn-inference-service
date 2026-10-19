from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Configuração da Documentação (Swagger UI) ---
descricao_api = """
API para predição em tempo real utilizando um modelo de Machine Learning semi-supervisionado (`SelfTrainingClassifier`).

O sistema recebe dados numéricos via JSON, converte os atributos estruturados pelo **Pydantic** e realiza a inferência em memória com **Scikit-Learn**, retornando a classificação de risco em milissegundos.
"""

app = FastAPI(
    title="ML Predict API | Portfólio de Inteligência Artificial",
    description=descricao_api,
    version="1.0.0",
    contact={
        "name": "Mariele Vieira da Silva",
        "url": "https://github.com/MarieleV",
        "email": "silvamariele797@gmail.com",
    }
)

# Variável global que vai guardar o modelo em memória
modelo_ml = None

# Evento de inicialização do FastAPI
@app.on_event("startup")
def carregar_modelo():
    global modelo_ml
    try:
        modelo_ml = joblib.load('modelo_anomalia.joblib')
        print("Modelo de ML carregado com sucesso na memória do servidor!")
    except Exception as e:
        print(f"ATENÇÃO: Erro ao carregar o modelo. Verifique se você rodou o train_model.py antes. Erro: {e}")

# --- Validação de Entrada (Pydantic) ---
class FeaturesConsumo(BaseModel):
    feature_1: float = Field(..., description="Volume registrado no hidrômetro/sensor")
    feature_2: float = Field(..., description="Variação de pressão hídrica na tubulação")
    feature_3: float = Field(..., description="Frequência de vazão noturna")
    feature_4: float = Field(..., description="Fator de sazonalidade climática")

@app.post(
    "/predict", 
    tags=["Machine Learning Inference"], 
    summary="Realizar predição de anomalias de consumo",
    description="Recebe 4 variáveis independentes (features) e retorna a classe prevista (0 = Padrão Normal, 1 = Possível Anomalia/Fraude)."
)
async def prever_anomalia(dados: FeaturesConsumo):
    if modelo_ml is None:
        raise HTTPException(status_code=500, detail="Modelo não disponível no servidor.")

    try:
        # Transforma os dados do JSON em um Array 2D que o Scikit-Learn exige
        input_array = np.array([[
            dados.feature_1, 
            dados.feature_2, 
            dados.feature_3, 
            dados.feature_4
        ]])
        
        # Fazendo a inferência
        predicao = modelo_ml.predict(input_array)
        resultado_int = int(predicao[0])
        
        return {
            "status": "sucesso",
            "resultado_classe": resultado_int,
            "interpretacao": "Anomalia Detectada" if resultado_int == 1 else "Consumo Normalizado"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante o processamento da inferência: {str(e)}")