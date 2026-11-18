"""Contratos de entrada e saída da API (validados pelo Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXEMPLO_LEITURA = {
    "volume_consumo": 1.25,
    "variacao_pressao": -0.40,
    "vazao_noturna": 2.10,
    "fator_sazonal": 0.30,
}


class LeituraConsumo(BaseModel):
    """Leitura de um sensor/hidrômetro. Valores já normalizados (escala do treino)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXEMPLO_LEITURA})

    volume_consumo: float = Field(..., allow_inf_nan=False, description="Volume registrado no hidrômetro")
    variacao_pressao: float = Field(..., allow_inf_nan=False, description="Variação de pressão na tubulação")
    vazao_noturna: float = Field(..., allow_inf_nan=False, description="Frequência de vazão noturna")
    fator_sazonal: float = Field(..., allow_inf_nan=False, description="Fator de sazonalidade climática")


class LoteLeituras(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"leituras": [EXEMPLO_LEITURA]}})

    leituras: list[LeituraConsumo] = Field(..., min_length=1)


class Predicao(BaseModel):
    classe: Literal[0, 1] = Field(..., description="0 = normal, 1 = anomalia")
    rotulo: Literal["normal", "anomalia"]
    probabilidade_anomalia: float = Field(..., ge=0, le=1)


class RespostaPredicao(Predicao):
    versao_modelo: str


class RespostaLote(BaseModel):
    versao_modelo: str
    total: int
    total_anomalias: int
    predicoes: list[Predicao]


class StatusSaude(BaseModel):
    status: Literal["ok", "indisponivel"]
    modelo_carregado: bool
    versao_modelo: str | None = None
