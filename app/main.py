"""API de inferência para detecção de anomalias de consumo."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from app.config import Settings
from app.model import ModelLoadError, ModelService
from app.schemas import (
    LeituraConsumo,
    LoteLeituras,
    RespostaLote,
    RespostaPredicao,
    StatusSaude,
)

logger = logging.getLogger(__name__)

DESCRICAO = """
Microsserviço que serve um modelo **semi-supervisionado** (`SelfTrainingClassifier` +
`RandomForestClassifier`) para identificar **anomalias de consumo de água** (vazamentos ou fraudes).

* O modelo é treinado offline (`python -m training.train`) e carregado **uma única vez** no startup.
* As entradas são validadas pelo **Pydantic** antes de chegarem ao modelo.
* A resposta traz a classe, a probabilidade e a versão do modelo usada (rastreabilidade).
"""


def get_model_service(request: Request) -> ModelService:
    """Dependência: injeta o modelo nas rotas ou responde 503 se ele não estiver carregado."""
    service: ModelService | None = request.app.state.model_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique os logs do servidor.",
        )
    return service


ModelDep = Annotated[ModelService, Depends(get_model_service)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory: facilita testes com configurações diferentes."""
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_service = None
        try:
            app.state.model_service = ModelService.load(settings.model_dir, settings.anomaly_threshold)
        except ModelLoadError as exc:
            # A API sobe mesmo assim: /health reporta o problema e as rotas de predição retornam 503.
            logger.error("Falha ao carregar o modelo: %s", exc)
        yield

    app = FastAPI(
        title="ML Inference Service — Detecção de Anomalias",
        description=DESCRICAO,
        version="2.0.0",
        lifespan=lifespan,
        contact={"name": "Mariele Vieira da Silva", "url": "https://github.com/MarieleV"},
        license_info={"name": "MIT"},
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.get("/health", response_model=StatusSaude, tags=["Operação"], summary="Health check")
    def health(request: Request, response: Response) -> StatusSaude:
        service: ModelService | None = request.app.state.model_service
        if service is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return StatusSaude(status="indisponivel", modelo_carregado=False)
        return StatusSaude(status="ok", modelo_carregado=True, versao_modelo=service.version)

    @app.get("/model/info", tags=["Operação"], summary="Metadados e métricas do modelo")
    def model_info(service: ModelDep) -> dict[str, Any]:
        return {**service.metadata, "anomaly_threshold": service.threshold}

    # Rotas síncronas de propósito: a inferência é CPU-bound, então o FastAPI as executa
    # em um threadpool em vez de bloquear o event loop (o que aconteceria com `async def`).
    @app.post(
        "/predict",
        response_model=RespostaPredicao,
        tags=["Inferência"],
        summary="Classifica uma leitura de consumo",
    )
    def predict(leitura: LeituraConsumo, service: ModelDep) -> RespostaPredicao:
        predicao = service.predict([leitura])[0]
        return RespostaPredicao(**predicao.model_dump(), versao_modelo=service.version)

    @app.post(
        "/predict/batch",
        response_model=RespostaLote,
        tags=["Inferência"],
        summary="Classifica várias leituras em uma única chamada ao modelo",
    )
    def predict_batch(lote: LoteLeituras, service: ModelDep) -> RespostaLote:
        if len(lote.leituras) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"O lote excede o limite de {settings.max_batch_size} leituras.",
            )
        predicoes = service.predict(lote.leituras)
        return RespostaLote(
            versao_modelo=service.version,
            total=len(predicoes),
            total_anomalias=sum(p.classe for p in predicoes),
            predicoes=predicoes,
        )

    return app


app = create_app()
