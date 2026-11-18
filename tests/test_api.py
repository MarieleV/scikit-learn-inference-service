from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealth:
    def test_health_ok_com_modelo_carregado(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["modelo_carregado"] is True
        assert body["versao_modelo"]

    def test_health_503_sem_modelo(self, client_sem_modelo: TestClient) -> None:
        response = client_sem_modelo.get("/health")
        assert response.status_code == 503
        assert response.json()["modelo_carregado"] is False

    def test_header_de_latencia(self, client: TestClient) -> None:
        assert "X-Process-Time-Ms" in client.get("/health").headers


class TestModelInfo:
    def test_retorna_metadados_e_metricas(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert body["features"] == ["volume_consumo", "variacao_pressao", "vazao_noturna", "fator_sazonal"]
        assert {"self_training", "baseline_supervised"} <= body["metrics"].keys()
        assert body["anomaly_threshold"] == 0.5


class TestPredict:
    def test_predicao_valida(self, client: TestClient, leitura: dict) -> None:
        response = client.post("/predict", json=leitura)
        assert response.status_code == 200
        body = response.json()
        assert body["classe"] in (0, 1)
        assert body["rotulo"] == {0: "normal", 1: "anomalia"}[body["classe"]]
        assert 0.0 <= body["probabilidade_anomalia"] <= 1.0
        assert body["versao_modelo"]

    def test_predicao_e_deterministica(self, client: TestClient, leitura: dict) -> None:
        assert client.post("/predict", json=leitura).json() == client.post("/predict", json=leitura).json()

    @pytest.mark.parametrize(
        ("alteracao", "descricao"),
        [
            ({"volume_consumo": "abc"}, "tipo inválido"),
            ({"volume_consumo": None}, "valor nulo"),
            ({"campo_extra": 1.0}, "campo não previsto"),
        ],
    )
    def test_payload_invalido_retorna_422(
        self, client: TestClient, leitura: dict, alteracao: dict, descricao: str
    ) -> None:
        response = client.post("/predict", json={**leitura, **alteracao})
        assert response.status_code == 422, descricao

    def test_campo_ausente_retorna_422(self, client: TestClient, leitura: dict) -> None:
        leitura.pop("fator_sazonal")
        assert client.post("/predict", json=leitura).status_code == 422

    def test_sem_modelo_retorna_503(self, client_sem_modelo: TestClient, leitura: dict) -> None:
        assert client_sem_modelo.post("/predict", json=leitura).status_code == 503


class TestPredictBatch:
    def test_lote_valido(self, client: TestClient, leitura: dict) -> None:
        response = client.post("/predict/batch", json={"leituras": [leitura] * 3})
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert len(body["predicoes"]) == 3
        assert body["total_anomalias"] == sum(p["classe"] for p in body["predicoes"])

    def test_lote_e_predicao_individual_sao_consistentes(self, client: TestClient, leitura: dict) -> None:
        individual = client.post("/predict", json=leitura).json()
        lote = client.post("/predict/batch", json={"leituras": [leitura]}).json()
        assert lote["predicoes"][0]["probabilidade_anomalia"] == individual["probabilidade_anomalia"]

    def test_lote_vazio_retorna_422(self, client: TestClient) -> None:
        assert client.post("/predict/batch", json={"leituras": []}).status_code == 422

    def test_lote_acima_do_limite_retorna_413(self, client: TestClient, leitura: dict) -> None:
        response = client.post("/predict/batch", json={"leituras": [leitura] * 6})
        assert response.status_code == 413
