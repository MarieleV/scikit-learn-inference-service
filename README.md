<div align="center">
  <img src="https://img.icons8.com/fluency/96/000000/brain.png" alt="Ícone de IA" width="80"/>
  
  <h1>FastAPI ML Predictor: Self-Training Classifier</h1>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-API_REST-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/scikit_learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/Joblib-Model_Export-8A2BE2?style=for-the-badge&logo=python&logoColor=white" alt="Joblib">
  </p>

  <p><em>Um microsserviço de inteligência artificial em tempo real, focado em produtizar algoritmos semi-supervisionados.</em></p>
</div>

<br>

<h2>🤖 Sobre o Projeto</h2>
<p>
  Este projeto demonstra a produtização (<i>deploy</i> local) de um modelo de Machine Learning, servindo como uma ponte arquitetural entre a Engenharia de Dados e a Engenharia de Software.
</p>
<p>
  O algoritmo utilizado foi o <b>SelfTrainingClassifier</b> (Semi-supervisionado) atuando sobre uma base de dados sintética de validação. O ciclo de vida do projeto é dividido em duas etapas rigorosas: um script de Ciências de Dados para a geração e serialização em binário do modelo (usando <code>joblib</code>) e uma API RESTful de alta performance em <b>FastAPI</b> encarregada exclusivamente de carregar o modelo em memória durante o <i>startup</i> e responder requisições JSON de inferência na rota <code>/predict</code>.
</p>

<br>

<h2>🚀 Principais Funcionalidades</h2>

<table>
  <tr>
    <td width="200"><b>Treinamento Isolado</b></td>
    <td>O treinamento não compromete os recursos da aplicação web. Ocorre em script separado (<code>train_model.py</code>), persistindo as regras em um artefato binário.</td>
  </tr>
  <tr>
    <td><b>Inferência em Memória</b></td>
    <td>No <i>startup</i>, a API aloca o modelo já treinado na RAM, permitindo classificações na casa dos milissegundos sem reprocessamento ou acessos a banco.</td>
  </tr>
  <tr>
    <td><b>Pydantic Validation</b></td>
    <td>Garante que o payload das requisições JSON possua a tipagem perfeita (<i>float</i>) exigida pelo <i>Array 2D</i> do Scikit-Learn para prevenir erros de <i>runtime</i>.</td>
  </tr>
</table>

<br>

<h2>⚙️ Como Executar o Projeto</h2>

<b>1. Clone o repositório e ative o ambiente virtual:</b>
```bash
git clone [https://github.com/MarieleV/fastapi-ml-predict.git](https://github.com/MarieleV/fastapi-ml-predict.git)
cd fastapi-ml-predict
python -m venv venv
.\venv\Scripts\Activate.ps1