## 🎬 Checkpoint CP2 — IMDB Top 250 (Recommender via Clusterização)

Aluno: Thiago Almança da Silva — RM558108

Disciplina: Front End & Mobile Development — 2º semestre (2025_2)
Entrega: 03/10/2025 às 23:59 via Teams
Apresentação: 03/10/2025 (todos do grupo)

## 🎯 Objetivo

Desenvolver um sistema de recomendação de filmes a partir da clusterização do IMDB Top 250 utilizando PyCaret e Streamlit, com dois métodos de interação:

Escolha de sinopse: usuário seleciona entre 3–5 sinopses anônimas → recebe recomendações do cluster correspondente.

Sinopse escrita: usuário digita uma descrição de filme → modelo classifica em cluster → recomendações geradas.

(Opcional: enriquecimento de sinopses com IA Generativa.)

## 🧭 Abordagem

Pré-processamento:

TF-IDF nas sinopses

One-Hot Encoding de gêneros

Padronização (z-score) em atributos numéricos (ano, votos, nota, duração)

Clusterização:

PyCaret (AutoML clustering)

Comparação de modelos com métricas: Silhouette, Calinski–Harabasz, Davies–Bouldin

Seleção final: melhor modelo escolhido pela combinação das métricas e testes práticos.

Recomendações: seleção dos 5 filmes mais próximos do centróide do cluster + ajuste por nota/votos.

## 🖥️ WebApp (Streamlit)

Interface simples e responsiva.

Dois métodos de recomendação (escolha ou texto).

Resultados apresentados com lista de 5 filmes recomendados.

Atualização automática com --server.runOnSave true.

## ⚙️ Como Executar
1. Pré-requisitos

Python 3.10+

Dependências listadas em requirements.txt

2. Instalar dependências
pip install -r requirements.txt

3. Executar o app

No terminal/PowerShell:

cd C:\Users\thi18\Downloads\projeto-imdb-recommender
streamlit run notebooks\webapp\app.py --server.runOnSave true


A aplicação abrirá automaticamente no navegador em http://localhost:8501/.

## 🗂️ Estrutura do Projeto
.
├── notebooks/
│   ├── 01_preprocessing.ipynb         # limpeza e preparação
│   ├── 02_pycaret_clustering.ipynb    # treino e comparação de modelos
│   └── webapp/
│       └── app.py                     # aplicação Streamlit
├── data/                              # datasets gerados (raw e clusters)
├── requirements.txt
└── README.md

## 📦 Entregáveis

PDF com nome dos integrantes + links do WebApp deployado e repositório GitHub.

WebApp publicado (Heroku, Streamlit Cloud ou similar).

Repositório GitHub com notebooks + app.

## 🚀 Melhorias Futuras

Testar embeddings semânticos (Sentence Transformers).

Explorar DBSCAN/HDBSCAN.

Ajustar pesos entre texto e atributos.

Deploy em Streamlit Cloud para compartilhamento rápido.
