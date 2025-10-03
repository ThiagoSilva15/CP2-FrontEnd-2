# 🎬 IMDB Recommender — CP1/CP2 (FIAP)

Estrutura pronta para VSCode com notebooks (01–04) e WebApp (Streamlit).

## Como usar
```bash
# 1) Ambiente e dependências (raiz)
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2) (opcional) Dependências do WebApp
cd webapp && pip install -r requirements.txt && cd ..
```

## Notebooks
- `01_scrape_and_kmeans_synopsis.ipynb` — CP1: scraping + KMeans (sinopse)
- `02_kmeans_all_features.ipynb` — CP1: KMeans (todas as features)
- `03_sklearn_synopsis_only.ipynb` — CP2: Sklearn (sinopse-only) → salva tfidf/svd/kmeans
- `04_pycaret_all_features.ipynb` — CP2: PyCaret (all-features) → salva best_cluster_model

## Rodar WebApp
```bash
cd webapp
streamlit run app.py
```


projeto-imdb-recommender/
├── data/
│   ├── imdb_top250_raw.csv
│   ├── imdb_top250_k5_synopsis.csv
│   ├── imdb_top250_k5_allfeatures.csv
│   ├── imdb_top250_synopsis_clusters.csv
│   └── imdb_top250_allfeat_clusters.csv
│
├── models/
│   ├── best_cluster_model.pkl        # Modelo PyCaret (todas as features)
│   ├── tfidf.pkl                # Modelo Sklearn (sinopse-only)
│   ├── svd.pkl                    # Redução dimensional para sinopses
│   └── kmeans_synopsis.pkl          # KMeans para sinopses
│
├── notebooks/
│   ├── 01_scrape_and_kmeans_synopsis.ipynb    # CP1: Clusterização por sinopse
│   ├── 02_kmeans_all_features.ipynb       # CP1: Clusterização com todas as features
│   ├── 03_sklearn_synopsis_only.ipynb    # CP2: Modelo Sklearn (sinopse-only)
│   └── 04_pycaret_all_features.ipynb         # CP2: PyCaret (todas as features)
│
├── webapp/
│   ├── app.py                        # Código do Streamlit
│   ├── recommender.py               # Lógica de recomendação
│   ├── models/ (link simbólico ou cópia dos modelos)
│   ├── data/ (link ou cópia dos CSVs finais)
│   └── requirements.txt
│
└── README.md                  # Documentação principal
