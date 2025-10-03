# ===============================================
# 🎬 IMDB Recommender Engine
# recommender.py 
# ===============================================
import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


class IMDBRecommender:
    """
    Classe que encapsula toda a lógica de carregamento de modelos
    e geração de recomendações de filmes.
    """

    def __init__(self, data_path="webapp/data/imdb_top250_k5_synopsis.csv",
                 models_path="webapp/models"):
        self.data_path = data_path
        self.models_path = models_path
        self.df = None
        self.models = {}
        self._load_data()
        self._load_models()

    # ===============================================
    # 📂 Carregar Dataset
    # ===============================================
    def _load_data(self):
        if os.path.exists(self.data_path):
            try:
                self.df = pd.read_csv(self.data_path)
                # Normalizar cluster se já existir
                if "cluster" in self.df.columns:
                    self.df["cluster"] = self.df["cluster"].astype(str)
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar dataset: {e}")
        else:
            raise FileNotFoundError(f"Dataset não encontrado em {self.data_path}")

    # ===============================================
    # 📦 Carregar Modelos
    # ===============================================
    def _load_models(self):
        expected_models = ["tfidf.pkl", "svd.pkl", "kmeans_synopsis.pkl", "best_cluster_model.pkl"]
        for model_file in expected_models:
            model_path = os.path.join(self.models_path, model_file)
            if os.path.exists(model_path):
                try:
                    self.models[model_file.replace(".pkl", "")] = joblib.load(model_path)
                except Exception as e:
                    print(f"⚠️ Falha ao carregar {model_file}: {e}")
            else:
                print(f"⚠️ Modelo não encontrado: {model_file}")

    # ===============================================
    # 🔎 Recomendação baseada em sinopse
    # ===============================================
    def recommend_by_synopsis(self, query, top_n=5):
        if not self.df is None and not self.df.empty:
            if "tfidf" not in self.models or "svd" not in self.models or "kmeans_synopsis" not in self.models:
                raise RuntimeError("Modelos necessários não carregados (tfidf, svd, kmeans).")

            tfidf = self.models["tfidf"]
            svd = self.models["svd"]
            kmeans = self.models["kmeans_synopsis"]

            # Vetorizar e reduzir a query
            query_vec = tfidf.transform([query])
            query_reduced = svd.transform(query_vec)

            # Predizer cluster
            cluster = kmeans.predict(query_reduced)[0]

            # Filtrar filmes do mesmo cluster
            if "cluster" not in self.df.columns:
                raise RuntimeError("Dataset não possui coluna 'cluster'. Rode os notebooks primeiro.")
            cluster_movies = self.df[self.df["cluster"] == cluster].copy()

            # Similaridade de cosseno
            synopsis_matrix = tfidf.transform(cluster_movies["synopsis"].fillna(""))
            query_sim = cosine_similarity(query_vec, synopsis_matrix).flatten()
            cluster_movies["similarity"] = query_sim

            cluster_movies = cluster_movies.sort_values("similarity", ascending=False)

            return cluster, cluster_movies.head(top_n)
        else:
            raise RuntimeError("Dataset vazio ou não carregado.")

    # ===============================================
    # 📊 Explorar Clusters
    # ===============================================
    def get_cluster_distribution(self):
        if self.df is None or "cluster" not in self.df.columns:
            raise RuntimeError("Dataset não possui coluna 'cluster'.")
        return self.df["cluster"].value_counts().reset_index().rename(columns={"index": "Cluster", "cluster": "Nº de Filmes"})

    def get_movies_by_cluster(self, cluster_id):
        if self.df is None or "cluster" not in self.df.columns:
            raise RuntimeError("Dataset não possui coluna 'cluster'.")
        return self.df[self.df["cluster"] == str(cluster_id)][["title", "year", "rating", "genres", "synopsis"]]

