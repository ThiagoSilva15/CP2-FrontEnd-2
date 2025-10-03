# ===============================================
# 🎬 IMDB Movie Recommender — UX polido (estilo iFood app)
# ===============================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --------------- Config de página + tema ---------------
st.set_page_config(
    page_title="IMDB Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS (mesmo “mood” do seu app anterior)
# CSS Customizado - Dark Theme
st.markdown(
    """
    <style>
    /* ===== Base ===== */
    .stApp { 
        background-color: #121212; 
        font-family: 'Helvetica', Arial, sans-serif; 
        color: #E0E0E0;
    }
    h1, h2, h3, h4, h5, h6 { color: #00bfa6; }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        padding: 1rem;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] * { color: #D0D0D0; }
    section[data-testid="stSidebar"] .st-bb { border-color: #333 !important; }

    /* ===== Cards / containers ===== */
    .card {
        background-color: #1E1E1E !important; 
        border-radius: 12px; 
        padding: 1rem; 
        box-shadow: 0 4px 14px rgba(0,0,0,0.35); 
        margin-bottom: 1rem;
        color: #F1F1F1;
        border: 1px solid #2A2A2A;
    }

    /* ===== Inputs ===== */
    .stSelectbox div[data-baseweb="select"],
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stDateInput input,
    .stFileUploader,
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #2A2A2A !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #3A3A3A !important;
    }
    .stSlider > div[data-baseweb="slider"] > div { background: #00bfa6 !important; }

    /* ===== Botões ===== */
    .stButton button {
        background-color: #00bfa6 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton button:hover { background-color: #009688 !important; }

    /* ===== Métricas ===== */
    [data-testid="stMetricValue"] { color: #00bfa6 !important; }
    [data-testid="stMetricDelta"] { color: #9ccc65 !important; }

    /* ===== Tabelas (pandas styler) ===== */
    .dataframe { color: #EAEAEA !important; }
    .dataframe tbody tr, .dataframe thead tr { background-color: #1E1E1E !important; }
    .dataframe th { background-color: #2C2C2C !important; color: #FFFFFF !important; }
    .dataframe td, .dataframe th { border-color: #333 !important; }

    /* ===== Mensagens ===== */
    .stAlert { background-color: #1E1E1E !important; border: 1px solid #333 !important; }
    .stAlert p, .stAlert div { color: #E0E0E0 !important; }

    /* Badges “chips” que você usa para colunas/arquivos */
    .chip {
        display:inline-block; padding: .25rem .6rem; border-radius: 999px;
        font-size:.85rem; background:#2a2a2a; color:#eaeaea; border:1px solid #3a3a3a;
        margin-right:.35rem; margin-bottom:.35rem;
    }
    .chip--ok { background:#153f3a; border-color:#1f5e54; color:#9be7d9; }
    .chip--warn { background:#3a2a23; border-color:#5f4032; color:#ffcc80; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🎬 IMDB Movie Recommender")
st.caption("Sistema de recomendação baseado em *clustering* de sinopses e (opcionalmente) todas as features.")

# --------------- Caminhos base ---------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = (APP_DIR / "data").resolve()
MODELS_DIR = (APP_DIR / "models").resolve()

# --------------- Helpers ---------------
def list_csvs(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.glob("*.csv")])

@st.cache_data(show_spinner=False)
def load_df(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name in ["tfidf.pkl", "svd.pkl", "kmeans_synopsis.pkl", "best_cluster_model.pkl"]:
        p = MODELS_DIR / name
        if p.exists():
            try:
                models[name.replace(".pkl", "")] = joblib.load(p)
            except Exception:
                pass
    return models

def ensure_columns(df: pd.DataFrame, must_have: list[str]) -> list[str]:
    return [c for c in must_have if c in df.columns]

def recommend_by_synopsis(df: pd.DataFrame, models: dict, query: str, top_n: int = 8, only_same_cluster: bool = True) -> pd.DataFrame:
    """TF-IDF → SVD → KMeans cluster → cosseno nas sinopses do cluster (ou de todo o dataset se desativar o filtro)"""

    # Checagens
    if "tfidf" not in models or "svd" not in models:
        st.error("❌ Modelos necessários não carregados: `tfidf` e `svd`.")
        return pd.DataFrame()

    if "synopsis" not in df.columns:
        st.error("❌ Dataset sem coluna `synopsis`.")
        return pd.DataFrame()

    tfidf = models["tfidf"]
    svd = models["svd"]

    # Vetoriza/Reduce a consulta
    query_vec = tfidf.transform([query])
    query_red = svd.transform(query_vec)

    # Se houver kmeans, descobrimos o cluster da consulta
    df_base = df.copy()
    sel_text = df_base["synopsis"].fillna("")

    if "kmeans_synopsis" in models and only_same_cluster and "cluster" in df_base.columns:
        km = models["kmeans_synopsis"]
        q_cluster = km.predict(query_red)[0]
        st.info(f"🔎 Sua sinopse foi classificada no **Cluster {q_cluster}**. Recomendando dentro dele.")
        df_base = df_base[df_base["cluster"] == q_cluster]
        sel_text = df_base["synopsis"].fillna("")

        # Se o cluster ficou vazio por qualquer motivo, relaxa o filtro
        if len(df_base) == 0:
            df_base = df.copy()
            sel_text = df_base["synopsis"].fillna("")
            st.warning("⚠️ Cluster vazio. Buscando no dataset inteiro.")

    # Similaridade de cosseno no espaço TF-IDF (mais robusto para texto curto)
    matrix = tfidf.transform(sel_text)
    sim = cosine_similarity(query_vec, matrix).flatten()
    out = df_base.copy()
    out["similarity"] = sim
    # Ordena e retorna top_n
    out = out.sort_values("similarity", ascending=False).head(top_n)
    return out

# --------------- Sidebar (config) ---------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/microsoft-teams/363/clapper-board_1f3ac.png", width=60)
    st.subheader("Configurações")

    st.markdown("**Pastas resolvidas**")
    st.code(str(DATA_DIR), language="bash")
    st.code(str(MODELS_DIR), language="bash")

    csv_list = list_csvs(DATA_DIR)
    if not csv_list:
        st.error("Nenhum CSV encontrado em `notebooks/webapp/data`.\nExporte pelos notebooks 03/04.")
        st.stop()

    selected_csv = st.selectbox("📂 Escolha o dataset", csv_list, index=0)
    st.markdown("---")

# --------------- Carregar dados + modelos ---------------
df = load_df(selected_csv)
models = load_models()

# --------------- Diagnóstico rápido ---------------
with st.expander("🔬 Diagnóstico do dataset carregado", expanded=True):
    left, right = st.columns([2,1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Arquivo:**", f"`{selected_csv}`")
        st.write("**Cols:**", ", ".join(df.columns))
        st.write("**Linhas:**", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        has_cluster = "cluster" in df.columns
        st.write("`cluster`:", "✅ presente" if has_cluster else "⚠️ ausente")
        st.write("`synopsis`:", "✅ presente" if "synopsis" in df.columns else "⚠️ ausente")
        st.markdown('</div>', unsafe_allow_html=True)

# --------------- Tabs ---------------
tab1, tab2, tab3 = st.tabs(["📊 Explorar Clusters", "🔎 Recomendar por Sinopse", "ℹ️ Sobre"])

# ---------- Tab 1: Explorar Clusters ----------
with tab1:
    st.header("📊 Explorar Clusters")
    if "cluster" not in df.columns:
        st.warning("Este dataset não possui coluna `cluster`. Gere pelos notebooks 03 (Sklearn) ou 04 (PyCaret).")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Distribuição
        dist = df["cluster"].value_counts().reset_index()
        dist.columns = ["Cluster", "Nº de Filmes"]
        fig = px.bar(dist, x="Cluster", y="Nº de Filmes", title="Distribuição de Filmes por Cluster")
        st.plotly_chart(fig, use_container_width=True)

        # Seleção + Tabela
        clust_id = st.selectbox("Selecione um cluster", sorted(df["cluster"].unique()))
        view_cols = ensure_columns(df, ["title", "year", "rating", "genres", "synopsis", "cluster"])
        st.dataframe(df[df["cluster"] == clust_id][view_cols], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 2: Recomendar por Sinopse ----------
with tab2:
    st.header("🔎 Recomendação por Sinopse")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    colA, colB = st.columns([3,1])
    with colA:
        query = st.text_area("Cole a sinopse aqui:", height=160,
                             placeholder="Coloque uma descrição/plot do filme…")
    with colB:
        topn = st.number_input("Qtd. recomendações", 3, 30, 8, 1)
        same_cluster = st.toggle("Filtrar pelo mesmo cluster (se disponível)", value=True)

    go = st.button("🔍 Recomendar")

    if go:
        if not query.strip():
            st.warning("Digite uma sinopse para continuar.")
        else:
            recs = recommend_by_synopsis(df, models, query.strip(), top_n=int(topn), only_same_cluster=same_cluster)
            if recs.empty:
                st.error("Nenhuma recomendação encontrada. Verifique se TF-IDF/SVD foram carregados e o dataset tem sinopses.")
            else:
                st.success(f"🎯 {len(recs)} recomendações geradas")
                show_cols = ensure_columns(recs, ["title", "year", "rating", "genres", "similarity", "cluster"])
                st.dataframe(recs[show_cols], use_container_width=True)

                # Download
                csv_bytes = recs[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar recomendações", csv_bytes, "recomendacoes.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 3: Sobre ----------
with tab3:
    st.header("ℹ️ Sobre o Projeto")
    st.markdown(
        """
        Este app usa:
        - **TF-IDF + SVD + KMeans** para clusterização e recomendação por similaridade.
        - CSVs gerados pelos notebooks **03 (Sklearn)** e **04 (PyCaret)**.
        - Coloque os arquivos em:
          - `notebooks/webapp/data/` → *imdb_top250_k5_synopsis.csv*, *imdb_top250_k5_allfeatures.csv*
          - `notebooks/webapp/models/` → *tfidf.pkl*, *svd.pkl*, *kmeans_synopsis.pkl* (e opcional *best_cluster_model.pkl*)
        
        **Dicas**
        - Se não aparecer gráfico de clusters, gere o CSV com a coluna `cluster`.
        - Se a recomendação falhar, verifique se `tfidf.pkl` e `svd.pkl` foram salvos em `models/`.
        """
    )
