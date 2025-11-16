# streamlit_app.py
"""
Advanced Autonomous Data Cleaning Dashboard (Streamlit)

Drop into repo root and run:
  streamlit run streamlit_app.py --server.port 8888

Notes:
 - Designed to handle large CSVs via chunking for cleaning.
 - Interactive dashboard built with plotly and sklearn (PCA, KMeans).
"""

import os
import io
import tempfile
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ----------------------
# Config / Paths
# ----------------------
ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "outputs")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

st.set_page_config(page_title="Autonomous Data Cleaning — Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🧹 Autonomous Data Cleaning — Advanced Dashboard")

# ----------------------
# Utilities & caching
# ----------------------
@st.cache_data(show_spinner=False)
def read_csv_peek(file_obj, nrows=50):
    try:
        file_obj.seek(0)
    except Exception:
        pass
    try:
        return pd.read_csv(file_obj, nrows=nrows)
    except Exception:
        # fallback iterator
        it = pd.read_csv(file_obj, iterator=True, chunksize=nrows)
        chunk = next(it)
        file_obj.seek(0)
        return chunk

def stream_read_in_chunks(uploaded_file, chunk_size=200000):
    """Yield pandas DataFrames for each chunk to process very large CSVs."""
    uploaded_file.seek(0)
    for chunk in pd.read_csv(uploaded_file, iterator=True, chunksize=chunk_size):
        yield chunk

@st.cache_data(show_spinner=False)
def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

@st.cache_data(show_spinner=False)
def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # numeric: median impute, clip 1-99 percentile
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        median = df[c].median() if not df[c].isnull().all() else 0.0
        df[c] = df[c].fillna(median)
        try:
            low, high = np.nanpercentile(df[c].values.astype(float), [1, 99])
            df[c] = df[c].clip(low, high)
        except Exception:
            pass
    # categorical: mode
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if df[c].isnull().any():
            modes = df[c].mode()
            fill = modes.iloc[0] if len(modes) else ""
            df[c] = df[c].fillna(fill)
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def save_report_text(text: str, base="autoclean_report"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(REPORTS_DIR, f"{base}_{ts}.txt")
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    return target

def make_pdf_from_text(text: str, base="autoclean_report"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORTS_DIR, f"{base}_{ts}.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line)
    pdf.output(filename)
    return filename

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b.read()

# ----------------------
# Sidebar: global controls
# ----------------------
st.sidebar.header("Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (use sample from repo if none)", type=["csv"])
use_sample = st.sidebar.button("Load sample CSV (app/data/processed or data/processed)")

preview_rows = st.sidebar.slider("Preview rows", min_value=5, max_value=200, value=12, step=1)
chunk_size = st.sidebar.number_input("Chunk size (rows)", min_value=50000, max_value=1000000, value=200000, step=50000)
run_full_clean = st.sidebar.button("Run full cleaning")

st.sidebar.markdown("---")
st.sidebar.header("Dashboard Controls")
selected_numeric_sample = st.sidebar.multiselect("Numeric columns for charts (select 1-6)", [], [])
cluster_k = st.sidebar.slider("KMeans clusters (PCA) k", 2, 10, 3)

# ----------------------
# Top-level helpers
# ----------------------
def locate_sample_csv():
    candidates = [
        os.path.join(ROOT, "app", "data", "processed", "train_clean.csv"),
        os.path.join(ROOT, "data", "processed", "train_clean.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: look in outputs for cleaned_*.csv
    out_files = sorted([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".csv")], reverse=True)
    if out_files:
        return os.path.join(OUT_DIR, out_files[0])
    return None

if use_sample and uploaded_file is None:
    sample_path = locate_sample_csv()
    if sample_path:
        uploaded_file = open(sample_path, "rb")
        st.sidebar.success(f"Loaded sample: {sample_path}")
    else:
        st.sidebar.warning("No sample file found. Please upload.")

# ----------------------
# Main: preview + diagnostics
# ----------------------
st.markdown("## 1) Data preview & diagnostics")
if uploaded_file is None:
    st.info("Upload a CSV from the sidebar or load the sample. The dashboard runs preview & diagnostics first.")
    st.stop()

# Preview safe peek
try:
    peek_df = read_csv_peek(uploaded_file, nrows=preview_rows)
    st.subheader("Raw Data Preview (head)")
    st.dataframe(peek_df, use_container_width=True)
except Exception as e:
    st.error(f"Preview failed: {e}")
    st.stop()

# Full small diagnostics
with st.expander("Quick dataset diagnostics", expanded=True):
    try:
        # attempt light-weight full read for metadata only (no heavy objects)
        uploaded_file.seek(0)
        dtypes = pd.read_csv(uploaded_file, nrows=2).dtypes.to_dict()
    except Exception:
        dtypes = peek_df.dtypes.to_dict()
    st.write("Columns / dtypes (sample):")
    col_info = pd.DataFrame([{"column": c, "dtype": str(dtypes.get(c, peek_df[c].dtype)), "sample_unique_preview": peek_df[c].nunique() if c in peek_df else "?" } for c in peek_df.columns])
    st.dataframe(col_info, use_container_width=True)

# compute numeric / categorical lists from peek (will be updated after full read)
numeric_cols = get_numeric_cols(peek_df)
categorical_cols = get_categorical_cols(peek_df)

if not selected_numeric_sample:
    # preselect top numeric cols by variance in peek
    try:
        var = peek_df[numeric_cols].var().sort_values(ascending=False)
        selected_numeric_sample = var.index[:min(6, len(var))].tolist()
    except Exception:
        selected_numeric_sample = numeric_cols[:6]

# ----------------------
# KPI row
# ----------------------
st.markdown("### Key metrics")
col1, col2, col3, col4 = st.columns(4)
try:
    uploaded_file.seek(0)
    # best-effort quick counts (use chunks if file large)
    total_rows = 0
    for chunk in pd.read_csv(uploaded_file, iterator=True, chunksize=100000):
        total_rows += len(chunk)
    uploaded_file.seek(0)
except Exception:
    total_rows = None

col1.metric("Columns", len(peek_df.columns))
col2.metric("Preview rows", len(peek_df))
col3.metric("File rows (approx)", total_rows if total_rows is not None else "unknown")
col4.metric("Numeric cols", len(numeric_cols))

# ----------------------
# Column diagnostics (missingness, uniques)
# ----------------------
st.markdown("## 2) Column diagnostics")
with st.expander("Missingness & uniques overview", expanded=True):
    uploaded_file.seek(0)
    # small sampling to estimate missingness (fast)
    sample_df = read_csv_peek(uploaded_file, nrows=2000)
    missing = sample_df.isna().mean().round(4).sort_values(ascending=False)
    uniques = sample_df.nunique().sort_values(ascending=False)
    mdf = pd.DataFrame({"missing_frac": missing, "nunique_sample": uniques})
    st.dataframe(mdf, use_container_width=True)

    # plot top missing
    top_miss = mdf[mdf["missing_frac"] > 0].sort_values("missing_frac", ascending=False).reset_index()
    if not top_miss.empty:
        fig = px.bar(top_miss, x="index", y="missing_frac", title="Columns with missing values (sample)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values detected in sample preview.")

# ----------------------
# Distributions / Correlations
# ----------------------
st.markdown("## 3) Distributions & correlations")
with st.expander("Interactive distributions and pairwise plots", expanded=True):
    # allow user to choose numeric columns for distribution
    cols_for_dist = st.multiselect("Pick numeric columns for distribution / boxplot", options=numeric_cols, default=selected_numeric_sample, max_selections=6)
    if cols_for_dist:
        for c in cols_for_dist:
            try:
                fig = px.histogram(peek_df, x=c, nbins=60, marginal="box", title=f"Distribution: {c}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write(f"Could not plot {c}")

    # correlation heatmap (sample-based)
    if len(numeric_cols) >= 2:
        corr = sample_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap (sample)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ----------------------
# PCA + Clustering
# ----------------------
st.markdown("## 4) Structure detection (PCA + KMeans)")
with st.expander("PCA projection + KMeans clustering", expanded=False):
    run_pca = st.button("Run PCA on sample (fast)")
    if run_pca:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for PCA.")
        else:
            df_sample = sample_df[numeric_cols].dropna().copy()
            # standardize
            scaler = StandardScaler()
            Xs = scaler.fit_transform(df_sample)
            pca = PCA(n_components=min(5, Xs.shape[1]))
            xp = pca.fit_transform(Xs)
            pc_df = pd.DataFrame(xp, columns=[f"PC{i+1}" for i in range(xp.shape[1])])
            # kmeans
            km = KMeans(n_clusters=cluster_k, random_state=42)
            labels = km.fit_predict(pc_df)
            pc_df["cluster"] = labels.astype(str)
            fig = px.scatter(pc_df, x="PC1", y="PC2", color="cluster", title="PCA 2D projection with KMeans clusters (sample)")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Explained variance:", pca.explained_variance_ratio_.round(4))

# ----------------------
# Full cleaning (streamed) and save
# ----------------------
st.markdown("## 5) Run full cleaning & export")
with st.expander("Full cleaning (chunked) and export", expanded=True):
    if st.button("Run cleaning pipeline on full file (chunked)"):
        try:
            uploaded_file.seek(0)
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_out_name = temp_out.name
            temp_out.close()
            first = True
            processed = 0
            # chunked processing to limit memory usage
            for chunk in stream_read_in_chunks(uploaded_file, chunk_size=chunk_size):
                cleaned_chunk = simple_clean(chunk)
                if first:
                    cleaned_chunk.to_csv(temp_out_name, index=False, mode="w", header=True)
                    first = False
                else:
                    cleaned_chunk.to_csv(temp_out_name, index=False, mode="a", header=False)
                processed += len(chunk)
                st.info(f"Processed rows (so far): {processed}")
            cleaned_df = pd.read_csv(temp_out_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = os.path.join(OUT_DIR, f"cleaned_{ts}.csv")
            cleaned_df.to_csv(out_csv, index=False)
            st.success(f"Cleaned file saved: {out_csv}  (rows: {len(cleaned_df)})")
            # summary report
            report_text = (
                f"Autonomous Cleaning Report — {ts}\n\n"
                f"Source: {getattr(uploaded_file, 'name', 'uploaded_file')}\n"
                f"Rows output: {len(cleaned_df)}\n"
                f"Numeric columns: {', '.join(cleaned_df.select_dtypes(include=['number']).columns.tolist())}\n"
                f"Object columns: {', '.join(cleaned_df.select_dtypes(include=['object','string']).columns.tolist())}\n"
            )
            rpt = save_report_text(report_text)
            pdf = make_pdf_from_text(report_text)
            st.success(f"Report: {rpt} | PDF: {pdf}")
            st.download_button("⬇️ Download cleaned CSV", data=df_to_csv_bytes(cleaned_df), file_name=os.path.basename(out_csv))
        except Exception as e:
            st.error(f"Full cleaning failed: {e}")

# ----------------------
# Export panel: list outputs & reports
# ----------------------
st.markdown("## 6) Outputs & saved reports")
out_files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".csv")], reverse=True)
rep_files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".txt")], reverse=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Saved cleaned CSVs")
    if out_files:
        for f in out_files[:8]:
            path = os.path.join(OUT_DIR, f)
            st.write(f"- {f} ({os.path.getsize(path)//1024} KB)")
            if st.button(f"Download {f}", key=f"dl_{f}"):
                st.download_button(label=f"Download {f}", data=open(path, "rb").read(), file_name=f)
    else:
        st.info("No cleaned CSVs saved yet.")

with col_b:
    st.subheader("Saved Reports")
    if rep_files:
        for r in rep_files[:8]:
            path = os.path.join(REPORTS_DIR, r)
            st.write(f"- {r}")
            if st.button(f"Open {r}", key=f"open_{r}"):
                st.code(open(path, "r", encoding="utf-8").read())
    else:
        st.info("No reports yet. Run the cleaning pipeline.")

st.markdown("---")
st.caption("Advanced dashboard: use the sidebar controls to tune chunk sizes, previews, and clustering. This is a cleaning-first, analysis-rich dashboard designed to scale to large CSVs.")
