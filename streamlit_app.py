
import os, pandas as pd, numpy as np, joblib, streamlit as st
from datetime import datetime

st.set_page_config(page_title="Autonomous Data Cleaning", layout="wide")

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "outputs")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============ Helper Functions ============

def simple_clean(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        lower, upper = np.percentile(df[col], [1, 99])
        df[col] = np.clip(df[col], lower, upper)
    df = df.drop_duplicates()
    return df

def load_model():
    outputs_dir = os.path.join(ROOT, "outputs")
    models = [f for f in os.listdir(outputs_dir) if f.endswith(".joblib")]
    if not models:
        return None, None
    latest = sorted(models)[-1]
    try:
        model = joblib.load(os.path.join(outputs_dir, latest))
        return model, latest
    except Exception:
        return None, None

# ============ Streamlit UI ============

st.title("🧹 Autonomous Data Cleaning & Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV only):", type=["csv"])

run_clean = st.checkbox("Run cleaning pipeline", value=True)
show_preview = st.slider("Preview rows", 1, 20, 5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(show_preview))

    if run_clean:
        cleaned_df = simple_clean(df)
        st.subheader("🧼 Cleaned Data Preview")
        st.dataframe(cleaned_df.head(show_preview))
    else:
        cleaned_df = df.copy()

    model, model_name = load_model()

    if st.button("🚀 Run Model (if available)"):
        if model:
            try:
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                X = cleaned_df[num_cols].fillna(0)
                preds = model.predict(X)
                st.success(f"✅ Model {model_name} predictions generated!")
                st.dataframe(pd.DataFrame(preds, columns=["Prediction"]).head(show_preview))
            except Exception as e:
                st.warning(f"⚠️ Model loaded but prediction failed: {e}")
        else:
            st.info("No trained model found in outputs/. Train one first.")

    # Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"cleaned_{ts}.csv")
    report_txt = os.path.join(REPORTS_DIR, f"report_{ts}.txt")
    cleaned_df.to_csv(out_csv, index=False)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Autonomous Cleaning Report — {ts}\n\n")
        f.write(f"Original rows: {len(df)}\nCleaned rows: {len(cleaned_df)}\n\n")
        f.write("Numeric features cleaned: " + ", ".join(cleaned_df.select_dtypes(include=['float64','int64']).columns))
    st.success(f"✅ Outputs saved! CSV: {out_csv} | Report: {report_txt}")

    with open(report_txt, "r") as f:
        report_text = f.read()
    st.text_area("🧠 AI-like Report Summary", report_text, height=250)
else:
    st.info("👈 Upload a CSV file to start cleaning and analysis.")
