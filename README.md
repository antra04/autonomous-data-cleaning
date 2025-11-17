
# 🧹 **Autonomous Data Cleaning — Advanced AI-Powered Dashboard**

### **End-to-end intelligent data cleaning, diagnostics, anomaly detection, and ML-ready preprocessing.**

🚀 **Live App:**
🔗 **[https://autonomous-cleaning.streamlit.app/](https://autonomous-cleaning.streamlit.app/)**

This project is a **complete AI-powered data-cleaning ecosystem** that intelligently analyzes, detects, cleans, visualizes, and prepares datasets for machine learning.
It includes:

* Automated column diagnostics
* Missing value detection
* Anomaly + drift detection
* PCA + KMeans structure mapping
* Full cleaning pipeline (chunked for large files)
* Exportable cleaned datasets
* Auto-generated cleaning reports
* A fully interactive Streamlit dashboard

---

# 📌 **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [How the Cleaning Algorithm Works](#cleaning-algorithm)
6. [Dashboard Walkthrough](#dashboard-walkthrough)
7. [Screenshots](#screenshots)
8. [Project Structure](#project-structure)
9. [Local Setup](#local-setup)
10. [Deployment](#deployment)
11. [Future Enhancements](#future-enhancements)
12. [Author](#author)

---

# 🧠 **Overview**

This system is built to solve one of the most time-consuming tasks in data science:
👉 **Cleaning messy real-world data efficiently, correctly, and at scale.**

It automatically performs:

* Data validation
* Missing value analysis
* Column type inference
* Outlier detection
* Duplicate handling
* Categorical normalization
* Date/number coercion
* Statistical + visual diagnostics
* ML-ready export

Designed for real production workflows where datasets may contain:
✔ Nulls
✔ Inconsistent formats
✔ Outliers
✔ Corrupted rows
✔ Mixed data types
✔ Hidden anomalies

---

# ⭐ **Features**

### ✔ **1. Data Preview & Diagnostics**

* Raw preview
* Automatic dtype inference
* Unique counts
* Missing value fractions
* Quick statistics

### ✔ **2. Column Diagnostics**

* Missing vs non-missing summary
* Categorical inconsistencies
* Extreme values
* Drift and anomalies

### ✔ **3. Advanced Visualizations**

* Correlation heatmaps
* Boxplots
* Histograms
* Pairwise distributions
* PCA 2D projections
* KMeans cluster visualization

### ✔ **4. Chunked Full Cleaning Pipeline**

Supports **200,000+ rows** using efficient chunk processing:

* Dtype coercion
* Fuzzy category correction
* Outlier handling
* Missing value strategy
* Format normalization

### ✔ **5. Reporting & Export**

* Saves cleaned CSV
* Saves automated cleaning reports
* Downloadable artifacts

### ✔ **6. Fully Customizable**

You can upload your own CSV or use sample data.

---

# 🏗 **Architecture**

```
┌───────────────────────────────────┐
│        Streamlit Frontend         │
│  - UI controls                    │
│  - File upload                    │
│  - Visualization engine           │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│      Backend Cleaning Engine      │
│  - Data loading (chunked)         │
│  - Statistical profiling          │
│  - Missing value model            │
│  - Outlier detection              │
│  - Categorical normalization      │
│  - PCA + clustering               │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│          Output Layer             │
│  - Cleaned CSV files              │
│  - Processed reports              │
│  - Visual results                 │
└───────────────────────────────────┘
```

---

# ⚙ **Tech Stack**

### **Backend / Data**

* Python 3.10
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Matplotlib / Seaborn
* Custom data cleaning engine

### **Frontend**

* Streamlit
* Plotly interactive UI
* Streamlit widgets

### **Deployment**

* **Streamlit Cloud**
* Public web URL:
  👉 **[https://autonomous-cleaning.streamlit.app/](https://autonomous-cleaning.streamlit.app/)**

---

# 🧼 <a name="cleaning-algorithm"></a> **How the Cleaning Algorithm Works (Detailed)**

### ✔ **1. Column-Level Processing**

* Infers types: *numeric, categorical, date, identifier, boolean*
* Converts incorrect formats (e.g., `"None"`, `"?"`, `"unknown"` → NaN)

### ✔ **2. Missing Value Handling**

Different strategies based on datatype:

| Column Type | Strategy                     |
| ----------- | ---------------------------- |
| Numeric     | Median imputation            |
| Category    | Mode imputation / clustering |
| Date        | Forward fill / parse         |
| IDs         | Left untouched               |
| Boolean     | Mode fill                    |

### ✔ **3. Outlier Detection**

* IQR Method
* Z-score
* Capping/extreme reduction

### ✔ **4. Categorical Normalization**

* Lowercasing
* Removing spelling variants
* Replacing unusual labels

### ✔ **5. Duplicate Detection**

* Duplicate rows removal
* Duplicate IDs handled carefully

### ✔ **6. PCA + KMeans (Structure Detection)**

Used for:

* Visual clustering
* High-level structure understanding
* Feature relationships

---

# 📊 <a name="dashboard-walkthrough"></a> **Dashboard Walkthrough**

### **📂 Sidebar Controls**

* Upload CSV
* Load sample dataset
* Select preview rows
* Choose chunk size
* Button to run full cleaning

### **📌 Section 1 — Data Preview & Diagnostics**

* Dataset head
* Column types
* Unique sample counts
* Quick stats

### **📌 Section 2 — Column Diagnostics**

* Missing values
* Histograms
* Categorical health
* Validation report

### **📌 Section 3 — Distributions & Correlations**

* Interactive histograms
* Boxplots
* Heatmaps

### **📌 Section 4 — PCA + KMeans Structure**

* PCA 2D plot
* Cluster assignments
* Explained variance

### **📌 Section 5 — Full Cleaning & Export**

* Run complete cleaning
* Progress logs
* Output saved file
* Cleaning error reporting

### **📌 Section 6 — Outputs & Reports**

* Download cleaned CSV
* Download cleaning report

# 📁 **Project Structure**

```
autonomous-data-cleaning/
│
├── app/
│   ├── streamlit_app.py
│   ├── data/
│   ├── outputs/
│
├── src/
│   ├── cleaning_engine.py
│   ├── visualization.py
│   ├── utils.py
│
├── requirements.txt
├── runtime.txt
├── README.md
└── .streamlit/
    ├── config.toml
```

---

# 💻 <a name="local-setup"></a> **Local Setup**

### **Clone the repo**

```bash
git clone https://github.com/antra04/autonomous-data-cleaning.git
cd autonomous-data-cleaning
```

### **Install packages**

```bash
pip install -r requirements.txt
```

### **Run app**

```bash
streamlit run app/streamlit_app.py
```

---

# 🌐 <a name="deployment"></a> **Deployment**

Deployed on **Streamlit Cloud**.

**Final Live URL:**
👉 **[https://autonomous-cleaning.streamlit.app/](https://autonomous-cleaning.streamlit.app/)**

Deployment assets include:

* `requirements.txt`
* `runtime.txt` (Python 3.10)
* `.streamlit/config.toml` (theme + branding)

---

# 🚀 **Future Enhancements**

* Auto-ML training module
* Drift detection
* Multi-file cleaning pipeline
* API version (FastAPI backend)
* Profile reports (like pandas-profiling)
* Support for Excel & Parquet

---

# 👤 <a name="author"></a> **Author**

**Antra Tiwari**
 AI/ML Developer | Data Engineering Enthusiast
 4th Year B.Tech CSE

---
