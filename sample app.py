import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 1. ADVANCED UI CONFIGURATION
# ===============================
st.set_page_config(
    page_title="HFR-MADM Clinical Portal",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }

.main-header {
    background: linear-gradient(90deg, #002b5b 0%, #004e92 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #002b5b 0%, #004e92 100%) !important;
}

[data-testid="stSidebar"] label, .sidebar-title {
    color: white !important;
    font-weight: 700 !important;
}

.sidebar-card {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px);
    padding: 16px;
    border-radius: 15px;
    margin-bottom: 1rem;
    color: white !important;
}

.rank-badge {
    background: linear-gradient(90deg, #ffd700 0%, #ffae00 100%);
    color: #002b5b;
    padding: 12px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. DATA LOADING
# ===============================
def load_datasets(folder="data"):
    datasets = {}
    if not os.path.exists(folder):
        return datasets

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file)).dropna().drop_duplicates()
            le = LabelEncoder()
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = le.fit_transform(df[col].astype(str))

            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            if y.nunique() >= 2:
                datasets[file] = (X, y, df)

    return datasets

# ===============================
# 3. HFR-MADM LOGIC
# ===============================
def hfr_madm_logic(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    results = []

    for name, (X, y, _) in datasets.items():
        f1 = min(len(X) / 1500, 1.0)
        f2 = 1 - abs(0.5 - y.value_counts(normalize=True).iloc[0])
        f3 = min(X.shape[1] / 25, 1.0)
        f4 = 0.95

        hesitant = [np.mean([m * 0.9, m, min(m * 1.1, 1.0)]) for m in [f1, f2, f3, f4]]

        results.append({
            "Dataset": name,
            "Score": round(np.dot(hesitant, weights), 4),
            "Samples": len(X),
            "Features": X.shape[1]
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False)

# ===============================
# 4. MODEL TRAINING (LR + RF)
# ===============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression (Primary)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_s, y_train)
    lr_preds = lr_model.predict(X_test_s)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)
    lr_report = classification_report(y_test, lr_preds, output_dict=True)

    # Random Forest (Added)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    return (
        lr_model,
        lr_acc,
        lr_cm,
        lr_report,
        scaler,
        X.columns,
        rf_acc
    )

# ===============================
# 5. LOAD & RANK DATASETS
# ===============================
all_data = load_datasets("data")
if not all_data:
    st.error("❌ Data folder is empty or missing CSV files.")
    st.stop()

rankings = hfr_madm_logic(all_data).reset_index(drop=True)
rankings.insert(0, "Rank", rankings.index + 1)

# ===============================
# 6. SIDEBAR
# ===============================
st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" width="90"/>
        <div class="sidebar-title">Navigation Menu</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-card"><b>📂 Dataset</b>', unsafe_allow_html=True)
dataset_choice = st.sidebar.selectbox("", list(all_data.keys()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card"><b>🏆 Top Ranked</b>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="rank-badge">{rankings.iloc[0]["Dataset"]}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 7. MAIN UI
# ===============================
st.markdown(f"""
<div class="main-header">
    <h1>🩺 Predictive Healthcare Decision System</h1>
    <p>HFR-MADM Optimized Analysis | Active Source: {dataset_choice}</p>
</div>
""", unsafe_allow_html=True)

X_sel, y_sel, raw_df = all_data[dataset_choice]
model, acc, cm, report, scaler, feature_names, rf_acc = train_model(X_sel, y_sel)

tab1, tab2, tab3 = st.tabs(["📊 Data Intelligence", "🧪 Model Performance", "🔍 Risk Diagnosis"])

# ===============================
# TAB 1
# ===============================
with tab1:
    st.dataframe(rankings, use_container_width=True)

# ===============================
# TAB 2
# ===============================
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logistic Regression Accuracy", f"{acc:.2%}")
    c2.metric("Random Forest Accuracy", f"{rf_acc:.2%}")
    c3.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    c4.metric("Samples", int(report['macro avg']['support']))

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

# ===============================
# TAB 3
# ===============================
with tab3:
    with st.form("clinical_form"):
        inputs = []
        cols = st.columns(2)
        for i, col in enumerate(feature_names):
            with cols[i % 2]:
                inputs.append(st.number_input(col, value=float(raw_df[col].median()), min_value=0.0))

        submit = st.form_submit_button("Generate Prediction")

    if submit:
        input_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled).max()

        if pred == 1:
            st.error(f"⚠️ HIGH RISK | Confidence: {prob:.2%}")
        else:
            st.success(f"✅ LOW RISK | Confidence: {prob:.2%}")
