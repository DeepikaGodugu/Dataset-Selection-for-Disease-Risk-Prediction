import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1.ADVANCED UI CONFIGURATION
st.set_page_config(
    page_title="HFR-MADM Clinical Portal",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
/* 1. Main Background */
.stApp { background-color: #f8f9fa; }

/* 2. Custom Header */
.main-header {
    background: linear-gradient(90deg, #002b5b 0%, #004e92 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

/* 3. FORCED SIDEBAR THEME */
[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #002b5b 0%, #004e92 100%) !important;
    background-color: #002b5b !important;
}

/* Force Sidebar labels to be white */
[data-testid="stSidebar"] label, .sidebar-title {
    color: white !important;
    font-weight: 700 !important;
}

/* FIX: Force Dataset Selection text to be BLACK */
div[data-baseweb="select"] > div {
    color: black !important;
}
input[data-testid="stWidgetInput-selectbox"] {
    color: black !important;
    -webkit-text-fill-color: black !important;
}

/* Sidebar Glass Cards - High Contrast */
.sidebar-card {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px);
    padding: 16px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin-bottom: 1rem;
    color: white !important; 
}

/* Force specific tags like <b> to be white */
.sidebar-card b, .sidebar-card span, .sidebar-card div {
    color: white !important;
}

/* Keep the dropdown text black so it's readable in the white box */
div[data-baseweb="select"] > div {
    color: black !important;
}

/* Gold Rank Badge */
.rank-badge {
    background: linear-gradient(90deg, #ffd700 0%, #ffae00 100%) !important;
    color: #002b5b !important;
    padding: 12px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
}

/* 4. Tabs & Metrics */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# 2. LOGIC (Logistic Regression Focus)
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

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    return (
        model,
        accuracy_score(y_test, preds),
        confusion_matrix(y_test, preds),
        classification_report(y_test, preds, output_dict=True),
        scaler,
        X.columns
    )
# Load datasets
all_data = load_datasets("data")
if not all_data:
    st.error("❌ Data folder is empty or missing CSV files.")
    st.stop()

# Rank datasets
rankings = hfr_madm_logic(all_data)
rankings = rankings.reset_index(drop=True)
rankings.insert(0, "Rank", rankings.index + 1)

# 3. SIDEBAR NAVIGATION
st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" width="90"/>
        <div class="sidebar-title">Navigation Menu</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-card">📂 <b>Database Access</b>', unsafe_allow_html=True)
dataset_choice = st.sidebar.selectbox("", list(all_data.keys()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-card">🏷️ <b>Top Ranked Dataset</b>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="rank-badge">{rankings.iloc[0]["Dataset"]}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# 4. MAIN INTERFACE
st.markdown(f"""
<div class="main-header">
    <h1>🩺Predictive Healthcare Decision System</h1>
    <p>HFR-MADM Optimized Analysis | Active Source: {dataset_choice}</p>
</div>
""", unsafe_allow_html=True)

X_sel, y_sel, raw_df = all_data[dataset_choice]
model, acc, cm, report, scaler, feature_names = train_model(X_sel, y_sel)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Intelligence", "🧪 Model Performance", "🔍 Risk Diagnosis"]
)

with tab1:
    st.markdown("### *HFR-MADM Quality Ranking*")
    st.dataframe(
        rankings.style
        .background_gradient(cmap="Blues", subset=["Score"])
        .format({"Score": "{:.3f}"}),
    use_container_width=True,
    hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("*Dataset Quality Scores*")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=rankings,
            x="Score",
            y="Dataset",
            palette="Blues_d",
            ax=ax1
        )
        st.pyplot(fig1)

    with col2:
        st.write("*Dataset Size vs Feature Count*")
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        sns.barplot(
            data=rankings,
            x="Samples",
            y="Dataset",
            palette="Blues_d",
            ax=ax2
        )

        ax2.set_xlabel("Number of Samples")
        ax2.set_ylabel("Dataset")

        st.pyplot(fig2)
with tab2:
    st.markdown("### **Metrics**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Diagnostic Accuracy", f"{acc:.2%}")
    m2.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    m3.metric("Processed Samples", f"{int(report['macro avg']['support'])}")

    st.markdown("---")
    l, r = st.columns(2)
    with l:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    with r:
        st.write("**Clinical Feature Weights**")
        coef = pd.Series(abs(model.coef_[0]), index=feature_names)
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        coef.nlargest(10).plot(kind="barh", color="#004e92", ax=ax_fi)
        ax_fi.invert_yaxis()
        st.pyplot(fig_fi)

with tab3:
    st.markdown("### **Patient Risk Predictor**")
    st.info("Input clinical parameters to generate a risk probability score.")

    # We create two columns: Left for the Form, Right for the Overall Context
    form_col, info_col = st.columns([2, 1])

    with info_col:
        st.write("**Overall Dataset Risk Context**")
        # ---- Small Pie Chart for Overall Risk ----
        risk_counts = y_sel.value_counts()
        labels = ['Low Risk', 'High Risk']
        sizes = [risk_counts.get(0, 0), risk_counts.get(1, 0)]
        colors = ['#4A90E2', '#E53935'] 

        fig_pie, ax_pie = plt.subplots(figsize=(3, 2))
        ax_pie.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            radius=1.0,
            textprops={'fontsize': 7},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie, bbox_inches='tight')
        plt.close(fig_pie)
        st.caption("This chart shows the distribution of the entire dataset.")

    with form_col:
        with st.form("clinical_form"):
            cols = st.columns(2) # Two columns for form fields to save space
            inputs = []
            for i, col in enumerate(feature_names):
                with cols[i % 2]:
                    # Added min_value=0.0 so any value can be entered
                    inputs.append(
                        st.number_input(col, value=float(raw_df[col].median()), min_value=0.0)
                    )

            submit = st.form_submit_button("Generate Individual Prediction", use_container_width=True)

    if submit:
        st.divider()
        input_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
        res = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled).max()

        if res == 1:
            st.error(f"### ⚠️ INDIVIDUAL DIAGNOSIS: HIGH RISK\nPersonalized Confidence: {prob:.2%}")
            st.warning("**Recommendation:** Clinical intervention and further diagnostic testing recommended.")
        else:
            st.success(f"### ✅ INDIVIDUAL DIAGNOSIS: LOW RISK\nPersonalized Confidence: {prob:.2%}")
            st.toast("Analysis Complete: Low Risk Detected", icon='✅')
   























