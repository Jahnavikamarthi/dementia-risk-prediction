
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="NeuroCare AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FB;
}
[data-testid="stSidebar"] {
    background-color: #101D35;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 35px;
    font-weight: 750;
    color: #172B4D;
}
.subtitle {
    color: #64748B;
    font-size: 16px;
}
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.section-title {
    color: #172B4D;
    font-size: 22px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🧠 NeuroCare AI")
    st.caption("Cognitive Health Research Platform")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("""
    - 🏠 Dashboard
    - 📂 Dataset Explorer
    - 🧪 Risk Prediction
    - 📊 Model Insights
    """)

    st.divider()
    st.caption("Research Project | ADNI Dataset")
    st.caption("AI-assisted research prototype")

# ---------------- MAIN DASHBOARD ----------------

st.markdown(
    '<div class="main-title">Dementia & Cognitive Decline Risk Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Multimodal AI research using MRI numerical measurements and clinical cognitive assessments.</div>',
    unsafe_allow_html=True
)

st.write("")

# Information cards

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h4>🧠 MRI Analysis</h4>
    <p>Brain structure and numerical MRI measurements.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h4>📋 Clinical Analysis</h4>
    <p>Cognitive assessments and clinical information.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h4>🤖 Machine Learning</h4>
    <p>XGBoost, Random Forest and SVM.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="section-title">Project Overview</div>',
    unsafe_allow_html=True
)

st.write("""
NeuroCare AI is a research prototype designed to investigate
cognitive decline and dementia progression using artificial
intelligence.

The system aims to combine MRI-derived numerical features,
clinical information and cognitive assessment scores to
develop machine-learning models for progression-risk prediction.
""")

# Workflow

st.markdown("### System Workflow")

steps = [
    ("01", "ADNI Dataset"),
    ("02", "Data Preprocessing"),
    ("03", "Feature Extraction"),
    ("04", "ML Prediction"),
    ("05", "SHAP Explanation")
]

cols = st.columns(5)

for col, (number, name) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
        <h2 style="color:#2563EB">{number}</h2>
        <b>{name}</b>
        </div>
        """, unsafe_allow_html=True)

st.write("")

st.info(
    "Development status: Dataset inspection completed. "
    "Clinical and MRI integration and model training are pending."
)

st.warning(
    "This is a research prototype, not a medical diagnostic tool. "
    "No validated individual risk predictions are available yet."
)

st.markdown("---")
st.caption("NeuroCare AI | Academic Research Project")
