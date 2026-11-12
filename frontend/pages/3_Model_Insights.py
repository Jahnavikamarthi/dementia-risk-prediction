
import streamlit as st

st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance & Explainable AI")

st.write(
    "This page will compare the machine-learning algorithms "
    "and explain the features influencing their predictions."
)

st.info(
    "No experimental performance results are available yet. "
    "Charts will be generated from actual test results after training."
)

st.subheader("Algorithms Used")

algorithms = {
    "XGBoost":
        "Gradient boosting algorithm that builds decision trees "
        "sequentially to improve prediction performance.",

    "Random Forest":
        "Ensemble learning algorithm that combines multiple "
        "decision trees to improve stability and reduce overfitting.",

    "SVM":
        "Support Vector Machine identifies a decision boundary "
        "that separates observations into different classes."
}

for name, description in algorithms.items():

    with st.expander(name):
        st.write(description)

st.divider()

st.subheader("Planned Evaluation Metrics")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "Confusion Matrix"
]

st.write(", ".join(metrics))

st.divider()

st.subheader("Explainable AI - SHAP")

st.write("""
SHAP (SHapley Additive exPlanations) will help explain
which MRI and clinical features influence the model's output.

Planned visualizations:

1. SHAP Summary Plot
2. Feature Importance Plot
3. Individual Prediction Explanation
""")

st.warning(
    "Performance graphs and SHAP visualizations will appear "
    "after the models have been trained and evaluated."
)
