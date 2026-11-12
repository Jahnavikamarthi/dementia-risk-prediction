
import streamlit as st

st.set_page_config(
    page_title="Risk Prediction",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Cognitive Decline Risk Prediction")

st.write(
    "This module will evaluate cognitive decline progression "
    "using trained machine-learning models."
)

st.warning(
    "Model training and clinical validation are not complete. "
    "The prediction function is intentionally unavailable."
)

st.markdown("### Planned Input Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=110,
        value=65
    )

    mmse = st.number_input(
        "MMSE Score",
        min_value=0,
        max_value=30,
        value=25
    )

    cdr = st.selectbox(
        "Clinical Dementia Rating",
        [0.0, 0.5, 1.0, 2.0, 3.0]
    )

with col2:
    st.subheader("MRI Numerical Features")

    hippocampus = st.number_input(
        "Hippocampal Volume",
        min_value=0.0,
        value=0.0,
        help="Enter a measurement only when the correct dataset units are verified."
    )

    entorhinal = st.number_input(
        "Entorhinal Cortex Measurement",
        min_value=0.0,
        value=0.0
    )

    brain_volume = st.number_input(
        "Whole Brain Volume",
        min_value=0.0,
        value=0.0
    )

st.divider()

st.subheader("Prediction Models")

models = [
    "XGBoost",
    "Random Forest",
    "Support Vector Machine (SVM)"
]

for model in models:
    st.write("🔹", model)

st.button(
    "Predict Cognitive Decline Risk",
    disabled=True,
    use_container_width=True
)

st.info(
    "Prediction will be enabled after ADNI preprocessing, "
    "target definition, model training and independent validation."
)

st.caption(
    "The displayed fields are a preliminary interface. "
    "They are not yet connected to a trained model."
)
