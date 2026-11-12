from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_model.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "merged_dataset.csv"

# Keep this consistent with FEATURE_COLUMNS in training_utils.py.
FEATURE_COLUMNS = []


def explain_model():

    print("SHAP Model Explanation")
    print("======================")

    if not MODEL_FILE.exists():
        print("Trained XGBoost model not found.")
        print("Train the model before generating SHAP explanations.")
        return

    if not DATA_FILE.exists():
        print("Dataset not found.")
        return

    if not FEATURE_COLUMNS:
        print("Set FEATURE_COLUMNS after inspecting your dataset.")
        return

    # Load trained pipeline and data
    pipeline = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        print("Missing feature columns:", missing)
        return

    X = df[FEATURE_COLUMNS].copy()

    # Limit the number of rows to keep explanation manageable.
    X = X.head(100)

    # Transform using the imputer fitted during model training.
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = imputer.transform(X)

    # Explain the tree-based XGBoost classifier.
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    # Global feature importance summary
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=FEATURE_COLUMNS,
        show=False
    )

    plt.tight_layout()

    reports_folder = PROJECT_ROOT / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(
            reports_folder / "shap_summary.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    print("SHAP summary saved in reports/shap_summary.png")


if __name__ == "__main__":
    explain_model()