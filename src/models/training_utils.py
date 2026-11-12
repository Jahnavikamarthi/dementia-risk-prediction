from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_FOLDER = PROJECT_ROOT / "models"

# These are placeholders. Do not guess these values.
TARGET_COLUMN = "REPLACE_WITH_TARGET_COLUMN"
SUBJECT_ID_COLUMN = "REPLACE_WITH_SUBJECT_ID_COLUMN"

# Leave empty until you inspect the real ADNI columns.
# Later, list only the approved predictor columns here.
FEATURE_COLUMNS = []


def load_project_data():
    """Load and validate the prepared dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "ADNI data has not been added yet."
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET_COLUMN.startswith("REPLACE_"):
        raise ValueError(
            "Set TARGET_COLUMN in training_utils.py "
            "after inspecting your dataset."
        )

    if SUBJECT_ID_COLUMN.startswith("REPLACE_"):
        raise ValueError(
            "Set SUBJECT_ID_COLUMN in training_utils.py "
            "after inspecting your dataset."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    if SUBJECT_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Subject ID column '{SUBJECT_ID_COLUMN}' was not found."
        )

    if not FEATURE_COLUMNS:
        raise ValueError(
            "FEATURE_COLUMNS is empty. Inspect the dataset and "
            "explicitly select valid predictor columns first."
        )

    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Predictor columns not found: {missing_features}")

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise ValueError("The target column cannot also be a predictor.")

    if SUBJECT_ID_COLUMN in FEATURE_COLUMNS:
        raise ValueError("The subject ID cannot be used as a predictor.")

    df = df.dropna(subset=[TARGET_COLUMN, SUBJECT_ID_COLUMN]).copy()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    groups = df[SUBJECT_ID_COLUMN].copy()

    # These starter models expect numeric predictors.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()

    if non_numeric:
        raise ValueError(
            "These predictors are not numeric: "
            f"{non_numeric}. Encode categorical columns before training."
        )

    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    return X, y, groups


def split_by_subject(X, y, groups):
    """Keep every participant entirely in either train or test."""

    if groups.nunique() < 2:
        raise ValueError("At least two different participants are required.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    if y_train.nunique() < 2:
        raise ValueError(
            "The training split contains fewer than two classes. "
            "Review the dataset and split strategy."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "The test split contains fewer than two classes. "
            "A more suitable grouped split is needed."
        )

    print("Training participants:", groups.iloc[train_indices].nunique())
    print("Testing participants:", groups.iloc[test_indices].nunique())

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Print basic classification metrics."""

    predictions = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print(
        "Balanced accuracy:",
        balanced_accuracy_score(y_test, predictions)
    )

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))


def save_model(model, filename):
    """Save a trained model in the project's models folder."""

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    output_path = MODEL_FOLDER / filename
    joblib.dump(model, output_path)

    print("\nModel saved to:", output_path)


def prepare_training_data():
    """Load data and create a participant-level train/test split."""

    X, y, groups = load_project_data()

    print("Dataset shape:", X.shape)
    print("\nTarget class counts:")
    print(y.value_counts())

    return split_by_subject(X, y, groups)