from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from training_utils import (
    prepare_training_data,
    evaluate_model,
    save_model,
)


def train_svm():

    print("Support Vector Machine Training")
    print("================================")

    # Load data and split by participant
    X_train, X_test, y_train, y_test = prepare_training_data()

    # SVM usually benefits from feature scaling.
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Train
    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, "svm_model.joblib")


if __name__ == "__main__":
    train_svm()