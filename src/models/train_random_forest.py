from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from training_utils import (
    prepare_training_data,
    evaluate_model,
    save_model,
)


def train_random_forest():

    print("Random Forest Model Training")
    print("============================")

    X_train, X_test, y_train, y_test = prepare_training_data()

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, "random_forest_model.joblib")


if __name__ == "__main__":
    train_random_forest()