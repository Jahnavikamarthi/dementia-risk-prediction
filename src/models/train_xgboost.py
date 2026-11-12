from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from training_utils import (
    prepare_training_data,
    evaluate_model,
    save_model,
)


def train_xgboost():

    print("XGBoost Model Training")
    print("======================")

    # Load data and split by participant
    X_train, X_test, y_train, y_test = prepare_training_data()

    # Create the model pipeline
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train
    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, "xgboost_model.joblib")


if __name__ == "__main__":
    train_xgboost()