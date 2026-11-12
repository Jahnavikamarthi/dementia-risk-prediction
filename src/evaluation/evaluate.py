from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true, y_pred):

    print("\nMODEL EVALUATION")
    print("================")

    print(
        "Accuracy:",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Balanced accuracy:",
        balanced_accuracy_score(y_true, y_pred)
    )

    print(
        "Macro F1-score:",
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    print("Evaluation module created.")
    print("It will be used to evaluate held-out test predictions.")