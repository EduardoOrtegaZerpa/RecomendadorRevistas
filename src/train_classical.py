import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from config import DATA_RAW_DIR, CLASSICAL_DIR, REPORTS_DIR
from data_loader import load_dataset_from_folders, split_train_test
from plots import plot_confusion_matrix, plot_f1_scores, plot_accuracy_per_class

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset_from_folders(DATA_RAW_DIR)
    train_df, test_df = split_train_test(df, test_size=0.2, seed=42)

    labels = sorted(df["label"].unique())

    # Model
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LinearSVC())
    ])

    pipe.fit(train_df["text"], train_df["label"])
    preds = pipe.predict(test_df["text"])

    # Metrics
    cm = confusion_matrix(test_df["label"], preds, labels=labels)

    # Plots
    plot_confusion_matrix(
    cm,
    labels,
    REPORTS_DIR / "confusion_matrix_classical.png",
    title="Normalized Confusion Matrix (TF-IDF + Linear SVM)"
    )

    plot_f1_scores(
        test_df["label"],
        preds,
        labels,
        REPORTS_DIR / "f1_scores_classical.png",
        title="F1-score per Journal (TF-IDF + Linear SVM)"
    )

    plot_accuracy_per_class(
        cm,
        labels,
        REPORTS_DIR / "accuracy_per_class_classical.png",
        title="Accuracy per Journal (TF-IDF + Linear SVM)"
    )

    # Save model
    joblib.dump(pipe, CLASSICAL_DIR / "model.joblib")

    print("Training finished. All plots saved in outputs/reports/")


if __name__ == "__main__":
    main()
