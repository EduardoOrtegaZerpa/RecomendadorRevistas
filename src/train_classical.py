import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW_DIR, CLASSICAL_DIR, REPORTS_DIR
from data_loader import load_dataset_from_folders
from plots import plot_confusion_matrix, plot_f1_scores, plot_accuracy_per_class


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset_from_folders(DATA_RAW_DIR)
    X = np.asarray(df["text"].tolist(), dtype=str)
    y = np.asarray(df["label"].tolist())
    labels = sorted(np.unique(y))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cms = []
    y_true_all = []
    y_pred_all = []
    pipe =  None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

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

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

        cm = confusion_matrix(y_test, preds, labels=labels)
        cms.append(cm)

    # Confusion matrix media
    cm_mean = np.mean(cms, axis=0)

    # Plots
    plot_confusion_matrix(
        cm_mean,
        labels,
        REPORTS_DIR / "confusion_matrix_classical.png",
        title="Normalized Confusion Matrix (TF-IDF + Linear SVM)"
    )

    plot_f1_scores(
        y_true_all,
        y_pred_all,
        labels,
        REPORTS_DIR / "f1_scores_classical.png",
        title="F1-score per Journal (TF-IDF + Linear SVM)"
    )

    plot_accuracy_per_class(
        cm_mean,
        labels,
        REPORTS_DIR / "accuracy_per_class_classical.png",
        title="Accuracy per Journal (TF-IDF + Linear SVM)"
    )

    # Entrenamiento final con TODO el dataset
    if pipe is None:
        raise Exception("Pipeline is not defined")
    pipe.fit(X, y)
    joblib.dump(pipe, CLASSICAL_DIR / "model.joblib")

    print("Classical model training finished (Stratified 5-Fold).")


if __name__ == "__main__":
    main()
