import os
import joblib
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

from config import DATA_RAW_DIR, BERT_DIR, REPORTS_DIR
from data_loader import load_dataset_from_folders, split_train_test
from plots import plot_confusion_matrix, plot_f1_scores, plot_accuracy_per_class


# DATASET
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# MAIN
def main():
    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BERT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset_from_folders(DATA_RAW_DIR)
    train_df, test_df = split_train_test(df, test_size=0.2, seed=42)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    labels = list(le.classes_)
    joblib.dump(le, BERT_DIR / "label_encoder.joblib")

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256
    )

    test_enc = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256
    )

    train_dataset = TextDataset(
        {k: torch.tensor(v) for k, v in train_enc.items()},
        y_train
    )

    test_dataset = TextDataset(
        {k: torch.tensor(v) for k, v in test_enc.items()},
        y_test
    )

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels)
    )

    args = TrainingArguments(
        output_dir=str(BERT_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    # Predictions
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_test_str = le.inverse_transform(y_test)
    y_pred_str = le.inverse_transform(y_pred)

    # Metrics
    cm = confusion_matrix(
        y_test_str,
        y_pred_str,
        labels=labels
    )

    # Plots
    plot_confusion_matrix(
        cm,
        labels,
        REPORTS_DIR / "confusion_matrix_bert.png",
        title="Normalized Confusion Matrix (BERT)"
    )

    plot_f1_scores(
        y_test_str,
        y_pred_str,
        labels,
        REPORTS_DIR / "f1_scores_bert.png",
        title="F1-score per Journal (BERT)"
    )

    plot_accuracy_per_class(
        cm,
        labels,
        REPORTS_DIR / "accuracy_per_class_bert.png",
        title="Accuracy per Journal (BERT)"
    )

    # Save model
    model.save_pretrained(BERT_DIR / "final_model")
    tokenizer.save_pretrained(BERT_DIR / "final_model")

    print("BERT training finished. All plots saved in outputs/reports/")


if __name__ == "__main__":
    main()
