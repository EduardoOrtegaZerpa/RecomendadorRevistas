import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from utils import shorten_labels


def plot_confusion_matrix(cm, labels, save_path, title):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_labels = shorten_labels(labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        xticklabels=short_labels,
        yticklabels=short_labels,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_f1_scores(y_true, y_pred, labels, save_path, title):
    f1 = np.asarray(
        f1_score(y_true, y_pred, labels=labels, average=None)
    ).tolist()

    short_labels = shorten_labels(labels)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=short_labels, y=f1)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(f1):
        ax.text(
            i,
            v - 0.05,
            f"{v:.2f}",
            ha="center",
            va="top",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy_per_class(cm, labels, save_path, title):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc_per_class = np.asarray(cm_norm.diagonal()).tolist()

    short_labels = shorten_labels(labels)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=short_labels, y=acc_per_class)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(acc_per_class):
        ax.text(
            i,
            v - 0.05,
            f"{v:.2f}",
            ha="center",
            va="top",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()