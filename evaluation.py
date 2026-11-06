import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def calculate_metrics(y_true, y_pred, classes):
    C = len(classes)
    t = np.searchsorted(classes, y_true)
    p = np.searchsorted(classes, y_pred)

    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (t, p), 1)

    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": TP.sum() / cm.sum(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
        "per_class": {"precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": cm,
    }


def print_results(results, classes, extra=None):
    """Imprime métricas globais e tabela por classe."""
    for label, value in (extra or {}).items():
        print(f"{label}: {value}")
    print(f"Accuracy (global): {results['accuracy']:.4f}")
    print(f"Precision (macro): {results['precision']:.4f}")
    print(f"Recall (macro):    {results['recall']:.4f}")
    print(f"F1-Score (macro):  {results['f1']:.4f}")

    pc = results["per_class"]
    print("\nPor classe:")
    print("Classe | Precision | Recall |   F1")
    for i, c in enumerate(classes):
        print(
            f"{str(c):>6} |   {pc['precision'][i]:.4f}  | {pc['recall'][i]:.4f} | {pc['f1'][i]:.4f}"
        )


def plot_confusion_matrix(results, classes, path, title, info=None):
    fig, ax = plt.subplots(figsize=(9, 9))

    ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"], display_labels=classes
    ).plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)

    legenda = (
        (f"{info}\n" if info else "")
        + f"Accuracy (Global): {results['accuracy']:.4f} | "
        f"Precision (Macro): {results['precision']:.4f}\n"
        f"Recall (Macro): {results['recall']:.4f} | "
        f"F1-Score (Macro): {results['f1']:.4f}"
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(legenda, fontsize=10, labelpad=15)
    ax.set_ylabel("Classe real", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_time(seconds):
    minutes, sec = divmod(seconds, 60)
    return (
        f"{seconds:.2f} s ({int(minutes)} min {sec:.1f} s)"
        if minutes
        else f"{seconds:.2f} s"
    )
