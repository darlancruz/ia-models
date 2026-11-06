import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluation import (
    calculate_metrics,
    format_time,
    plot_confusion_matrix,
    print_results,
)
from objects.data import Dataset

VAR_SMOOTHINGS = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
OUTPUT_DIR = Path("./output")


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9, batch_size=2000):
        self.var_smoothing = var_smoothing
        self.batch_size = batch_size

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n, n_feat = X.shape
        C = len(self.classes_)

        self.theta_ = np.zeros((C, n_feat))  # médias por classe
        self.var_ = np.zeros((C, n_feat))  # variâncias por classe
        self.log_priors_ = np.zeros(C)

        for c in range(C):
            Xc = X[y_idx == c]
            self.theta_[c] = Xc.mean(axis=0, dtype=np.float64)
            self.var_[c] = Xc.var(axis=0, dtype=np.float64)
            self.log_priors_[c] = np.log(len(Xc) / n)

        # suavisa os valores para evitar variância 0
        # para que não ocorra divisão por 0 causaria divisão por zero.
        eps = self.var_smoothing * X.var(axis=0, dtype=np.float64).max()
        self.epsilon_ = max(eps, np.finfo(np.float64).tiny)
        self.var_ += self.epsilon_
        return self

    def _joint_log_likelihood(self, X):
        out = np.empty((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            const = -0.5 * np.sum(np.log(2 * np.pi * self.var_[c]))
            sq = (X - self.theta_[c]) ** 2 / self.var_[c]
            out[:, c] = self.log_priors_[c] + const - 0.5 * sq.sum(axis=1)
        return out

    def predict_log_proba(self, X):
        X = np.asarray(X)
        parts = [
            self._joint_log_likelihood(X[i : i + self.batch_size])
            for i in range(0, len(X), self.batch_size)
        ]
        jll = np.vstack(parts)
        m = jll.max(axis=1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True)))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        X = np.asarray(X)
        idx = np.concatenate(
            [
                np.argmax(
                    self._joint_log_likelihood(X[i : i + self.batch_size]), axis=1
                )
                for i in range(0, len(X), self.batch_size)
            ]
        )
        return self.classes_[idx]

    @classmethod
    def select_var_smoothing(cls, dataset: Dataset, var_smoothings=VAR_SMOOTHINGS):
        summary = {}
        for vs in var_smoothings:
            model = cls(var_smoothing=vs).fit(dataset.X_tr, dataset.y_tr)
            preds = model.predict(dataset.X_val)
            summary[vs] = s = calculate_metrics(dataset.y_val, preds, dataset.classes)
            print(
                f"var_smoothing = {vs:<8.0e} | F1 = {s['f1']:.4f} | Recall = {s['recall']:.4f} | "
                f"Precision = {s['precision']:.4f} | Accuracy = {s['accuracy']:.4f}"
            )

        best = max(
            summary,
            key=lambda v: (
                round(summary[v]["f1"], 6),
                round(summary[v]["accuracy"], 6),
            ),
        )
        return best, summary

    @staticmethod
    def plot_smoothing_curve(summary, best, path):
        vals = list(summary.keys())

        fig, ax = plt.subplots(figsize=(9, 5))
        for name, label in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision (macro)"),
            ("recall", "Recall (macro)"),
            ("f1", "F1 (macro)"),
        ]:
            ax.plot(vals, [summary[v][name] for v in vals], marker="o", label=label)

        ax.axvline(best, color="gray", linestyle="--", label=f"Melhor = {best:.0e}")
        ax.set_xscale("log")
        ax.set_xlabel("var_smoothing")
        ax.set_ylabel("Desempenho na validação")
        ax.set_title("Desempenho por var_smoothing (Naive Bayes)")
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def run(
        cls,
        dataset: Dataset,
        var_smoothings=VAR_SMOOTHINGS,
        output_dir=OUTPUT_DIR,
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        print("=" * 60)
        print(f"CLASSIFICAÇÃO NAIVE BAYES - {dataset.name}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("ETAPA 1: SELEÇÃO DE var_smoothing NA VALIDAÇÃO")
        print("Critério: F1 macro")
        print("=" * 60)

        t = time.perf_counter()
        best, summary = cls.select_var_smoothing(dataset, var_smoothings)
        print(
            f"\nMelhor var_smoothing: {best:.0e}  (etapa: {format_time(time.perf_counter() - t)})"
        )

        curve_path = output_dir / f"{dataset.name} - metricas_por_var_smoothing_nb.png"
        cls.plot_smoothing_curve(summary, best, curve_path)

        print("\n" + "=" * 60)
        print("ETAPA 2: TREINAMENTO FINAL")
        print("=" * 60)

        model = cls(var_smoothing=best).fit(dataset.X_dev, dataset.y_dev)

        print("\n" + "=" * 60)
        print("ETAPA 3: AVALIAÇÃO FINAL NO TESTE")
        print("=" * 60)

        t = time.perf_counter()
        preds = model.predict(dataset.X_test)
        print(f"(classificação do teste: {format_time(time.perf_counter() - t)})")

        results = calculate_metrics(dataset.y_test, preds, dataset.classes)
        print()
        print_results(
            results, dataset.classes, extra={"Melhor var_smoothing": f"{best:.0e}"}
        )

        cm_path = output_dir / f"{dataset.name} - matriz_confusao_nb.png"
        plot_confusion_matrix(
            results,
            dataset.classes,
            cm_path,
            title="Naive Bayes Gaussiano",
            info=f"var_smoothing = {best:.0e}",
        )
        print(f"\nMatriz de confusão salva em: {cm_path.resolve()}")
        print(f"Gráfico de métricas por var_smoothing salvo em: {curve_path.resolve()}")

        print("\n" + "=" * 60)
        print(f"Duração Total: {format_time(time.perf_counter() - start)}")
        print("=" * 60)

        return {"best_var_smoothing": best, "val_summary": summary, "test": results}
