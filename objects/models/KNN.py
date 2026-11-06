import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from objects.data import Dataset
from evaluation import (
    calculate_metrics,
    format_time,
    plot_confusion_matrix,
    print_results,
)

K_RANGE = range(3, 16, 2)  # K ímpares
METRICS = ["euclidean", "manhattan"]
TEST_BATCH_SIZE = 250
TRAIN_CHUNK_SIZE = 5000
OUTPUT_DIR = Path("../../output")


class KNN:
    VALID_METRICS = ("euclidean", "manhattan")

    def __init__(
        self,
        k=5,
        metric="manhattan",
        test_batch_size=TEST_BATCH_SIZE,
        train_chunk_size=TRAIN_CHUNK_SIZE,
    ):
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"metric deve ser um de {self.VALID_METRICS}, não '{metric}'"
            )
        self.k = k
        self.metric = metric
        self.test_batch_size = test_batch_size
        self.train_chunk_size = train_chunk_size

    def fit(self, X, y):
        self.X_train = np.ascontiguousarray(X, dtype=np.float32)
        self.classes_, self.y_idx = np.unique(np.asarray(y), return_inverse=True)

        if self.metric == "manhattan":
            self.X_train_T = np.ascontiguousarray(self.X_train.T)
        else:
            self.sq_norms = np.sum(self.X_train**2, axis=1)
        return self

    def manhattan_metric(self, XbT, start, end):
        n_feat, n = XbT.shape
        out = np.zeros((n, end - start), dtype=np.float32)
        tmp = np.empty_like(out)
        for f in range(n_feat):
            np.subtract(XbT[f][:, None], self.X_train_T[f, start:end][None, :], out=tmp)
            np.abs(tmp, out=tmp)
            out += tmp
        return out

    def euclidean_metric(self, X_batch, b2, start, end):
        # ||a||² + ||b||² - 2a·b  (mais rápido que a versão convencional)
        d2 = (
            b2
            + self.sq_norms[start:end][None, :]
            - 2.0 * (X_batch @ self.X_train[start:end].T)
        )
        return np.sqrt(np.maximum(d2, 0.0))

    def kneighbors_batch(self, X_batch, k):
        n = X_batch.shape[0]
        best_d = np.full((n, k), np.inf, dtype=np.float32)
        best_l = np.zeros((n, k), dtype=np.int64)
        rows = np.arange(n)[:, None]

        if self.metric == "manhattan":
            XbT = np.ascontiguousarray(X_batch.T)
        else:
            b2 = np.sum(X_batch**2, axis=1, keepdims=True)

        for start in range(0, len(self.X_train), self.train_chunk_size):
            end = min(start + self.train_chunk_size, len(self.X_train))

            if self.metric == "manhattan":
                d = self.manhattan_metric(XbT, start, end)
            else:
                d = self.euclidean_metric(X_batch, b2, start, end)

            l = np.broadcast_to(self.y_idx[start:end], d.shape)

            all_d = np.hstack((best_d, d.astype(np.float32, copy=False)))
            all_l = np.hstack((best_l, l))

            idx = np.argpartition(all_d, k - 1, axis=1)[:, :k]
            best_d = all_d[rows, idx]
            best_l = all_l[rows, idx]

        order = np.argsort(best_d, axis=1, kind="stable")
        return np.take_along_axis(best_l, order, axis=1)

    def kneighbors_labels(self, X, k, desc="Classificando"):
        X = np.ascontiguousarray(X, dtype=np.float32)

        results = [
            self.kneighbors_batch(X[i : i + self.test_batch_size], k)
            for i in tqdm(
                range(0, len(X), self.test_batch_size), desc=desc, leave=False
            )
        ]
        return np.vstack(results)

    def predict_multiple_k(self, X, ks, desc="Classificando"):
        labels = self.kneighbors_labels(X, max(ks), desc=desc)
        return {k: self.classes_[self.vote(labels[:, :k])] for k in ks}

    def predict(self, X, desc="Classificando"):
        return self.predict_multiple_k(X, [self.k], desc=desc)[self.k]

    def vote(self, label_idx):
        n, k = label_idx.shape
        onehot = label_idx[:, :, None] == np.arange(len(self.classes_))
        counts = onehot.sum(axis=1)

        rank_w = 0.5 ** np.arange(k)
        tiebreak = (onehot * rank_w[None, :, None]).sum(axis=1)

        return np.argmax(counts * 4.0 + tiebreak, axis=1)

    @classmethod
    def select_k(cls, dataset: Dataset, metric, k_range=K_RANGE):
        ks = list(k_range)
        knn = cls(k=max(ks), metric=metric).fit(dataset.X_tr, dataset.y_tr)
        preds_by_k = knn.predict_multiple_k(
            dataset.X_val, ks, desc="Validação (vizinhos)"
        )

        summary = {}
        for k in ks:
            summary[k] = s = calculate_metrics(
                dataset.y_val, preds_by_k[k], dataset.classes
            )
            print(
                f"K = {k:2d} | F1 = {s['f1']:.4f} | Recall = {s['recall']:.4f} | "
                f"Precision = {s['precision']:.4f} | Accuracy = {s['accuracy']:.4f}"
            )

        best_k = max(
            summary,
            key=lambda k: (
                round(summary[k]["f1"], 6),
                round(summary[k]["accuracy"], 6),
            ),
        )
        return best_k, summary

    @staticmethod
    def plot_k_curve(summary, best_k, path, metric):
        ks = list(summary.keys())

        fig, ax = plt.subplots(figsize=(9, 5))
        for name, label in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision (macro)"),
            ("recall", "Recall (macro)"),
            ("f1", "F1 (macro)"),
        ]:
            ax.plot(ks, [summary[k][name] for k in ks], marker="o", label=label)

        ax.axvline(best_k, color="gray", linestyle="--", label=f"Melhor K = {best_k}")
        ax.set_xlabel("K")
        ax.set_ylabel("Desempenho na validação")
        ax.set_title(f"Desempenho por K ({metric})")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def run(
        cls, dataset: Dataset, metrics=METRICS, k_range=K_RANGE, output_dir=OUTPUT_DIR
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        all_results = {}

        print("=" * 60)
        print(f"CLASSIFICAÇÃO KNN - {dataset.name}")
        print("=" * 60)

        for metric in metrics:
            metric_start = time.perf_counter()

            print("\n" + "=" * 60)
            print(f"ETAPA 1: SELEÇÃO DE K NA VALIDAÇÃO ({metric})")
            print("Critério: F1 macro")
            print("=" * 60)

            t = time.perf_counter()
            best_k, summary = cls.select_k(dataset, metric, k_range)
            print(
                f"\nMelhor K: {best_k}  (etapa: {format_time(time.perf_counter() - t)})"
            )

            k_curve_path = output_dir / f"{dataset.name} - metricas_por_k_{metric}.png"
            cls.plot_k_curve(summary, best_k, k_curve_path, metric)

            print("\n" + "=" * 60)
            print("ETAPA 2: TREINAMENTO FINAL")
            print("=" * 60)

            knn = cls(best_k, metric).fit(dataset.X_dev, dataset.y_dev)

            print("\n" + "=" * 60)
            print("ETAPA 3: AVALIAÇÃO FINAL NO TESTE")
            print("=" * 60)

            t = time.perf_counter()
            preds = knn.predict(dataset.X_test, desc="Classificando teste")
            print(f"(classificação do teste: {format_time(time.perf_counter() - t)})")

            results = calculate_metrics(dataset.y_test, preds, dataset.classes)
            print()
            print_results(results, dataset.classes, extra={"Melhor K": best_k})

            cm_path = output_dir / f"{dataset.name} - matriz_confusao_knn_{metric}.png"
            plot_confusion_matrix(
                results,
                dataset.classes,
                cm_path,
                title=f"KNN\nDistância {metric}",
                info=f"Melhor K = {best_k}",
            )
            print(f"\nMatriz de confusão salva em: {cm_path.resolve()}")
            print(f"Gráfico de métricas por K salvo em: {k_curve_path.resolve()}")

            print("\n" + "=" * 60)
            print(
                f"{metric.capitalize()} finalizado! \n Tempo de execução: "
                f"{format_time(time.perf_counter() - metric_start)}"
            )
            print("=" * 60)

            all_results[metric] = {
                "best_k": best_k,
                "val_summary": summary,
                "test": results,
            }

        print("\n" + "=" * 60)
        print(f"Duração Total: {format_time(time.perf_counter() - start)}")
        print("=" * 60)
        return all_results
