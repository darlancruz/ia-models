import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluation import (
    calculate_metrics,
    format_time,
    plot_confusion_matrix,
    print_results,
)
from objects.data import SEED, Dataset

# Hiperparâmetros FIXOS por dataset (a chave deve ser igual ao nome passado em prepare_dataset)
PARAMS = {
    "Mnist": {"hidden_layer_size": 128, "learning_rate": 0.10, "epochs": 3000},
    "Breast Cancer": {"hidden_layer_size": 8, "learning_rate": 0.05, "epochs": 500},
}
OUTPUT_DIR = Path("./output")


class NeuralNetwork:
    def __init__(self, hidden_layer_size=10, learning_rate=0.10, random_state=None):
        self.h = hidden_layer_size
        self.ln = learning_rate
        self.random_state = random_state

    # ------------------------------------------------------------------ modelo
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.normal(0, np.sqrt(2 / self.n_feat), (self.h, self.n_feat)).astype(
            np.float32
        )
        self.b1 = np.zeros((self.h, 1), dtype=np.float32)

        self.W2 = rng.normal(0, np.sqrt(2 / self.h), (self.n_classes, self.h)).astype(
            np.float32
        )
        self.b2 = np.zeros((self.n_classes, 1), dtype=np.float32)

    def _feedforward(self, X):
        # X no formato (n_features, n_amostras).
        Z1 = self.W1.dot(X) + self.b1
        A1 = self._relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self._softmax(Z2)
        return Z1, A1, Z2, A2

    def _backpropagation(self, Z1, A1, A2):
        def relu_deriv(x):
            return x > 0

        dZ2 = A2 - self.y
        dW2 = 1 / self.m * dZ2.dot(A1.T)
        db2 = (1 / self.m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * relu_deriv(Z1)
        dW1 = 1 / self.m * dZ1.dot(self.X.T)
        db1 = (1 / self.m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def _update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.ln * dW1
        self.b1 -= self.ln * db1
        self.W2 -= self.ln * dW2
        self.b2 -= self.ln * db2

    @staticmethod
    def _cross_entropy(A2, Y, eps=1e-12):
        m = Y.shape[1]
        return -np.sum(Y * np.log(np.clip(A2, eps, 1.0))) / m

    def fit(
        self,
        X_train,
        y_train,
        epochs=1000,
        verbose=False,
        X_val=None,
        y_val=None,
        desc=None,
    ):
        self.classes_, y_idx = np.unique(np.asarray(y_train), return_inverse=True)
        self.m, self.n_feat = X_train.shape
        self.n_classes = len(self.classes_)

        self.X = np.ascontiguousarray(X_train.T, dtype=np.float32)
        self.y = np.eye(self.n_classes, dtype=np.float32)[y_idx].T

        track_val = X_val is not None and y_val is not None
        if track_val:
            Xv = np.ascontiguousarray(X_val.T, dtype=np.float32)
            yv_idx = np.searchsorted(self.classes_, np.asarray(y_val))
            Yv = np.eye(self.n_classes, dtype=np.float32)[yv_idx].T

        self._init_params()

        self.history = {"loss": [], "accuracy": [], "cost": [], "acc": []}
        if track_val:
            self.history.update({"val_loss": [], "val_accuracy": []})

        iterator = (
            tqdm(range(epochs), desc=desc, leave=False) if desc else range(epochs)
        )
        for i in iterator:
            Z1, A1, _, A2 = self._feedforward(self.X)

            loss = self._cross_entropy(A2, self.y)
            accuracy = np.mean(np.argmax(A2, axis=0) == y_idx)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)
            self.history["cost"].append(loss)
            self.history["acc"].append(accuracy)

            if track_val:
                A2v = self._feedforward(Xv)[3]
                self.history["val_loss"].append(self._cross_entropy(A2v, Yv))
                self.history["val_accuracy"].append(
                    np.mean(np.argmax(A2v, axis=0) == yv_idx)
                )

            dW1, db1, dW2, db2 = self._backpropagation(Z1, A1, A2)
            self._update_params(dW1, db1, dW2, db2)

            if verbose and ((i + 1) % 50 == 0 or i == 0):
                print(f"Época {i + 1}/{epochs} - Acurácia: {accuracy:.2%}")

        self.n_epochs_ = len(self.history["loss"])
        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X).T, dtype=np.float32)
        return self._feedforward(X)[3]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=0)]

    @staticmethod
    def plot_learning_curve(history, path, title):
        epochs = np.arange(1, len(history["loss"]) + 1)
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_l.plot(epochs, history["loss"], label="Treino")
        if "val_loss" in history:
            ax_l.plot(epochs, history["val_loss"], label="Validação")
        ax_l.set_xlabel("Época")
        ax_l.set_ylabel("Cross-entropy")
        ax_l.set_title("Loss")

        ax_a.plot(epochs, history["accuracy"], label="Treino")
        if "val_accuracy" in history:
            ax_a.plot(epochs, history["val_accuracy"], label="Validação")
        ax_a.set_xlabel("Época")
        ax_a.set_ylabel("Accuracy")
        ax_a.set_title("Acurácia")

        for ax in (ax_l, ax_a):
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def run(
        cls,
        dataset: Dataset,
        hidden_layer_size=None,
        learning_rate=None,
        epochs=None,
        output_dir=OUTPUT_DIR,
        random_state=SEED,
    ):
        """Treina com hiperparâmetros fixos (PARAMS[dataset.name]), avalia no teste e gera gráficos.

        Os argumentos hidden_layer_size, learning_rate e epochs, se informados,
        sobrescrevem os valores fixos do dataset.
        Retorna {"params", "test", "history"}.
        """
        params = dict(PARAMS.get(dataset.name, {}))
        for key, value in [
            ("hidden_layer_size", hidden_layer_size),
            ("learning_rate", learning_rate),
            ("epochs", epochs),
        ]:
            if value is not None:
                params[key] = value

        missing = {"hidden_layer_size", "learning_rate", "epochs"} - params.keys()
        if missing:
            raise ValueError(
                f"Sem hiperparâmetros para o dataset '{dataset.name}' (faltam: {sorted(missing)}). "
                f"Adicione-o em PARAMS ou passe os valores em run()."
            )
        h, lr, ep = (
            params["hidden_layer_size"],
            params["learning_rate"],
            params["epochs"],
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        print("=" * 60)
        print(f"CLASSIFICAÇÃO REDE NEURAL - {dataset.name}")
        print(f"Hiperparâmetros fixos: h = {h} | lr = {lr} | épocas = {ep}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("ETAPA 1: TREINAMENTO (treino + validação)")
        print("=" * 60)

        t = time.perf_counter()
        model = cls(h, lr, random_state).fit(
            dataset.X_dev, dataset.y_dev, epochs=ep, desc="Treino"
        )
        print(f"(treino: {format_time(time.perf_counter() - t)})")

        curve_path = output_dir / f"{dataset.name} - curva_aprendizado_ann.png"
        cls.plot_learning_curve(
            model.history, curve_path, title=f"Rede Neural (h={h}, lr={lr}) - treino"
        )

        print("\n" + "=" * 60)
        print("ETAPA 2: AVALIAÇÃO FINAL NO TESTE")
        print("=" * 60)

        preds = model.predict(dataset.X_test)
        results = calculate_metrics(dataset.y_test, preds, dataset.classes)
        print()
        print_results(
            results,
            dataset.classes,
            extra={"Camada oculta": h, "Learning rate": lr, "Épocas": ep},
        )

        cm_path = output_dir / f"{dataset.name} - matriz_confusao_ann.png"
        plot_confusion_matrix(
            results,
            dataset.classes,
            cm_path,
            title=f"Rede Neural\nh = {h} | lr = {lr}",
            info=f"h = {h} | lr = {lr} | épocas = {ep}",
        )
        print(f"\nMatriz de confusão salva em: {cm_path.resolve()}")
        print(f"Curva de aprendizado salva em: {curve_path.resolve()}")

        print("\n" + "=" * 60)
        print(f"Duração Total: {format_time(time.perf_counter() - start)}")
        print("=" * 60)

        return {"params": params, "test": results, "history": model.history}
