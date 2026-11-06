from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10


@dataclass
class Dataset:
    name: str
    classes: np.ndarray
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_dev: np.ndarray
    y_dev: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_csv(path):
    print(f"Carregando: {path}")

    df = pd.read_csv(path)
    if "diagnosis" in df.columns:
        df = df.dropna(axis=1, how="all")
        y = df["diagnosis"].to_numpy()
        X = df.drop(columns=["diagnosis", "id"], errors="ignore")
        X = X.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    else:
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    return X, y


def prepare_dataset(
    path, name, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED
) -> Dataset:
    X, y = load_csv(Path(path))
    classes = np.unique(y)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=val_size, random_state=seed, stratify=y_dev
    )

    scaler_val = StandardScaler()
    X_tr_s = scaler_val.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler_val.transform(X_val).astype(np.float32)

    scaler_final = StandardScaler()
    X_dev_s = scaler_final.fit_transform(X_dev).astype(np.float32)
    X_test_s = scaler_final.transform(X_test).astype(np.float32)

    print(
        f"Treino + Validação: {len(X_dev)} amostras \n Teste: {len(X_test)} amostras \n "
        f"Atributos: {X_dev.shape[1]} \n Classes: {len(classes)}"
    )
    print(f"Treinamento: {len(X_tr)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

    return Dataset(
        name=name,
        classes=classes,
        X_tr=X_tr_s,
        y_tr=y_tr,
        X_val=X_val_s,
        y_val=y_val,
        X_dev=X_dev_s,
        y_dev=y_dev,
        X_test=X_test_s,
        y_test=y_test,
    )
