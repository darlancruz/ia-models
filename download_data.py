import argparse
import shutil
import struct
import sys
from pathlib import Path

MNIST_SLUG = "hojjatk/mnist-dataset"
BREAST_SLUG = "uciml/breast-cancer-wisconsin-data"
DATA_DIR = Path("./data")
EXPECTED = ["mnist_train.csv", "breast_cancer.csv"]  # usados pelo main.py


def find_file(root: Path, *candidates: str) -> Path:
    """Procura um arquivo (não pasta) cujo nome seja um dos candidatos."""
    for p in root.rglob("*"):
        if p.is_file() and p.name in candidates:
            return p
    raise FileNotFoundError(f"Não encontrei {' / '.join(candidates)} em {root}")


def read_idx_images(path: Path):
    import numpy as np

    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Arquivo de imagens inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def read_idx_labels(path: Path):
    import numpy as np

    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Arquivo de rótulos inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def idx_to_csv(images_path: Path, labels_path: Path, out_csv: Path):
    import numpy as np

    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)
    table = np.column_stack((labels, images))
    header = "label," + ",".join(f"pixel{i}" for i in range(images.shape[1]))
    np.savetxt(out_csv, table, fmt="%d", delimiter=",", header=header, comments="")


def prepare_mnist(src: Path):
    print("Convertendo MNIST (IDX -> CSV)...")
    for split, prefix in (("train", "train"), ("test", "t10k")):
        images = find_file(
            src, f"{prefix}-images.idx3-ubyte", f"{prefix}-images-idx3-ubyte"
        )
        labels = find_file(
            src, f"{prefix}-labels.idx1-ubyte", f"{prefix}-labels-idx1-ubyte"
        )
        out = DATA_DIR / f"mnist_{split}.csv"
        idx_to_csv(images, labels, out)
        print(f"  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")


def prepare_breast_cancer(src: Path):
    csv = find_file(src, "data.csv", "breast_cancer.csv")
    out = DATA_DIR / "breast_cancer.csv"
    shutil.copy(csv, out)
    print(f"  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa os datasets do Kaggle para ./data"
    )
    parser.add_argument(
        "--force", action="store_true", help="baixa mesmo que já existam"
    )
    args = parser.parse_args()

    missing = [f for f in EXPECTED if not (DATA_DIR / f).exists()]
    if not missing and not args.force:
        print(f"Tudo já está em {DATA_DIR.resolve()}: {', '.join(EXPECTED)}")
        print("Use --force para baixar novamente.")
        return

    try:
        import kagglehub
    except ImportError:
        sys.exit("Falta o kagglehub. Instale com: pip install kagglehub numpy")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("Falta o numpy. Instale com: pip install numpy")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Baixando {MNIST_SLUG} ...")
        mnist_path = Path(
            kagglehub.dataset_download(MNIST_SLUG, force_download=args.force)
        )
        print(f"Baixando {BREAST_SLUG} ...")
        breast_path = Path(
            kagglehub.dataset_download(BREAST_SLUG, force_download=args.force)
        )
    except Exception as e:
        sys.exit(
            f"Falha no download: {e}\n"
            "Verifique a internet e as credenciais do Kaggle "
            "(KAGGLE_USERNAME/KAGGLE_KEY ou ~/.kaggle/kaggle.json)."
        )

    try:
        prepare_mnist(mnist_path)
        print("Copiando breast cancer...")
        prepare_breast_cancer(breast_path)
    except Exception as e:
        sys.exit(f"Falha ao preparar os arquivos: {e}")

    print("\nArquivos em ./data:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    still_missing = [f for f in EXPECTED if not (DATA_DIR / f).exists()]
    if still_missing:
        print(
            f"\nATENÇÃO: não encontrei {', '.join(still_missing)}. "
            "Confira os nomes em ./data e ajuste os caminhos no main.py."
        )


if __name__ == "__main__":
    main()
