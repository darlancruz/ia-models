"""Baixa os CSVs da pasta do Google Drive para ./data.

Uso (a partir da raiz do projeto):
    pip install gdown
    python download_data.py            # baixa só o que faltar
    python download_data.py --force    # baixa tudo de novo
"""

import argparse
import shutil
import sys
from pathlib import Path

FOLDER_ID = "1vSpJhuWcuzzJfWooVnMi5letwM9M-lnf"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
DATA_DIR = Path("./data")
EXPECTED = ["mnist_train.csv", "breast_cancer.csv"]  # usados pelo main.py


def flatten_csvs(data_dir: Path):
    """Garante que todos os CSVs fiquem direto em data_dir (sem subpastas)."""
    for csv in list(data_dir.rglob("*.csv")):
        if csv.parent != data_dir:
            target = data_dir / csv.name
            shutil.move(str(csv), str(target))
    # remove subpastas que sobraram vazias
    for d in sorted((p for p in data_dir.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Baixa os datasets para ./data")
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
        import gdown
    except ImportError:
        sys.exit("Falta o gdown. Instale com: pip install gdown")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Baixando de {FOLDER_URL}")

    try:
        files = gdown.download_folder(
            url=FOLDER_URL, output=str(DATA_DIR), quiet=False, use_cookies=False
        )
    except Exception as e:
        sys.exit(
            f"Falha no download: {e}\n"
            "Verifique a internet e se a pasta está compartilhada como "
            "'Qualquer pessoa com o link'."
        )

    if not files:
        sys.exit(
            "Nenhum arquivo foi baixado (pasta vazia, sem permissão ou limite do Drive)."
        )

    flatten_csvs(DATA_DIR)

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
