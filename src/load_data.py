from pathlib import Path

import pandas as pd


CURATED_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "curated"
    / "properties.parquet"
)


def load_properties() -> pd.DataFrame:
    return pd.read_parquet(CURATED_PATH)


def main() -> None:
    properties = load_properties()

    print(f"Quantidade de registros: {len(properties)}")
    print(f"Colunas: {list(properties.columns)}")
    print("Tipos de dados:")
    print(properties.dtypes)
    print("Primeiras 5 linhas:")
    print(properties.head(5))


if __name__ == "__main__":
    main()
