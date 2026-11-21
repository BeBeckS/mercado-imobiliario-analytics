import numpy as np
import pandas as pd

if __package__:
    from .load_data import load_properties
else:
    from load_data import load_properties


def count_records(properties: pd.DataFrame) -> int:
    return len(properties)


def missing_values_by_column(properties: pd.DataFrame) -> pd.DataFrame:
    missing_count = properties.isna().sum()
    total_records = len(properties)
    missing_percentage = (
        missing_count / total_records * 100
        if total_records
        else missing_count.astype(float)
    )
    return pd.DataFrame(
        {
            "quantidade_ausente": missing_count,
            "percentual_ausente": missing_percentage,
        }
    )


def numerical_descriptive_statistics(properties: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = properties.select_dtypes(include=[np.number])
    return numerical_columns.describe()


def _categorical_distribution(
    properties: pd.DataFrame, column: str
) -> pd.Series | None:
    if column not in properties.columns:
        return None

    return properties[column].value_counts(dropna=False).sort_values(
        ascending=False, kind="stable"
    )


def distribution_tipo_imovel(properties: pd.DataFrame) -> pd.Series | None:
    return _categorical_distribution(properties, "tipo_imovel")


def distribution_cidade(properties: pd.DataFrame) -> pd.Series | None:
    return _categorical_distribution(properties, "cidade")


def distribution_bairro(properties: pd.DataFrame) -> pd.Series | None:
    return _categorical_distribution(properties, "bairro")


def distribution_finalidade(properties: pd.DataFrame) -> pd.Series | None:
    return _categorical_distribution(properties, "finalidade")


def count_unique_advertisement_ids(properties: pd.DataFrame) -> int | None:
    if "id_anuncio" not in properties.columns:
        return None

    return int(properties["id_anuncio"].nunique())


def _column_statistics(
    properties: pd.DataFrame, column: str
) -> dict[str, float] | None:
    if column not in properties.columns:
        return None

    values = properties[column]
    if not pd.api.types.is_numeric_dtype(values) or not values.notna().any():
        return None

    return {
        "Média": float(values.mean()),
        "Mediana": float(values.median()),
        "Mínimo": float(values.min()),
        "Máximo": float(values.max()),
    }


def price_statistics(properties: pd.DataFrame) -> dict[str, float] | None:
    return _column_statistics(properties, "preco")


def price_per_square_meter_statistics(
    properties: pd.DataFrame,
) -> dict[str, float] | None:
    return _column_statistics(properties, "preco_m2")


def area_statistics(properties: pd.DataFrame) -> dict[str, float] | None:
    return _column_statistics(properties, "area_m2")


def _print_section(title: str, result: object) -> None:
    print(f"\n{title}")
    print("=" * len(title))

    if result is None:
        print("Dados indisponíveis: coluna ausente ou sem valores suficientes.")
    elif isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
        print("Sem dados para exibir.")
    elif isinstance(result, dict):
        for statistic, value in result.items():
            print(f"{statistic}: {value}")
    else:
        print(result)


def main() -> None:
    properties = load_properties()

    _print_section("Quantidade de registros", count_records(properties))
    _print_section("Valores ausentes por coluna", missing_values_by_column(properties))
    _print_section(
        "Estatísticas descritivas das colunas numéricas",
        numerical_descriptive_statistics(properties),
    )
    _print_section(
        "Distribuição por tipo de imóvel", distribution_tipo_imovel(properties)
    )
    _print_section("Distribuição por cidade", distribution_cidade(properties))
    _print_section("Distribuição por bairro", distribution_bairro(properties))
    _print_section("Distribuição por finalidade", distribution_finalidade(properties))
    _print_section(
        "IDs de anúncio únicos", count_unique_advertisement_ids(properties)
    )
    _print_section("Estatísticas de preço", price_statistics(properties))
    _print_section(
        "Estatísticas de preço por metro quadrado",
        price_per_square_meter_statistics(properties),
    )
    _print_section("Estatísticas de área", area_statistics(properties))


if __name__ == "__main__":
    main()
