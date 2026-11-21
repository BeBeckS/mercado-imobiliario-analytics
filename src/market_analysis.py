import numpy as np
import pandas as pd

if __package__:
    from .load_data import load_properties
else:
    from load_data import load_properties


_TOP_PROPERTY_COLUMNS = [
    "titulo",
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "preco",
    "preco_m2",
    "url",
]


def _valid_positive_price_per_m2(df: pd.DataFrame) -> pd.Series:
    price_per_m2 = df["preco_m2"]
    nonzero_area = df["area_m2"].ne(0).fillna(True)
    return (
        price_per_m2.notna()
        & np.isfinite(price_per_m2)
        & price_per_m2.gt(0)
        & nonzero_area
    )


def analyze_price_by_property_type(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("tipo_imovel", dropna=False, observed=True, sort=False)
        .agg(
            quantidade=("tipo_imovel", "size"),
            preco_medio=("preco", "mean"),
            preco_mediano=("preco", "median"),
            preco_minimo=("preco", "min"),
            preco_maximo=("preco", "max"),
        )
        .sort_values("quantidade", ascending=False, kind="stable")
    )
    return result


def analyze_price_per_m2_by_property_type(df: pd.DataFrame) -> pd.DataFrame:
    valid_properties = df.loc[_valid_positive_price_per_m2(df)]
    return (
        valid_properties.groupby(
            "tipo_imovel", dropna=False, observed=True, sort=False
        )
        .agg(
            quantidade=("preco_m2", "size"),
            preco_m2_medio=("preco_m2", "mean"),
            preco_m2_mediano=("preco_m2", "median"),
            preco_m2_minimo=("preco_m2", "min"),
            preco_m2_maximo=("preco_m2", "max"),
        )
        .sort_values("preco_m2_mediano", ascending=False, kind="stable")
    )


def analyze_properties_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby("bairro", dropna=False, observed=True, sort=False).size()
    result = result.rename("quantidade").to_frame()
    result["percentual_do_total"] = result["quantidade"] / len(df) * 100
    return result.sort_values("quantidade", ascending=False, kind="stable")


def analyze_price_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("bairro", dropna=False, observed=True, sort=False)
        .agg(
            quantidade=("bairro", "size"),
            preco_medio=("preco", "mean"),
            preco_mediano=("preco", "median"),
        )
    )
    valid_properties = df.loc[_valid_positive_price_per_m2(df)]
    median_price_per_m2 = valid_properties.groupby(
        "bairro", dropna=False, observed=True, sort=False
    ).agg(preco_m2_mediano=("preco_m2", "median"))
    result = result.join(median_price_per_m2)
    return result.sort_values("quantidade", ascending=False, kind="stable")


def analyze_area_price_relationship(df: pd.DataFrame) -> dict[str, int | float | None]:
    valid_values = (
        df["area_m2"].notna()
        & df["preco"].notna()
        & np.isfinite(df["area_m2"])
        & np.isfinite(df["preco"])
    )
    values = df.loc[valid_values, ["area_m2", "preco"]]
    correlation = (
        values["area_m2"].corr(values["preco"], method="pearson")
        if len(values) >= 2
        else None
    )
    if correlation is not None and not np.isfinite(correlation):
        correlation = None

    return {
        "quantidade_registros": len(values),
        "correlacao_pearson": (
            float(correlation) if correlation is not None else None
        ),
    }


def _top_properties(df: pd.DataFrame, n: int, largest: bool) -> pd.DataFrame:
    if n <= 0:
        return df.loc[:, _TOP_PROPERTY_COLUMNS].iloc[0:0]

    valid_prices = df["preco"].notna() & np.isfinite(df["preco"])
    properties = df.loc[valid_prices, _TOP_PROPERTY_COLUMNS]
    if largest:
        return properties.nlargest(n, "preco")
    return properties.nsmallest(n, "preco")


def top_expensive_properties(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return _top_properties(df, n, largest=True)


def top_cheap_properties(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return _top_properties(df, n, largest=False)


def analyze_concentration(df: pd.DataFrame) -> dict[str, object]:
    total_properties = len(df)
    if total_properties:
        porto_alegre_percentage = (
            df["cidade"].eq("Porto Alegre").sum() / total_properties * 100
        )
        top_neighborhood_count = (
            df["bairro"].value_counts().nlargest(5).sum()
        )
        top_neighborhood_percentage = (
            top_neighborhood_count / total_properties * 100
        )
        property_type_percentages = (
            df["tipo_imovel"].value_counts(dropna=False)
            / total_properties
            * 100
        )
    else:
        porto_alegre_percentage = None
        top_neighborhood_percentage = None
        property_type_percentages = pd.Series(dtype=float, name="percentual")

    property_type_percentages.name = "percentual"
    return {
        "percentual_porto_alegre": float(porto_alegre_percentage)
        if porto_alegre_percentage is not None
        else None,
        "percentual_top_5_bairros": float(top_neighborhood_percentage)
        if top_neighborhood_percentage is not None
        else None,
        "percentual_por_tipo_imovel": property_type_percentages,
    }


def _format_currency(value: object) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "N/D"

    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"R$ {formatted}"


def _format_percentage(value: object) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "N/D"
    return f"{value:.2f}%"


def _format_table(
    table: pd.DataFrame,
    currency_columns: tuple[str, ...] = (),
    percentage_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    formatted = table.copy()
    for column in currency_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_currency)
    for column in percentage_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percentage)
    return formatted


def _print_section(title: str, result: object) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(result)


def main() -> None:
    properties = load_properties()

    _print_section(
        "Preço por tipo de imóvel",
        _format_table(
            analyze_price_by_property_type(properties),
            currency_columns=(
                "preco_medio",
                "preco_mediano",
                "preco_minimo",
                "preco_maximo",
            ),
        ),
    )
    _print_section(
        "Preço por m² por tipo de imóvel (mediana como indicador principal)",
        _format_table(
            analyze_price_per_m2_by_property_type(properties),
            currency_columns=(
                "preco_m2_medio",
                "preco_m2_mediano",
                "preco_m2_minimo",
                "preco_m2_maximo",
            ),
        ),
    )
    _print_section(
        "Imóveis por bairro",
        _format_table(
            analyze_properties_by_neighborhood(properties),
            percentage_columns=("percentual_do_total",),
        ),
    )
    _print_section(
        "Preço por bairro",
        _format_table(
            analyze_price_by_neighborhood(properties),
            currency_columns=("preco_medio", "preco_mediano", "preco_m2_mediano"),
        ),
    )

    relationship = analyze_area_price_relationship(properties)
    correlation = relationship["correlacao_pearson"]
    correlation_display = (
        f"{correlation:.4f}" if correlation is not None else "N/D"
    )
    _print_section(
        "Relação entre área e preço (correlação não implica causalidade)",
        f"Registros utilizados: {relationship['quantidade_registros']}\n"
        f"Correlação de Pearson: {correlation_display}",
    )
    _print_section(
        "10 imóveis mais caros",
        _format_table(
            top_expensive_properties(properties),
            currency_columns=("preco", "preco_m2"),
        ),
    )
    _print_section(
        "10 imóveis mais baratos",
        _format_table(
            top_cheap_properties(properties),
            currency_columns=("preco", "preco_m2"),
        ),
    )

    concentration = analyze_concentration(properties)
    _print_section(
        "Concentração de imóveis",
        f"Em Porto Alegre: {_format_percentage(concentration['percentual_porto_alegre'])}\n"
        "Nos 5 bairros com maior quantidade: "
        f"{_format_percentage(concentration['percentual_top_5_bairros'])}\n"
        "Percentual por tipo de imóvel:\n"
        f"{concentration['percentual_por_tipo_imovel'].map(_format_percentage)}",
    )


if __name__ == "__main__":
    main()
