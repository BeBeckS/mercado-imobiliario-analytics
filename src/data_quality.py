import numpy as np
import pandas as pd

if __package__:
    from .load_data import load_properties
else:
    from load_data import load_properties


_MIN_VALID_GROUP_SIZE = 4
_IQR_COLUMNS = ("preco", "area_m2", "preco_m2")


def _valid_values_mask(df: pd.DataFrame, column: str) -> pd.Series:
    mask = df[column].notna()
    if column == "preco_m2":
        mask &= df[column].gt(0).fillna(False)
    return mask.fillna(False)


def check_invalid_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    total_records = len(df)
    invalid_values = []

    for column in _IQR_COLUMNS:
        invalid_mask = df[column].le(0).fillna(False)
        count = int(invalid_mask.sum())
        percentage = count / total_records * 100 if total_records else np.nan
        invalid_values.append(
            {
                "coluna": column,
                "quantidade": count,
                "percentual": percentage,
            }
        )

    return pd.DataFrame(invalid_values).set_index("coluna")


def calculate_iqr_bounds(
    df: pd.DataFrame, column: str
) -> dict[str, float | None]:
    values = df.loc[_valid_values_mask(df, column), column]
    if values.empty:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "limite_inferior": None,
            "limite_superior": None,
        }

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": q1 - 1.5 * iqr,
        "limite_superior": q3 + 1.5 * iqr,
    }


def detect_outliers(df: pd.DataFrame, column: str) -> dict[str, object]:
    bounds = calculate_iqr_bounds(df, column)
    if bounds["limite_inferior"] is None:
        return {
            "quantidade_outliers": None,
            "percentual_outliers": None,
            "indices_outliers": [],
            "ids_outliers": [],
            "limites_iqr": bounds,
        }

    valid_positions = np.flatnonzero(
        _valid_values_mask(df, column).to_numpy(dtype=bool)
    )
    values = df.iloc[valid_positions][column]
    outlier_mask = (
        values.lt(bounds["limite_inferior"])
        | values.gt(bounds["limite_superior"])
    ).to_numpy(dtype=bool)
    outlier_positions = valid_positions[outlier_mask]
    outlier_count = len(outlier_positions)

    return {
        "quantidade_outliers": outlier_count,
        "percentual_outliers": outlier_count / len(valid_positions) * 100,
        "indices_outliers": df.index.take(outlier_positions).tolist(),
        "ids_outliers": (
            df.iloc[outlier_positions]["id_anuncio"].tolist()
            if "id_anuncio" in df.columns
            else []
        ),
        "limites_iqr": bounds,
    }


def outliers_by_property_type(df: pd.DataFrame, column: str) -> pd.DataFrame:
    results = []

    for property_type, group in df.groupby(
        "tipo_imovel", dropna=False, observed=True, sort=False
    ):
        total_count = len(group)
        valid_count = int(_valid_values_mask(group, column).sum())

        if valid_count < _MIN_VALID_GROUP_SIZE:
            outlier_count = pd.NA
            outlier_percentage = pd.NA
        else:
            detected = detect_outliers(group, column)
            outlier_count = detected["quantidade_outliers"]
            outlier_percentage = outlier_count / total_count * 100

        results.append(
            {
                "tipo_imovel": property_type,
                "quantidade_total": total_count,
                "quantidade_outliers": outlier_count,
                "percentual_outliers": outlier_percentage,
            }
        )

    return pd.DataFrame(
        results,
        columns=[
            "tipo_imovel",
            "quantidade_total",
            "quantidade_outliers",
            "percentual_outliers",
        ],
    ).sort_values("quantidade_total", ascending=False, kind="stable")


def summarize_data_quality(df: pd.DataFrame) -> pd.Series:
    invalid_values = check_invalid_numeric_values(df)["quantidade"]
    outlier_counts = {
        column: detect_outliers(df, column)["quantidade_outliers"]
        for column in _IQR_COLUMNS
    }

    return pd.Series(
        {
            "registros_totais": len(df),
            "ids_duplicados": int(df["id_anuncio"].dropna().duplicated().sum()),
            "precos_invalidos": int(invalid_values["preco"]),
            "areas_invalidas": int(invalid_values["area_m2"]),
            "preco_m2_invalido": int(invalid_values["preco_m2"]),
            "valores_ausentes": int(df.isna().sum().sum()),
            "outliers_preco": outlier_counts["preco"],
            "outliers_area_m2": outlier_counts["area_m2"],
            "outliers_preco_m2": outlier_counts["preco_m2"],
        },
        dtype=object,
    )


def _format_percentage(value: object) -> str:
    if pd.isna(value):
        return "N/D"
    return f"{value:.2f}%"


def _print_section(title: str, result: object) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(result)


def main() -> None:
    properties = load_properties()

    invalid_values = check_invalid_numeric_values(properties).copy()
    invalid_values["percentual"] = invalid_values["percentual"].map(
        _format_percentage
    )
    _print_section("Valores numéricos inválidos", invalid_values)

    missing_values = (
        properties.isna().sum().rename("quantidade_ausente").to_frame()
    )
    _print_section("Valores ausentes por coluna", missing_values)

    for column in _IQR_COLUMNS:
        bounds = calculate_iqr_bounds(properties, column)
        _print_section(f"Limites IQR - {column}", pd.Series(bounds))

        outliers = detect_outliers(properties, column)
        outlier_details = (
            f"Quantidade: {outliers['quantidade_outliers']}\n"
            f"Percentual: {_format_percentage(outliers['percentual_outliers'])}\n"
            f"Índices: {outliers['indices_outliers']}\n"
            f"IDs de anúncio: {outliers['ids_outliers']}"
        )
        _print_section(f"Outliers - {column}", outlier_details)

        grouped_outliers = outliers_by_property_type(properties, column).copy()
        grouped_outliers["percentual_outliers"] = grouped_outliers[
            "percentual_outliers"
        ].map(_format_percentage)
        _print_section(
            f"Outliers por tipo de imóvel - {column} "
            f"(grupos com menos de {_MIN_VALID_GROUP_SIZE} valores válidos: N/D)",
            grouped_outliers,
        )

    _print_section("Resumo consolidado da qualidade", summarize_data_quality(properties))


if __name__ == "__main__":
    main()
