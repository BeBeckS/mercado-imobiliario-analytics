from pathlib import Path

import numpy as np
import pandas as pd

from src.load_data import load_properties


OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "outputs"
)


def prepare_analytics_data(properties: pd.DataFrame) -> pd.DataFrame:
    """Prepara uma cópia dos dados para as análises estatísticas."""
    analytics = properties.copy()

    # Áreas menores que 10 m² são tratadas como inválidas para
    # análises que dependem de área e preço por m².
    analytics.loc[analytics["area_m2"] < 10, "area_m2"] = np.nan

    # Recalcula o preço por m² após o tratamento das áreas inválidas.
    analytics["preco_m2"] = np.where(
        (analytics["preco"] > 0) & (analytics["area_m2"] > 0),
        analytics["preco"] / analytics["area_m2"],
        np.nan,
    )

    return analytics


def analyze_neighborhoods(properties: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores de mercado por bairro."""
    return (
        properties.groupby("bairro", dropna=False)
        .agg(
            quantidade_imoveis=("bairro", "size"),
            preco_medio=("preco", "mean"),
            preco_mediano=("preco", "median"),
            area_media_m2=("area_m2", "mean"),
            preco_m2_medio=("preco_m2", "mean"),
            preco_m2_mediano=("preco_m2", "median"),
        )
        .reset_index()
        .sort_values("quantidade_imoveis", ascending=False)
    )


def analyze_market_by_type(properties: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores de mercado por tipo de imóvel."""
    total = len(properties)

    result = (
        properties.groupby("tipo_imovel", dropna=False)
        .agg(
            quantidade_imoveis=("tipo_imovel", "size"),
            preco_medio=("preco", "mean"),
            preco_mediano=("preco", "median"),
            area_media_m2=("area_m2", "mean"),
            preco_m2_medio=("preco_m2", "mean"),
        )
        .reset_index()
    )

    result["participacao_percentual"] = (
        result["quantidade_imoveis"] / total * 100
        if total
        else np.nan
    )

    return result.sort_values("quantidade_imoveis", ascending=False)


def analyze_market_by_neighborhood(
    properties: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula a concentração de imóveis por bairro."""
    total = len(properties)

    result = (
        properties.groupby("bairro", dropna=False)
        .size()
        .reset_index(name="quantidade_imoveis")
    )

    result["participacao_percentual"] = (
        result["quantidade_imoveis"] / total * 100
        if total
        else np.nan
    )

    return result.sort_values("quantidade_imoveis", ascending=False)


def create_data_quality_summary(
    properties: pd.DataFrame,
) -> pd.DataFrame:
    """Cria um resumo de qualidade das colunas."""
    total = len(properties)

    rows = []

    for column in properties.columns:
        rows.append(
            {
                "coluna": column,
                "tipo": str(properties[column].dtype),
                "total_registros": total,
                "valores_nulos": int(properties[column].isna().sum()),
                "percentual_nulos": (
                    properties[column].isna().mean() * 100
                    if total
                    else np.nan
                ),
                "valores_unicos": properties[column].nunique(dropna=True),
            }
        )

    return pd.DataFrame(rows)


def calculate_correlations(
    properties: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula correlações de Pearson entre variáveis numéricas."""
    columns = [
        "preco",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
        "preco_m2",
    ]

    numeric = properties[columns].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = numeric.corr(method="pearson")

    rows = []

    for index, variable_a in enumerate(columns):
        for variable_b in columns[index + 1 :]:
            correlation = correlation_matrix.loc[
                variable_a, variable_b
            ]

            rows.append(
                {
                    "variavel_a": variable_a,
                    "variavel_b": variable_b,
                    "correlacao": correlation,
                }
            )

    return pd.DataFrame(rows)


def save_analysis_outputs(
    properties: pd.DataFrame,
    analytics: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Gera e salva os cinco artefatos analíticos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "neighborhood_analysis.csv": analyze_neighborhoods(analytics),
        "market_by_type.csv": analyze_market_by_type(analytics),
        "market_by_neighborhood.csv": analyze_market_by_neighborhood(
            analytics
        ),
        "data_quality_summary.csv": create_data_quality_summary(properties),
        "correlations.csv": calculate_correlations(analytics),
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    """Executa a etapa de Analytics."""
    properties = load_properties()

    print(f"Registros carregados: {len(properties)}")

    analytics = prepare_analytics_data(properties)

    invalid_area_count = properties["area_m2"].lt(10).sum()

    print(
        "Áreas menores que 10 m² tratadas como inválidas para análise "
        f"de preço por m²: {invalid_area_count}"
    )

    save_analysis_outputs(properties, analytics)

    print("\nArtefatos analíticos gerados:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {path}")


if __name__ == "__main__":
    main()