from pathlib import Path

import pandas as pd

if __package__:
    from .data_quality import summarize_data_quality
    from .investment_analysis import analyze_neighborhoods, classify_opportunity
    from .load_data import load_properties
    from .market_analysis import (
        analyze_area_price_relationship,
        analyze_price_by_neighborhood,
        analyze_price_by_property_type,
        analyze_price_per_m2_by_property_type,
        analyze_properties_by_neighborhood,
    )
else:
    from data_quality import summarize_data_quality
    from investment_analysis import analyze_neighborhoods, classify_opportunity
    from load_data import load_properties
    from market_analysis import (
        analyze_area_price_relationship,
        analyze_price_by_neighborhood,
        analyze_price_by_property_type,
        analyze_price_per_m2_by_property_type,
        analyze_properties_by_neighborhood,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


def _build_neighborhood_analysis(properties: pd.DataFrame) -> pd.DataFrame:
    neighborhoods = analyze_neighborhoods(properties)
    return classify_opportunity(neighborhoods)


def _build_market_by_type(properties: pd.DataFrame) -> pd.DataFrame:
    by_type = analyze_price_by_property_type(properties).rename(
        columns={"quantidade": "quantidade_imoveis"}
    )
    price_per_m2 = analyze_price_per_m2_by_property_type(properties).rename(
        columns={"quantidade": "quantidade_imoveis_validos_preco_m2"}
    )

    result = by_type.join(price_per_m2, how="left")
    valid_count_column = "quantidade_imoveis_validos_preco_m2"
    result[valid_count_column] = result[valid_count_column].fillna(0).astype(int)
    return result.reset_index()


def _build_market_by_neighborhood(properties: pd.DataFrame) -> pd.DataFrame:
    by_count = analyze_properties_by_neighborhood(properties).rename(
        columns={
            "quantidade": "quantidade_imoveis",
            "percentual_do_total": "participacao_amostra",
        }
    )
    by_price = analyze_price_by_neighborhood(properties).drop(
        columns="quantidade"
    )
    return by_count.join(by_price, how="left").reset_index()


def _build_data_quality_summary(properties: pd.DataFrame) -> pd.DataFrame:
    return summarize_data_quality(properties).to_frame().T.reset_index(drop=True)


def _build_correlations(properties: pd.DataFrame) -> pd.DataFrame:
    relationship = analyze_area_price_relationship(properties)
    return pd.DataFrame(
        [
            {
                "variavel_a": "area_m2",
                "variavel_b": "preco",
                "correlacao": relationship["correlacao_pearson"],
                "metodo": "Pearson",
                "registros_utilizados": relationship["quantidade_registros"],
            }
        ]
    )


def export_analysis() -> None:
    properties = load_properties()
    artifacts = {
        "neighborhood_analysis.csv": _build_neighborhood_analysis(properties),
        "market_by_type.csv": _build_market_by_type(properties),
        "market_by_neighborhood.csv": _build_market_by_neighborhood(properties),
        "data_quality_summary.csv": _build_data_quality_summary(properties),
        "correlations.csv": _build_correlations(properties),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, result in artifacts.items():
        output_path = OUTPUT_DIR / filename
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        relative_path = output_path.relative_to(PROJECT_ROOT)
        print(
            f"Gerado: {relative_path} "
            f"({result.shape[0]} registros, {result.shape[1]} colunas)"
        )


if __name__ == "__main__":
    export_analysis()
