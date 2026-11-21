from pathlib import Path

import pandas as pd

if __package__:
    from .load_data import load_properties
else:
    from load_data import load_properties


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"
MIN_VALID_PROPERTIES = 3

CLASSIFICATIONS = (
    "Oportunidade relativa",
    "Preço acessível por m²",
    "Ticket menor, preço/m² elevado",
    "Segmento premium",
    "Sem evidência suficiente",
)


def analyze_neighborhoods(df: pd.DataFrame) -> pd.DataFrame:
    total_properties = len(df)
    grouped = df.groupby("bairro", dropna=False, sort=False)

    analysis = grouped.size().rename("quantidade_imoveis").to_frame()
    analysis["preco_mediano"] = grouped["preco"].median()
    analysis["area_mediana"] = grouped["area_m2"].median()

    # Exigir preço e área positivos evita que anúncios inválidos distorçam o preço/m².
    valid_mask = (
        df["preco"].gt(0).fillna(False)
        & df["area_m2"].gt(0).fillna(False)
        & df["preco_m2"].notna()
    )
    valid_properties = df.loc[valid_mask]
    valid_grouped = valid_properties.groupby("bairro", dropna=False, sort=False)
    analysis["quantidade_imoveis_validos"] = valid_grouped.size()
    analysis["preco_m2_mediano"] = valid_grouped["preco_m2"].median()
    analysis["quantidade_imoveis_validos"] = (
        analysis["quantidade_imoveis_validos"].fillna(0).astype(int)
    )

    analysis["participacao_amostra"] = (
        analysis["quantidade_imoveis"] / total_properties * 100
    )
    return (
        analysis.reset_index()
        .sort_values("quantidade_imoveis", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _reference_medians(
    analysis: pd.DataFrame, min_properties: int
) -> tuple[float | None, float | None]:
    reference = analysis.loc[
        analysis["quantidade_imoveis_validos"].ge(min_properties)
        & analysis["preco_mediano"].notna()
        & analysis["preco_m2_mediano"].notna()
    ]
    if reference.empty:
        return None, None

    return (
        float(reference["preco_mediano"].median()),
        float(reference["preco_m2_mediano"].median()),
    )


def classify_opportunity(
    analysis: pd.DataFrame,
    min_properties: int = MIN_VALID_PROPERTIES,
) -> pd.DataFrame:
    classified = analysis.copy()
    classified["classificacao"] = "Sem evidência suficiente"

    price_reference, price_m2_reference = _reference_medians(
        classified, min_properties
    )
    if price_reference is None or price_m2_reference is None:
        return classified

    enough_data = (
        classified["quantidade_imoveis_validos"].ge(min_properties)
        & classified["preco_mediano"].notna()
        & classified["preco_m2_mediano"].notna()
    )
    price_below = classified["preco_mediano"].lt(price_reference)
    price_m2_below = classified["preco_m2_mediano"].lt(price_m2_reference)
    price_not_below = classified["preco_mediano"].ge(price_reference)
    price_m2_not_below = classified["preco_m2_mediano"].ge(price_m2_reference)

    # Empates com a referência ficam no lado "não abaixo", de forma conservadora.
    categories = (
        (
            enough_data & price_below & price_m2_below,
            "Oportunidade relativa",
        ),
        (
            enough_data & price_not_below & price_m2_below,
            "Preço acessível por m²",
        ),
        (
            enough_data & price_below & price_m2_not_below,
            "Ticket menor, preço/m² elevado",
        ),
        (
            enough_data & price_not_below & price_m2_not_below,
            "Segmento premium",
        ),
    )
    for mask, label in categories:
        classified.loc[mask, "classificacao"] = label

    return classified


def _format_currency(value: float | None) -> str:
    if pd.isna(value):
        return "N/D"
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"R$ {formatted}"


def main() -> None:
    properties = load_properties()
    analysis = analyze_neighborhoods(properties)
    analysis = classify_opportunity(analysis)
    price_reference, price_m2_reference = _reference_medians(
        analysis, MIN_VALID_PROPERTIES
    )

    print("\n=== ANÁLISE COMPLETA POR BAIRRO ===")
    print(analysis.to_string(index=False))

    print("\n=== MEDIANAS DE REFERÊNCIA ===")
    print(f"Preço mediano: {_format_currency(price_reference)}")
    print(f"Preço mediano/m²: {_format_currency(price_m2_reference)}")

    print("\n=== DISTRIBUIÇÃO DAS CLASSIFICAÇÕES ===")
    print(
        analysis["classificacao"]
        .value_counts()
        .reindex(CLASSIFICATIONS, fill_value=0)
        .to_string()
    )

    opportunities = analysis.loc[
        analysis["classificacao"] == "Oportunidade relativa"
    ]
    print('\n=== BAIRROS "OPORTUNIDADE RELATIVA" ===')
    print(
        opportunities.to_string(index=False)
        if not opportunities.empty
        else "Nenhum bairro classificado."
    )

    premium = analysis.loc[analysis["classificacao"] == "Segmento premium"]
    print('\n=== BAIRROS "SEGMENTO PREMIUM" ===')
    print(
        premium.to_string(index=False)
        if not premium.empty
        else "Nenhum bairro classificado."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "neighborhood_analysis.csv"
    analysis.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nAnálise salva em: {output_path}")


if __name__ == "__main__":
    main()
