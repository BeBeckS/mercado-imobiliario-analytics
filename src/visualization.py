from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_properties


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"


def save_figure(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_price_distribution(df: pd.DataFrame) -> None:
    prices = df["preco"].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(prices, bins=30)

    plt.title("Distribuição dos preços dos imóveis")
    plt.xlabel("Preço (R$)")
    plt.ylabel("Quantidade de imóveis")

    save_figure("price_distribution.png")


def plot_properties_by_type(df: pd.DataFrame) -> None:
    counts = (
        df["tipo_imovel"]
        .value_counts()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    counts.plot(kind="barh")

    plt.title("Quantidade de imóveis por tipo")
    plt.xlabel("Quantidade de imóveis")
    plt.ylabel("Tipo de imóvel")

    save_figure("properties_by_type.png")


def plot_median_price_by_type(df: pd.DataFrame) -> None:
    median_prices = (
        df.groupby("tipo_imovel")["preco"]
        .median()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    median_prices.plot(kind="barh")

    plt.title("Preço mediano por tipo de imóvel")
    plt.xlabel("Preço mediano (R$)")
    plt.ylabel("Tipo de imóvel")

    save_figure("median_price_by_type.png")


def plot_median_price_m2_by_type(df: pd.DataFrame) -> None:
    median_price_m2 = (
        df.dropna(subset=["preco_m2"])
        .groupby("tipo_imovel")["preco_m2"]
        .median()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    median_price_m2.plot(kind="barh")

    plt.title("Preço mediano por m² por tipo de imóvel")
    plt.xlabel("Preço mediano por m² (R$)")
    plt.ylabel("Tipo de imóvel")

    save_figure("median_price_m2_by_type.png")


def plot_properties_by_neighborhood(df: pd.DataFrame) -> None:
    top_neighborhoods = (
        df["bairro"]
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    top_neighborhoods.plot(kind="barh")

    plt.title("Top 10 bairros por quantidade de imóveis")
    plt.xlabel("Quantidade de imóveis")
    plt.ylabel("Bairro")

    save_figure("properties_by_neighborhood.png")


def plot_neighborhood_concentration(df: pd.DataFrame) -> None:
    neighborhood_counts = df["bairro"].value_counts()

    cumulative_share = (
        neighborhood_counts.cumsum()
        / neighborhood_counts.sum()
        * 100
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_share) + 1),
        cumulative_share,
        marker="o",
    )

    plt.axhline(80, linestyle="--", label="80% dos imóveis")

    plt.title("Concentração acumulada dos imóveis por bairro")
    plt.xlabel("Quantidade de bairros")
    plt.ylabel("Percentual acumulado (%)")
    plt.legend()

    save_figure("neighborhood_concentration.png")


def plot_area_vs_price(df: pd.DataFrame) -> None:
    valid = df[
        (df["area_m2"] > 0)
        & (df["preco"] > 0)
    ].copy()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        valid["area_m2"],
        valid["preco"],
        alpha=0.6,
    )

    plt.title("Relação entre área e preço dos imóveis")
    plt.xlabel("Área (m²)")
    plt.ylabel("Preço (R$)")

    save_figure("area_vs_price.png")


def main() -> None:
    df = load_properties()

    print(f"Gerando visualizações para {len(df)} registros...")

    plot_price_distribution(df)
    plot_properties_by_type(df)
    plot_median_price_by_type(df)
    plot_median_price_m2_by_type(df)
    plot_properties_by_neighborhood(df)
    plot_neighborhood_concentration(df)
    plot_area_vs_price(df)

    print(f"Visualizações salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()