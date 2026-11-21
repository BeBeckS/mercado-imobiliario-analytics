import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import export_analysis


def test_export_analysis_creates_expected_artifacts_without_changing_parquet(
    tmp_path, monkeypatch
):
    properties = pd.DataFrame(
        {
            "id_anuncio": [1, 2, 3, 4, 5, 6],
            "tipo_imovel": [
                "Apartamento",
                "Apartamento",
                "Casa",
                "Casa",
                "Casa",
                "Apartamento",
            ],
            "bairro": ["Centro", "Centro", "Centro", "Bairro A", "Bairro A", "Bairro A"],
            "preco": [100000, 120000, 140000, 200000, 220000, 240000],
            "area_m2": [50, 60, 70, 80, 90, 100],
            "preco_m2": [2000, 2000, 2000, 2500, 2444, 2400],
        }
    )
    input_path = tmp_path / "properties.parquet"
    properties.to_parquet(input_path, index=False)
    input_hash_before = hashlib.sha256(input_path.read_bytes()).digest()
    loaded_properties = pd.read_parquet(input_path)
    loaded_properties_before = loaded_properties.copy(deep=True)

    output_dir = tmp_path / "data" / "outputs"
    monkeypatch.setattr(export_analysis, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export_analysis, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        export_analysis, "load_properties", lambda: loaded_properties
    )

    export_analysis.export_analysis()

    expected_files = {
        "neighborhood_analysis.csv",
        "market_by_type.csv",
        "market_by_neighborhood.csv",
        "data_quality_summary.csv",
        "correlations.csv",
    }
    generated_files = {path.name for path in output_dir.glob("*.csv")}
    assert generated_files == expected_files

    artifacts = {
        path.name: pd.read_csv(path, encoding="utf-8-sig")
        for path in output_dir.glob("*.csv")
    }
    assert all(not artifact.empty for artifact in artifacts.values())

    assert {
        "bairro",
        "quantidade_imoveis",
        "preco_mediano",
        "area_mediana",
        "quantidade_imoveis_validos",
        "preco_m2_mediano",
        "participacao_amostra",
        "classificacao",
    }.issubset(artifacts["neighborhood_analysis.csv"].columns)
    assert {
        "tipo_imovel",
        "quantidade_imoveis",
        "preco_medio",
        "preco_mediano",
        "preco_minimo",
        "preco_maximo",
        "quantidade_imoveis_validos_preco_m2",
        "preco_m2_medio",
        "preco_m2_mediano",
        "preco_m2_maximo",
    }.issubset(artifacts["market_by_type.csv"].columns)
    assert {
        "registros_totais",
        "ids_duplicados",
        "precos_invalidos",
        "areas_invalidas",
        "preco_m2_invalido",
        "valores_ausentes",
        "outliers_preco",
        "outliers_area_m2",
        "outliers_preco_m2",
    }.issubset(artifacts["data_quality_summary.csv"].columns)
    assert {
        "bairro",
        "quantidade_imoveis",
        "participacao_amostra",
        "preco_medio",
        "preco_mediano",
        "preco_m2_mediano",
    }.issubset(artifacts["market_by_neighborhood.csv"].columns)

    correlations = artifacts["correlations.csv"]
    area_price = correlations.loc[
        (correlations["variavel_a"] == "area_m2")
        & (correlations["variavel_b"] == "preco")
    ]
    assert len(area_price) == 1
    assert area_price.iloc[0]["metodo"] == "Pearson"
    assert pd.notna(area_price.iloc[0]["correlacao"])

    assert hashlib.sha256(input_path.read_bytes()).digest() == input_hash_before
    pd.testing.assert_frame_equal(loaded_properties, loaded_properties_before)
