# Mercado Imobiliário Analytics

Este repositório representa a etapa de Analytics do projeto Mercado Imobiliário
Regional. Ele consumirá dados curados em formato Parquet produzidos pelo
repositório `mercado-imobiliario-batch`.

## Artefatos de análise

O Analytics consolida e exporta cinco arquivos CSV em `data/outputs/` para
consumo posterior pelo repositório de Storytelling:

- `neighborhood_analysis.csv`
- `market_by_type.csv`
- `market_by_neighborhood.csv`
- `data_quality_summary.csv`
- `correlations.csv`
