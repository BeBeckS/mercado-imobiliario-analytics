FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN mkdir -p data && ln -s /data/analytics data/outputs

ENV CURATED_DATA_PATH=/data/curated/properties.parquet

CMD ["sh", "-c", "mkdir -p /data/analytics && exec python src/export_analysis.py"]
