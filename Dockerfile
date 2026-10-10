# Serving image. Build after running the pipeline so artifacts/ and data/processed/ exist:
#   docker build -t pathrank .   &&   docker run -p 8080:8080 pathrank
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080 PATHRANK_CONFIG=configs/careerbuilder.yaml
# tracing: set EVALWRIGHT_EXPORT=otlp:http://<collector>:4318/v1/traces (or http:<url>, jsonl:<path>) to enable
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
COPY sdk/python sdk/python
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serve.txt
COPY pathrank/ pathrank/
COPY configs/ configs/
COPY data/processed/ data/processed/
COPY artifacts/ artifacts/
EXPOSE 8080
CMD ["sh", "-c", "uvicorn pathrank.serve.app:app --host 0.0.0.0 --port ${PORT}"]
