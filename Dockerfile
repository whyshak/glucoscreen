FROM python:3.11.15-slim

WORKDIR /app

# build-essential is needed because some packages (faiss-cpu, scipy, sentence-transformers deps)
# may need to compile from source depending on the platform's available wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the PORT env var (defaults to 8080) - bind gunicorn to it
ENV PORT=8080
EXPOSE 8080

# 1 worker, 4 threads keeps memory down (matches your single-instance/low-traffic setup)
# timeout 120s gives room for the first-request embedding model load
CMD exec gunicorn "app:create_app()" \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 120
