FROM python:3.13-slim

# curl is used by the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies using uv (FAST ⚡)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code (secrets come from env_file at runtime, never the image)
COPY app/            ./app/
COPY charts/         ./charts/
COPY mlflow_track/   ./mlflow_track/
COPY scripts/        ./scripts/

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
