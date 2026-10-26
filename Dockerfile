FROM python:3.13-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies using uv (FAST ⚡)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY app/            ./app/
COPY charts/         ./charts/
COPY mlflow_track/   ./mlflow_track/
COPY scripts/        ./scripts/
COPY .env .env
# Make scripts importable
RUN touch app/__init__.py charts/__init__.py mlflow_track/__init__.py scripts/__init__.py

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]