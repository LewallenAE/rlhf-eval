FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer cache)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy rest of project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rlhf_eval.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
