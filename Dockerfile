# ============================================================
# Dockerfile — Credit Risk Prediction API
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Tejas Subhash Patil"
LABEL description="Credit Risk ML Inference API"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY api /app/api
COPY monitoring /app/monitoring
COPY models /app/models

ENV MODEL_PATH=/app/models/credit_model.pkl
ENV DECISION_THRESHOLD=0.167
ENV PYTHONPATH=/app/src:/app/monitoring

RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]