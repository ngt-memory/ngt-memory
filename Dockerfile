FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Зависимости Python (сначала — кешируются отдельным слоем)
# torch CPU-only — CUDA не нужен, всё считает OpenAI API
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Исходный код
COPY ngt/ ./ngt/
COPY api/ ./api/

# Не копируем: .env, results/, experiments/, tests/, .venv/
# API ключ передаётся через переменные окружения

EXPOSE 9190

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9190", "--workers", "1"]
