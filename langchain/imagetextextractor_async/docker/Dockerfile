FROM python:3.11-slim

# 1. Install system dependencies (Commonly needed for OCR/Image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy requirements first to leverage Docker cache
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the application
# This will copy 'app/', 'src/', 'data/', etc., into '/app/'
COPY . .
COPY logging_config.yaml /app/logging_config.yaml
# 4. Set Environment Variables
# PYTHONPATH=/app ensures that 'app/main.py' can 'import src.rag...'


# Ensure the logs directory exists
ENV PYTHONPATH=/app
ENV LOG_DIR=/app/logs
ENV LOG_CONFIG_PATH=/app/logging_config.yaml
ENV PYTHONUNBUFFERED=1

# 5. Ensure the logs directory exists so the app doesn't crash on write
RUN mkdir -p /app/logs

EXPOSE 8000

# 6. Start the application
# 'app.main:app' refers to: folder 'app' -> file 'main.py' -> variable 'app'
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/logging_config.yaml"]