FROM python:3.11-slim-buster

WORKDIR /app

# Install dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory (CORRECT PATH)
RUN mkdir -p /app/logs

# Copy application files
COPY ImageTextExtractor.py .
COPY main_fastapi.py .
COPY logging_config.yaml .
COPY __init__.py .

# Copy data
COPY data/ /app/data/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]