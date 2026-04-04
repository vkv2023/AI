FROM python:3.11-slim-buster

WORKDIR /app

# 1. Copy requirements
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Create logs directory
RUN mkdir -p ../logs

# 3. Copy Python application files directly (they are in the build context root)
COPY ImageTextExtractor.py .
COPY main_fastapi.py .
COPY logging_config.yaml .
COPY __init__.py .

# 4. Copy the data folder (Required for PDF ingestion)
COPY data/ /app/data/

EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Run FastAPI directly
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
