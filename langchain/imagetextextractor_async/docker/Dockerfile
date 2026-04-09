FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
COPY app/ /app/app/
COPY src/ /app/src/
COPY data/ /app/data/
COPY logging_config.yaml /app/

ENV PYTHONPATH=/app
ENV LOG_DIR=/app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]