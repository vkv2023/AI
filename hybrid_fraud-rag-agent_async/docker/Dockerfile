FROM python:3.9-slim-buster

WORKDIR /app

COPY docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

ENV WEAVIATE_HOST=weaviate
ENV REDIS_HOST=redis

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
