FROM python:3.9-slim-buster

WORKDIR /app

COPY docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY langchain ./langchain

EXPOSE 8000

ENV WEAVIATE_HOST=weaviate
ENV REDIS_HOST=redis

CMD ["uvicorn", "langchain.imageTextExtractor.main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
