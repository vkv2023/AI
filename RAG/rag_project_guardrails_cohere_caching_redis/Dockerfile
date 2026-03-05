# =========================
# Base Python Image
# =========================
FROM python:3.11-slim

# =========================
# Set Working Directory
# =========================
WORKDIR /app

# =========================
# Install System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Copy Requirements
# =========================
COPY requirements.txt .

# =========================
# Install Python Libraries
# =========================
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy Application Code
# =========================
COPY . .

# =========================
# Environment Variables
# =========================
ENV PYTHONUNBUFFERED=1

# =========================
# Run Application
# =========================
CMD ["python", "rag_project_guardrails_cohere_caching_redis.py"]