FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV LANGSMITH_TRACING="true"
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_PROJECT="customer-rag-agent"
ENV PORT 8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
