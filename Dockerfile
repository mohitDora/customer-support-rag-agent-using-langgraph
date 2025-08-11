FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

RUN pip install -r requirements.txt

COPY . .

ENV LANGSMITH_TRACING="true"
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_PROJECT="customer-rag-agent"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
