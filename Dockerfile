FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV LANGSMITH_TRACING="true"
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_PROJECT="customer-rag-agent"
ENV PORT 8080

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
