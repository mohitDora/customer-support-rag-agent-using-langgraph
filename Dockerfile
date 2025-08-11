FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
COPY setup.py /app/
COPY pyproject.toml /app/

RUN pip install -r requirements.txt

COPY . /app/

ENV LANGSMITH_TRACING="true"
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_PROJECT="customer-rag-agent"

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]