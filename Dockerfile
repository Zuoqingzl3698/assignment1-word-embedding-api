FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

RUN uv pip install --system "fastapi[standard]" spacy

RUN python -m spacy download en_core_web_md

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
