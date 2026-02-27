FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides $PORT; we use a shell form to expand it.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"
