FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/output

ENTRYPOINT ["python", "run_pipeline.py"]