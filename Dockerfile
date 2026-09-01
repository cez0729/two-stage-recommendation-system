FROM python:3.12-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
    && pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "recsys.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
