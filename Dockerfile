# Single-bundle HTTP server: `axiom serve` (no .axb baked into the image).
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    AXIOM_REQUIRE_API_KEY=1

COPY pyproject.toml readme.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[serve,lock]"

EXPOSE 8000

CMD ["axiom", "serve"]
