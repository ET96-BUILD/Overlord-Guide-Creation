FROM python:3.12-slim

# ffmpeg is required for video frame extraction
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code
COPY sopgen ./sopgen

# Run as non-root for safety
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Cloud Run injects PORT=8080; default for local runs
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "hypercorn sopgen.api.main:app --bind 0.0.0.0:${PORT}"]
