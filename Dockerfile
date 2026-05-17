FROM python:3.11-slim

# PLAN.md Phase A Step 5 (A-05): non-root user, no repo volume mount
RUN groupadd -r inids && useradd -r -g inids inids

WORKDIR /app

# Copy application source — NOT the full repo (no ../../:/app volume mount)
COPY src/ ./src/
COPY web_app/ ./web_app/
COPY rules/ ./rules/
COPY requirements.txt requirements.in ./

# A-07: --require-hashes enforces the pinned SHA-256 manifest; --no-deps prevents
# the resolver from silently pulling in unverified transitive packages.
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.txt

# Create data and model directories owned by non-root user
RUN mkdir -p /data /models && chown -R inids:inids /data /models

USER inids

# Python-native health check — no curl required in the image
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=4)"

EXPOSE 5000

# gunicorn with eventlet worker — required for Flask-SocketIO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "web_app.app:app"]
