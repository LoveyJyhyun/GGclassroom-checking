FROM python:3.12-slim

# Docker CLI only (no daemon): used to start sandbox containers for student code
# through the host's Docker socket.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    REPORTS_DIR=/data/reports \
    ANSWER_KEYS_DIR=/data/answer_keys \
    GOOGLE_TOKEN_FILE=/data/token.json \
    GOOGLE_CLIENT_SECRETS=/secrets/client_secret.json

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt gunicorn==26.2.0

COPY app.py .
COPY checker ./checker
COPY templates ./templates
COPY static ./static
COPY examples ./examples

EXPOSE 8080
# One worker so sessions and in-memory state stay consistent; threads for concurrency.
# Long timeout because a full check with a local LLM can take many minutes.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 3600 app:app"]
