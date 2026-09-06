# MedVision AI — Phase 0 image
# Python 3.11 pinned per DECISIONS.md D-02 (verified against torch/monai/torchxrayvision).
FROM python:3.11-slim

# No .pyc files; unbuffered stdout so logs stream in `docker compose up`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install deps first so this layer caches unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
