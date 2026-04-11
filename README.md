# Container Number Recognition AI

OCR-based container number detection using Azure Computer Vision API or Llama Extract. Validates ISO 6346 container numbers with check-digit verification and supports both CLI and API modes.

## Setup

```bash
pip install -r requirements.txt
cp .env-sample .env
# Edit .env with your VISION_ENDPOINT and VISION_KEY
```

## Usage

```bash
# CLI – process images in ./data/
python main.py

# API – FastAPI server on port 8000
python main.py api
```

## API

- `POST /detect` — upload image, returns container number, type, bounding box, color
- `GET /health` — health check
