# Container Number Recognition AI

OCR-based container number detection using Azure Computer Vision API.

## Setup

```bash
pip install -r requirements.txt
```

Rename `.env-sample` to `.env` and fill in `VISION_ENDPOINT` and `VISION_KEY`.

## Usage

```bash
# CLI mode – process images in ./data/
python main.py

# API mode – start FastAPI server on port 8000
python main.py api
```

## API

`POST /detect` — upload an image, returns container number, type, bounding box, and color.

`GET /health` — health check.
