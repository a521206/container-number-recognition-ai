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

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point. Run `python main.py` for CLI mode or `python main.py api` to start the FastAPI server. |
| `src/api.py` | FastAPI app with `/detect` and `/health` endpoints. Accepts image uploads and returns container detection results. |
| `src/cli.py` | CLI interface. Scans `./data/` for images and prints detection results for each file. |
| `src/config.py` | Configuration constants and environment variable loading. Loads carrier prefixes from `container_prefix.txt` and builds OCR character confusion maps for fuzzy matching. |
| `src/ocr_client.py` | Azure Computer Vision OCR client wrapper. Sends images to Azure Read API and polls for results. |
| `src/extraction.py` | Core extraction logic. Regex-based and location-based container number detection with fuzzy prefix matching and ISO 6346 validation. |
| `src/geometry.py` | Bounding-box and spatial utilities for OCR word processing (bounding box parsing, orientation checks, coordinate distance). |
| `src/preprocessing.py` | Image utilities. Downscaling large images and extracting dominant container color from cropped regions. |
| `src/validation.py` | ISO 6346 check-digit validation for container numbers. |
| `src/__init__.py` | Package init file. |
| `tests/test_validation.py` | Pytest tests for ISO 6346 check-digit validation. |
| `container_prefix.txt` | Comma-separated list of valid 4-letter carrier prefixes (e.g. CSQU, TEMU). |
| `requirements.txt` | Python dependencies. |
| `.env-sample` | Template for environment variables (`VISION_ENDPOINT`, `VISION_KEY`). |
