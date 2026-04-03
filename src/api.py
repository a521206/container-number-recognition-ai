"""FastAPI application for container detection."""

import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from .ocr_client import OCRClient
from .extraction import ContainerResult, run_extraction_pipeline
from .preprocessing import downscale_image

log = logging.getLogger(__name__)

app = FastAPI(title="Container Number Recognition AI")
ocr_client = OCRClient()


@app.post("/detect")
async def detect_container(file: UploadFile = File(...)):
    """Detect container number from uploaded image."""
    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Accepted: .jpg, .jpeg, .png, .bmp")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image – unsupported format or corrupted data")

    img = downscale_image(img)
    _, encoded = cv2.imencode('.JPG', img)
    img_bytes = encoded.tobytes()

    ocr_result = ocr_client.recognize_printed_text(img_bytes)
    if not ocr_result.regions:
        log.info("No text detected in %s", file.filename)
        return ContainerResult(error="No text detected").to_dict()

    result = run_extraction_pipeline(ocr_result, img_bytes)
    log.info("Detection result for %s: container=%s type=%s", file.filename, result.container_number or "none", result.container_type or "none")
    return result.to_dict()


@app.post("/detect/llama-extract")
async def detect_container_llama_extract(file: UploadFile = File(...)):
    """Detect container number from uploaded image using Llama Extract."""
    from .llama_extract_client import LlamaExtractClient

    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Accepted: .jpg, .jpeg, .png, .bmp")

    contents = await file.read()

    try:
        client = LlamaExtractClient()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = client.extract_from_bytes(contents, filename=file.filename)
    log.info(
        "Llama Extract result for %s: container=%s type=%s",
        file.filename,
        result.container_number or "none",
        result.container_type or "none",
    )
    return result.to_dict()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
