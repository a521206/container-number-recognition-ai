"""FastAPI application for container detection."""

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from .ocr_client import OCRClient
from .extraction import ContainerResult, run_extraction_pipeline
from .preprocessing import downscale_image

app = FastAPI(title="Container Number Recognition AI")
ocr_client = OCRClient()


@app.post("/detect")
async def detect_container(file: UploadFile = File(...)):
    """Detect container number from uploaded image."""
    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = downscale_image(img)
        _, encoded = cv2.imencode('.JPG', img)
        img_bytes = encoded.tobytes()

        ocr_result = ocr_client.recognize_printed_text(img_bytes)
        if not ocr_result.regions:
            return ContainerResult(error="No text detected").__dict__

        return run_extraction_pipeline(ocr_result, img_bytes).__dict__

    except Exception as e:
        return ContainerResult(error=str(e)).__dict__


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}