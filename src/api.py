"""FastAPI application for container detection."""

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .ocr_client import OCRClient
from .extraction import extract_container_regex, extract_container_location
from .preprocessing import get_container_color_from_bytes, downscale_image
from .models import ContainerResult

app = FastAPI(title="Container Number Recognition AI")
ocr_client = OCRClient()


@app.post("/detect")
async def detect_container(file: UploadFile = File(...)):
    """
    Detect container number from uploaded image.

    Args:
        file: Image file

    Returns:
        JSON with detection results
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Read and process image
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = downscale_image(img)
        _, encoded = cv2.imencode('.JPG', img)
        processed_bytes = encoded.tobytes()

        # OCR
        ocr_result = ocr_client.recognize_printed_text(processed_bytes)

        if not ocr_result or not hasattr(ocr_result, 'regions') or not ocr_result.regions:
            return ContainerResult(error="No text detected").__dict__

        # Extract
        result = extract_container_regex(ocr_result)
        if not result.container_number or not result.container_type:
            location_result = extract_container_location(ocr_result)
            if not result.container_number:
                result.container_number = location_result.container_number
                result.bounding_box = location_result.bounding_box
            if not result.container_type:
                result.container_type = location_result.container_type

        # Color
        if result.bounding_box != [0, 0, 0, 0]:
            result.container_color = get_container_color_from_bytes(contents, result.bounding_box)

        return result.__dict__

    except Exception as e:
        return ContainerResult(error=str(e)).__dict__


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}