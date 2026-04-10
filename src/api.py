"""FastAPI application for container detection."""

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException

from .combined_pipeline import run_combined_extraction_from_bytes

log = logging.getLogger(__name__)

app = FastAPI(title="Container Number Recognition AI")


@app.post("/detect")
async def detect_container(file: UploadFile = File(...)):
    """Detect container number from uploaded image using combined OCR + Llama Extract."""
    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Accepted: .jpg, .jpeg, .png, .bmp")

    contents = await file.read()
    
    result, method = run_combined_extraction_from_bytes(contents, filename=file.filename)
    log.info("Combined result for %s: container=%s type=%s method=%s", file.filename, result.container_number or "none", result.container_type or "none", method)
    return result.to_dict()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}