"""
Harness EDA Advisor — FastAPI Backend
AI-Assisted Data Discovery and EDA Readiness Platform
"""
import os
import uuid
import json
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.analyser import run_analysis

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 600 * 1024 * 1024  # 600 MB hard ceiling

app = FastAPI(title="Harness EDA Advisor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main frontend HTML."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "Harness EDA Advisor API — frontend not found"})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a CSV/Excel/Parquet upload and persist it to disk.
    Returns a file_id the client uses to trigger analysis.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls", ".parquet"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use CSV, Excel, or Parquet.")

    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}{ext}"

    total = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds 600 MB limit.")
            out.write(chunk)

    logger.info(f"Uploaded {file.filename} → {dest} ({total/1024/1024:.1f} MB)")
    return {"file_id": file_id, "filename": file.filename, "size_bytes": total, "ext": ext}


class AnalyseRequest(BaseModel):
    file_id: str


@app.post("/api/analyse")
async def analyse(req: AnalyseRequest):
    """
    Run the full Harness EDA analysis pipeline on a previously uploaded file.
    Returns all 10-page data payloads as a single JSON object.
    """
    # Locate the file
    matches = list(UPLOAD_DIR.glob(f"{req.file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found. Upload first.")

    file_path = matches[0]
    logger.info(f"Analysing {file_path}")

    try:
        result = run_analysis(file_path)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "harness-eda-advisor"}
