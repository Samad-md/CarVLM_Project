import os
import asyncio
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import CarVLMPredictor


BASE = Path(__file__).resolve().parent

MODEL_DIR = Path(
    os.getenv("CARVLM_MODEL_DIR", BASE / "model")
)

SPECS_PATH = Path(
    os.getenv("CARVLM_SPECS_PATH", BASE / "vehicle_specs.json")
)


app = FastAPI(
    title="CarVLM API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def predictor():
    print("Loading CarVLM model...")
    model = CarVLMPredictor(
        MODEL_DIR,
        SPECS_PATH
    )
    print("Model loaded successfully")
    return model


@app.get("/")
def root():
    return {
        "name": "CarVLM API",
        "docs": "/docs",
        "predict": "POST /predict"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "CarVLM API"
    }


@app.get("/classes")
def classes():
    try:
        p = predictor()

        return {
            "count": len(p.labels),
            "classes": p.labels
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty file."
        )


    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 15 MB."
        )


    try:

        result = await asyncio.to_thread(
            predictor().predict,
            data
        )

        result["filename"] = file.filename

        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )