import os
import asyncio
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


# Global model holder
MODEL = None



@app.on_event("startup")
def startup_event():

    global MODEL

    print("================================")
    print("Loading CarVLM model...")
    print("================================")

    MODEL = CarVLMPredictor(
        MODEL_DIR,
        SPECS_PATH
    )

    print("================================")
    print("CarVLM model loaded successfully")
    print("================================")



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
        "service": "CarVLM API",
        "model_loaded": MODEL is not None
    }



@app.get("/classes")
def classes():

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model loading"
        )

    return {
        "count": len(MODEL.labels),
        "classes": MODEL.labels
    }



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )


    data = await file.read()


    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty file"
        )


    if len(data) > 15 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 15MB"
        )


    try:

        result = await asyncio.to_thread(
            MODEL.predict,
            data
        )


        result["filename"] = file.filename


        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )