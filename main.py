from model_service import router as model_router, create_db
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import os
from sizing_logic import estimate_gpu_requirement

app = FastAPI()

create_db()  # initialize model DB table

app.include_router(model_router)

# Allow CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and GPU catalogs from JSON files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CATALOG_PATH = os.path.join(BASE_DIR, "data", "model_catalog.json")
GPU_CATALOG_PATH = os.path.join(BASE_DIR, "data", "gpu_catalog.json")

with open(MODEL_CATALOG_PATH, "r") as f:
    model_catalog = json.load(f)

with open(GPU_CATALOG_PATH, "r") as f:
    gpu_catalog = json.load(f)

@app.get("/")
def read_root():
    return {"message": "Welcome to the GPU Sizer API"}

@app.get("/models", response_model=List[dict])
def get_models():
    return model_catalog

@app.get("/gpus", response_model=List[dict])
def get_gpus():
    return gpu_catalog

@app.get("/recommendation")
def get_recommendation(model: str, users: int, latency: float):
    try:
        matching_model = next((m for m in model_catalog if m["Model"] == model), None)
        if not matching_model:
            raise HTTPException(status_code=404, detail="Model not found")

        result = estimate_gpu_requirement(matching_model, users, latency, gpu_catalog)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recommendation: {str(e)}")