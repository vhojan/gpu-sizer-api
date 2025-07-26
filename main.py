from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import os
from sizing_logic import estimate_gpu_requirement

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and GPU catalogs
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

@app.get("/models")
def get_models():
    return model_catalog

@app.get("/gpus")
def get_gpus():
    return gpu_catalog

@app.get("/recommendation")
def get_recommendation(model: str, users: int, latency: float):
    try:
        matching_model = next((m for m in model_catalog if m["Model"] == model), None)
        if not matching_model:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")

        result = estimate_gpu_requirement(matching_model, users, latency, gpu_catalog)
        return {
            "model": model,
            "users": users,
            "latency_ms": latency,
            "recommendation": result[0],
            "alternatives": result[1]
        }
    except Exception as e:
        print(f"[ERROR] in /recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during recommendation: {str(e)}")