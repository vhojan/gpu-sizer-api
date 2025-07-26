import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from sizing_logic import estimate_gpu_requirement

app = FastAPI()

# Enable CORS for local and hosted frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all during dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GPU and model catalogs once at startup
with open("data/gpu_catalog.json") as f:
    gpu_catalog = json.load(f)

with open("data/model_catalog.json") as f:
    model_catalog = json.load(f)


class ModelConfig(BaseModel):
    Model: str
    Size: str
    VRAM_Required_GB: int
    Base_Latency_s: float
    KV_Cache_Per_User_GB: float = 0  # Optional; default to 0


@app.get("/models")
def get_models():
    return model_catalog


@app.get("/gpus")
def get_gpus():
    return gpu_catalog


@app.get("/recommendation")
def get_recommendation(
    model_name: str = Query(..., alias="model"),
    users: int = Query(...),
    latency: int = Query(...)
):
    # Find model by name
    model_entry = next((m for m in model_catalog if m["Model"] == model_name), None)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found")

    # Map JSON keys to ModelConfig fields
    model_data = {
        "Model": model_entry["Model"],
        "Size": model_entry["Size"],
        "VRAM_Required_GB": model_entry["VRAM Required (GB)"],
        "Base_Latency_s": model_entry["Base Latency (s)"],
        "KV_Cache_Per_User_GB": model_entry.get("KV Cache per User (GB)", 0),
    }

    model = ModelConfig(**model_data)

    result = estimate_gpu_requirement(model, users, latency, gpu_catalog)

    return result