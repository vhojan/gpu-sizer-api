from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from sizing_logic import estimate_gpu_requirement

app = FastAPI()

# Load the catalogs at startup
with open("data/model_catalog.json") as f:
    model_catalog = json.load(f)

with open("data/gpu_catalog.json") as f:
    gpu_catalog = json.load(f)

class Recommendation(BaseModel):
    gpu: str
    quantity: int
    gpu_memory: int

class RecommendationResponse(BaseModel):
    recommendation: Optional[Recommendation]
    alternatives: List[Recommendation]

@app.get("/recommendation", response_model=RecommendationResponse)
def get_recommendation(
    model: str = Query(...),
    users: int = Query(..., gt=0),
    latency: int = Query(..., gt=0)
):
    matching_model = next((m for m in model_catalog if m["Model"] == model), None)
    if not matching_model:
        return {"recommendation": None, "alternatives": []}

    result = estimate_gpu_requirement(matching_model, users, latency, gpu_catalog)
    return result

@app.get("/models", response_model=List[str])
def list_models():
    return [m["Model"] for m in model_catalog]

@app.get("/gpus", response_model=List[str])
def list_gpus():
    return [g["Name"] for g in gpu_catalog]