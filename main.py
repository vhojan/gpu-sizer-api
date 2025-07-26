from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
from sizing_logic import estimate_gpu_requirement

# Load model and GPU catalog
with open("data/model_catalog.json") as f:
    model_catalog = json.load(f)

with open("data/gpu_catalog.json") as f:
    gpu_catalog = json.load(f)

# FastAPI app
app = FastAPI()


class Recommendation(BaseModel):
    gpu: str
    quantity: int
    gpu_memory: int


class RecommendationResponse(BaseModel):
    recommendation: Optional[Recommendation]
    alternatives: List[Recommendation]


@app.get("/recommendation", response_model=RecommendationResponse)
def get_recommendation(model: str, users: int, latency: int):
    try:
        selected_model = next((m for m in model_catalog if m["Model"] == model), None)
        if not selected_model:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        result = estimate_gpu_requirement(selected_model, users, latency, gpu_catalog)
        return result
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
def get_models():
    return [m["Model"] for m in model_catalog]


@app.get("/gpus")
def get_gpus():
    return [g["Name"] for g in gpu_catalog]