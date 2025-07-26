import json
import math
from typing import Optional, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sizing_logic import estimate_gpu_requirement

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Recommendation(BaseModel):
    gpu: str
    quantity: int
    gpu_memory: int


class RecommendationResponse(BaseModel):
    recommendation: Optional[Recommendation]
    alternatives: List[Recommendation]


# Load model and GPU catalogs
with open("data/model_catalog.json") as f:
    model_catalog = json.load(f)

with open("data/gpu_catalog.json") as f:
    gpu_catalog = json.load(f)


@app.get("/recommendation", response_model=RecommendationResponse)
def get_recommendation(
    model: str = Query(...), users: int = Query(...), latency: int = Query(...)
):
    # Find the model data
    model_data = next((m for m in model_catalog if m["Model"] == model), None)
    if not model_data:
        return {"recommendation": None, "alternatives": []}

    result = estimate_gpu_requirement(model_data, users, latency, gpu_catalog)
    return result