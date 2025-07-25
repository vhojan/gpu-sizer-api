from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import math
import json

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://gpu-sizer-ui.onrender.com",  # if deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model matching JSON fields from frontend
class ModelInput(BaseModel):
    Model: str
    Size: str
    VRAM_Required_GB: float = Field(..., alias="VRAM Required (GB)")
    Base_Latency_s: float = Field(..., alias="Base Latency (s)")
    Parameters: str
    Weights_Size_FP16_GB: float = Field(..., alias="Weights Size (FP16, GB)")
    Architecture: str
    Intended_Use: str = Field(..., alias="Intended Use")

    class Config:
        allow_population_by_field_name = True  # allow internal names in backend

# Wrapper request model
class RecommendationRequest(BaseModel):
    model: ModelInput
    users: int
    latency: int

# Load data once
with open("data/model_catalog.json") as f:
    model_catalog = json.load(f)

with open("data/gpu_catalog.json") as f:
    gpu_catalog = json.load(f)

# Core logic
def estimate_gpu_requirement(model: ModelInput, users: int, latency: int):
    base_latency = model.Base_Latency_s
    required_latency = latency / 1000  # convert ms to seconds

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    # Estimate concurrency
    parallelism = max(1, math.floor(required_latency / base_latency))
    concurrent_per_gpu = parallelism
    total_concurrency_required = users
    total_inferences_needed = math.ceil(total_concurrency_required / concurrent_per_gpu)

    # STEP 1: Prefer a single GPU solution
    single_gpu_candidates = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= model.VRAM_Required_GB
    ]
    if not single_gpu_candidates:
        raise HTTPException(status_code=400, detail="No single GPU has enough memory to run this model.")

    # Choose the one with the smallest VRAM that satisfies the requirement
    best_single_gpu = sorted(single_gpu_candidates, key=lambda g: g["VRAM (GB)"])[0]

    # STEP 2: Optional NVLink-enabled multi-GPU alternatives
    nvlink_gpus = [
        gpu for gpu in gpu_catalog
        if gpu["NVLink"] is True and
           gpu["VRAM (GB)"] * 2 >= model.VRAM_Required_GB and
           int(gpu.get("Max NVLink GPUs", 2)) >= 2
    ]
    multi_gpu_alternatives = [
        {
            "gpu": gpu["GPU Type"],
            "quantity": 2,
            "gpu_memory": gpu["VRAM (GB)"],
            "note": "Requires NVLink"
        }
        for gpu in nvlink_gpus
    ]

    return {
        "recommendation": {
            "gpu": best_single_gpu["GPU Type"],
            "quantity": 1,
            "gpu_memory": model.VRAM_Required_GB
        },
        "alternatives": multi_gpu_alternatives
    }

@app.get("/models")
def get_models():
    return model_catalog

@app.post("/recommend")
def recommend_gpu(req: RecommendationRequest):
    return estimate_gpu_requirement(req.model, req.users, req.latency)