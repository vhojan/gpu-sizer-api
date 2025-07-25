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
    allow_origins=origins,  # instead of ["*"]
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
    required_latency = latency / 1000  # convert ms to s

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    parallelism = math.floor(required_latency / base_latency)
    concurrent_per_gpu = parallelism if parallelism > 0 else 1
    required_gpus = math.ceil(users / concurrent_per_gpu)

    suitable_gpus = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= model.VRAM_Required_GB
    ]

    if not suitable_gpus:
        return {"recommendation": None}

    sorted_gpus = sorted(suitable_gpus, key=lambda x: x["TFLOPs (FP16)"], reverse=True)
    best_gpu = sorted_gpus[0]
    top_gpu = suitable_gpus[0]  # Now safe because we checked it's not empty

    return {
    "recommendation": {
        "gpu": top_gpu["GPU Type"],
        "quantity": top_gpu["Quantity"],
        "gpu_memory": top_gpu["Memory Needed"],
    },
    "alternatives": [
        {
            "gpu": gpu["GPU Type"],
            "quantity": gpu["Quantity"],
            "gpu_memory": gpu["Memory Needed"]
        }
        for gpu in suitable_gpus[1:5]  # Limit to top 4 alternatives
    ]
}

@app.get("/models")
def get_models():
    return model_catalog

@app.post("/recommend")
def recommend_gpu(req: RecommendationRequest):
    return estimate_gpu_requirement(req.model, req.users, req.latency)