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
    required_latency = latency / 1000  # ms to s

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    # Determine concurrency and required GPU count
    parallelism = max(1, math.floor(required_latency / base_latency))
    concurrent_per_gpu = parallelism
    required_gpus = math.ceil(users / concurrent_per_gpu)

    # Try single-GPU candidates first (preferred)
    single_gpu_candidates = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= model.VRAM_Required_GB
    ]

    if single_gpu_candidates:
        sorted_single = sorted(single_gpu_candidates, key=lambda x: x["VRAM (GB)"])
        best = sorted_single[0]
        return {
            "recommendation": {
                "gpu": best["GPU Type"],
                "quantity": 1,
                "gpu_memory": model.VRAM_Required_GB
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": 1,
                    "gpu_memory": model.VRAM_Required_GB
                }
                for gpu in sorted_single[1:5]
            ]
        }

    # Else, try multi-GPU with NVLink
    multi_gpu_candidates = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] * int(gpu.get("Max NVLink GPUs", 0)) >= model.VRAM_Required_GB and
        gpu.get("NVLink") is True and
        int(gpu.get("Max NVLink GPUs", 0)) >= required_gpus
    ]

    if not multi_gpu_candidates:
        raise HTTPException(status_code=400, detail="No suitable GPUs found.")

    sorted_multi = sorted(multi_gpu_candidates, key=lambda x: x["VRAM (GB)"])
    top = sorted_multi[0]

    return {
        "recommendation": {
            "gpu": top["GPU Type"],
            "quantity": required_gpus,
            "gpu_memory": model.VRAM_Required_GB
        },
        "alternatives": [
            {
                "gpu": gpu["GPU Type"],
                "quantity": required_gpus,
                "gpu_memory": model.VRAM_Required_GB
            }
            for gpu in sorted_multi[1:5]
        ]
    }

@app.get("/models")
def get_models():
    return model_catalog

@app.post("/recommend")
def recommend_gpu(req: RecommendationRequest):
    return estimate_gpu_requirement(req.model, req.users, req.latency)