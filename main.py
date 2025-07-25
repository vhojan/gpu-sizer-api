
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import math

app = FastAPI()

# Load data
try:
    model_df = pd.read_csv("model_catalog.csv")
    gpu_df = pd.read_csv("gpu_catalog.csv")
except Exception as e:
    raise RuntimeError(f"Failed to load CSVs: {e}")

# Input schema for recommendations
class RecommendationRequest(BaseModel):
    model: str
    users: int
    latency_target: float

@app.get("/")
def read_root():
    return {"message": "AI Model GPU Sizer API"}

@app.get("/models")
def get_models():
    return model_df.to_dict(orient="records")

@app.get("/gpus")
def get_gpus():
    return gpu_df.to_dict(orient="records")

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    selected = model_df[model_df["Model"] == req.model]
    if selected.empty:
        raise HTTPException(status_code=404, detail="Model not found")

    model = selected.iloc[0]
    total_vram = model["VRAM Required (GB)"] * req.users

    results = []
    for _, gpu in gpu_df.iterrows():
        mem = gpu["Memory (GB)"]
        nvlink = gpu["NVLink Support"]
        max_nvlink = gpu.get("Max NVLink GPUs", 1)

        if mem >= total_vram:
            needed_gpus = 1
            supported = True
        elif nvlink and (math.ceil(total_vram / mem) <= max_nvlink):
            needed_gpus = math.ceil(total_vram / mem)
            supported = True
        else:
            needed_gpus = math.ceil(total_vram / mem)
            supported = False

        results.append({
            "model": gpu["Model"],
            "memory": mem,
            "fp16_tflops": gpu.get("FP16 TFLOPs", "-"),
            "fp8_tflops": gpu.get("FP8 TFLOPs", "-"),
            "fp4_tflops": gpu.get("FP4 TFLOPs", "-"),
            "nvlink": nvlink,
            "max_nvlink_gpus": max_nvlink,
            "needed_gpus": needed_gpus,
            "supported": supported,
            "color": "green" if supported else "red"
        })

    return results
