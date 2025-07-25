from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import math

app = FastAPI()

# Load and clean CSVs
def load_csv(filename):
    df = pd.read_csv(filename)
    df = df.fillna("")  # Replace NaN with empty strings
    return df

model_df = load_csv("model_catalog.csv")
gpu_df = load_csv("gpu_catalog.csv")

# Request schema for recommendation
class RecommendRequest(BaseModel):
    model: str
    users: int
    latency_target: float

@app.get("/")
def root():
    return {"message": "AI Model GPU Sizer API is running!"}

@app.get("/models")
def get_models():
    return model_df.to_dict(orient="records")

@app.get("/gpus")
def get_gpus():
    return gpu_df.to_dict(orient="records")

@app.post("/recommend")
def recommend_gpu(req: RecommendRequest):
    # Get the selected model
    model_row = model_df[model_df["Model"] == req.model]

    if model_row.empty:
        raise HTTPException(status_code=404, detail="Model not found")

    required_vram = float(model_row["Min GPU Memory (GB)"].values[0])
    latency_factor = float(model_row["Latency Factor"].values[0])
    target_latency = req.latency_target

    # Filter GPUs that meet the VRAM requirement
    supported_gpus = []
    for _, row in gpu_df.iterrows():
        try:
            gpu_vram = float(row["VRAM (GB)"])
            max_nvlink_gpus = int(row["Max NVLink GPUs"]) if row["Max NVLink GPUs"] != "" else 1
            total_vram = gpu_vram * max_nvlink_gpus

            if total_vram >= required_vram:
                estimated_latency = latency_factor / float(row["Latency Factor"]) if row["Latency Factor"] else math.inf
                supported_gpus.append({
                    "GPU Type": row["GPU Type"],
                    "Total VRAM": total_vram,
                    "Estimated Latency": round(estimated_latency, 2),
                    "Max NVLink GPUs": max_nvlink_gpus,
                    "Architecture": row["Arch"],
                    "Supported": True
                })
        except:
            continue

    if not supported_gpus:
        raise HTTPException(status_code=400, detail="No compatible GPUs found for this model.")

    # Sort by estimated latency
    supported_gpus = sorted(supported_gpus, key=lambda x: x["Estimated Latency"])
    return {"recommendation": supported_gpus[0], "alternatives": supported_gpus}