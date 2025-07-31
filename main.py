from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json

from model_service import ModelService
from sizing_logic import get_gpu_recommendation

#DB_PATH = os.environ.get("DB_PATH", "models.db")
DB_PATH = os.environ.get("MODELS_DB_PATH", "/home/models.db")

MODEL_SERVICE = ModelService(DB_PATH)

app = FastAPI()

# Enable CORS for dev (adjust for prod!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your UI's URL in prod!
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
def list_models():
    return MODEL_SERVICE.list_models()

@app.get("/models/search")
def search_models(q: str):
    return MODEL_SERVICE.search_models(q)

@app.get("/models/{model_id}")
def get_model(model_id: str, force_recalc_kv: bool = Query(False)):
    details = MODEL_SERVICE.get_model_details(model_id, force_recalc_kv=force_recalc_kv)
    if not details:
        raise HTTPException(status_code=404, detail="Model not found or inaccessible.")
    return details


from fastapi import Path

@app.post("/models/{model_id:path}/recalc")
def recalc_model(model_id: str = Path(..., description="Model ID with slashes allowed")):
    details = MODEL_SERVICE.recalc_model_details(model_id)
    if not details:
        raise HTTPException(status_code=404, detail="Model not found or inaccessible.")
    return details

@app.get("/gpus")
def list_gpus():
    catalog_path = os.path.join(os.path.dirname(__file__), "gpu_catalog.json")
    try:
        with open(catalog_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {catalog_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load GPU catalog.")

@app.get("/recommendation")
def recommend_gpu(
    model: str,
    users: int,
    latency: float,
    kv_cache_override: float = Query(None),
    force_recalc_kv: bool = Query(False),
):
    model_details = MODEL_SERVICE.get_model_details(model, force_recalc_kv=force_recalc_kv)
    if not model_details:
        raise HTTPException(status_code=404, detail="Model not found or inaccessible.")
    catalog_path = os.path.join(os.path.dirname(__file__), "gpu_catalog.json")
    try:
        with open(catalog_path, "r") as f:
            gpu_catalog = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not load GPU catalog.")
    rec = get_gpu_recommendation(
        model_details,  # model_info
        gpu_catalog,    # gpus
        users,          # users
        latency,        # latency
        kv_cache_override=kv_cache_override,
        force_recalc_kv=force_recalc_kv,
    )
    if "error" in rec:
        raise HTTPException(status_code=400, detail=rec["error"])
    return rec