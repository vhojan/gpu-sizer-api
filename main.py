from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json

from model_service import ModelService
from sizing_logic import get_gpu_recommendation

def get_db_path():
    # Are we on Azure App Service?
    on_azure = "WEBSITE_INSTANCE_ID" in os.environ

    if on_azure:
        db_path = "/home/models.db"
    else:
        # Always use the directory where this file lives for local
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "models.db")
    return db_path

DB_PATH = get_db_path()
print(f"[INFO] Using SQLite DB at: {DB_PATH}")

from model_service import ModelService
MODEL_SERVICE = ModelService(DB_PATH)

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
    local_ids = MODEL_SERVICE.search_models(q)
    # Format DB results for clarity
    db_results = [
        {"model_id": mid, "source": "local"} for mid in local_ids
    ]
    hf_results = MODEL_SERVICE.search_hf_models(q, exclude_ids=local_ids)
    # Combine, locals first
    return db_results + hf_results

@app.get("/models/{model_id:path}")
def get_model(model_id: str, force_recalc_kv: bool = Query(False)):
    print(f"[HANDLER DEBUG] /models/{model_id} force_recalc_kv={force_recalc_kv}")
    # Always try the DB, then Hugging Face if missing
    details = MODEL_SERVICE.get_model_details(model_id, force_recalc_kv=force_recalc_kv)
    if not details:
        print(f"[INFO] Not found in DB, attempting force fetch for: {model_id}")
        details = MODEL_SERVICE.get_model_details(model_id, force_recalc_kv=True)
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