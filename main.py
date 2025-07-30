from fastapi import FastAPI, HTTPException, Query
from model_service import ModelService
from sizing_logic import get_gpu_recommendation
import json

app = FastAPI()
model_service = ModelService("models.db")

@app.get("/models")
def list_models():
    return model_service.list_models()

@app.get("/models/search")
def search_models(q: str = Query(..., min_length=1)):
    results = model_service.search_models(q)
    if not results:
        raise HTTPException(status_code=404, detail="404: Model not found")
    return results

@app.get("/models/{model_id:path}")
def get_model(model_id: str):
    # :path allows slashes in model_id
    print(f"[DEBUG] /models/{model_id} called")
    model = model_service.get_model_details(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found (in DB or Hugging Face)")
    return model

CATALOG_PATH = "gpu_catalog.json"  # root folder, no slash needed

@app.get("/gpus")
def list_gpus():
    with open(CATALOG_PATH) as f:
        return json.load(f)

@app.get("/recommendation")
def recommend(
    model: str = Query(..., description="Model ID"),
    users: int = Query(..., ge=1, description="Concurrent users"),
    latency: float = Query(..., gt=0, description="Latency target (ms or s)")
):
    print(f"[DEBUG] /recommendation: model={model}, users={users}, latency={latency}")
    details = model_service.get_model_details(model)
    if not details:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        with open(CATALOG_PATH) as f:
            gpu_catalog = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load gpu_catalog.json: {e}")
        raise HTTPException(status_code=500, detail="Could not load GPU catalog")

    try:
        rec = get_gpu_recommendation(details, users, latency, gpu_catalog)
    except Exception as e:
        import traceback
        print(f"[ERROR] Sizing logic failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sizing logic error: {e}")
    if not rec:
        raise HTTPException(status_code=400, detail="No suitable GPU found for the given parameters")
    return rec