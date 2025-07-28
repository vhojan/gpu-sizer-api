import sqlite3
import json
import os
from datetime import datetime
from transformers import AutoConfig
from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

# Load .env and authenticate with Hugging Face
load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)

DB_PATH = "models.db"
SEQ_LEN = 2048

router = APIRouter()

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            architecture TEXT,
            hidden_size INTEGER,
            num_hidden_layers INTEGER,
            num_attention_heads INTEGER,
            use_cache BOOLEAN,
            kv_cache_fp16_gb FLOAT,
            kv_cache_bf16_gb FLOAT,
            kv_cache_fp32_gb FLOAT,
            config_json TEXT,
            query_count INTEGER DEFAULT 0,
            last_accessed_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def calculate_kv_cache_bytes(config, dtype_bytes=2):
    try:
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        return num_layers * num_heads * head_dim * SEQ_LEN * 2 * dtype_bytes
    except Exception:
        return None

def fetch_model_info(model_id):
    print(f"Fetching model: {model_id}")
    config = AutoConfig.from_pretrained(model_id, token=True, trust_remote_code=True)

    # Ensure required fields exist
    required = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
    for attr in required:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required config field: '{attr}'")

    kv_fp16 = calculate_kv_cache_bytes(config, dtype_bytes=2)
    kv_fp32 = calculate_kv_cache_bytes(config, dtype_bytes=4)

    return {
        "model_id": model_id,
        "architecture": getattr(config, "architectures", ["unknown"])[0],
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "use_cache": getattr(config, "use_cache", True),
        "kv_cache_fp16_gb": round(kv_fp16 / 1e9, 3) if kv_fp16 else None,
        "kv_cache_bf16_gb": round(kv_fp16 / 1e9, 3) if kv_fp16 else None,
        "kv_cache_fp32_gb": round(kv_fp32 / 1e9, 3) if kv_fp32 else None,
        "config_json": json.dumps(config.to_dict()),
        "query_count": 1,
        "last_accessed_at": datetime.utcnow().isoformat()
    }

def store_model_info(model_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_data["model_id"],
        model_data["architecture"],
        model_data["hidden_size"],
        model_data["num_hidden_layers"],
        model_data["num_attention_heads"],
        model_data["use_cache"],
        model_data["kv_cache_fp16_gb"],
        model_data["kv_cache_bf16_gb"],
        model_data["kv_cache_fp32_gb"],
        model_data["config_json"],
        model_data["query_count"],
        model_data["last_accessed_at"]
    ))
    conn.commit()
    conn.close()

def increment_model_usage(model_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE models
        SET query_count = query_count + 1,
            last_accessed_at = ?
        WHERE model_id = ?
    ''', (datetime.utcnow().isoformat(), model_id))
    conn.commit()
    conn.close()

def get_model_from_db(model_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    keys = [
        "model_id", "architecture", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "use_cache",
        "kv_cache_fp16_gb", "kv_cache_bf16_gb", "kv_cache_fp32_gb", "config_json",
        "query_count", "last_accessed_at"
    ]
    return dict(zip(keys, row))

@router.get("/models/search")
def search_models(q: str):
    db_results = []
    hf_results = []
    seen = set()

    # Search in local DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT model_id FROM models WHERE model_id LIKE ? LIMIT 20", (f"%{q}%",))
    matches = c.fetchall()
    conn.close()

    for m in matches:
        model_id = m[0]
        db_results.append({"label": model_id, "value": model_id})
        seen.add(model_id)

    # Fallback to Hugging Face Hub
    api = HfApi()
    try:
        results = api.list_models(search=q, limit=20)
        for model in results:
            if model.modelId not in seen:
                hf_results.append({"label": model.modelId, "value": model.modelId})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face search failed: {str(e)}")

    return db_results + hf_results

@router.get("/models/{model_id:path}")
def get_model(model_id: str):
    model = get_model_from_db(model_id)
    if model:
        increment_model_usage(model_id)
        return model
    try:
        model_data = fetch_model_info(model_id)
        store_model_info(model_data)
        return model_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e).lower()
        if "403" in error_msg and "gated repo" in error_msg:
            raise HTTPException(
                status_code=403,
                detail=f"Access to '{model_id}' is restricted. You may request access at https://huggingface.co/{model_id}"
            )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found or failed to load: {str(e)}"
        )
