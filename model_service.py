import sqlite3
import json
from huggingface_hub import hf_hub_download
import os

from sizing_logic import estimate_kv_cache_gb

class ModelService:
    def __init__(self, db_path):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                architecture TEXT,
                hidden_size INTEGER,
                num_hidden_layers INTEGER,
                num_attention_heads INTEGER,
                use_cache INTEGER,
                kv_cache_fp16_gb REAL,
                kv_cache_bf16_gb REAL,
                kv_cache_fp32_gb REAL,
                config_json TEXT,
                query_count INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                missing_kv_cache BOOLEAN DEFAULT 0
            );
            """)
            conn.commit()

    def list_models(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT model_id FROM models")
            return [row[0] for row in c.fetchall()]

    def search_models(self, query):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT model_id FROM models WHERE model_id LIKE ?", (f"%{query}%",))
            return [row[0] for row in c.fetchall()]

    def get_model_details(self, model_id, force_recalc_kv=False):
        # Always prefer DB if available, otherwise try fetch
        result = self._lookup_model(model_id)
        if result and not force_recalc_kv and not result.get("missing_kv_cache"):
            print(f"[DEBUG] Found in DB: {model_id}")
            return result

        # Either missing or force recalc, so (re)fetch
        try:
            print(f"[DEBUG] Fetching from Hugging Face: {model_id}")
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            result = self._extract_from_config(model_id, config)
            # Save to DB (including updated kv_cache fields!)
            self.save_model(result)
            return result
        except Exception as e:
            print(f"[ERROR] Failed to fetch from HF for model_id={model_id}: {e}")
            return None

    def _lookup_model(self, model_id):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = c.fetchone()
            if not row:
                return None
            columns = [desc[0] for desc in c.description]
            return dict(zip(columns, row))

    def _extract_from_config(self, model_id, config):
        # Extract the relevant fields and calculate kv_cache if needed
        result = {
            "model_id": model_id,
            "architecture": config.get("architectures", [None])[0] or config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers") or config.get("num_layers"),
            "num_attention_heads": config.get("num_attention_heads") or config.get("n_head"),
            "use_cache": int(config.get("use_cache", True)),
            "kv_cache_fp16_gb": None,
            "kv_cache_bf16_gb": None,
            "kv_cache_fp32_gb": None,
            "config_json": json.dumps(config),
            "query_count": 0,
            "last_accessed_at": None,
            "missing_kv_cache": False,
        }

        # Try to find kv_cache from config or estimate
        kv_config_keys = [
            "kv_cache_fp16_gb", "kv_cache_bf16_gb", "kv_cache_fp32_gb"
        ]
        found = False
        for key in kv_config_keys:
            if key in config:
                result[key] = config[key]
                found = True
        if not found:
            # Estimate FP16 as default
            kv_est = estimate_kv_cache_gb(
                num_layers=result["num_hidden_layers"],
                num_attention_heads=result["num_attention_heads"],
                hidden_size=result["hidden_size"],
                seq_len=config.get("max_position_embeddings", 2048),
                dtype_bytes=2
            )
            result["kv_cache_fp16_gb"] = kv_est
            result["missing_kv_cache"] = True if kv_est is None else False
        return result

    def save_model(self, model_info):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            columns = list(model_info.keys())
            values = list(model_info.values())
            placeholders = ",".join(["?"] * len(columns))
            update_expr = ",".join([f"{k}=excluded.{k}" for k in columns if k != "model_id"])
            sql = f"""
            INSERT INTO models ({",".join(columns)}) VALUES ({placeholders})
            ON CONFLICT(model_id) DO UPDATE SET {update_expr}
            """
            c.execute(sql, values)
            conn.commit()

    def recalc_model_details(self, model_id):
        return self.get_model_details(model_id, force_recalc_kv=True)