import sqlite3
import json
from transformers import AutoConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError
from requests.exceptions import HTTPError
import os

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

class ModelService:
    def __init__(self, db_path):
        self.db_path = db_path
        self._ensure_db_schema()  # Ensures the table exists at startup

    def _ensure_db_schema(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                architecture TEXT,
                hidden_size INTEGER,
                num_hidden_layers INTEGER,
                num_attention_heads INTEGER,
                use_cache BOOLEAN,
                kv_cache_fp16_gb REAL,
                kv_cache_bf16_gb REAL,
                kv_cache_fp32_gb REAL,
                config_json TEXT,
                query_count INTEGER,
                last_accessed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def list_models(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT model_id FROM models")
        models = [row[0] for row in c.fetchall()]
        conn.close()
        return models

    def search_models(self, query):
        # Local DB search
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        like_query = f"%{query}%"
        c.execute("SELECT model_id FROM models WHERE model_id LIKE ?", (like_query,))
        local_matches = {row[0] for row in c.fetchall()}
        conn.close()

        # Hugging Face Hub search (top 5 models)
        api = HfApi()
        remote_matches = set()
        try:
            results = api.list_models(search=query, limit=5)
            for model in results:
                remote_matches.add(model.modelId)
        except Exception as e:
            print(f"[ERROR] Hugging Face Hub search failed: {e}")

        # Combine, keeping order (local first)
        combined = list(local_matches) + [m for m in remote_matches if m not in local_matches]
        return combined

    def get_model_details(self, model_id: str):
        print(f"[DEBUG] Looking up model_id: {model_id}")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM models WHERE model_id=?", (model_id,))
        row = c.fetchone()
        conn.close()
        if row:
            print(f"[DEBUG] Found in DB: {model_id}")
            return self._row_to_dict(row)
        # Try HuggingFace fallback
        try:
            print(f"[DEBUG] Fetching from Hugging Face: {model_id}")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, use_auth_token=HF_TOKEN)
            hidden_size = getattr(config, "hidden_size", None)
            num_layers = getattr(config, "num_hidden_layers", None)
            num_heads = getattr(config, "num_attention_heads", None)
            use_cache = getattr(config, "use_cache", None)
            seq_len = getattr(config, "max_position_embeddings", 2048)
            fp16_bytes = 2
            kv_cache_fp16_gb = None
            if hidden_size and num_layers and num_heads and seq_len:
                kv_cache_fp16_gb = (
                    num_layers * num_heads * seq_len * hidden_size * 2 * fp16_bytes / (1024 ** 3)
                )
            result = {
                "model_id": model_id,
                "architecture": getattr(config, "architectures", ["Unknown"])[0],
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "use_cache": use_cache,
                "kv_cache_fp16_gb": round(kv_cache_fp16_gb, 3) if kv_cache_fp16_gb else None,
                "kv_cache_bf16_gb": None,
                "kv_cache_fp32_gb": None,
                "config_json": config.to_json_string(),
                "query_count": 0,
                "last_accessed_at": None,
            }
            self.save_model(result)
            print(f"[DEBUG] Model fetched and saved: {model_id}")
            return result
        except (GatedRepoError, HTTPError) as e:
            msg = str(e)
            if "gated repo" in msg or "restricted" in msg or "403" in msg or "401" in msg:
                print(f"[INFO] Model {model_id} is gated/restricted on Hugging Face (GatedRepoError/HTTPError).")
                return {
                    "model_id": model_id,
                    "restricted": True,
                    "message": "This model is restricted or gated on Hugging Face. Please log in, request access, or agree to the repository terms."
                }
            else:
                print(f"[ERROR] HTTP/GatedRepoError for model_id={model_id}: {e}")
                return None
        except OSError as e:
            msg = str(e)
            if "gated repo" in msg or "restricted" in msg or "403" in msg or "401" in msg:
                print(f"[INFO] Model {model_id} is gated/restricted on Hugging Face (OSError).")
                return {
                    "model_id": model_id,
                    "restricted": True,
                    "message": "This model is restricted or gated on Hugging Face. Please log in, request access, or agree to the repository terms."
                }
            else:
                print(f"[ERROR] OSError for model_id={model_id}: {e}")
                return None
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to fetch from HF for model_id={model_id}: {e}")
            traceback.print_exc()
            return None

    def save_model(self, model_info):
        # Save model details to DB, upsert-style
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        columns = ', '.join(model_info.keys())
        placeholders = ', '.join('?' * len(model_info))
        update_assignments = ', '.join(f"{col}=excluded.{col}" for col in model_info)
        sql = f"""
            INSERT INTO models ({columns})
            VALUES ({placeholders})
            ON CONFLICT(model_id) DO UPDATE SET {update_assignments}
        """
        c.execute(sql, tuple(model_info.values()))
        conn.commit()
        conn.close()

    def _row_to_dict(self, row):
        # Map DB row to dict, column order must match table!
        keys = [
            "model_id", "architecture", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "use_cache", "kv_cache_fp16_gb",
            "kv_cache_bf16_gb", "kv_cache_fp32_gb", "config_json",
            "query_count", "last_accessed_at"
        ]
        return dict(zip(keys, row))