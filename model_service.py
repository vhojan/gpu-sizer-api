import sqlite3
import json
import requests
from huggingface_hub import hf_hub_download
import os
import traceback

from sizing_logic import estimate_kv_cache_gb, estimate_weights_gib


class ModelService:
    def __init__(self, db_path):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Base schema (older deployments may already have this without new cols)
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
                last_accessed_at TEXT
                -- NOTE: columns may be added below via ALTER (idempotent)
            );
            """)
            # Idempotent migration: add missing columns if needed
            c.execute("PRAGMA table_info(models)")
            existing = {row[1] for row in c.fetchall()}

            if "missing_kv_cache" not in existing:
                c.execute("ALTER TABLE models ADD COLUMN missing_kv_cache BOOLEAN DEFAULT 0")
            if "minimal_gpu_memory_gb" not in existing:
                c.execute("ALTER TABLE models ADD COLUMN minimal_gpu_memory_gb REAL")

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
        print(f"[DEBUG] get_model_details called for: {model_id}, force_recalc_kv={force_recalc_kv}")
        result = self._lookup_model(model_id)
        print(f"[DEBUG] DB lookup for {model_id}: {result}")

        # If we have a cached row and we're not forcing recompute, try to backfill missing fields and return.
        if result and not force_recalc_kv:
            changed = False

            # Backfill minimal_gpu_memory_gb if missing, using stored config_json.
            if not result.get("minimal_gpu_memory_gb"):
                try:
                    cfg = json.loads(result.get("config_json") or "{}")
                except Exception:
                    cfg = {}
                try:
                    w_gib = estimate_weights_gib(cfg)
                    if w_gib is not None:
                        result["minimal_gpu_memory_gb"] = float(w_gib)
                        changed = True
                except Exception as e:
                    print(f"[WARN] backfill weights_vram failed: {e}")

            # (Optional) If kv missing you could recompute here too; for now we keep your current behavior.

            if changed:
                try:
                    self.save_model(result)
                    print(f"[DEBUG] Backfilled and saved row for {model_id}")
                except Exception as e:
                    print(f"[WARN] Failed to persist backfilled row: {e}")

            if result.get("missing_kv_cache"):
                print(f"[WARNING] Found in DB, but missing KV cache: {model_id}")
            else:
                print(f"[DEBUG] Found in DB: {model_id}")
            return result   # return cached row (possibly backfilled)

        # Only fetch if not in DB or force_recalc_kv!
        try:
            print(f"[DEBUG] Fetching from Hugging Face: {model_id}")
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            print(f"[DEBUG] Downloaded config.json to: {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"[DEBUG] Loaded config JSON (truncated): {str(config)[:200]}...")
            result = self._extract_from_config(model_id, config)
            print(f"[DEBUG] Extracted config: {result}")
            self.save_model(result)
            print(f"[DEBUG] Saved model to DB: {model_id}")
            return result
        except Exception as e:
            print(f"[ERROR] Failed to fetch from HF for model_id={model_id}: {e}")
            traceback.print_exc()
            if result:
                print(f"[WARNING] Returning possibly incomplete DB row for {model_id}")
                return result
            print(f"[ERROR] No data for {model_id}, returning None")
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
        # Helper to prefer text_config, fallback to root/alternatives
        def get_config_key(key, alt_keys=None, default=None):
            alt_keys = alt_keys or []
            tc = config.get("text_config", {})
            if key in tc:
                return tc[key]
            for alt in alt_keys:
                if alt in tc:
                    return tc[alt]
            if key in config:
                return config[key]
            for alt in alt_keys:
                if alt in config:
                    return config[alt]
            return default

        # For kv_cache keys, search all configs
        def get_kv_cache_value(key):
            for section in ("", "text_config", "audio_config", "vision_config"):
                d = config if not section else config.get(section, {})
                if key in d:
                    return d[key]
            return None

        result = {
            "model_id": model_id,
            "architecture": config.get("architectures", [None])[0] or config.get("model_type"),
            "hidden_size": get_config_key("hidden_size"),
            "num_hidden_layers": get_config_key("num_hidden_layers", alt_keys=["num_layers"]),
            "num_attention_heads": get_config_key("num_attention_heads", alt_keys=["n_head"]),
            "use_cache": int(bool(get_config_key("use_cache", default=True))),
            "kv_cache_fp16_gb": None,
            "kv_cache_bf16_gb": None,
            "kv_cache_fp32_gb": None,
            "config_json": json.dumps(config),
            "query_count": 0,
            "last_accessed_at": None,
            "missing_kv_cache": False,
            "minimal_gpu_memory_gb": None,  # filled below
        }

        # --- Weights VRAM (bf16/fp16) from config ---
        try:
            # estimate_weights_gib accepts dict or JSON string; try dict first
            w_gib = estimate_weights_gib(config)
            if w_gib is not None:
                result["minimal_gpu_memory_gb"] = float(w_gib)
            else:
                w2 = estimate_weights_gib(json.dumps(config))
                if w2 is not None:
                    result["minimal_gpu_memory_gb"] = float(w2)
        except Exception as e:
            print(f"[WARN] weights estimate failed: {e}")

        # --- KV per user: from config if present, else estimate ---
        kv_config_keys = ["kv_cache_fp16_gb", "kv_cache_bf16_gb", "kv_cache_fp32_gb"]
        found = False
        for key in kv_config_keys:
            value = get_kv_cache_value(key)
            if value is not None:
                result[key] = value
                found = True

        if not found:
            # Robust sequence length lookup (fallback to 2048)
            seq_len = (
                get_config_key("max_position_embeddings")
                or get_config_key("seq_length")
                or get_config_key("n_positions")
                or 2048
            )
            try:
                seq_len = int(seq_len)
            except Exception:
                seq_len = 2048

            # Use GQA/MQA KV heads when available; fallback to attention heads.
            kv_heads = get_config_key("num_key_value_heads") or result["num_attention_heads"]

            # Derive head_dim if not explicitly provided
            head_dim = get_config_key("head_dim")
            if not head_dim and result["hidden_size"] and result["num_attention_heads"]:
                try:
                    head_dim = int(result["hidden_size"]) // int(result["num_attention_heads"])
                except Exception:
                    head_dim = None

            kv_est = estimate_kv_cache_gb(
                num_layers=int(result["num_hidden_layers"]),
                num_attention_heads=int(kv_heads),        # pass KV heads here
                hidden_size=int(result["hidden_size"]),
                seq_len=seq_len,
                dtype_bytes=2,
                head_dim=int(head_dim) if head_dim is not None else None,
            )

            result["kv_cache_fp16_gb"] = kv_est
            result["missing_kv_cache"] = kv_est is None

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

    def search_hf_models(self, query, exclude_ids=None):
        HF_SEARCH_URL = "https://huggingface.co/api/models"
        params = {"search": query, "limit": 10}
        exclude_ids = set(exclude_ids or [])
        try:
            resp = requests.get(HF_SEARCH_URL, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for m in data:
                model_id = m.get("modelId") or m.get("id")
                if model_id not in exclude_ids:
                    results.append({
                        "model_id": model_id,
                        "source": "huggingface",
                        "likes": m.get("likes", 0),
                        "tags": m.get("tags", []),
                        "private": m.get("private", False),
                        "downloads": m.get("downloads", 0)
                    })
            return results
        except Exception as e:
            print(f"[ERROR] Hugging Face search failed: {e}")
            return []