# sizing_logic.py
import os
import json
import math
from typing import Optional, Union, Dict, Any

Cfg = Union[dict, str]

def estimate_weights_gib(cfg_or_json: Cfg, overhead_frac: float = 0.10) -> Optional[float]:
    """
    Estimate VRAM for model weights (GiB).

    - Assumes bf16/fp16 unless torch_dtype explicitly says fp32.
    - Reads from both root and text_config; supports common alt names.
    - Handles MoE: if num_experts present, counts expert MLP params.
      By default we assume **all experts are resident** on the GPU.
      Set env MOE_LOAD_ALL_EXPERTS="0" to scale by num_experts_per_tok instead.

    Returns None if required keys are unavailable.
    """

    # Parse config input
    if isinstance(cfg_or_json, str):
        try:
            cfg: Dict[str, Any] = json.loads(cfg_or_json)
        except Exception:
            return None
    elif isinstance(cfg_or_json, dict):
        cfg = cfg_or_json
    else:
        return None

    txt = cfg.get("text_config", {}) or {}

    def _get(key, *alts):
        # prefer text_config, then root
        for src in (txt, cfg):
            if key in src:
                return src[key]
            for a in alts:
                if a in src:
                    return src[a]
        return None

    def _num(x):
        try:
            n = float(x)
            return n if math.isfinite(n) else None
        except Exception:
            return None

    # Core sizes
    d = _num(_get("hidden_size", "n_embd", "d_model"))
    L = _num(_get("num_hidden_layers", "num_layers", "n_layer"))
    # Vocab: some configs omit it; default to a reasonable 256k
    v = _num(_get("vocab_size", "vocab_sz"))
    if v is None:
        v = 256000.0

    # MLP widths
    i = _num(_get("intermediate_size", "ffn_hidden_size", "mlp_dim")) or 0.0

    if not (d and L and v):
        return None

    # dtype → bytes per param
    dtype = str(_get("torch_dtype")).lower()
    bpp = 4 if ("float32" in dtype or "fp32" in dtype) else 2  # default bf16/fp16

    # ---- Dense (non‑MoE) baseline formulas ----
    # 1) per-layer + MLP + embeddings
    per_layer_mlp = L * (4 * d * d + 3 * d * i) + v * d
    # 2) LLaMA-ish scaling (attn-heavy)
    llama_like    = 12 * d * d * L + v * d
    dense_params  = max(per_layer_mlp, llama_like)

    # ---- MoE awareness ----
    num_experts          = _num(_get("num_experts"))
    num_experts_per_tok  = _num(_get("num_experts_per_tok", "top_k", "k")) or 0.0
    moe_intermediate     = _num(_get("moe_intermediate_size")) or 0.0
    shared_mlp_i         = _num(_get("shared_expert_intermediate_size")) or 0.0

    is_moe = bool(num_experts and num_experts > 0)

    if not is_moe:
        total_params = dense_params
    else:
        # Attention params (still dense)
        attn_params = L * (4 * d * d)

        # Shared (non-expert) MLP, if present
        shared_params = L * (3 * d * shared_mlp_i) if shared_mlp_i > 0 else 0.0

        # Expert MLP params per expert per layer: roughly 3 * d * moe_intermediate
        expert_mlp_per_expert = 3 * d * (moe_intermediate if moe_intermediate > 0 else i)

        load_all = os.getenv("MOE_LOAD_ALL_EXPERTS", "1") not in ("0", "false", "False")
        if load_all or not num_experts_per_tok:
            # safer: assume all experts need to be resident on one GPU
            expert_params = L * (num_experts * expert_mlp_per_expert)
        else:
            # lighter: only active experts per token resident
            expert_params = L * (num_experts_per_tok * expert_mlp_per_expert)

        total_params = attn_params + shared_params + expert_params + v * d

        # If MoE estimate is still suspiciously *smaller* than dense baseline,
        # fall back to dense (avoid undercounting).
        total_params = max(total_params, dense_params)

    gib = (total_params * bpp * (1.0 + overhead_frac)) / (1024 ** 3)
    return round(gib, 2)

def estimate_kv_cache_gb(
    *,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    seq_len: int,
    dtype_bytes: int = 2,
    head_dim: Optional[int] = None,
) -> Optional[float]:
    """
    Estimate KV cache size (GiB per user).

    IMPORTANT:
    - Pass **KV heads** here (i.e., num_key_value_heads if present).
    - If head_dim is None, it is derived as hidden_size / attention_heads.
    """
    if not (num_layers and num_attention_heads and hidden_size and seq_len and dtype_bytes):
        return None

    if head_dim is None:
        if num_attention_heads == 0:
            return None
        head_dim = hidden_size // num_attention_heads
        if head_dim <= 0:
            return None

    bytes_total = num_layers * seq_len * num_attention_heads * head_dim * 2 * dtype_bytes
    return bytes_total / (1024 ** 3)

def get_effective_kv_cache(model_info, seq_len=None, dtype_bytes=2, users=1, kv_cache_override=None, force_recalc_kv=False):
    """
    Returns the KV cache requirement in GB, supporting user override and forced recalculation.
    """
    if kv_cache_override is not None:
        try:
            return float(kv_cache_override)
        except Exception:
            pass
    if force_recalc_kv:
        # Ignore saved value and estimate anew
        return estimate_kv_cache_gb(
            num_layers=model_info.get("num_hidden_layers"),
            num_attention_heads=model_info.get("num_attention_heads"),
            hidden_size=model_info.get("hidden_size"),
            seq_len=seq_len or 2048,
            dtype_bytes=dtype_bytes,
            users=users,
        )
    # Otherwise use stored value if available, else estimate
    for k in ["kv_cache_fp16_gb", "kv_cache_bf16_gb", "kv_cache_fp32_gb"]:
        v = model_info.get(k)
        if v is not None:
            try:
                val = float(v)
                if val > 0:
                    return val
            except Exception:
                continue
    # Fallback: estimate
    return estimate_kv_cache_gb(
        num_layers=model_info.get("num_hidden_layers"),
        num_attention_heads=model_info.get("num_attention_heads"),
        hidden_size=model_info.get("hidden_size"),
        seq_len=seq_len or 2048,
        dtype_bytes=dtype_bytes,
        users=users,
    )

def get_gpu_recommendation(
    model_info, gpus, users, latency, kv_cache_override=None, force_recalc_kv=False
):
    # 1. Calculate/override/force-recalc KV cache
    kv_cache_gb = get_effective_kv_cache(
        model_info, users=users, kv_cache_override=kv_cache_override, force_recalc_kv=force_recalc_kv
    )
    if kv_cache_gb is None or kv_cache_gb == 0:
        return {"error": "Missing or invalid KV cache size for model."}

    # 2. Find all eligible GPUs with enough VRAM
    eligible_gpus = [gpu for gpu in gpus if (gpu.get("VRAM (GB)", 0) >= kv_cache_gb)]
    if not eligible_gpus:
        return {"error": "No suitable GPU found."}

    # 3. Recommend the cheapest (lowest VRAM) GPU
    eligible_gpus.sort(key=lambda x: x.get("VRAM (GB)", 9999))
    recommended = eligible_gpus[0]
    alternatives = eligible_gpus[1:]

    return {
        "recommended": recommended,
        "alternatives": alternatives,
        "required_vram": kv_cache_gb,
        "kv_cache_fp16_gb": kv_cache_gb,
        "users": users,
        "latency": latency,
    }