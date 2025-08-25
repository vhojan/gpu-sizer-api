from typing import Optional

from typing import Optional

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