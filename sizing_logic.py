def estimate_kv_cache_gb(
    num_layers,
    num_attention_heads,
    hidden_size,
    seq_len=2048,
    dtype_bytes=2,  # FP16/BF16=2 bytes, FP32=4 bytes
    users=1,
):
    """
    Estimate the KV cache requirement in GB.
    """
    try:
        if not all([num_layers, num_attention_heads, hidden_size, seq_len]):
            return None
        head_dim = hidden_size // num_attention_heads
        kv_cache_bytes = num_layers * num_attention_heads * 2 * seq_len * head_dim * dtype_bytes * users
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        return round(kv_cache_gb, 3)
    except Exception as e:
        print(f"[ERROR] KV cache estimation failed: {e}")
        return None

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