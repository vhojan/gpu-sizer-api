import math
from fastapi import HTTPException

def estimate_gpu_requirement(model, users: int, latency: int, gpu_catalog: list, kv_cache_enabled: bool = False):
    base_latency = model.Base_Latency_s
    required_latency = latency / 1000  # ms → seconds

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    # Adjust VRAM requirement based on KV cache
    kv_factor = 2 if kv_cache_enabled else 1
    vram_per_user = model.VRAM_Required_GB * kv_factor
    total_vram = users * vram_per_user

    # Calculate concurrency per GPU based on latency
    parallelism = max(1, math.floor(required_latency / base_latency))
    concurrent_per_gpu = parallelism
    required_gpus = math.ceil(users / concurrent_per_gpu)

    # Step 1: Filter GPUs that meet per-GPU VRAM requirement
    suitable_gpus = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= vram_per_user
    ]

    if not suitable_gpus:
        return {"recommendation": None, "alternatives": []}

    # Step 2: Single-GPU option if enough
    if required_gpus == 1:
        sorted_single = sorted(suitable_gpus, key=lambda g: (g["VRAM (GB)"], -g["TFLOPs (FP16)"]))
        best_gpu = sorted_single[0]
        alternatives = sorted_single[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": 1,
                "gpu_memory": vram_per_user
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": 1,
                    "gpu_memory": vram_per_user
                } for gpu in alternatives
            ]
        }

    # Step 3: Multi-GPU fallback with NVLink
    nvlink_candidates = [
        gpu for gpu in suitable_gpus
        if gpu.get("NVLink", False) and required_gpus <= int(gpu.get("Max NVLink GPUs", "1"))
    ]

    if nvlink_candidates:
        sorted_nvlink = sorted(nvlink_candidates, key=lambda g: (g["VRAM (GB)"], -g["TFLOPs (FP16)"]))
        best_gpu = sorted_nvlink[0]
        alternatives = sorted_nvlink[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": required_gpus,
                "gpu_memory": vram_per_user
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": required_gpus,
                    "gpu_memory": vram_per_user
                } for gpu in alternatives
            ]
        }

    return {"recommendation": None, "alternatives": []}