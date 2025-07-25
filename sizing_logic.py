import math
from fastapi import HTTPException

def estimate_gpu_requirement(model, users: int, latency: int, gpu_catalog: list):
    base_latency = model.Base_Latency_s
    required_latency = latency / 1000  # ms → seconds

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    # Calculate concurrency per GPU
    parallelism = max(1, math.floor(required_latency / base_latency))
    concurrent_per_gpu = parallelism
    required_gpus = math.ceil(users / concurrent_per_gpu)

    # Step 1: Filter GPUs that meet minimum VRAM requirement
    suitable_gpus = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= model.VRAM_Required_GB
    ]

    if not suitable_gpus:
        return {"recommendation": None, "alternatives": []}

    # Step 2: Filter for single-GPU solutions first (only if enough VRAM per GPU)
    single_gpu_candidates = [
        gpu for gpu in suitable_gpus
        if model.VRAM_Required_GB <= gpu["VRAM (GB)"]
    ]

    if single_gpu_candidates:
        sorted_single = sorted(single_gpu_candidates, key=lambda g: g["VRAM (GB)"])
        best_gpu = sorted_single[0]
        alternatives = sorted_single[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": required_gpus,
                "gpu_memory": required_gpus * model.VRAM_Required_GB
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": required_gpus,
                    "gpu_memory": required_gpus * model.VRAM_Required_GB
                } for gpu in alternatives
            ]
        }

    # Step 3: Fall back to multi-GPU only if NVLink is supported
    nvlink_multi_gpu_candidates = [
        gpu for gpu in suitable_gpus
        if gpu.get("NVLink", False) and required_gpus <= int(gpu.get("Max NVLink GPUs", "1"))
    ]

    if nvlink_multi_gpu_candidates:
        sorted_nvlink = sorted(nvlink_multi_gpu_candidates, key=lambda g: g["VRAM (GB)"])
        best_gpu = sorted_nvlink[0]
        alternatives = sorted_nvlink[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": required_gpus,
                "gpu_memory": required_gpus * model.VRAM_Required_GB
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": required_gpus,
                    "gpu_memory": required_gpus * model.VRAM_Required_GB
                } for gpu in alternatives
            ]
        }

    # Step 4: No viable solution
    return {"recommendation": None, "alternatives": []}