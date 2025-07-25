import math
from fastapi import HTTPException

def estimate_gpu_requirement(model, users: int, latency: int, gpu_catalog: list):
    base_latency = model.Base_Latency_s
    required_latency = latency / 1000  # Convert ms → seconds

    if required_latency < base_latency:
        raise HTTPException(status_code=400, detail="Requested latency is lower than model base latency.")

    # Calculate how many users a single GPU can serve within the latency budget
    concurrent_per_gpu = max(1, math.floor(required_latency / base_latency))
    required_gpus = math.ceil(users / concurrent_per_gpu)

    # Calculate VRAM needed per GPU (each user needs full context memory)
    vram_needed_per_gpu = concurrent_per_gpu * model.VRAM_Required_GB

    # Step 1: Filter GPUs with enough VRAM
    suitable_gpus = [
        gpu for gpu in gpu_catalog
        if gpu["VRAM (GB)"] >= vram_needed_per_gpu
    ]

    if not suitable_gpus:
        return {"recommendation": None, "alternatives": []}

    # Step 2: Prefer single GPU options
    single_gpu_candidates = [gpu for gpu in suitable_gpus if not gpu.get("NVLink", False) or required_gpus == 1]

    if single_gpu_candidates:
        sorted_single = sorted(single_gpu_candidates, key=lambda g: g["VRAM (GB)"])  # Cheapest VRAM-first
        best_gpu = sorted_single[0]
        alternatives = sorted_single[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": 1,
                "gpu_memory": vram_needed_per_gpu
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": 1,
                    "gpu_memory": vram_needed_per_gpu
                } for gpu in alternatives
            ]
        }

    # Step 3: Fall back to multi-GPU options with NVLink
    nvlink_candidates = [
        gpu for gpu in suitable_gpus
        if gpu.get("NVLink", False) and required_gpus <= int(gpu.get("Max NVLink GPUs", "1"))
    ]

    if nvlink_candidates:
        sorted_nvlink = sorted(nvlink_candidates, key=lambda g: g["VRAM (GB)"])  # Cheapest VRAM-first
        best_gpu = sorted_nvlink[0]
        alternatives = sorted_nvlink[1:5]
        return {
            "recommendation": {
                "gpu": best_gpu["GPU Type"],
                "quantity": required_gpus,
                "gpu_memory": vram_needed_per_gpu
            },
            "alternatives": [
                {
                    "gpu": gpu["GPU Type"],
                    "quantity": required_gpus,
                    "gpu_memory": vram_needed_per_gpu
                } for gpu in alternatives
            ]
        }

    # Step 4: No suitable GPU found
    return {"recommendation": None, "alternatives": []}