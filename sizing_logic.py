from typing import Dict, List, Optional


def estimate_gpu_requirement(model: Dict, users: int, latency: int, gpu_catalog: List[Dict]) -> Dict:
    try:
        vram_per_user = model.get("VRAM (GB) per user", 0)
        tps_required = model.get("Tokens per second per user", 0)
        base_latency = model.get("Base Latency (s)", 1)

        if not vram_per_user or not tps_required:
            raise ValueError("Missing VRAM or TPS values in model")

        total_vram_required = vram_per_user * users
        total_tps_required = tps_required * users

        matching_gpus = []

        for gpu in gpu_catalog:
            gpu_name = gpu["Name"]
            gpu_vram = gpu.get("VRAM (GB)", 0)
            gpu_tps = gpu.get("Tokens per second", 0)

            if not gpu_tps or not gpu_vram:
                continue

            # How many GPUs needed to meet TPS and VRAM demand
            gpu_count_tps = (total_tps_required / gpu_tps)
            gpu_count_vram = (total_vram_required / gpu_vram)
            gpu_count = max(gpu_count_tps, gpu_count_vram)

            # Round up GPU count
            quantity = int(gpu_count) + (0 if gpu_count.is_integer() else 1)

            matching_gpus.append({
                "gpu": gpu_name,
                "quantity": quantity,
                "gpu_memory": gpu_vram
            })

        # Sort by total GPU memory (quantity * vram), then by quantity
        matching_gpus.sort(key=lambda g: (g["quantity"] * g["gpu_memory"], g["quantity"]))

        if matching_gpus:
            return {
                "recommendation": matching_gpus[0],
                "alternatives": matching_gpus[1:5]
            }
        else:
            return {"recommendation": None, "alternatives": []}

    except Exception as e:
        print(f"Error in estimate_gpu_requirement: {e}")
        raise