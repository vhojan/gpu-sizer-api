from typing import Dict, List


def estimate_gpu_requirement(model: Dict, users: int, latency: int, gpu_catalog: List[Dict]) -> Dict:
    try:
        vram_per_user = model.get("VRAM (GB) per user")
        tps_per_user = model.get("Tokens per second per user")
        model_name = model.get("Model")

        if vram_per_user is None or tps_per_user is None:
            raise ValueError(f"Missing VRAM or TPS data for model '{model_name}'")

        total_vram = users * vram_per_user
        total_tps = users * tps_per_user

        print(f"[INFO] Model: {model_name}, Users: {users}, Latency: {latency}")
        print(f"[INFO] Required VRAM: {total_vram} GB, Required TPS: {total_tps}")

        matches = []

        for gpu in gpu_catalog:
            gpu_name = gpu.get("Name")
            gpu_vram = gpu.get("VRAM (GB)", 0)
            gpu_tps = gpu.get("Tokens per second", 0)

            if not gpu_name or gpu_vram <= 0 or gpu_tps <= 0:
                print(f"[WARN] Skipping GPU due to missing VRAM/TPS: {gpu_name}")
                continue

            quantity_vram = total_vram / gpu_vram
            quantity_tps = total_tps / gpu_tps
            quantity = max(quantity_vram, quantity_tps)
            quantity = int(quantity) + (0 if quantity.is_integer() else 1)

            matches.append({
                "gpu": gpu_name,
                "quantity": quantity,
                "gpu_memory": gpu_vram
            })

        if not matches:
            print(f"[WARN] No suitable GPUs found for {model_name}")
            return {"recommendation": None, "alternatives": []}

        matches.sort(key=lambda g: (g["quantity"] * g["gpu_memory"], g["quantity"]))
        return {
            "recommendation": matches[0],
            "alternatives": matches[1:5]
        }

    except Exception as e:
        print(f"[ERROR] in estimate_gpu_requirement: {e}")
        raise