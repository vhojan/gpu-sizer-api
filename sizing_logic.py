import math
from typing import Dict, List

def estimate_gpu_requirement(model: Dict, users: int, latency: float, gpu_catalog: List[Dict]) -> Dict:
    model_name = model.get("Model", "Unknown")
    try:
        base_latency = model.get("Base_Latency_s")
        base_vram = model.get("Base_VRAM_GB")
        tps = model.get("Tokens_per_second")

        if base_latency is None or base_vram is None or tps is None:
            raise ValueError(f"Missing VRAM or TPS data for model '{model_name}'")

        # Estimate required VRAM and TFLOPs
        required_vram = base_vram * users
        required_tps = tps * users

        # Allow some overhead buffer
        required_vram *= 1.2
        required_tps *= 1.2

        # Filter GPUs that meet VRAM and TPS requirement
        suitable_gpus = []
        for gpu in gpu_catalog:
            try:
                gpu_name = gpu["Name"]
                gpu_vram = float(gpu["VRAM_GB"])
                gpu_tps = float(gpu["Tokens_per_second"])

                if gpu_vram >= required_vram and gpu_tps >= required_tps:
                    suitable_gpus.append({
                        "GPU": gpu_name,
                        "Count": 1,
                        "VRAM_GB": gpu_vram,
                        "Tokens_per_second": gpu_tps,
                    })
                elif "NVLink" in gpu.get("Features", []) and gpu_vram * 2 >= required_vram and gpu_tps * 2 >= required_tps:
                    suitable_gpus.append({
                        "GPU": gpu_name,
                        "Count": 2,
                        "VRAM_GB": gpu_vram,
                        "Tokens_per_second": gpu_tps,
                        "Note": "Uses NVLink"
                    })
            except Exception as e:
                print(f"Skipping GPU due to error: {e}")

        if not suitable_gpus:
            raise ValueError(f"No suitable GPU found for model {model_name}")

        # Sort by GPU count first, then VRAM ascending
        suitable_gpus.sort(key=lambda x: (x["Count"], x["VRAM_GB"]))

        return {
            "model": model_name,
            "required_vram_gb": round(required_vram, 2),
            "required_tokens_per_second": round(required_tps, 2),
            "recommendation": suitable_gpus[0],
            "alternatives": suitable_gpus[1:],
        }

    except Exception as e:
        raise RuntimeError(f"Error in estimate_gpu_requirement: {str(e)}")