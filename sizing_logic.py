# sizing_logic.py

def estimate_gpu_requirement(model: dict, users: int, latency_target_ms: float, gpu_catalog: list) -> dict:
    try:
        # Validate fields
        required_fields = ["kv_cache_fp16_gb"]
        for field in required_fields:
            if field not in model:
                raise ValueError(f"Missing required field '{field}' in model '{model.get('model_id', '')}'")

        # For compatibility with your new backend/DB fields:
        model_vram = model.get("vram_required_gb") or model.get("VRAM Required (GB)") or 0
        kv_cache_per_user = model.get("kv_cache_fp16_gb") or model.get("KV Cache (GB per user)") or 0
        base_latency_s = model.get("base_latency_s") or model.get("Base Latency (s)") or 1.0

        total_vram_required = float(model_vram) + (float(kv_cache_per_user) * users)
        latency_target_s = float(latency_target_ms) / 1000.0

        candidates = []

        for gpu in gpu_catalog:
            gpu_type = gpu["GPU Type"]
            gpu_vram = gpu["VRAM (GB)"]
            gpu_tflops = gpu["TFLOPs (FP16)"]
            latency_factor = gpu.get("Latency Factor", 1)
            tokens_per_second = gpu.get("Tokens/s", 0)
            supports_nvlink = gpu.get("NVLink", False)
            max_nvlink = int(gpu.get("Max NVLink GPUs", 1)) if str(gpu.get("Max NVLink GPUs", "1")).isdigit() else 1

            adjusted_latency = base_latency_s * latency_factor
            meets_latency = adjusted_latency <= latency_target_s

            if meets_latency:
                # Try to fit model on single GPU
                if gpu_vram >= total_vram_required:
                    candidates.append({
                        "GPU Type": gpu_type,
                        "Config": "1x",
                        "Total VRAM (GB)": gpu_vram,
                        "Total TFLOPs": gpu_tflops,
                        "Latency (s)": round(adjusted_latency, 3),
                        "Meets Latency": True,
                        "Tokens/s": tokens_per_second
                    })
                # Try multi-GPU with NVLink
                elif supports_nvlink:
                    for num_gpus in range(2, max_nvlink + 1):
                        combined_vram = gpu_vram * num_gpus
                        if combined_vram >= total_vram_required:
                            candidates.append({
                                "GPU Type": gpu_type,
                                "Config": f"{num_gpus}x NVLink",
                                "Total VRAM (GB)": combined_vram,
                                "Total TFLOPs": gpu_tflops * num_gpus,
                                "Latency (s)": round(adjusted_latency, 3),
                                "Meets Latency": True,
                                "Tokens/s": tokens_per_second * num_gpus
                            })
                            break  # Stop after first fitting NVLink combo

        if not candidates:
            return {"error": "No suitable GPU configuration found"}

        sorted_candidates = sorted(candidates, key=lambda x: (x["Total VRAM (GB)"], x["Latency (s)"]))
        return {
            "model": model.get("model_id") or model.get("Model"),
            "users": users,
            "latency_target_ms": latency_target_ms,
            "total_vram_required_gb": total_vram_required,
            "recommended": sorted_candidates[0],
            "alternatives": sorted_candidates[1:3]  # up to 2 alternatives
        }

    except Exception as e:
        raise RuntimeError(f"Error in estimate_gpu_requirement: {str(e)}")

# You might also need a get_gpu_recommendation wrapper if your main.py expects it:
def get_gpu_recommendation(model: dict, users: int, latency: float, gpu_catalog: list):
    return estimate_gpu_requirement(model, users, latency, gpu_catalog)