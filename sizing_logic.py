def estimate_gpu_requirement(model: dict, users: int, latency: float, gpu_catalog: list):
    try:
        required_vram = model.get("VRAM_GB")
        required_tps = model.get("Tokens_per_second")

        if required_vram is None or required_tps is None:
            raise ValueError(f"Missing VRAM or TPS data for model '{model.get('Model', 'Unknown')}'")

        total_tps_required = users * (1000 / latency) * required_tps
        total_vram_required = users * required_vram

        suitable_gpus = []

        for gpu in gpu_catalog:
            gpu_name = gpu.get("Name")
            gpu_vram = gpu.get("Memory (GB)")
            gpu_tps = gpu.get("Tokens/sec")

            if gpu_vram is None or gpu_tps is None:
                continue

            if gpu_vram >= required_vram and gpu_tps > 0:
                gpu_count_vram = -(-total_vram_required // gpu_vram)
                gpu_count_tps = -(-total_tps_required // gpu_tps)
                quantity = int(max(gpu_count_vram, gpu_count_tps))
                suitable_gpus.append({
                    "gpu": gpu_name,
                    "quantity": quantity,
                    "gpu_memory": gpu_vram
                })

        if not suitable_gpus:
            raise ValueError(f"No suitable GPU found for model {model['Model']}")

        best = sorted(suitable_gpus, key=lambda x: (x['quantity'], x['gpu_memory']))[0]
        alternatives = [gpu for gpu in suitable_gpus if gpu != best]

        return best, alternatives

    except Exception as e:
        raise RuntimeError(f"Error in estimate_gpu_requirement: {e}")