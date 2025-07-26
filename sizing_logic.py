def estimate_gpu_requirement(model: dict, users: int, latency: int, gpu_catalog: list):
    try:
        base_latency = model["Base Latency (s)"]
        base_vram = model["VRAM Required (GB)"]
        kv_cache_per_user = model.get("KV Cache (GB per user)", 0)

        if base_latency is None or base_vram is None:
            raise ValueError("Missing VRAM or latency info in model catalog")

        required_vram = base_vram + kv_cache_per_user * users
        required_latency = latency / 1000.0  # convert to seconds

        print(f"[INFO] Base VRAM: {base_vram} GB")
        print(f"[INFO] KV Cache total: {kv_cache_per_user * users} GB")
        print(f"[INFO] Required VRAM: {required_vram} GB for {users} users")
        print(f"[INFO] Latency target: {required_latency}s")

        suitable_gpus = []
        for gpu in gpu_catalog:
            try:
                name = gpu.get("Name")
                vram = gpu.get("Memory (GB)")
                tps = gpu.get("Tokens/sec")

                if not all([name, vram, tps]):
                    continue  # Skip incomplete entries

                if vram >= required_vram:
                    # Estimate how many users this GPU can handle within latency
                    # E.g., latency * tps = token budget
                    # Assume 100 tokens per user as a simplification
                    supported_users = int((tps * required_latency) / 100)

                    if supported_users >= users:
                        suitable_gpus.append({
                            "gpu": name,
                            "gpu_memory": vram,
                            "quantity": 1
                        })
                    else:
                        quantity = int(users / supported_users) + 1
                        suitable_gpus.append({
                            "gpu": name,
                            "gpu_memory": vram,
                            "quantity": quantity
                        })
            except Exception as gpu_err:
                print(f"[WARNING] Skipping GPU due to error: {gpu_err}")

        suitable_gpus.sort(key=lambda g: (g["quantity"], g["gpu_memory"]))

        if not suitable_gpus:
            raise ValueError(f"No suitable GPU found for model {model['Model']}")

        best = suitable_gpus[0]
        alternatives = suitable_gpus[1:3]

        return best, alternatives

    except Exception as e:
        print(f"[ERROR] in estimate_gpu_requirement: {e}")
        raise