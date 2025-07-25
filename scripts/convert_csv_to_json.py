import pandas as pd
import os

def convert(csv_filename, json_filename):
    df = pd.read_csv(csv_filename)
    df.to_json(json_filename, orient="records", indent=2)
    print(f"✅ Converted {csv_filename} to {json_filename}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    convert("data/model_catalog.csv", "data/model_catalog.json")
    convert("data/gpu_catalog.csv", "data/gpu_catalog.json")import pandas as pd
import os

def convert(csv_filename, json_filename):
    df = pd.read_csv(csv_filename)
    df.to_json(json_filename, orient="records", indent=2)
    print(f"✅ Converted {csv_filename} to {json_filename}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    convert("data/model_catalog.csv", "data/model_catalog.json")
    convert("data/gpu_catalog.csv", "data/gpu_catalog.json")