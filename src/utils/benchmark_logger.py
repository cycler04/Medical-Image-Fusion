import os
import csv
import time
from datetime import datetime
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset", "Harvard-AANLIB")
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
LOG_FILE = os.path.join(RESULTS_ROOT, "run_log.txt")

MODALITY_MAP = {
    "CT-MRI": "MRI-CT",
    "MRI-CT": "MRI-CT",
    "CT": "MRI-CT",
    "PET-MRI": "MRI-PET",
    "MRI-PET": "MRI-PET",
    "PET": "MRI-PET",
    "SPECT-MRI": "MRI-SPECT",
    "MRI-SPECT": "MRI-SPECT",
    "SPECT": "MRI-SPECT",
}

def get_modality_info(mod_name):
    clean = mod_name.replace("/", "").replace("\\", "").strip()
    mod_out = MODALITY_MAP.get(clean, clean)
    return mod_out, "time_metrics.csv"

def save_time_metrics(repo_name, mod_name, latency, gflop, parameters, vram, batch_size=1):
    mod_out, _ = get_modality_info(mod_name)
    repo_results_dir = os.path.join(RESULTS_ROOT, repo_name)
    os.makedirs(repo_results_dir, exist_ok=True)
    csv_path = os.path.join(repo_results_dir, "time_metrics.csv")
    
    rows = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "modality" in r:
                        rows[r["modality"]] = r
        except Exception:
            pass

    rows[mod_out] = {
        "modality": mod_out,
        "batch_size": batch_size,
        "latency": latency,
        "GFLOP": gflop,
        "parameters": parameters,
        "peak_VRAM": vram
    }

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "batch_size", "latency", "GFLOP", "parameters", "peak_VRAM"])
        for m in ["MRI-CT", "MRI-PET", "MRI-SPECT"]:
            if m in rows:
                writer.writerow([
                    rows[m]["modality"],
                    rows[m].get("batch_size", 1),
                    rows[m]["latency"],
                    rows[m]["GFLOP"],
                    rows[m]["parameters"],
                    rows[m].get("peak_VRAM", rows[m].get("VRAM"))
                ])
        for m, r in rows.items():
            if m not in ["MRI-CT", "MRI-PET", "MRI-SPECT"]:
                writer.writerow([
                    r["modality"],
                    r.get("batch_size", 1),
                    r["latency"],
                    r["GFLOP"],
                    r["parameters"],
                    r.get("peak_VRAM", r.get("VRAM"))
                ])
    return csv_path

def log_status(repo_name, status, message=""):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {repo_name}: {status.upper()} - {message}\n"
    with open(LOG_FILE, mode="a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())

def compute_gflops_and_params(model, sample_inputs):
    params = "N/A"
    gflops = "N/A"
    try:
        if isinstance(model, torch.nn.Module):
            params = sum(p.numel() for p in model.parameters())
    except Exception:
        pass

    try:
        from thop import profile
        flops, _ = profile(model, inputs=sample_inputs, verbose=False)
        gflops = round(flops / 1e9, 4)
    except Exception:
        pass
    return gflops, params
