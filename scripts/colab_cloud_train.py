#!/usr/bin/env python3
"""
colab_cloud_train.py: End-to-End Cloud Orchestrator for Colab GPU Training via GitHub Actions.

Features:
1. Provisions Colab GPU (Tesla T4) directly in the cloud.
2. Clones Harvard-AANLIB dataset and syncs repository code.
3. Launches dual parallel training (g16_l4 and g32_l3) for N epochs.
4. Maintains active WebSocket keep-alive connection from GitHub Actions.
5. Continuously uploads checkpoints to private Kaggle dataset.
6. Automatically cleans up GPU session upon completion.
"""

import os
import sys
import time
import json
import base64
import tarfile
import io
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

SESSION_NAME = "gha-colab"
DEFAULT_KAGGLE_TOKEN = "KGAT_04de0289207c7e9829974cd62511fda0"


def run_cmd(cmd_list, capture=True, check=False) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd_list, capture_output=True, text=True, check=check)
    return subprocess.run(cmd_list, check=check)


def run_colab_exec(session_name: str, python_code: str, timeout: float = 30.0) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(python_code)
        f_name = f.name

    try:
        cmd = ["colab", "exec", "-s", session_name, "-f", f_name, "--timeout", str(timeout)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)


def ensure_gpu_session(session_name: str) -> str:
    print(f"[*] Checking active Colab sessions on server...")
    res = run_cmd(["colab", "sessions"])
    print(res.stdout)

    # Check if session_name exists locally
    check_code = f"print('Session {session_name} probe')"
    code, out, _ = run_colab_exec(session_name, check_code, timeout=10.0)
    if code == 0:
        print(f"[+] Reusing existing active session: '{session_name}'")
        return session_name

    print(f"[*] Provisioning fresh GPU session: '{session_name}' (Tesla T4)...")
    create_res = run_cmd(["colab", "new", "-s", session_name, "--gpu", "T4"])
    print(create_res.stdout)
    if create_res.returncode != 0:
        print(f"[-] Error creating session: {create_res.stderr}")
        # Try without name or check assignments
        sys.exit(1)

    print(f"[+] Successfully provisioned GPU session '{session_name}'!")
    return session_name


def prepare_code_payload() -> str:
    """Packages local core code and ablation scripts into base64 tar archive."""
    buf = io.BytesIO()
    project_root = Path(__file__).resolve().parent.parent

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for base in ["experiments/inet_core", "experiments/inet_ablation/01_cnn_architectures/01_dense_residual", "scripts"]:
            base_path = project_root / base
            if not base_path.exists():
                continue
            for root, dirs, files in os.walk(base_path):
                if any(x in root for x in ["runs", "__pycache__", "results", "fused_images"]):
                    continue
                for f in files:
                    p = Path(root) / f
                    rel_p = p.relative_to(project_root)
                    tar.add(str(p), arcname=str(rel_p))

    return base64.b64encode(buf.getvalue()).decode("ascii")


def setup_remote_environment(session_name: str, kaggle_token: str):
    print("[*] Setting up remote Colab VM (cloning dataset, repo, dependencies)...")

    # 1. Clone dataset and repo if missing
    clone_script = """
import os, subprocess

def run_sh(cmd):
    print(">>>", cmd)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

os.makedirs("/content/dataset", exist_ok=True)
if not os.path.exists("/content/dataset/Harvard-AANLIB"):
    run_sh("git clone --depth 1 https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets.git /content/dataset/Harvard-AANLIB")

if not os.path.exists("/content/Medical-Image-Fusion"):
    run_sh("git clone https://github.com/cycler04/Medical-Image-Fusion.git /content/Medical-Image-Fusion")

# Ensure kornia and kaggle
run_sh("pip install kornia kaggle")
"""
    code, out, err = run_colab_exec(session_name, clone_script, timeout=180.0)
    print(out)
    if code != 0:
        print("[-] Remote setup failed:", err)
        sys.exit(1)

    # 2. Transfer code payload
    b64_payload = prepare_code_payload()
    print(f"[*] Transferring updated code payload ({len(b64_payload)} chars)...")

    unpack_script = f"""
import base64, io, tarfile, os
b64 = \"\"\"{b64_payload}\"\"\"
data = base64.b64decode(b64)
buf = io.BytesIO(data)
os.makedirs("/content/Medical-Image-Fusion", exist_ok=True)
with tarfile.open(fileobj=buf, mode="r:gz") as tar:
    tar.extractall("/content/Medical-Image-Fusion")
print("Code payload unpacked successfully.")
"""
    code, out, err = run_colab_exec(session_name, unpack_script, timeout=45.0)
    print(out)
    if code != 0:
        print("[-] Code unpack failed:", err)
        sys.exit(1)

    # 3. Configure Kaggle backup script on VM
    backup_script = f"""
import os
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/access_token", "w") as f:
    f.write("{kaggle_token}")
"""
    run_colab_exec(session_name, backup_script, timeout=15.0)
    print("[+] Remote environment is fully configured!")


def launch_training(session_name: str, epochs: int, val_freq: int, batch_size: int, kaggle_token: str):
    print(f"[*] Launching dual parallel training ({epochs} epochs, batch {batch_size}, val_freq {val_freq})...")
    
    launch_code = f"""
import subprocess, time, os

os.chmod("/content/Medical-Image-Fusion/experiments/inet_ablation/01_cnn_architectures/01_dense_residual/run_dual_dense_colab.sh", 0o755)

# 1. Launch dual training
res = subprocess.run(["bash", "/content/Medical-Image-Fusion/experiments/inet_ablation/01_cnn_architectures/01_dense_residual/run_dual_dense_colab.sh", "{epochs}", "{val_freq}", "{batch_size}"], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

time.sleep(3)

# 2. Launch Kaggle backup daemon
cmd = "nohup python3 /content/Medical-Image-Fusion/scripts/colab_kaggle_backup.py > /content/kaggle_backup.log 2>&1 &"
subprocess.run(cmd, shell=True)
print("Dual training and Kaggle backup daemon launched successfully.")
"""
    code, out, err = run_colab_exec(session_name, launch_code, timeout=60.0)
    print(out)
    if code != 0:
        print("[-] Launch failed:", err)
        sys.exit(1)


def monitor_and_keepalive(session_name: str, max_hours: float, interval_sec: int = 30):
    start_time = time.time()
    max_duration_sec = max_hours * 3600
    cycle = 0

    print("=" * 75)
    print("🚀 DUAL TRAINING ACTIVE — COMMENCING CLOUD KEEPALIVE & LOG STREAM")
    print("=" * 75)

    probe_code = """
import os, json, subprocess
from pathlib import Path

def run_sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

gpu_line = run_sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
proc_out = run_sh("ps -ef | grep train.py | grep -v grep")
procs = [l for l in proc_out.splitlines() if l.strip()]

runs_dir = Path("/content/Medical-Image-Fusion/experiments/inet_ablation/01_cnn_architectures/01_dense_residual/runs")
g16_tail = run_sh(f"tail -n 1 {runs_dir}/run_g16_l4.log") if (runs_dir / "run_g16_l4.log").exists() else "No log"
g32_tail = run_sh(f"tail -n 1 {runs_dir}/run_g32_l3.log") if (runs_dir / "run_g32_l3.log").exists() else "No log"

print("PROBE_JSON_START")
print(json.dumps({
    "gpu": gpu_line,
    "active_trainers": len(procs),
    "g16": g16_tail,
    "g32": g32_tail
}))
print("PROBE_JSON_END")
"""

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_duration_sec:
            print(f"\n[+] Maximum run duration ({max_hours:.1f}h) reached. Checkpoints are safely on Kaggle.")
            break

        cycle += 1
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        code, stdout, stderr = run_colab_exec(session_name, probe_code, timeout=25.0)
        if code == 0 and "PROBE_JSON_START" in stdout and "PROBE_JSON_END" in stdout:
            json_str = stdout.split("PROBE_JSON_START")[1].split("PROBE_JSON_END")[0].strip()
            try:
                data = json.loads(json_str)
                gpu_str = data.get("gpu", "N/A")
                active_t = data.get("active_trainers", 0)
                g16_str = data.get("g16", "")
                g32_str = data.get("g32", "")

                print(f"[{now_str} | Cycle {cycle:04d}] GPU: {gpu_str} | Active Trainers: {active_t}")
                if "Epoch [" in g16_str:
                    print(f"         ├─ g16_l4: {g16_str.split('INFO: ')[-1]}")
                if "Epoch [" in g32_str:
                    print(f"         └─ g32_l3: {g32_str.split('INFO: ')[-1]}")

                if active_t == 0 and cycle > 4:
                    print("\n🎉 Training completed for all epochs!")
                    break
            except Exception as e:
                print(f"[{now_str}] JSON Parse warning: {e}")
        else:
            print(f"[{now_str}] Probe warning: {stderr or stdout}")

        time.sleep(interval_sec)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Cloud Colab Trainer")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--val-freq", type=int, default=50, help="Validation frequency")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--duration-hours", type=float, default=5.5, help="Keep-alive max duration in hours")
    parser.add_argument("--kaggle-token", type=str, default="", help="Kaggle API token")
    args = parser.parse_args()

    token = args.kaggle_token or os.environ.get("KAGGLE_API_TOKEN") or DEFAULT_KAGGLE_TOKEN

    session = ensure_gpu_session(SESSION_NAME)
    setup_remote_environment(session, token)
    launch_training(session, args.epochs, args.val_freq, args.batch_size, token)
    monitor_and_keepalive(session, args.duration_hours)

    print("[*] Releasing GPU session...")
    run_cmd(["colab", "stop", "-s", session])
    print("[+] Done! All checkpoints backed up to Kaggle.")


if __name__ == "__main__":
    main()
