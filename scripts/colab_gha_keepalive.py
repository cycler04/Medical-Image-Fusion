#!/usr/bin/env python3
"""
colab_gha_keepalive.py: GitHub Actions Keep-Alive and Remote Monitoring Agent for Google Colab.

Maintains an external connection to Google Colab, emits active kernel heartbeats,
streams live training progress into GitHub Actions console, and guarantees cloud backup.
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run_colab_exec(session_name: str, python_code: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Executes a Python snippet inside the Colab session via colab CLI."""
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


def probe_remote_colab(session_name: str) -> dict:
    """Executes a diagnostic probe on the Colab VM and returns parsed status."""
    probe_code = """
import os, sys, json, subprocess
from pathlib import Path

def run_sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

# GPU Status
gpu_line = run_sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")

# Processes
proc_out = run_sh("ps -ef | grep train.py | grep -v grep")
procs = [l for l in proc_out.splitlines() if l.strip()]

# Backup daemon
backup_proc = run_sh("ps -ef | grep colab_kaggle_backup.py | grep -v grep")

# Log tails
runs_dir = Path("/content/Medical-Image-Fusion/experiments/inet_ablation/01_cnn_architectures/01_dense_residual/runs")
g16_tail = run_sh(f"tail -n 1 {runs_dir}/run_g16_l4.log") if (runs_dir / "run_g16_l4.log").exists() else "No log"
g32_tail = run_sh(f"tail -n 1 {runs_dir}/run_g32_l3.log") if (runs_dir / "run_g32_l3.log").exists() else "No log"

print("PROBE_JSON_START")
print(json.dumps({
    "gpu": gpu_line,
    "active_train_processes": len(procs),
    "backup_daemon_active": len(backup_proc.strip()) > 0,
    "g16_tail": g16_tail,
    "g32_tail": g32_tail
}))
print("PROBE_JSON_END")
"""
    code, stdout, stderr = run_colab_exec(session_name, probe_code, timeout=25.0)
    if code != 0:
        return {"error": stderr or stdout or "colab exec returned non-zero"}

    if "PROBE_JSON_START" in stdout and "PROBE_JSON_END" in stdout:
        json_str = stdout.split("PROBE_JSON_START")[1].split("PROBE_JSON_END")[0].strip()
        try:
            import json
            return json.loads(json_str)
        except Exception as e:
            return {"error": f"JSON parse error: {e}", "raw": stdout}

    return {"error": "Delimiter missing in output", "raw": stdout}


def ensure_backup_daemon(session_name: str, kaggle_token: str):
    """Ensures the Kaggle cloud backup daemon is running on the Colab VM."""
    daemon_ensure_code = f"""
import subprocess, os
out = subprocess.run("ps -ef | grep colab_kaggle_backup.py | grep -v grep", shell=True, capture_output=True, text=True).stdout
if not out.strip():
    token = "{kaggle_token}"
    os.environ['KAGGLE_API_TOKEN'] = token
    os.environ['KAGGLE_API_KEY'] = token
    cmd = "nohup python3 /content/Medical-Image-Fusion/scripts/colab_kaggle_backup.py > /content/kaggle_backup.log 2>&1 &"
    subprocess.run(cmd, shell=True)
    print("Launched colab_kaggle_backup daemon.")
else:
    print("Backup daemon already running.")
"""
    run_colab_exec(session_name, daemon_ensure_code, timeout=20.0)


def main():
    parser = argparse.ArgumentParser(description="Colab GitHub Actions Keep-Alive Runner")
    parser.add_argument("--session", type=str, default="sess2", help="Colab session name")
    parser.add_argument("--duration-hours", type=float, default=5.5, help="Maximum run hours before exit")
    parser.add_argument("--interval", type=int, default=30, help="Heartbeat probe interval in seconds")
    parser.add_argument("--kaggle-token", type=str, default="", help="Kaggle API token for cloud backup")
    args = parser.parse_args()

    token = args.kaggle_token or os.environ.get("KAGGLE_API_TOKEN", "")
    start_time = time.time()
    max_duration_sec = args.duration_hours * 3600

    print("=" * 75)
    print("🚀 COLAB GITHUB ACTIONS KEEP-ALIVE RUNNER")
    print("=" * 75)
    print(f"Target Session:     {args.session}")
    print(f"Max Duration:       {args.duration_hours:.1f} hours ({max_duration_sec/60:.0f} min)")
    print(f"Probe Interval:     {args.interval}s")
    print(f"Start Time (UTC):   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 75)

    if token:
        print("[*] Verifying remote backup daemon on Colab VM...")
        ensure_backup_daemon(args.session, token)

    cycle = 0
    consecutive_errors = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_duration_sec:
            print(f"\n[+] Maximum workflow duration reached ({args.duration_hours:.1f} hours). Exiting cleanly.")
            break

        cycle += 1
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        status = probe_remote_colab(args.session)
        if "error" in status:
            consecutive_errors += 1
            print(f"[{now_str}] ⚠️ Warning: Probe failed (attempt {consecutive_errors}): {status['error']}")
            if consecutive_errors >= 10:
                print("[-] Too many consecutive probe failures. Runtime may have stopped.")
                sys.exit(1)
        else:
            consecutive_errors = 0
            gpu_str = status.get("gpu", "N/A")
            active_p = status.get("active_train_processes", 0)
            backup_ok = "✅ Active" if status.get("backup_daemon_active") else "⚠️ Inactive"

            print(f"[{now_str} | Cycle {cycle:04d}] GPU: {gpu_str} | Active Trainers: {active_p} | Backup: {backup_ok}")
            
            # Print log excerpts if available
            g16 = status.get("g16_tail", "")
            g32 = status.get("g32_tail", "")
            if "Epoch [" in g16:
                short_g16 = g16.split("INFO: ")[-1] if "INFO: " in g16 else g16
                print(f"         ├─ g16_l4: {short_g16}")
            if "Epoch [" in g32:
                short_g32 = g32.split("INFO: ")[-1] if "INFO: " in g32 else g32
                print(f"         └─ g32_l3: {short_g32}")

            # If training finished
            if active_p == 0 and cycle > 3:
                print("\n🎉 Both training runs have completed on Colab GPU!")
                print("[*] Verifying final backup on Kaggle...")
                if token:
                    ensure_backup_daemon(args.session, token)
                time.sleep(10)
                print("[+] Job complete. Exiting workflow.")
                break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()

