#!/usr/bin/env python3
"""
Colab Experiment Manager for MMIF-INet
======================================
Automated deployment, monitoring, log inspection, and weight retrieval for
Google Colab GPU runtimes, mirroring the Kaggle workflow (kaggle_manager.py).

Commands:
  # Check active sessions on Google Colab:
  python scripts/colab_manager.py status

  # Push and train an experiment on Colab GPU:
  python scripts/colab_manager.py push --exp 05_simple_cnn --gpu T4 --epochs 10

  # Check logs of a running session:
  python scripts/colab_manager.py log --session mmif-colab-05-simple-cnn

  # Download checkpoints and results from Colab into local runs/:
  python scripts/colab_manager.py download --exp 05_simple_cnn

  # Release the GPU session:
  python scripts/colab_manager.py stop --session mmif-colab-05-simple-cnn
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.inet_core.config import BaseConfig
from experiments.inet_core.colab_notebook_creator import (
    ColabNotebookCreator,
    find_experiment_dir
)


def run_cmd(cmd_list, capture=True, check=False):
    """Executes a command and returns completed process."""
    if capture:
        return subprocess.run(cmd_list, capture_output=True, text=True, check=check)
    return subprocess.run(cmd_list, check=check)


def get_active_sessions():
    """Returns a list of active Colab sessions."""
    res = run_cmd(["colab", "sessions"])
    if res.returncode != 0:
        return []
    lines = res.stdout.strip().splitlines()
    sessions = []
    for line in lines:
        if line.startswith("- ") or line.startswith("* "):
            parts = line.strip().split()
            if len(parts) >= 2:
                sessions.append(parts[1])
    return sessions


def check_status():
    """Prints status of active Colab sessions and account usage."""
    print("=== Colab Account Status ===")
    res_who = run_cmd(["colab", "whoami"])
    print(res_who.stdout.strip() if res_who.returncode == 0 else res_who.stderr.strip())

    print("\n=== Active Sessions ===")
    res_sess = run_cmd(["colab", "sessions"])
    print(res_sess.stdout.strip() if res_sess.returncode == 0 else "No active sessions found.")


def push_experiment(exp_target: str, gpu: str = "T4", epochs: int = None, batch_size: int = None, session_name: str = None, async_exec: bool = False):
    """Generates notebook, provisions Colab GPU session, and launches training."""
    exp_dir = Path(find_experiment_dir(exp_target))
    if not exp_dir.exists():
        raise FileNotFoundError(f"Could not find experiment folder for: {exp_target}")

    exp_clean = exp_dir.name.replace('_', '-')
    if not session_name:
        session_name = f"colab-{exp_clean}"

    print(f"[*] Preparing Colab deployment for: {exp_dir.name}")

    # 1. Generate train_colab.ipynb
    cfg = BaseConfig()
    if epochs is not None:
        cfg.epochs = epochs
    if batch_size is not None:
        cfg.batch_size = batch_size
    cfg.lr_decay = False

    creator = ColabNotebookCreator(config=cfg)
    nb_path = creator.generate_for_experiment(str(exp_dir), config=cfg, output_file="train_colab.ipynb")
    print(f"[+] Generated Colab notebook: {nb_path}")

    # 2. Provision or verify GPU session
    active = get_active_sessions()
    if session_name not in active:
        print(f"[*] Allocating fresh Colab session '{session_name}' with GPU: {gpu}...")
        res_new = run_cmd(["colab", "new", "-s", session_name, "--gpu", gpu])
        if res_new.returncode != 0:
            print(f"[-] Failed to provision GPU session:\n{res_new.stderr}")
            if "quota" in res_new.stderr.lower() or "400" in res_new.stderr:
                print("[-] Quota limit or accelerator unavailable. Trying without accelerator or check tier.")
            return False
        print(f"[+] Session '{session_name}' allocated successfully.")
    else:
        print(f"[i] Using existing active session '{session_name}'.")

    # 3. Execute notebook on Colab GPU
    print(f"[*] Dispatching notebook execution on Colab GPU ({session_name})...")
    exec_cmd = ["colab", "exec", "-s", session_name, "--timeout", "7200", "-f", nb_path]

    if async_exec:
        proc = subprocess.Popen(exec_cmd)
        print(f"[+] Execution launched in background (PID: {proc.pid}).")
        print(f"    Monitor with: python scripts/colab_manager.py log --session {session_name}")
        return True
    else:
        print("[*] Running training (streaming logs)...")
        res_exec = run_cmd(exec_cmd, capture=False)
        return res_exec.returncode == 0


def view_logs(session_name: str):
    """Displays execution history and logs for a session."""
    run_cmd(["colab", "log", "-s", session_name], capture=False)


def download_run(exp_target: str, session_name: str = None):
    """Downloads outputs and checkpoints from Colab session into local runs directory."""
    exp_dir = Path(find_experiment_dir(exp_target))
    if not session_name:
        session_name = f"colab-{exp_dir.name.replace('_', '-')}"

    local_runs_dir = exp_dir / "runs"
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Packaging and downloading runs from Colab session '{session_name}' into {local_runs_dir}...")
    
    # 1. Package /content/runs on Colab VM into a tar.gz archive
    archive_script = "/tmp/pack_archive.py"
    with open(archive_script, "w") as f:
        f.write("import os\nos.system('tar -czf /content/runs_archive.tar.gz -C /content runs')\n")
    run_cmd(["colab", "exec", "-s", session_name, "--timeout", "120", "-f", archive_script])

    # 2. Download the archive
    local_archive = "/tmp/colab_runs_archive.tar.gz"
    res = run_cmd(["colab", "download", "-s", session_name, "/content/runs_archive.tar.gz", local_archive])
    if res.returncode == 0 and os.path.exists(local_archive):
        # 3. Extract into local_runs_dir
        import tarfile
        with tarfile.open(local_archive, "r:gz") as tar:
            tar.extractall(path=local_runs_dir.parent)
        print(f"[+] Download and extraction complete: {local_runs_dir}")
        if os.path.exists(local_archive):
            os.remove(local_archive)
    else:
        print(f"[-] Download returned error:\n{res.stderr}")


def download_kaggle_backup(dataset_slug="cyclerlol/med-image-fusion-colab-checkpoints", dest_dir=None):
    """Downloads all backed up runs from private Kaggle dataset into local experiment runs/."""
    default_dest = Path(_project_root) / "experiments" / "inet_ablation" / "01_cnn_architectures" / "01_dense_residual" / "runs"
    dest = Path(dest_dir) if dest_dir else default_dest
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading latest cloud backup from Kaggle dataset: {dataset_slug}...")
    tmp_dl = Path("/tmp/kaggle_dl_temp")
    if tmp_dl.exists():
        shutil.rmtree(tmp_dl)
    tmp_dl.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi
    # Ensure env auth
    if not os.environ.get("KAGGLE_API_TOKEN"):
        env_path = Path(_project_root) / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("KAGGLE_API_KEY_LIST="):
                        val = line.split("=", 1)[1].split(",")[0].strip()
                        os.environ["KAGGLE_API_TOKEN"] = val
                        os.environ["KAGGLE_API_KEY"] = val
                        break

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(tmp_dl), unzip=True, quiet=False)

    # If unpacked contains 'runs/', copy contents
    unpacked_runs = tmp_dl / "runs"
    src = unpacked_runs if unpacked_runs.exists() else tmp_dl
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    shutil.rmtree(tmp_dl, ignore_errors=True)
    print(f"Checkpoints and logs successfully downloaded and restored to:\n  {dest}")


def trigger_gha_keepalive(session_name="sess2", duration_hours="5.5"):
    """Dispatches the GitHub Actions Keep-Alive workflow via GitHub REST API."""
    import requests
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        env_file = Path(_project_root) / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    if not token:
        print("[-] GITHUB_TOKEN not found in environment or .env file.")
        return False

    url = "https://api.github.com/repos/cycler04/Medical-Image-Fusion/actions/workflows/colab_runner.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main",
        "inputs": {
            "session_name": session_name,
            "duration_hours": str(duration_hours)
        }
    }
    print(f"[*] Triggering GitHub Actions Keep-Alive for session '{session_name}' ({duration_hours}h)...")
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 204:
        print("[+] SUCCESS! GitHub Actions cloud runner dispatched and active.")
        print("    URL: https://github.com/cycler04/Medical-Image-Fusion/actions")
        return True
    else:
        print(f"[-] Failed to trigger workflow ({res.status_code}): {res.text}")
        return False


def stop_session(session_name: str):
    """Stops and releases a Colab session."""
    print(f"[*] Stopping session '{session_name}'...")
    res = run_cmd(["colab", "stop", "-s", session_name])
    if res.returncode == 0:
        print(f"[+] Session '{session_name}' stopped.")
    else:
        print(f"[-] Error stopping session: {res.stderr}")


def main():
    parser = argparse.ArgumentParser(description="MMIF-INet Colab Experiment Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Status
    subparsers.add_parser("status", help="Check Colab account & active sessions")

    # Push / Deploy
    push_p = subparsers.add_parser("push", help="Deploy experiment to Colab GPU")
    push_p.add_argument("--exp", type=str, required=True, help="Experiment name (e.g. 05_simple_cnn)")
    push_p.add_argument("--gpu", type=str, default="T4", choices=["T4", "L4", "A100"], help="GPU variant (default: T4)")
    push_p.add_argument("--epochs", type=int, default=None, help="Epoch count")
    push_p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    push_p.add_argument("--session", type=str, default=None, help="Custom session name")
    push_p.add_argument("--async", dest="async_exec", action="store_true", help="Launch in background")

    # Log
    log_p = subparsers.add_parser("log", help="View logs of a Colab session")
    log_p.add_argument("--session", type=str, required=True, help="Session name")

    # Download
    dl_p = subparsers.add_parser("download", help="Download results from Colab into local experiment runs/")
    dl_p.add_argument("--exp", type=str, required=True, help="Experiment name")
    dl_p.add_argument("--session", type=str, default=None, help="Session name")

    # Download from Kaggle Backup
    dl_k_p = subparsers.add_parser("download-backup", help="Download checkpoints from Kaggle cloud backup")
    dl_k_p.add_argument("--dataset", type=str, default="cyclerlol/med-image-fusion-colab-checkpoints", help="Dataset slug")
    dl_k_p.add_argument("--dest", type=str, default=None, help="Destination directory (default: current experiment runs/)")

    # Trigger GitHub Actions Keep-Alive
    gha_p = subparsers.add_parser("trigger-keepalive", help="Automatically dispatch GitHub Actions keep-alive runner")
    gha_p.add_argument("--session", type=str, default="sess2", help="Colab session name")
    gha_p.add_argument("--duration", type=float, default=5.5, help="Duration in hours")

    # Stop
    stop_p = subparsers.add_parser("stop", help="Stop and release a Colab session")
    stop_p.add_argument("--session", type=str, required=True, help="Session name")

    args = parser.parse_args()

    if args.command == "status":
        check_status()
    elif args.command == "push":
        push_experiment(
            exp_target=args.exp,
            gpu=args.gpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            session_name=args.session,
            async_exec=args.async_exec
        )
    elif args.command == "log":
        view_logs(args.session)
    elif args.command == "download":
        download_run(args.exp, session_name=args.session)
    elif args.command == "download-backup":
        download_kaggle_backup(dataset_slug=args.dataset, dest_dir=args.dest)
    elif args.command == "trigger-keepalive":
        trigger_gha_keepalive(session_name=args.session, duration_hours=args.duration)
    elif args.command == "stop":
        stop_session(args.session)
    else:
        check_status()


if __name__ == '__main__':
    main()
