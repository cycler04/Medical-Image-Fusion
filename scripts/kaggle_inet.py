#!/usr/bin/env python3
"""
CLI Helper to manage Kaggle MMIF-INet training runs:
- Check kernel status & weekly GPU quota
- Stream Kaggle execution logs
- Push / restart Kaggle training jobs
- Download / pull checkpoints and results
- Monitor local training job
"""
import os
import sys
import glob
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_FILE = os.path.join(PROJECT_ROOT, '.env')
KERNEL_SLUG = "cyclerlol/mmif-inet-distributed-training-2x-t4-gpu"
KERNEL_DIR = os.path.join(PROJECT_ROOT, 'kaggle_kernel')
VENV_KAGGLE = os.path.join(PROJECT_ROOT, '.venv_main', 'bin', 'kaggle')


def get_kaggle_cmd():
    if os.path.exists(VENV_KAGGLE):
        return VENV_KAGGLE
    return "kaggle"


def get_env():
    env = os.environ.copy()
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if k == 'KAGGLE_API_KEY':
                        env['KAGGLE_API_TOKEN'] = v
                        env['KAGGLE_API_KEY'] = v
                    else:
                        env[k] = v
    return env


def run_cmd(args):
    cmd = [get_kaggle_cmd()] + args
    res = subprocess.run(cmd, env=get_env(), text=True)
    return res.returncode


def cmd_status(args):
    print("=" * 60)
    print(f"KAGGLE KERNEL STATUS: {KERNEL_SLUG}")
    print("=" * 60)
    run_cmd(["kernels", "status", KERNEL_SLUG])
    print("\n" + "=" * 60)
    print("KAGGLE HARDWARE QUOTA:")
    print("=" * 60)
    run_cmd(["quota"])


def cmd_logs(args):
    print("=" * 60)
    print(f"STREAMING LOGS: {KERNEL_SLUG}")
    print("=" * 60)
    run_cmd(["kernels", "logs", KERNEL_SLUG])


def cmd_push(args):
    print("=" * 60)
    print(f"PUSHING NOTEBOOK TO KAGGLE WITH T4 GPU ACCELERATOR...")
    print("=" * 60)
    # Sync notebook to kaggle_kernel folder
    src_nb = os.path.join(PROJECT_ROOT, 'notebooks', 'train_inet_kaggle.ipynb')
    dst_nb = os.path.join(KERNEL_DIR, 'train_inet_kaggle.ipynb')
    if os.path.exists(src_nb):
        subprocess.run(['cp', src_nb, dst_nb])
    run_cmd(["kernels", "push", "-p", KERNEL_DIR, "--accelerator", "NvidiaTeslaT4"])


def cmd_pull(args):
    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, 'results', 'kaggle_checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(f"DOWNLOADING KERNEL OUTPUTS TO: {out_dir}")
    print("=" * 60)
    run_cmd(["kernels", "output", KERNEL_SLUG, "-p", out_dir])


def cmd_local_status(args):
    print("=" * 60)
    print("LOCAL INET PROCESS & GPU MONITOR:")
    print("=" * 60)
    subprocess.run("ps aux | grep -E 'train_job|MMIF' | grep -v grep", shell=True)
    print("\n--- GPU Utilization ---")
    subprocess.run("nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv", shell=True)
    print("\n--- Recent Local Log Tail ---")
    log_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'logging', 'train_job2_with_ct_*.log')), key=os.path.getmtime)
    if log_files:
        subprocess.run(['tail', '-n', '5', log_files[-1]])
    else:
        print("No local train_job2 logs found.")


def main():
    import glob
    parser = argparse.ArgumentParser(description="MMIF-INet Kaggle Training Manager")
    subparsers = parser.add_subparsers(dest="action", help="Action to execute")

    subparsers.add_parser("status", help="Check Kaggle kernel status & quota")
    subparsers.add_parser("logs", help="View Kaggle execution logs")
    subparsers.add_parser("push", help="Push & launch training on Kaggle T4 GPU")
    p_pull = subparsers.add_parser("pull", help="Download trained checkpoints from Kaggle")
    p_pull.add_argument("-o", "--output-dir", help="Directory to save downloaded files")
    subparsers.add_parser("local", help="Check local training job status")

    args = parser.parse_args()
    if not args.action:
        parser.print_help()
        sys.exit(0)

    if args.action == "status":
        cmd_status(args)
    elif args.action == "logs":
        cmd_logs(args)
    elif args.action == "push":
        cmd_push(args)
    elif args.action == "pull":
        cmd_pull(args)
    elif args.action == "local":
        cmd_local_status(args)


if __name__ == '__main__':
    main()
