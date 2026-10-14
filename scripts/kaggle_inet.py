#!/usr/bin/env python3
"""
CLI Helper to manage Kaggle MMIF-INet training runs across all 3 tasks:
  - Task 1: Baseline reproduction (PET-MRI, SPECT-MRI, standard color loss)
  - Task 2: Job 1 with CT (PET-MRI, SPECT-MRI, CT-MRI, ZERO color loss)
  - Task 3: Job 2 with CT (PET-MRI, SPECT-MRI, CT-MRI, conditional color loss)

Commands:
  - status: Check kernel status across all 3 tasks & GPU quota
  - logs:   Stream logs for a specific job
  - push:   Push and start training on Kaggle Dual Tesla T4 GPUs
  - pull:   Download checkpoints and visual results
  - local:  Monitor local training jobs & GPU
"""
import os
import sys
import glob
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_FILE = os.path.join(PROJECT_ROOT, '.env')
VENV_KAGGLE = os.path.join(PROJECT_ROOT, '.venv_main', 'bin', 'kaggle')

JOBS = {
    'task1': {
        'name': 'Task 1: Baseline (PET+SPECT, Paper loss)',
        'slug': 'cyclerlol/mmif-inet-task-1-baseline',
        'dir': os.path.join(PROJECT_ROOT, 'kaggle_jobs', 'job1_baseline'),
        'file': 'train_inet_baseline.ipynb'
    },
    'task2': {
        'name': 'Task 2: Job 1 with CT (PET+SPECT+CT, ZERO color loss)',
        'slug': 'cyclerlol/mmif-inet-task-2-ct-no-color',
        'dir': os.path.join(PROJECT_ROOT, 'kaggle_jobs', 'job2_ct_no_color'),
        'file': 'train_inet_job1_with_ct.ipynb'
    },
    'task3': {
        'name': 'Task 3: Job 2 with CT (PET+SPECT+CT, Conditional color loss)',
        'slug': 'cyclerlol/mmif-inet-task-3-ct-conditional-color',
        'dir': os.path.join(PROJECT_ROOT, 'kaggle_jobs', 'job3_ct_conditional_color'),
        'file': 'train_inet_job2_with_ct.ipynb'
    },
    'master': {
        'name': 'Master Notebook (Distributed 2x T4)',
        'slug': 'cyclerlol/mmif-inet-distributed-training-2x-t4-gpu',
        'dir': os.path.join(PROJECT_ROOT, 'kaggle_kernel'),
        'file': 'train_inet_kaggle.ipynb'
    }
}

ALIASES = {
    'baseline': 'task1',
    'job1_baseline': 'task1',
    '1': 'task1',
    'ct_no_color': 'task2',
    'job1': 'task2',
    'job1_with_ct': 'task2',
    'job2_ct_no_color': 'task2',
    '2': 'task2',
    'ct_conditional': 'task3',
    'ct_conditional_color': 'task3',
    'job2': 'task3',
    'job2_with_ct': 'task3',
    'job3_ct_conditional_color': 'task3',
    '3': 'task3',
    'legacy': 'master'
}


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


def resolve_job_key(job_str):
    if not job_str:
        return 'all'
    clean = job_str.lower().strip()
    if clean in JOBS:
        return clean
    if clean in ALIASES:
        return ALIASES[clean]
    return clean


def cmd_status(args):
    target = resolve_job_key(args.job)
    print("=" * 65)
    print("KAGGLE KERNEL STATUS")
    print("=" * 65)
    
    if target == 'all':
        for key in ['task1', 'task2', 'task3']:
            j = JOBS[key]
            print(f"\n--> [{key.upper()}] {j['name']}")
            print(f"    Slug: {j['slug']}")
            run_cmd(["kernels", "status", j['slug']])
    elif target in JOBS:
        j = JOBS[target]
        print(f"\n--> [{target.upper()}] {j['name']}")
        print(f"    Slug: {j['slug']}")
        run_cmd(["kernels", "status", j['slug']])
    else:
        print(f"Unknown job '{target}'. Choose from: task1, task2, task3, master, all")

    print("\n" + "=" * 65)
    print("KAGGLE HARDWARE QUOTA:")
    print("=" * 65)
    run_cmd(["quota"])


def cmd_logs(args):
    target = resolve_job_key(args.job)
    if target == 'all' or target not in JOBS:
        target = 'task2'  # Default to job1_with_ct (task2)
    j = JOBS[target]
    print("=" * 65)
    print(f"STREAMING LOGS: [{target.upper()}] {j['name']}")
    print(f"Kernel Slug:   {j['slug']}")
    print("=" * 65)
    run_cmd(["kernels", "logs", j['slug']])


def cmd_push(args):
    target = resolve_job_key(args.job)
    if target == 'all':
        targets = ['task1', 'task2', 'task3']
    elif target in JOBS:
        targets = [target]
    else:
        print(f"Unknown job '{target}'. Choose from: task1, task2, task3, master, all")
        return

    for t in targets:
        j = JOBS[t]
        print("=" * 65)
        print(f"PUSHING [{t.upper()}] TO KAGGLE (2x Tesla T4 GPU)...")
        print(f"Title:     {j['name']}")
        print(f"Slug:      {j['slug']}")
        print(f"Directory: {j['dir']}")
        print("=" * 65)

        if not os.path.exists(j['dir']):
            print(f"Directory {j['dir']} does not exist! Running create_kaggle_notebook.py...")
            subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'create_kaggle_notebook.py')])

        # Special sync for master
        if t == 'master':
            src_nb = os.path.join(PROJECT_ROOT, 'notebooks', 'train_inet_kaggle.ipynb')
            dst_nb = os.path.join(j['dir'], 'train_inet_kaggle.ipynb')
            if os.path.exists(src_nb):
                subprocess.run(['cp', src_nb, dst_nb])

        run_cmd(["kernels", "push", "-p", j['dir'], "--accelerator", "NvidiaTeslaT4"])
        print()


def cmd_pull(args):
    target = resolve_job_key(args.job)
    base_out = args.output_dir or os.path.join(PROJECT_ROOT, 'results', 'kaggle_checkpoints')

    if target == 'all':
        targets = ['task1', 'task2', 'task3']
    elif target in JOBS:
        targets = [target]
    else:
        print(f"Unknown job '{target}'. Choose from: task1, task2, task3, master, all")
        return

    for t in targets:
        j = JOBS[t]
        out_dir = os.path.join(base_out, t) if len(targets) > 1 else base_out
        os.makedirs(out_dir, exist_ok=True)
        print("=" * 65)
        print(f"DOWNLOADING OUTPUTS FOR [{t.upper()}] TO: {out_dir}")
        print("=" * 65)
        run_cmd(["kernels", "output", j['slug'], "-p", out_dir])


def cmd_local_status(args):
    print("=" * 65)
    print("LOCAL INET PROCESS & GPU MONITOR:")
    print("=" * 65)
    subprocess.run("ps aux | grep -E 'train_job|MMIF' | grep -v grep", shell=True)
    print("\n--- GPU Utilization ---")
    subprocess.run("nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv", shell=True)
    print("\n--- Recent Local Log Tail ---")
    log_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'logging', 'train_job*.log')), key=os.path.getmtime)
    if log_files:
        print(f"Tailing: {log_files[-1]}")
        subprocess.run(['tail', '-n', '5', log_files[-1]])
    else:
        print("No local train_job logs found.")


def main():
    parser = argparse.ArgumentParser(description="MMIF-INet Multi-Task Kaggle Manager")
    subparsers = parser.add_subparsers(dest="action", help="Action to execute")

    # Status
    p_status = subparsers.add_parser("status", help="Check Kaggle kernel status & GPU quota")
    p_status.add_argument("-j", "--job", default="all", help="Job target: task1, task2, task3, master, or all (default)")

    # Logs
    p_logs = subparsers.add_parser("logs", help="View Kaggle execution logs")
    p_logs.add_argument("-j", "--job", default="task2", help="Job target: task1, task2 (default: job1_with_ct), task3, or master")

    # Push
    p_push = subparsers.add_parser("push", help="Push & launch training on Kaggle (2x Tesla T4)")
    p_push.add_argument("-j", "--job", default="task2", help="Job target: task1, task2 (job1_with_ct), task3, or all")

    # Pull
    p_pull = subparsers.add_parser("pull", help="Download trained checkpoints and plots from Kaggle")
    p_pull.add_argument("-j", "--job", default="all", help="Job target: task1, task2, task3, or all (default)")
    p_pull.add_argument("-o", "--output-dir", help="Directory to save downloaded files")

    # Local
    subparsers.add_parser("local", help="Check local training job status & GPU")

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
