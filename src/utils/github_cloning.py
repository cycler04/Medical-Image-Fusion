import json
import shutil
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("ref_repos")
OUTPUT_DIR.mkdir(exist_ok=True)

with open("ref_method_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()

failed = []

for paper in data:
    github_url = paper.get("url")

    if not github_url :
        print(f"Skipping {paper['stem']}, don't have link")
        continue

    github_url = github_url.strip()

    # skip duplicates
    if github_url in seen:
        continue
    seen.add(github_url)

    # repo name
    repo_name = github_url.rstrip("/").split("/")[-1]

    target_dir = OUTPUT_DIR / repo_name

    if target_dir.exists():
        print(f"[SKIP] {repo_name} already exists")
        continue

    print(f"[CLONE] {github_url}")

    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth", "1",
            github_url,
            str(target_dir)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[FAILED] {github_url}")
        print(result.stderr)

        failed.append({
            "url": github_url,
            "error": result.stderr
        })

        # remove broken partial folder automatically
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

print(f"\nFailed repos: {len(failed)}")