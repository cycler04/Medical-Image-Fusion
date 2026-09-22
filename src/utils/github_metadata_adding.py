import json
import time
import requests
from pathlib import Path

INPUT_JSON = "ref_method_data.json"
OUTPUT_JSON = "ref_method_data_with_repo_metadata.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()

for i, paper in enumerate(data):
    github_url = paper.get("url")

    if not github_url:
        print(f"[{i}] Skipping: no GitHub URL")
        continue

    github_url = github_url.strip()

    # skip duplicate URLs
    if github_url in seen:
        continue
    seen.add(github_url)

    try:
        parts = github_url.rstrip("/").split("/")

        # validate github repo url
        if len(parts) < 5 or "github.com" not in github_url:
            print(f"[INVALID] {github_url}")
            continue

        owner = parts[-2]
        repo = parts[-1]

        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        response = requests.get(api_url)

        if response.status_code != 200:
            print(f"[FAILED] {repo}: HTTP {response.status_code}")
            continue

        repo_info = response.json()

        # -------------------------
        # Add metadata fields
        # -------------------------
        paper["repo_metadata"] = {
            "full_name": repo_info.get("full_name"),
            "description": repo_info.get("description"),

            "created_at": repo_info.get("created_at"),
            "updated_at": repo_info.get("updated_at"),
            "pushed_at": repo_info.get("pushed_at"),

            "language": repo_info.get("language"),
            "default_branch": repo_info.get("default_branch"),

            "stars": repo_info.get("stargazers_count"),
            "forks": repo_info.get("forks_count"),
            "watchers": repo_info.get("watchers_count"),
            "open_issues": repo_info.get("open_issues_count"),

            "archived": repo_info.get("archived"),
            "disabled": repo_info.get("disabled"),

            "has_issues": repo_info.get("has_issues"),
            "has_wiki": repo_info.get("has_wiki"),
            "has_projects": repo_info.get("has_projects"),

            "size_kb": repo_info.get("size"),

            "license": (
                repo_info.get("license", {}).get("spdx_id")
                if repo_info.get("license")
                else None
            ),

            "topics": repo_info.get("topics", []),
        }

        print(
            f"[OK] {repo} | "
            f"stars={paper['repo_metadata']['stars']} | "
            f"created={paper['repo_metadata']['created_at']}"
        )

        # avoid GitHub API rate spikes
        time.sleep(0.2)

    except Exception as e:
        print(f"[ERROR] {github_url}")
        print(e)

# save enriched json
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nSaved updated metadata to: {OUTPUT_JSON}")