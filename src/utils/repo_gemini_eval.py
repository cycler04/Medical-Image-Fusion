"""
analyze_repos.py
────────────────
Iterates a parent folder that contains sub-folders of GitHub repos.
For each repo it:
  1. Extracts the full file/folder tree
  2. Reads the README (any variant)
  3. Sends both to Gemini with a JSON-schema response to evaluate
     how easy the model/project is to run locally.

Skips repos that:
  • Are empty (no files at all)
  • Have no README

Rotates through multiple API keys (key-first) then models on failure.

Usage:
    pip install google-genai pydantic python-dotenv

    # .env  (or export directly)
    GOOGLE_API_KEY_LIST="key1 , key2 , key3"

    python analyze_repos.py /path/to/repos/folder
    python analyze_repos.py /path/to/repos/folder --reprocess        # ignore prior results
    python analyze_repos.py /path/to/repos/folder --output out.json  # custom output path
"""

import os
import sys
import json
import time
import pathlib
import argparse
from typing import Optional, List, Literal

import dotenv
from pydantic import BaseModel, Field
from google import genai

dotenv.load_dotenv()

# ──────────────────────────────────────────────
# 0.  Config
# ──────────────────────────────────────────────

MODEL_LIST = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]

_raw_keys = os.getenv("GOOGLE_API_KEY_LIST", "")
GOOGLE_API_KEY_LIST = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# Fallback: single key env var
if not GOOGLE_API_KEY_LIST:
    single = os.getenv("GEMINI_API_KEY", "")
    if single:
        GOOGLE_API_KEY_LIST = [single]

README_NAMES = {
    "readme.md", "readme.txt", "readme.rst", "readme",
    "readme.markdown", "readme.adoc", "readme.org",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
}


# ──────────────────────────────────────────────
# 1.  Pydantic schema — "runability" evaluation
# ──────────────────────────────────────────────

class DependencyInfo(BaseModel):
    name: str = Field(description="Dependency or framework name")
    version_pinned: bool = Field(description="Whether a specific version is pinned")

class SetupStep(BaseModel):
    order: int = Field(description="Step number")
    command: str = Field(description="Shell command or action required")
    description: str = Field(description="What this step does")

class UrlDetails(BaseModel):
    url: str = Field(description="the url")
    description: str = Field(description="Where it redirect too")


class RunabilityEvaluation(BaseModel):
    repo_name: str = Field(description="Name of the repository folder")

    overall_ease_score: int = Field(
        description="Overall ease-of-running score from 1 (very hard) to 10 (trivial)",
        ge=1, le=10,
    )
    ease_label: Literal["trivial", "easy", "moderate", "hard", "very hard"] = Field(
        description="Human-readable ease label"
    )
    has_lockfile: bool = Field(
        description="Whether a lock-file or pinned requirements file is present in the tree"
    )
    has_checkpoint: bool = Field(
        description="A checkpoint or pre-train weight exist, for test without retrain."
    )
    key_dependencies: List[DependencyInfo] = Field(
        description="Up to 8 noteworthy dependencies/frameworks"
    )
    linked_url: List[UrlDetails] = Field(
        default=None,
        description="List of url the Readme redirect to"
    )
    dataset_used: List[str] = Field(
        default=None,
        description="List of metioned dataset name used for training"
    )
    setup_steps: List[SetupStep] = Field(
        description="Ordered steps to get the project running (max 10)"
    )
    one_liner_possible: bool = Field(
        description="Can the project be started with a single command"
    )
    one_liner_command: Optional[str] = Field(
        default=None,
        description="The one-liner command if applicable"
    )
    readme_quality: Literal["excellent", "good", "minimal", "missing"] = Field(
        description="Quality/completeness of the README"
    )
    potential_blockers: List[str] = Field(
        description="List of things that could make running the project difficult"
    )
    summary: str = Field(
        description="2-3 sentence plain-English summary of how easy this repo is to run"
    )


# ──────────────────────────────────────────────
# 2.  Key / model rotation
# ──────────────────────────────────────────────

def rotate_to_next(key_idx: int, model_idx: int) -> tuple[int, int]:
    """
    Advance one step: exhaust all keys on the current model first,
    then advance the model index.
    """
    next_key = (key_idx + 1) % len(GOOGLE_API_KEY_LIST)
    if next_key != 0:
        # Still have more keys for the current model
        return next_key, model_idx
    else:
        # Wrapped around all keys — advance model
        next_model = (model_idx + 1) % len(MODEL_LIST)
        return 0, next_model


# ──────────────────────────────────────────────
# 3.  Repo helpers
# ──────────────────────────────────────────────

def repo_has_files(repo_path: pathlib.Path) -> bool:
    """Return True if the repo contains at least one non-.git file."""
    for entry in repo_path.rglob("*"):
        if entry.is_file() and ".git" not in entry.parts:
            return True
    return False


def find_readme(repo_path: pathlib.Path) -> Optional[str]:
    """Return text content of the first README found in repo root, or None."""
    for child in repo_path.iterdir():
        if child.is_file() and child.name.lower() in README_NAMES:
            try:
                return child.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None
    return None


def build_tree(root: pathlib.Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> str:
    """Return an indented tree string, similar to the `tree` command."""
    if depth > max_depth:
        return prefix + "..."

    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return prefix + "[permission denied]"

    entries = [e for e in entries if e.name not in SKIP_DIRS]

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            sub = build_tree(entry, prefix + extension, max_depth, depth + 1)
            if sub:
                lines.append(sub)

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 4.  Gemini call with key/model rotation
# ──────────────────────────────────────────────

def analyze_repo(
    repo_path: pathlib.Path,
    key_idx: int,
    model_idx: int,
) -> tuple[RunabilityEvaluation, int, int]:
    """
    Try to analyze *repo_path* starting from the given key/model indices.
    On failure, rotates key-first then model, up to max_rotations attempts.
    Returns (evaluation, final_key_idx, final_model_idx).
    """
    tree_text = build_tree(repo_path)
    readme_text = find_readme(repo_path) or "(No README found)"

    if len(readme_text) > 8000:
        readme_text = readme_text[:8000] + "\n\n[... README truncated ...]"

    prompt = (
        "You are a developer-experience evaluator. Analyze the repository below and fill in "
        "the JSON schema completely and accurately.\n\n"
        f"Repository name: {repo_path.name}\n\n"
        "=== FILE / FOLDER TREE ===\n"
        f"{tree_text}\n\n"
        "=== README CONTENT ===\n"
        f"{readme_text}\n\n"
        "Evaluate how easy it is for a new developer to clone this repo and get it running "
        "locally. Be precise; base your answers on evidence from the tree and README."
    )

    max_rotations = len(GOOGLE_API_KEY_LIST) * len(MODEL_LIST)
    current_key_idx = key_idx
    current_model_idx = model_idx

    for attempt in range(1, max_rotations + 2):
        current_model = MODEL_LIST[current_model_idx]
        current_key = GOOGLE_API_KEY_LIST[current_key_idx]

        print(f"   Attempt {attempt}: model={current_model}, key_idx={current_key_idx}")
        try:
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RunabilityEvaluation,
                    "temperature": 0.2,
                },
            )
            evaluation = RunabilityEvaluation.model_validate_json(response.text)
            return evaluation, current_key_idx, current_model_idx

        except Exception as e:
            print(f"   ✗ Attempt {attempt} failed: {e}")

            if attempt > max_rotations:
                raise RuntimeError(
                    f"Aborted after {max_rotations} rotation attempts. Last error: {e}"
                )

            current_key_idx, current_model_idx = rotate_to_next(current_key_idx, current_model_idx)
            print(
                f"   → Rotating to key_idx={current_key_idx}, "
                f"model={MODEL_LIST[current_model_idx]}"
            )
            time.sleep(1)


# ──────────────────────────────────────────────
# 5.  Main batch loop
# ──────────────────────────────────────────────

def process_all_repos(
    repos_root: pathlib.Path,
    output_path: pathlib.Path,
    skip_existing: bool = True,
) -> list[dict]:

    repo_dirs = sorted([p for p in repos_root.iterdir() if p.is_dir()])
    print(f"Found {len(repo_dirs)} sub-folder(s) in '{repos_root}'\n{'─'*60}")

    all_results: list[dict] = []

    # Load prior results for skip-existing
    if skip_existing and output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s) from '{output_path}'.\n")

    processed_names = {r["repo_name"] for r in all_results} if skip_existing else set()

    skipped_empty: list[str] = []
    skipped_no_readme: list[str] = []
    failed: list[str] = []

    # Start from the first key/model; rotate forward for every new repo
    key_idx = 0
    model_idx = 0

    for idx, repo in enumerate(repo_dirs, start=1):
        print(f"\n[{idx}/{len(repo_dirs)}] 📁  {repo.name}")

        # ── Skip: already processed ──────────────────────────────────────
        if skip_existing and repo.name in processed_names:
            print("   ⏭  Skipping (already processed)")
            continue

        # ── Skip: empty repo ─────────────────────────────────────────────
        if not repo_has_files(repo):
            print("   ⚠  Skipping — folder is empty (no files found)")
            skipped_empty.append(repo.name)
            continue

        # ── Skip: no README ──────────────────────────────────────────────
        if find_readme(repo) is None:
            print("   ⚠  Skipping — no README found")
            skipped_no_readme.append(repo.name)
            continue

        # ── Analyze with Gemini ───────────────────────────────────────────
        try:
            evaluation, key_idx, model_idx = analyze_repo(repo, key_idx, model_idx)
            result = {
                "repo_name": evaluation.repo_name,
                "model_used": MODEL_LIST[model_idx],
                **evaluation.model_dump(),
            }
            all_results.append(result)

            score = evaluation.overall_ease_score
            bar = "█" * score + "░" * (10 - score)
            print(f"   ✓  Score : [{bar}] {score}/10  ({evaluation.ease_label})")
            print(f"      README: {evaluation.readme_quality}")

            if evaluation.potential_blockers:
                print(f"      ⚠  Blockers: {'; '.join(evaluation.potential_blockers[:3])}")
            print(f"      💬 {evaluation.summary}")

        except Exception as exc:
            print(f"   ❌ Failed after all rotation attempts: {exc}")
            failed.append(repo.name)
            all_results.append({"repo_name": repo.name, "error": str(exc)})

        # Rotate key/model for the next repo regardless of success/failure
        key_idx, model_idx = rotate_to_next(key_idx, model_idx)

        # Save incrementally after every repo
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Processed           : {len(all_results) - len(failed)} repo(s)")
    print(f"  Skipped (empty)     : {len(skipped_empty)}"
          + (f"  → {skipped_empty}" if skipped_empty else ""))
    print(f"  Skipped (no readme) : {len(skipped_no_readme)}"
          + (f"  → {skipped_no_readme}" if skipped_no_readme else ""))
    print(f"  Failed              : {len(failed)}"
          + (f"  → {failed}" if failed else ""))
    print(f"  Output              : {output_path}")

    return all_results


# ──────────────────────────────────────────────
# 6.  Entry point
# ──────────────────────────────────────────────

def main():

    if not GOOGLE_API_KEY_LIST:
        print(
            "Error: No API keys found.\n"
            "Set GOOGLE_API_KEY_LIST='key1 , key2' or GEMINI_API_KEY='key' "
            "in your .env file or environment."
        )
        sys.exit(1)

    repos_root = pathlib.Path("ref_repos").expanduser().resolve()
    if not repos_root.is_dir():
        print(f"Error: '{repos_root}' is not a directory.")
        sys.exit(1)

    out_path = pathlib.Path("runability_report.json")

    print(f"Keys loaded   : {len(GOOGLE_API_KEY_LIST)}")
    print(f"Models        : {MODEL_LIST}")
    print(f"Output        : {out_path}\n")

    process_all_repos(
        repos_root=repos_root,
        output_path=out_path,
        skip_existing=True,
    )


if __name__ == "__main__":
    main()