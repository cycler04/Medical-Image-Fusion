#!/usr/bin/env python3
"""
find_fusion_papers.py
---------------------
Scans a folder of research paper PDFs and identifies image fusion papers
by detecting evaluation metrics commonly used in image fusion research.

A paper is classified as an image fusion paper if it uses at least 3 of
the known image fusion metrics.

Detected fusion papers are automatically copied to a separate output folder.

Usage:
    python find_fusion_papers.py
"""

import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_FOLDER  = Path("data/research_paper/new_papers")
OUTPUT_FOLDER  = Path("data/research_paper/fusion_papers")   # copies go here
RESULTS_FILE   = Path("filter_paper.txt")
MIN_METRICS    = 3

# ---------------------------------------------------------------------------
# Metric patterns
# ---------------------------------------------------------------------------
# Each entry: (canonical_name, [regex_patterns_to_match])
# Patterns are matched case-insensitively against extracted text.

FUSION_METRICS = [
    ("NMI",    [r"\bNMI\b"]),
    ("SSIM",    [r"\bSSIM\b", r"\bstructural\s+similarity\b"]),
    ("VIF",     [r"\bVIF\b", r"\bvisual\s+information\s+fidelity\b"]),
    ("QAB/F",   [r"\bQ[_\-]?AB/?F\b", r"\bQABF\b"]),
    ("QCB",     [r"\bQ[_\-]?CB\b", r"\bQCB\b"]),
    ("QCV",     [r"\bQ[_\-]?CV\b"]),
    ("QE",      [r"\bQ[_\-]?E\b(?!\w)"]),
    ("QG",      [r"\bQ[_\-]?G\b(?!\w)"]),
    ("QM",      [r"\bQ[_\-]?M\b(?!\w)"]),
    ("QP",      [r"\bQ[_\-]?P\b(?!\w)"]),
    ("QS",      [r"\bQ[_\-]?S\b(?!\w)"]),
    ("QW",      [r"\bQ[_\-]?W\b(?!\w)"]),
    ("Q0",      [r"\bQ[_\-]?0\b"]),
    ("PSNR",    [r"\bPSNR\b", r"\bpeak\s+signal.to.noise\b"]),
    ("SD",      [r"\bSD\b", r"\bstandard\s+deviation\b"]),
    ("SF",      [r"\bSF\b", r"\bspatial\s+frequency\b"]),
    ("AG",      [r"\bAG\b", r"\baverage\s+gradient\b"]),
    ("FMI",     [r"\bFMI\b", r"\bfeature\s+mutual\s+information\b"]),
    ("NCIE",    [r"\bNCIE\b"]),
    ("NABF",    [r"\bNABF\b"]),
    ("SCD",     [r"\bSCD\b"]),
    ("MS-SSIM", [r"\bMS.SSIM\b", r"\bmulti.scale\s+SSIM\b"]),
    ("VIFF",    [r"\bVIFF\b"]),
    ("VIFP",    [r"\bVIFP\b"]),
]

# Compile all patterns once
COMPILED_METRICS = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for name, patterns in FUSION_METRICS
]

# Any paper containing at least one of these metrics is immediately classified
# as a fusion paper, regardless of MIN_METRICS.
STRONG_METRICS = {"SSIM", "QG", "QP", "QW", "MS-SSIM", "QAB/F", "VIFP", "U2Fusion"}


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber, with pypdf as fallback."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        print(f"  [!] Could not extract text from {pdf_path.name}: {e}")

    return text


# ---------------------------------------------------------------------------
# Metric detection
# ---------------------------------------------------------------------------

def detect_metrics(text: str) -> list[str]:
    """Return a sorted list of metric names found in the text."""
    found = []
    for name, patterns in COMPILED_METRICS:
        if any(p.search(text) for p in patterns):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def copy_paper(src: Path, dest_folder: Path) -> Path | None:
    """
    Copy a single PDF into *dest_folder* immediately.
    Handles name collisions by appending a numeric suffix.
    Returns the destination path, or None on failure.
    """
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / src.name

    counter = 1
    while dest.exists():
        dest = dest_folder / f"{src.stem}({counter}){src.suffix}"
        counter += 1

    try:
        shutil.copy2(src, dest)
        return dest
    except Exception as e:
        print(f"  [!] Failed to copy {src.name}: {e}")
        return None


def scan_folder(folder: Path, dest_folder: Path, min_metrics: int = MIN_METRICS) -> list[dict]:
    """
    Scan all PDFs in *folder* recursively.
    - Skips files already present in *dest_folder* (rerun-safe).
    - A paper is fusion if it contains any STRONG_METRICS, OR >= min_metrics total.
    - Fusion papers are copied to *dest_folder* immediately upon detection.
    Returns a list of dicts for all papers that passed the fusion filter.
    """
    pdf_files = sorted(folder.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {folder}")
        return []

    # Build set of filenames already in the output folder for fast lookup
    already_copied = {p.name for p in dest_folder.glob("*.pdf")} if dest_folder.exists() else set()

    print(f"Found {len(pdf_files)} PDF file(s). Scanning ΓÇª\n")

    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}", end=" ΓÇª ", flush=True)

        # Skip files already in the output folder
        if pdf_path.name in already_copied:
            print("(already copied, skipped)")
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print("(no text extracted, skipped)")
            continue

        metrics = detect_metrics(text)
        has_strong  = bool(STRONG_METRICS & set(metrics))
        is_fusion   = has_strong or len(metrics) >= min_metrics

        if is_fusion:
            reason = f"strong metric" if has_strong else f"{len(metrics)} metrics"
            dest = copy_paper(pdf_path.resolve(), dest_folder)
            copy_status = f"  ΓåÆ  copied to '{dest.name}'" if dest else "  ΓåÆ  copy FAILED"
            print(f"Γ£ô FUSION  [{', '.join(metrics)}]  ({reason}){copy_status}")
        elif metrics:
            print(
                f"Γ£ù  [{', '.join(metrics)}]"
                f"  (only {len(metrics)}, need {min_metrics})"
            )
        else:
            print("Γ£ù  (no fusion metrics found)")

        if is_fusion:
            results.append({
                "path": pdf_path.resolve(),
                "name": pdf_path.name,
                "metrics": metrics,
                "metric_count": len(metrics),
            })

    return results


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------

def write_output(results: list[dict], output_path: Path) -> None:
    """Write fusion paper paths (and detected metrics) to a text file."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Image Fusion Papers\n")
        f.write(f"# Total: {len(results)} paper(s)\n")
        f.write("#\n")
        f.write("# Format: <path>  |  metrics: <list>\n")
        f.write("#" + "-" * 78 + "\n\n")
        for r in results:
            f.write(f"{r['path']}\n")
            f.write(
                f"  metrics ({r['metric_count']}): {', '.join(r['metrics'])}\n\n"
            )
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SOURCE_FOLDER.is_dir():
        print(f"Error: '{SOURCE_FOLDER}' is not a valid directory.")
        sys.exit(1)

    # Ensure required libraries are available
    for lib in ("pdfplumber", "pypdf"):
        try:
            __import__(lib)
        except ImportError:
            print(f"Installing {lib} ΓÇª")
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", lib, "--quiet",
                 "--break-system-packages"]
            )

    results = scan_folder(SOURCE_FOLDER, dest_folder=OUTPUT_FOLDER, min_metrics=MIN_METRICS)

    print(f"\n{'='*60}")
    print(f"  Image fusion papers found: {len(results)}")
    print(f"{'='*60}\n")

    if results:
        write_output(results, RESULTS_FILE)
        print("\nDetected papers:")
        for r in results:
            print(f"  ΓÇó {r['name']}")
            print(f"    metrics: {', '.join(r['metrics'])}")
    else:
        print("No image fusion papers detected with the current settings.")
        print(f"Try lowering MIN_METRICS (currently {MIN_METRICS}).")


if __name__ == "__main__":
    main()
