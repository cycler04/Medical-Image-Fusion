"""
verify_paper_fields.py
──────────────────────
Reads the extracted-info JSON produced by the paper-reading pipeline and lets
you choose a subset of fields (including nested sub-fields) to verify with a
Gemini model by re-reading the original PDF.

Usage
-----
    python verify_paper_fields.py

Configuration is done via the CONFIG block at the top of this file.

Output
------
A new JSON file (output_verification_path) where every entry gains a
"verification" key that maps each verified field path to:
  {
    "is_correct":   bool | null,   # null = Gemini couldn't determine
    "confidence":   "high" | "medium" | "low",
    "original":     <the value from extraction>,
    "corrected":    <Gemini's corrected value, or null if is_correct=True>,
    "explanation":  "..."
  }
"""

import os
import json
import time
from pathlib import Path

import dotenv
from google import genai

dotenv.load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Where the extracted JSON lives
INPUT_JSON_PATH = "docs/extracted_info/all_papers_structured_raw_v2.json"

# Where PDFs are stored (script looks for <stem>.pdf here)
PDF_FOLDER = "docs/papers"

# Where to write verification results
OUTPUT_JSON_PATH = "docs/extracted_info/all_papers_verified.json"

# Skip papers that already have a "verification" key
SKIP_EXISTING = False

# Limit for testing (None = process all)
MAX_PAPERS = None

# ── FIELDS TO VERIFY ──────────────────────────────────────────────────────────
FIELDS_TO_VERIFY = [
    "proposed_method_detail.model_family",
    "proposed_method_detail.architecture_backbone",
]

# ── FIELD CONSTRAINTS ─────────────────────────────────────────────────────────
FIELD_CONSTRAINTS: dict[str, dict] = {
    "proposed_method_detail.model_family": {
        "enum": [
            "Traditional non-DL", "CNN", "U-Net", "Transformer",
            "AutoEncoder", "GAN", "Diffusion", "Mamba", "VLM",
        ],
        "description": (
            "Must be an ORDERED array of enum values. "
            "First element = primary family (the paper's main contribution). "
            "Remaining elements = supporting families in order of importance. "
            "Every element must be exactly one of the allowed enum values — "
            "Don't mistake custom module for a model family, like CNN with Attention enhance as Transfromer"
            "The paper should explicitly mention the models to be include here"
            "Examples: Diffusion model with Mamba encoder → ['Diffusion', 'Mamba']. "
            "Pure CNN → ['CNN']. "
            "A string instead of an array is WRONG; correct it to a single-element array."
        ),
    },
}

# ── API CONFIG ─────────────────────────────────────────────────────────────────
MODEL = "gemini-2.5-flash"
GOOGLE_API_KEY_LIST = [k.strip() for k in os.getenv("GOOGLE_API_KEY_LIST", "").split(",") if k.strip()]

# Simple round-robin key index (one key is fine too)
_key_idx = 0


def _get_client() -> tuple[genai.Client, int]:
    """Return (client, key_id) using round-robin. key_id is 1-based for logging."""
    global _key_idx
    kid = _key_idx % len(GOOGLE_API_KEY_LIST)
    key = GOOGLE_API_KEY_LIST[kid]
    _key_idx += 1
    return genai.Client(api_key=key), kid + 1  # 1-based


# ── VERIFICATION SCHEMA ────────────────────────────────────────────────────────
VERIFICATION_SCHEMA = {
    "type": "object",
    "description": "Verification results for each requested field",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "is_correct": {
                "type": ["boolean", "null"],
                "description": "True if the extracted value matches the paper; False if wrong/incomplete; null if undeterminable."
            },
            "corrected": {
                "description": "The correct value extracted from the paper (same type as original). null if is_correct=True.",
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of the verdict."
            }
        },
        "required": ["is_correct", "confidence", "corrected", "explanation"]
    }
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_nested(obj: dict, path: str):
    """Retrieve a value from a nested dict using dot-notation path."""
    parts = path.split(".")
    cur = obj
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def build_verification_prompt(entry: dict, fields: list[str]) -> str:
    """Build a prompt asking Gemini to verify the listed fields."""
    field_block = {}
    for f in fields:
        field_block[f] = get_nested(entry, f)

    constraint_lines: list[str] = []
    for f in fields:
        c = FIELD_CONSTRAINTS.get(f)
        if not c:
            continue
        lines = [f'### "{f}"']
        if "enum" in c:
            enum_str = ", ".join(f'"{v}"' for v in c["enum"])
            lines.append(
                f"STRICT ENUM — the corrected value MUST be exactly one of: {enum_str}. "
                "Any value not in this list is WRONG, even if it is semantically related. "
                "Never invent a new label; always map to the closest enum member."
            )
        if "description" in c:
            lines.append(c["description"])
        constraint_lines.append("\n".join(lines))

    constraints_section = ""
    if constraint_lines:
        constraints_section = (
            "\n## Per-field constraints (MUST be followed exactly)\n"
            + "\n\n".join(constraint_lines)
            + "\n"
        )

    prompt = (
        "You are a rigorous research-paper fact-checker.\n\n"
        "Below is structured information that was automatically extracted from the attached PDF paper. "
        "Your task is to verify whether each extracted value is correct, incomplete, or wrong "
        "by carefully reading the paper.\n\n"
        "## Extracted values to verify\n"
        f"```json\n{json.dumps(field_block, indent=2, ensure_ascii=False)}\n```\n"
        f"{constraints_section}\n"
        "## General instructions\n"
        "For every key in the JSON above, return a verification object with:\n"
        "  - is_correct: true if the value is fully correct per the paper AND satisfies any "
        "constraints above; false otherwise; null if the paper does not contain enough info\n"
        "  - confidence: 'high' | 'medium' | 'low'\n"
        "  - corrected: the correct value (same structure/type as the original), "
        "respecting any enum constraints; null if is_correct=true\n"
        "  - explanation: one or two sentences explaining your verdict\n\n"
        "Return ONLY a JSON object whose keys are the same dot-separated paths as above. "
        "No preamble, no markdown fences."
    )
    return prompt


def verify_paper(pdf_path: str, entry: dict, fields: list[str]) -> dict:
    """Upload PDF and ask Gemini to verify the chosen fields.

    Tries each API key in round-robin order. Falls back to the next key on any
    error (rate-limit, auth, network, etc.). Raises if all keys are exhausted.
    """
    prompt = build_verification_prompt(entry, fields)
    n_keys = len(GOOGLE_API_KEY_LIST)
    last_exc: Exception | None = None

    for attempt in range(n_keys):
        client, kid = _get_client()
        try:
            print(f"    [key {kid}/{n_keys}] Uploading PDF …")
            file = client.files.upload(file=pdf_path)

            print(f"    [key {kid}/{n_keys}] Sending verification request …")
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"file_data": {"mime_type": "application/pdf", "file_uri": file.uri}},
                        ],
                    }
                ],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0,
                },
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw.strip())
            print(f"    [key {kid}/{n_keys}] ✓ Success")
            return result

        except Exception as e:
            last_exc = e
            print(f"    [key {kid}/{n_keys}] ✗ Error: {e}")
            if attempt < n_keys - 1:
                print(f"    Falling back to next key …")
                time.sleep(1)

    raise RuntimeError(f"All {n_keys} API key(s) failed. Last error: {last_exc}") from last_exc


def enrich_verification(verification: dict, entry: dict, fields: list[str]) -> dict:
    """Attach the original value to each verification result for convenience."""
    enriched = {}
    for f in fields:
        v = verification.get(f, {})
        enriched[f] = {
            "original": get_nested(entry, f),
            "is_correct": v.get("is_correct"),
            "confidence": v.get("confidence", "low"),
            "corrected": v.get("corrected"),
            "explanation": v.get("explanation", ""),
        }
    return enriched


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        all_entries: list[dict] = json.load(f)
    print(f"Loaded {len(all_entries)} paper(s) from '{INPUT_JSON_PATH}'.")
    print(f"Using {len(GOOGLE_API_KEY_LIST)} API key(s).")

    verified_entries: list[dict] = []
    if SKIP_EXISTING and os.path.isfile(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
            verified_entries = json.load(f)
        print(f"Loaded {len(verified_entries)} already-verified paper(s).")

    verified_stems = {e["stem"] for e in verified_entries if "stem" in e}

    target = all_entries[:MAX_PAPERS] if MAX_PAPERS else all_entries
    failed: list[str] = []

    for idx, entry in enumerate(target, 1):
        stem = entry.get("stem", f"paper_{idx}")

        if SKIP_EXISTING and stem in verified_stems:
            print(f"[{idx}/{len(target)}] Skipping (already verified): {stem}")
            continue

        pdf_path = str(Path(PDF_FOLDER) / f"{stem}.pdf")
        if not os.path.isfile(pdf_path):
            print(f"[{idx}/{len(target)}] ✗ PDF not found, skipping: {pdf_path}")
            failed.append(stem)
            continue

        print(f"[{idx}/{len(target)}] Verifying: {stem}")
        try:
            raw_verification = verify_paper(pdf_path, entry, FIELDS_TO_VERIFY)
            enriched = enrich_verification(raw_verification, entry, FIELDS_TO_VERIFY)

            verified_entry = {**{"stem": stem}, "verification": enriched}
            verified_entries.append(verified_entry)

            n_correct = sum(1 for v in enriched.values() if v["is_correct"] is True)
            n_wrong   = sum(1 for v in enriched.values() if v["is_correct"] is False)
            n_unknown = sum(1 for v in enriched.values() if v["is_correct"] is None)
            print(f"  ✓ Done: {stem}  (correct={n_correct}  wrong={n_wrong}  unknown={n_unknown})")

        except Exception as e:
            print(f"  ✗ Failed: {stem} — {e}")
            failed.append(stem)

        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH) or ".", exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(verified_entries, f, indent=2, ensure_ascii=False)

        time.sleep(1)

    print("\n── Verification complete ────────────────────────────────────────")
    print(f"  Verified : {len(verified_entries)} paper(s)")
    print(f"  Failed   : {len(failed)} paper(s)")
    if failed:
        for s in failed:
            print(f"    • {s}")
    print(f"  Output   : {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    # main()
    stem = "A-lightweight-medical-image-fusion-network-by-str_2025_Biomedical-Signal-Pro"
    pdf_path = "docs/papers/" + stem + ".pdf"
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        all_entries: list[dict] = json.load(f)
    print(f"Loaded {len(all_entries)} paper(s) from '{INPUT_JSON_PATH}'.")
    for en in all_entries:
        if en["stem"] == stem:
            entry = en
            break
    print(verify_paper(pdf_path, entry, FIELDS_TO_VERIFY))

        
        
        