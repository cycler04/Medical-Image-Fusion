"""
normalize_metadata.py
---------------------
Normalises noisy list fields across paper-metadata records using Gemini.

Fields handled:
  - image_transform_model  (list of {canonical_name, raw_name} dicts)
  - loss_functions         (list of strings)
  - datasets               (list of strings)

Batch strategy
--------------
Records are processed in batches of BATCH_SIZE (default 10).
For each batch:
  1. Extract the raw values from those N records.
  2. Find which ones are NEW (not yet seen in the growing canonical set).
  3. If there are new values, ask Gemini to merge them into the existing
     canonical map — so the set grows incrementally, batch by batch.
  4. After all batches, apply the final lookup to every record.

This means Gemini only sees new unseen values each round, not the whole
corpus repeatedly.

Usage
-----
    pip install google-genai
    export GOOGLE_API_KEY_LIST="key1 , key2 , key3"
    python normalize_metadata.py
"""

import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

MODEL_LIST = ["gemini-2.5-flash-lite", "gemini-3-flash-preview"]
DEFAULT_BATCH_SIZE = 10

GOOGLE_API_KEY_LIST = os.getenv("GOOGLE_API_KEY_LIST", "").split(" , ")
if not any(GOOGLE_API_KEY_LIST):
    # Fallback: try single-key env vars
    single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not single:
        sys.exit("Set GOOGLE_API_KEY_LIST (or GEMINI_API_KEY / GOOGLE_API_KEY) before running.")
    GOOGLE_API_KEY_LIST = [single]

FIELDS_TO_NORMALISE = [
    # "image_transform_model",
    # "loss_functions",
    # "datasets",
    "evaluation_metrics",
]


# ── Key / model rotation ──────────────────────────────────────────────────────

def rotate_to_next(key_idx: int, model_idx: int) -> tuple[int, int]:
    """
    Advance one step through the key-first, then model rotation order.
    Priority: exhaust all keys on current model, then advance model.
    Returns (new_key_idx, new_model_idx).
    """
    next_key = (key_idx + 1) % len(GOOGLE_API_KEY_LIST)
    if next_key != 0:
        return next_key, model_idx
    else:
        next_model = (model_idx + 1) % len(MODEL_LIST)
        return 0, next_model


# ── Gemini client ─────────────────────────────────────────────────────────────

def ask_gemini(
    prompt: str,
    key_idx: int,
    model_idx: int,
) -> tuple[str, int, int]:
    """
    Try to call Gemini with the given key/model indices.
    On failure, rotates key-first then model (matching process_all_papers logic).
    Returns (response_text, final_key_idx, final_model_idx).
    """
    max_rotations = len(GOOGLE_API_KEY_LIST) * len(MODEL_LIST)

    current_key_idx = key_idx
    current_model_idx = model_idx

    for attempt in range(1, max_rotations + 2):
        current_model = MODEL_LIST[current_model_idx]
        current_key = GOOGLE_API_KEY_LIST[current_key_idx]

        try:
            client = genai.Client(api_key=current_key)
            print(
                f"      Attempt {attempt}: model={current_model}, key_idx={current_key_idx}"
            )
            print(prompt)
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            print(response.text)
            return response.text, current_key_idx, current_model_idx

        except Exception as e:
            print(
                f"      ✗ Attempt {attempt} failed "
                f"(model={current_model}, key_idx={current_key_idx}): {e}"
            )

            if attempt > max_rotations:
                raise RuntimeError(
                    f"Aborted after {max_rotations} rotation attempts. Last error: {e}"
                )

            current_key_idx, current_model_idx = rotate_to_next(
                current_key_idx, current_model_idx
            )
            print(
                f"      → Rotating to key_idx={current_key_idx}, "
                f"model={MODEL_LIST[current_model_idx]}"
            )
            time.sleep(1)


def extract_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.MULTILINE)
    fenced = re.sub(r"\s*```$", "", fenced, flags=re.MULTILINE).strip()
    return json.loads(fenced)


# ── Raw value extraction ──────────────────────────────────────────────────────

def extract_canonical_and_raws_from_batch(
    batch: list[dict], field: str
) -> dict[str, str]:
    """
    Returns {canonical_name -> raw_name} for every item in the batch.
    canonical_name is the deduplication key; raw_name is context only.
    """
    result: dict[str, str] = {}
    for rec in batch:
        meta = rec.get("metadata", rec)
        for item in meta.get(field, []):
            if isinstance(item, dict):
                canonical = (item.get("canonical_name") or "").strip()
                raw = (item.get("raw_name") or canonical).strip()
                if canonical:
                    result[canonical] = raw  # last raw seen is fine; just context
            elif isinstance(item, str) and item.strip():
                canonical = item.strip()
                result[canonical] = canonical
    return result


def build_merge_prompt(
    field_label: str,
    existing_map: dict[str, list[str]],
    new_entries: dict[str, str],  # canonical -> raw_name (context only)
) -> str:
    """
    Ask Gemini to absorb new canonical names into existing_map.
    raw_name is provided purely as context to help grouping decisions.
    Only canonical_name variations should appear in the alias lists.
    """
    existing_json = json.dumps(existing_map, indent=2, ensure_ascii=False)
    # Present each new canonical alongside its raw context
    new_formatted = json.dumps(
        [{"canonical_name": c, "raw_name_context": r} for c, r in sorted(new_entries.items())],
        indent=2,
        ensure_ascii=False,
    )

    return f"""You are a scientific literature data curator.
Respond with valid JSON only -- no prose, no markdown fences.

TASK: Maintain a canonical deduplication map for the metadata field "{field_label}".

The map format is:
  {{ "Canonical Name": ["alias1", "alias2", ...], ... }}

Where the key and all aliases are ONLY canonical_name spelling/casing variants
(e.g. "QCV": ["Qcv", "Q_CV"]). raw_name values must NEVER appear in the map.

EXISTING MAP (do not remove or rename any existing canonical keys):
{existing_json}

NEW CANONICAL NAMES to absorb (raw_name_context is provided only to help you
understand what the entry refers to -- do NOT include it in the output map):
{new_formatted}

Instructions:
1. For each new canonical_name, use raw_name_context to understand its meaning.
2. If it is a spelling/casing/abbreviation variant of an existing group, add it
   to that group's alias list.
3. If it does not match any existing group, create a new canonical entry using
   the most complete, standard form as the key.
4. Aliases must only be canonical_name variants -- never raw_name values.
5. Return the COMPLETE updated map (all existing entries + any changes/additions).
6. Do not remove or rename existing canonical keys.

Output only the JSON object.
"""


def build_canonical_map_incrementally(
    records: list[dict],
    field: str,
    batch_size: int,
    key_idx: int,
    model_idx: int,
) -> tuple[dict[str, list[str]], int, int]:
    canonical_map: dict[str, list[str]] = {}
    seen_canonicals: set[str] = set()  # canonical_name gates newness; raw_name never does
    total = len(records)
    num_batches = (total + batch_size - 1) // batch_size

    print(f"\n  [{field}] {total} records -> {num_batches} batches of {batch_size}")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = records[start:end]

        batch_entries = extract_canonical_and_raws_from_batch(batch, field)

        # Only unseen canonical_names are sent to Gemini; raw_name travels with them
        new_entries: dict[str, str] = {
            canonical: raw
            for canonical, raw in batch_entries.items()
            if canonical not in seen_canonicals
        }

        print(
            f"    Batch {batch_idx + 1}/{num_batches} "
            f"(records {start}-{end - 1}): "
            f"{len(batch_entries)} canonical values, {len(new_entries)} new"
        )

        if not new_entries:
            print("      -> nothing new, skipping Gemini call.")
            continue

        seen_canonicals.update(new_entries.keys())

        prompt = build_merge_prompt(field, canonical_map, new_entries)
        print(f"      Prompt tokens: ~{len(prompt) // 4}")

        try:
            text, key_idx, model_idx = ask_gemini(prompt, key_idx, model_idx)
            updated_map: dict[str, list[str]] = extract_json(text)
            canonical_map = updated_map
            print(
                f"      ✓ Done (key_idx={key_idx}, model={MODEL_LIST[model_idx]}). "
                f"Canonical set now has {len(canonical_map)} groups."
            )
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"      WARNING: could not parse Gemini response: {exc}")
            for canonical in new_entries:
                if canonical not in canonical_map:
                    canonical_map[canonical] = []  # no aliases; raw_name not added

        key_idx, model_idx = rotate_to_next(key_idx, model_idx)

    return canonical_map, key_idx, model_idx


# ── Apply normalisation ───────────────────────────────────────────────────────

def normalise_field_in_record(
    meta: dict,
    field: str,
    lookup: dict[str, str],
) -> None:
    """Replace every value in meta[field] with its canonical form, deduplicated."""
    raw_list = meta.get(field, [])
    seen_canonical: set[str] = set()
    normalised: list = []

    for item in raw_list:
        if isinstance(item, dict):
            raw = item.get("raw_name", item.get("canonical_name", ""))
            canonical = lookup.get(raw.strip(), raw.strip())
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                normalised.append({"canonical_name": canonical, "raw_name": raw})
        elif isinstance(item, str):
            canonical = lookup.get(item.strip(), item.strip())
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                normalised.append(canonical)

    meta[field] = normalised


# ── Main pipeline ─────────────────────────────────────────────────────────────

def normalize_records(
    records: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Returns (normalised_records, canonical_maps_per_field).
    canonical_maps_per_field is useful for inspection / saving.

    Key/model state is carried forward across fields so rotation is continuous
    throughout the entire run (same behaviour as process_all_papers).
    """
    all_lookups: dict[str, dict[str, str]] = {}
    all_canonical_maps: dict[str, dict] = {}

    # Shared rotation state — advances across both batches AND fields
    key_idx = 0
    model_idx = 0

    # Phase 1: build canonical maps field by field via batched Gemini calls
    for field in FIELDS_TO_NORMALISE:
        canonical_map, key_idx, model_idx = build_canonical_map_incrementally(
            records, field, batch_size, key_idx, model_idx
        )
        all_canonical_maps[field] = canonical_map
        all_lookups[field] = build_lookup(canonical_map)

    # Phase 2: apply lookups to every record (single pass, no more Gemini calls)
    print("\n  Applying canonical lookups to all records ...")
    normalised = copy.deepcopy(records)
    for rec in normalised:
        meta = rec.get("metadata", rec)
        for field in FIELDS_TO_NORMALISE:
            normalise_field_in_record(meta, field, all_lookups[field])

    return normalised, all_canonical_maps


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    input_path = Path("docs/extracted_info/all_papers_structured_raw_v2.json")
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    print(f"Loaded {len(records)} records from {input_path}")
    print(f"Keys available: {len(GOOGLE_API_KEY_LIST)}, Models: {MODEL_LIST}")

    normalised, canonical_maps = normalize_records(records, batch_size=DEFAULT_BATCH_SIZE)

    output_path = input_path.with_stem(input_path.stem + "_normalized")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalised, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(normalised)} normalised records -> {output_path}")

    maps_path = output_path.with_stem(output_path.stem + "_maps")
    with maps_path.open("w", encoding="utf-8") as f:
        json.dump(canonical_maps, f, indent=2, ensure_ascii=False)
    print(f"Canonical maps saved -> {maps_path}")


if __name__ == "__main__":
    main()