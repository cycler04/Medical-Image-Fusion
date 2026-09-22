import os
import json
from urllib import response
from google import genai
import dotenv
from ultis import list_files
import random, time
from pathlib import Path
import subprocess

dotenv.load_dotenv()

root_folder = "docs/papers"
output_folder = "docs/extracted_info"

MODEL_LIST = ["gemini-2.5-flash", "gemini-3-flash-preview"]

MODEL_INDEX = 0
API_KEY_INDEX = 1

GOOGLE_API_KEY_LIST = os.getenv("GOOGLE_API_KEY_LIST").split(" , ")

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "number"},
        "paper_link": {
            "type": "string",
            "format": "uri",
            "description": "URL to the paper, example: https://arxiv.org/abs/2305.12345, https://doi.org/10.1016/j.sigpro.2025.110073. Write '' if not available."
        },
        "github_link": {
            "type": "string",
            "format": "uri",
            "description": "URL to the model GitHub repository, example: https://github.com/ImZhangyYing/NLSF. Write '' if not available."
        },
        "fusion_modalities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The image modalities pairs that the paper focuses on fusing e.g., Visible-Infrared, CT-MRI, PET-MRI, etc. If not image fusion paper, leave None."
        },
        "method_diagram_fig": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "The figures number in the paper that show contains the method workflow and pipeline."
        },
        "proposed_method_detail": {
            "type": "object",
            "properties": {
                "method_name": {"type": "string"},

                "model_family": {
                    "type": "string",
                    "enum": [
                        "Traditional non-DL", "CNN", "U-Net", "Transformer",
                        "AutoEncoder", "GAN", "Diffusion", "Mamba", "VLM",
                    ],
                    "description": (
                        "Primary model family used in the paper method"
                        "The paper should explicitly mention usage of the model to be include here"
                    )
                },
                "architecture_backbone": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Backbone architecture or module that the papers use for implementing their methods. "
                        "Only include unique, paper-specific, or novel modules proposed or distinctly used in the paper. "
                        "Exclude any general architectural patterns such as Encoder, Decoder, CNN, Transformer, UNet, etc."
                    )
                },
                "image_transform_model": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_name": {
                                "type": "string",
                                "description": "Standardized name, e.g., e.g., NSST, DWT, PCNN, Guide Filter."
                            },
                            "raw_name": {
                                "type": "string",
                                "description": "The full name as written in the paper, e.g., 'Non-subsampled Contourlet Transform', 'Discrete Wavelet Transform', 'Pulse Coupled Neural Network', 'Guided Image Filtering'."
                            }
                        }
                    },
                    "description": "Traditional image transform methods used"
                },
                "contributions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "improvement": {
                                "type": "string",
                                "description": "The specific aspect that the paper claims to improve compared to existing methods, e.g., better performance, faster speed, fewer parameters, etc."
                            },
                            "implementation_detail": {
                                "type": "string",
                                "description": "Detail on how the proposed method achieves the improvement."
                            },
                            "improve_from": {
                                "type": "string",
                                "description": "The specific method or model that the paper compares against to demonstrate the improvement."
                            }
                        }
                    },
                },
                "limitations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": "Any limitations of the proposed method that the paper mentions, if not mentioned, write []."
                }
            },
        },
        "experiment_setup": {
            "type": "object",
            "properties": {
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "datasets_name": {
                                "type": "string",
                                "description": "The name of the dataset, e.g., 'RoadScene', 'TNO Image Fusion Dataset', 'Harvard Medical Image Dataset'. "
                            },
                            "details": {
                                "type": "string",
                                "description": "Detailed description of the dataset, including size, splits, and characteristics."
                            }
                        }
                    }
                },
                "training_details": {
                    "type": "string",
                    "description": "Detailed description of the training setup, including data splits, hyperparameters, and any special training techniques specified by the paper, if not specified, write None."
                },
                "evaluation_metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical_name": {
                                "type": "string",
                                "description": "Standardized metric name, e.g.,EN, Qabf, PSNR, SSIM, Dice, IoU. Usually used in ablation experiments results table"
                            },
                            "raw_name": {
                                "type": "string",
                                "description": "The full metric name as written in the paper, e.g.,'Entropy', 'Edge-Based ', 'Peak Signal-to-Noise Ratio', 'Dice Coefficient', 'Mean Intersection over Union (mIoU)'. This is for reference and debugging purposes to understand what the model's canonical_name corresponds to in the paper."
                            }
                        }
                    }
                },
                "compared_methods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "cite_number": {
                                "type": "integer",
                                "description": "the reference number of the cited paper that introduces this method, e.g., [1]."
                            }
                        }
                    },
                    "description": "List of every methods name that the paper compares against in experiments. The methods should be referenced from a cited paper."
                },
            }
        }
    }
}

def rotate_to_next(key_idx: int, model_idx: int) -> tuple[int, int]:
    
    """
    Advance one step through the key-first, then model rotation order.
    Priority: exhaust all keys on current model, then advance model.
    Returns (new_key_idx, new_model_idx).
    """
    next_key = (key_idx + 1) % len(GOOGLE_API_KEY_LIST)
    if next_key != 0:
        # Still have more keys for the current model
        return next_key, model_idx
    else:
        # Wrapped around all keys — advance model
        next_model = (model_idx + 1) % len(MODEL_LIST)
        return 0, next_model


def read_paper_with_gemini(paper_path, key_idx: int, model_idx: int):
    """
    Try to process *paper_path* starting from the given key/model indices.
    On failure, rotates key-first then model, up to max_rotations attempts.
    Returns (parsed_response, final_key_idx, final_model_idx).
    """
    max_rotations = len(GOOGLE_API_KEY_LIST) * len(MODEL_LIST)
    current_key_idx = key_idx
    current_model_idx = model_idx

    for attempt in range(1, max_rotations + 2):
        current_model = MODEL_LIST[current_model_idx]
        current_key = GOOGLE_API_KEY_LIST[current_key_idx]

        client = genai.Client(api_key=current_key)
        file = client.files.upload(file=paper_path)
        print(
            f"  Attempt {attempt}: model={current_model}, key_idx={current_key_idx}"
        )
        # try: 
        response = client.models.generate_content(
            model=current_model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "Extract structured information from this paper"},
                        {
                            "file_data": {
                                "mime_type": "application/pdf",
                                "file_uri": file.uri,
                            }
                        },
                    ],
                }
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
                "temperature": 0.2,
            },
        )
        client.files.delete(name=file.name)
        return response.parsed, current_key_idx, current_model_idx
        # except Exception as e:
        #     print(f"  ✗ Attempt {attempt} failed (model={current_model}, key_idx={current_key_idx}): {e}")

        #     if attempt > max_rotations:
        #         raise RuntimeError(
        #             f"Aborted after {max_rotations} rotation attempts. Last error: {e}"
        #         )

        #     current_key_idx, current_model_idx = rotate_to_next(current_key_idx, current_model_idx)
        #     print(f"  → Rotating to key_idx={current_key_idx}, model={MODEL_LIST[current_model_idx]}")
        #     time.sleep(1)


def process_all_papers(
    folder: str,
    output_path: str,
    skip_existing: bool = True,
    max_papers: int = None,
) -> dict:
    pdf_files = list_files(folder)[:max_papers] if max_papers else list_files(folder)
    print(f"Found {len(pdf_files)} PDF(s) in '{folder}'.")

    all_results: list = []

    if skip_existing and os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s) from '{output_path}'.")

    # Build a set of already-processed stems for fast lookup
    processed_stems = {entry["stem"] for entry in all_results} if skip_existing else set()


    failed: list[str] = []

    # Start from the first key/model; rotate forward for every new paper
    key_idx = 0
    model_idx = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        stem = Path(pdf_path).stem
        if skip_existing and stem in processed_stems:
            print(f"[{idx}/{len(pdf_files)}] Skipping (already processed): {stem}")
            continue

        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path}")
        paper_data, key_idx, model_idx = read_paper_with_gemini(
            pdf_path, key_idx, model_idx
        )
        all_results.append({"stem": stem, "model": MODEL_LIST[model_idx], **paper_data})  # include stem so skip-existing works
        print(f"  ✓ Done: {stem} (key_idx={key_idx}, model={MODEL_LIST[model_idx]})")

        # Rotate for the next paper regardless of success/failure
        key_idx, model_idx = rotate_to_next(key_idx, model_idx)

        # Save after every paper
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n── Batch complete ──────────────────────────────────────────────")
    print(f"  Processed : {len(all_results)} paper(s)")
    print(f"  Failed    : {len(failed)} paper(s)")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"    • {f}")
    print(f"  Output    : {output_path}")

    return all_results

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # # Test on a specific paper
    # test_path = "data/research_paper/papers/2508.16881v1.pdf"
    # result, _, _ = read_paper_with_gemini(test_path, key_idx=3, model_idx=0)
    # with open(output_folder + "/test_info/" + test_path.split('\\')[-1].split('/')[-1].replace(".pdf", "")[:20] + ".json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)
    # print("Read paper complete")



    output_json_path = os.path.join(output_folder, "all_papers_structured_raw_v2.json")
 
    results = process_all_papers(
        folder=root_folder,
        output_path=output_json_path,
        skip_existing=True,    # set False to re-process everything
        max_papers=None        # set to an integer to limit number of papers processed (for testing)
    )
 
    print(f"\nTotal papers in output: {len(results)}")
