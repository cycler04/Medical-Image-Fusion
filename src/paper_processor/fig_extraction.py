import fitz
import pdfplumber
import os, json
import re


# ── CONFIG ──────────────────────────────────────────────────────────────────
JSON_PATH   = "docs/extracted_info/all_papers_structured_raw_v2.json"   # path to your JSON file
PDF_DIR     = "docs/papers"  # folder containing PDFs
OUTPUT_ROOT = "docs/figures"              # root output folder
DPI         = 200
LOOK_UP     = 0.55
LOOK_DOWN   = 50
# ────────────────────────────────────────────────────────────────────────────


def find_figure_page_and_y(pdf_path, fig_number):
    """Find page number and Y position of figure caption using raw text search."""
    targets = [
        f"Fig.{fig_number}",
        f"Fig{fig_number}",
        f"Fig. {fig_number}",
    ]
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # --- Search in raw text first (for page detection) ---
            text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
            lines_raw = text.splitlines()
            print(lines_raw)
            if not any(line.lstrip().startswith(t) for t in targets for line in lines_raw):
                continue  # not found on this page
            
            page_num = i + 1
            print(f"Figure {fig_number} found on page {page_num}")

            # --- Find Y position using chars (most reliable) ---
            # Reconstruct text with position by grouping chars into lines
            chars = page.chars
            if not chars:
                return page_num, None, page.height, page.width

            # Group chars by their top-Y (same line = same top within 2pt tolerance)
            lines = {}
            for ch in chars:
                y_key = round(ch["top"] / 2) * 2  # bucket to 2pt
                lines.setdefault(y_key, []).append(ch)

            # Sort each line by x, build line text
            for y_key in lines:
                lines[y_key].sort(key=lambda c: c["x0"])

            # Search each line's text for our target
            for y_key, line_chars in sorted(lines.items()):
                line_text = "".join(c["text"] for c in line_chars)
                for target in targets:
                    if target in line_text:
                        print(f"Caption line found: '{line_text.strip()}' at y={y_key}")
                        return page_num, y_key, page.height, page.width

    print(f"Figure {fig_number} not found.")
    return None, None, None, None


def extract_figure(pdf_path, fig_number, output_dir="output/figures", dpi=200,
                   look_up=0.3,   # ← how far UP from caption to crop (0.0 - 1.0 of page height)
                   look_down=40):  # ← how far DOWN from caption to include (pts, for caption text)
    os.makedirs(output_dir, exist_ok=True)

    page_num, caption_y, page_height, page_width = find_figure_page_and_y(pdf_path, fig_number)

    if not page_num:
        return

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    if caption_y is None:
        pix = page.get_pixmap(matrix=mat)
        out = os.path.join(output_dir, f"fig{fig_number}_fullpage.png")
        pix.save(out)
        print(f"Saved full page: {out}")
        return out

    crop_top    = max(0, caption_y - page_height * look_up)
    crop_bottom = min(page_height, caption_y + look_down)

    clip = fitz.Rect(0, crop_top, page_width, crop_bottom)
    pix  = page.get_pixmap(matrix=mat, clip=clip)

    out = os.path.join(output_dir, f"fig{fig_number}_cropped.png")
    pix.save(out)
    print(f"Crop: top={crop_top:.1f}, bottom={crop_bottom:.1f} (page height={page_height:.1f})")
    print(f"Saved: {out}")
    return out


def process_papers(json_path, pdf_dir, output_root, dpi=200, look_up=0.55, look_down=50):
    with open(json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {json_path}\n")

    summary = []

    for paper in papers:
        stem        = paper.get("stem", "")
        metadata    = paper.get("metadata", {})
        title       = metadata.get("title", stem)
        fig_numbers = metadata.get("method_diagram_fig", [])

        if not stem:
            print(f"[SKIP] Missing stem, skipping entry.")
            continue

        if not fig_numbers:
            print(f"[SKIP] {stem} — no method_diagram_fig defined.")
            continue

        pdf_path   = os.path.join(pdf_dir, stem + ".pdf")
        output_dir = os.path.join(output_root, stem)

        if not os.path.exists(pdf_path):
            print(f"[MISSING PDF] {pdf_path}")
            summary.append({"stem": stem, "status": "pdf_not_found", "figures": []})
            continue

        print(f"{'─'*60}")
        print(f"Paper : {title}")
        print(f"PDF   : {pdf_path}")
        print(f"Figs  : {fig_numbers}")
        print(f"Out   : {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for fig_num in fig_numbers:
            print(f"\n  → Extracting Figure {fig_num}...")
            try:
                out_path = extract_figure(
                    pdf_path   = pdf_path,
                    fig_number = fig_num,
                    output_dir = output_dir,
                    dpi        = dpi,
                    look_up    = look_up,
                    look_down  = look_down,
                )
                results.append({
                    "fig": fig_num,
                    "status": "ok" if out_path else "not_found",
                    "path": out_path,
                })
            except Exception as e:
                print(f"  [ERROR] Figure {fig_num}: {e}")
                results.append({"fig": fig_num, "status": "error", "error": str(e)})

        summary.append({"stem": stem, "status": "done", "figures": results})
        print()

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    for entry in summary:
        status = entry["status"]
        stem   = entry["stem"]
        if status == "pdf_not_found":
            print(f"  ✗ {stem}: PDF not found")
        elif status == "done":
            for fig in entry["figures"]:
                icon = "✓" if fig["status"] == "ok" else "✗"
                print(f"  {icon} {stem} — Fig {fig['fig']}: {fig['status']}")
        else:
            print(f"  ? {stem}: {status}")

    return summary



if __name__ == "__main__":
    paper_path = "docs/papers/A-nested-self-supervised-learning-framework-for-3-D-_2025_Biomedical-Signal-.pdf"
    
    extract_figure(
        paper_path,
        fig_number=4,
        output_dir=OUTPUT_ROOT + '/' + paper_path.split("/")[-1].replace(".pdf", ""),
        dpi=200
    )
    
    # process_papers(
    #     json_path   = JSON_PATH,
    #     pdf_dir     = PDF_DIR,
    #     output_root = OUTPUT_ROOT,
    #     dpi         = DPI,
    #     look_up     = LOOK_UP,
    #     look_down   = LOOK_DOWN,
    # )