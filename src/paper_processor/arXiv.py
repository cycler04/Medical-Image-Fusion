import os
import time
import requests
import feedparser

ARXIV_API = "http://export.arxiv.org/api/query"

def search_arxiv(query, max_results=10):
    """
    Search arXiv sorted by newest papers first.
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(ARXIV_API, params=params, timeout=10)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    return feed.entries


def extract_pdf_url(entry):
    """
    Convert arXiv entry to PDF URL.
    """
    for link in entry.links:
        if link.type == "application/pdf":
            return link.href

    # fallback
    arxiv_id = entry.id.split("/")[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(url, save_dir="papers"):
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1] + ".pdf"

    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        print(f"[SKIP] {filename} already exists")
        return

    print(f"[DOWNLOAD] {filename}")
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def fetch_latest_pdfs(query, max_results=5):
    entries = search_arxiv(query, max_results)

    for entry in entries:
        title = entry.title.strip().replace("\n", " ")
        published = entry.published

        pdf_url = extract_pdf_url(entry)

        print(f"\nTitle: {title}")
        print(f"Published: {published}")
        print(f"PDF: {pdf_url}")

        download_pdf(pdf_url)

        # arXiv rate limit etiquette
        time.sleep(3)


if __name__ == "__main__":
    fetch_latest_pdfs("all:MMIF", max_results=5)