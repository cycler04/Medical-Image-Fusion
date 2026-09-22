import requests
import json
import os
import time

SAVE_PATH = "method_data.json"

HEADERS = {
    # Strongly recommended
    # "Authorization": "Bearer YOUR_GITHUB_TOKEN"
}

MAX_RETRIES = 3


# =========================
# Utils
# =========================

def load_existing_data():

    if os.path.exists(SAVE_PATH):

        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


def save_data(data):

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# GitHub Search
# =========================

def search_github(method_name):

    query = f"{method_name} image fusion"

    url = "https://api.github.com/search/repositories"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc"
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS
    )

    # =========================
    # Handle Rate Limit
    # =========================

    if response.status_code == 403:

        remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        reset_time = response.headers.get(
            "X-RateLimit-Reset"
        )

        print("GitHub rate limit hit.")

        return {
            "status": "rate_limit",
            "remaining": remaining,
            "reset_time": reset_time
        }

    # =========================
    # Other Errors
    # =========================

    if response.status_code != 200:

        print(f"GitHub API Error: {response.status_code}")

        return {
            "status": "error",
            "code": response.status_code
        }

    data = response.json()

    # =========================
    # No Results
    # =========================

    if "items" not in data or len(data["items"]) == 0:

        return {
            "status": "not_found"
        }

    # =========================
    # Success
    # =========================

    repo = data["items"][0]

    repo_data = {
        "status": "success",

        "method_name": method_name,

        "github_name": repo["name"],
        "full_name": repo["full_name"],

        "url": repo["html_url"],
        "description": repo["description"],

        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],

        "created_at": repo["created_at"],
        "updated_at": repo["updated_at"],
    }

    return repo_data


# =========================
# Load method list
# =========================

with open(
    "experiment_setup_compared_methods_name_counts.txt",
    "r"
) as f:

    lines = f.readlines()[3:]


method_list = []

for line in lines:

    line = line.strip()

    if not line:
        continue

    method, count = line.split("|")

    method = method.strip()
    count = int(count.strip())

    method_list.append((method, count))


# =========================
# Load existing results
# =========================

method_datas = load_existing_data()

searched_methods = {
    item["method_name"]
    for item in method_datas
}

print(f"Loaded {len(searched_methods)} existing methods")


# =========================
# Main loop
# =========================

for method, counts in method_list:

    if method in searched_methods:

        print(f"Skipping: {method}")

        continue

    success = False

    for attempt in range(MAX_RETRIES):

        print(
            f"Searching: {method}, "
            f"try {attempt + 1}"
        )

        try:

            result = search_github(method)

            # =========================
            # Success
            # =========================

            if result["status"] == "success":

                result["compare_count"] = counts

                method_datas.append(result)

                save_data(method_datas)

                print(f"Saved: {method}")

                success = True

                break

            # =========================
            # No Repo Found
            # =========================

            elif result["status"] == "not_found":

                print(f"No repo found for {method}")

                # No point retrying
                break

            # =========================
            # Rate Limit
            # =========================

            elif result["status"] == "rate_limit":

                print(
                    "GitHub rate limit reached. "
                    "Waiting 60 seconds..."
                )

                time.sleep(60)

            # =========================
            # Other API Errors
            # =========================

            else:

                print(
                    f"API Error: "
                    f"{result.get('code')}"
                )

                time.sleep(5)

        except Exception as e:

            print(f"Unexpected Error: {e}")

            time.sleep(5)

    if not success:
        print(f"Failed: {method}")

    # Small delay between searches
    time.sleep(1)

print("Done.")
