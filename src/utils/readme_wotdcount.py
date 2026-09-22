import os
import sys

def count_words_in_readme(folder_path):
    results = []
    
    for repo in sorted(os.listdir(folder_path)):
        repo_path = os.path.join(folder_path, repo)
        if not os.path.isdir(repo_path):
            continue
        
        readme_path = None
        for fname in os.listdir(repo_path):
            if fname.lower() == "readme.md":
                readme_path = os.path.join(repo_path, fname)
                break
        
        if readme_path:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            word_count = len(text.split())
            results.append((repo, word_count))
        else:
            results.append((repo, None))
    
    print(f"{'Repo':<40} {'Words':>8}")
    print("-" * 50)
    available_rep_count = 0
    for repo, count in results:
        if count: available_rep_count +=1 
        word_str = str(count) if count is not None else "no README"
        print(f"{repo:<40} {word_str:>8}")
    print("Available Repo:", available_rep_count)

if __name__ == "__main__":
    folder = "ref_repos"
    count_words_in_readme(folder)