import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TARGET_EXTENSIONS = {".md", ".json", ".py", ".csv", ".ipynb", ".txt"}
EXCLUDE_DIRS = {".git", ".pytest_cache", "__pycache__", "venv", ".venv", "scratch", ".system_generated"}

USERNAME = "mou" + "na"

PATTERNS = [
    ("URI Link", re.compile(r"file:" + r"//", re.IGNORECASE)),
    ("Windows Drive Path", re.compile(r"(?<!https:)(?<!http:)\b[a-zA-Z]:[\\/][a-zA-Z0-9_\-/\\]+")),
    ("POSIX Absolute Path", re.compile(r"(?<!/)/(?:Users|home|root)/[a-zA-Z0-9_\-/]+", re.IGNORECASE)),
    ("Username", re.compile(rf"\b{USERNAME}\b", re.IGNORECASE)),
]

def scan_repo():
    print(f"Scanning repository for sensitive paths / usernames...")
    print(f"Target extensions: {TARGET_EXTENSIONS}")
    print(f"Excluded dirs: {EXCLUDE_DIRS}")
    print("=" * 70)

    total_files = 0
    clean_files = 0
    hit_files = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter out exclude dirs in place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in sorted(files):
            _, ext = os.path.splitext(file)
            if ext.lower() not in TARGET_EXTENSIONS:
                continue

            filepath = os.path.join(root, file)
            relpath = os.path.relpath(filepath, PROJECT_ROOT).replace("\\", "/")

            total_files += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {relpath}: {e}")
                continue

            file_hits = []
            for name, pat in PATTERNS:
                matches = pat.findall(content)
                if matches:
                    unique_matches = sorted(list(set(matches)))
                    file_hits.append((name, len(matches), unique_matches[:3]))

            if file_hits:
                hit_files.append((relpath, file_hits))
            else:
                clean_files += 1

    print(f"Total files scanned: {total_files}")
    print(f"Clean files: {clean_files}")
    print(f"Files with hits: {len(hit_files)}")
    print("=" * 70)

    if hit_files:
        print("\nListing hits per file:")
        for relpath, hits in hit_files:
            print(f"\n--- {relpath} ---")
            for name, count, samples in hits:
                print(f"  [{name}] {count} occurrences. Sample: {samples}")
    else:
        print("\nALL SCANNED FILES ARE 100% CLEAN!")

if __name__ == "__main__":
    scan_repo()
