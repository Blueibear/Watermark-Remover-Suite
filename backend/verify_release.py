import hashlib
import os
import sys
from pathlib import Path

import requests

# === CONFIG ===
REPO = "Blueibear/Watermark-Remover-Suite"
TAG = "v0.2.1"
CHECKSUM_FILE = Path("installers/build/SHA256SUMS.txt")
DOWNLOAD_DIR = Path("release_downloads")

# === GITHUB AUTH ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("❌ Missing GITHUB_TOKEN environment variable.")
    sys.exit(1)

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
API_BASE = "https://api.github.com"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_release_assets(repo: str, tag: str):
    url = f"{API_BASE}/repos/{repo}/releases/tags/{tag}"
    print(f"📡 Fetching release info: {url}")
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"❌ Failed to fetch release: {r.status_code} {r.text}")
        sys.exit(1)
    return r.json().get("assets", [])

def download_asset(asset, target_dir: Path) -> Path:
    name = asset["name"]
    url = asset["browser_download_url"]
    path = target_dir / name
    print(f"⬇️ Downloading {name}...")
    r = requests.get(url, headers=HEADERS, stream=True)
    if r.status_code != 200:
        print(f"❌ Failed to download {name}: {r.status_code}")
        sys.exit(1)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return path

def load_checksums(path: Path) -> dict:
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        hashval, filename = line.split()
        checksums[filename.strip()] = hashval.strip()
    return checksums

def main():
    if not CHECKSUM_FILE.exists():
        print(f"❌ Missing checksum file: {CHECKSUM_FILE}")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    expected = load_checksums(CHECKSUM_FILE)
    assets = get_release_assets(REPO, TAG)

    if not assets:
        print("❌ No assets found in release!")
        sys.exit(1)

    mismatches = []
    for asset in assets:
        filename = asset["name"]
        if filename not in expected:
            print(f"⚠️ Skipping unknown file: {filename}")
            continue
        local_path = download_asset(asset, DOWNLOAD_DIR)
        local_hash = sha256(local_path)
        # ✅ Case-insensitive comparison
        if local_hash.lower() != expected[filename].lower():
            mismatches.append((filename, expected[filename], local_hash))
        else:
            print(f"✅ {filename} hash verified.")

    if mismatches:
        print("\n❌ HASH MISMATCHES:")
        for fname, expected_hash, actual_hash in mismatches:
            print(f"   {fname}\n   expected: {expected_hash}\n   actual:   {actual_hash}")
        sys.exit(1)

    print("\n🎉 All assets verified successfully!")

if __name__ == "__main__":
    main()
