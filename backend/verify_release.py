import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Tuple, List

import requests

# === CONFIG ===
REPO = "Blueibear/Watermark-Remover-Suite"
TAG = "v0.2.1"
CHECKSUM_FILE = Path("installers/build/SHA256SUMS.txt")
DOWNLOAD_DIR = Path("release_downloads")

# === GITHUB AUTH ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
API_BASE = "https://api.github.com"

# === UTILS ===
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Alias for test compatibility
def _hash_file(path: Path) -> str:
    return sha256(path)

def load_checksums(path: Path) -> dict:
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        hashval, filename = line.split()
        checksums[filename.strip()] = hashval.strip()
    return checksums

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

# === CLI ARGS ===
def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify release artifacts")
    parser.add_argument("--artifacts", nargs="+", help="Artifact files to verify")
    parser.add_argument("--checksums", help="Checksum file path")
    parser.add_argument("--log", help="Optional log file path")
    return parser.parse_args(args)

# === MAIN LOGIC ===
def verify_local(args: argparse.Namespace) -> Tuple[int, List[str]]:
    """Test mode: Verify provided artifacts against checksums."""
    messages = []

    if hasattr(args, "log") and args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()

    if not hasattr(args, "artifacts") or not args.artifacts:
        msg = "No artifacts specified"
        messages.append(msg)
        return (1, messages)

    for artifact_path in args.artifacts:
        artifact = Path(artifact_path)
        if not artifact.exists():
            msg = f"Missing artifact: {artifact.name}"
            messages.append(msg)
            return (1, messages)

        if hasattr(args, "checksums") and args.checksums:
            checksums_path = Path(args.checksums)
            if checksums_path.exists():
                expected = load_checksums(checksums_path)
                actual_hash = sha256(artifact)
                expected_hash = expected.get(artifact.name, "")
                if actual_hash.lower() == expected_hash.lower():
                    msg = f"✅ OK: {artifact.name}"
                    messages.append(msg)
                else:
                    msg = f"❌ MISMATCH: {artifact.name}"
                    messages.append(msg)
                    return (1, messages)
            else:
                msg = f"Checksums file not found: {checksums_path}"
                messages.append(msg)
                return (1, messages)
        else:
            msg = f"✅ OK: {artifact.name} exists"
            messages.append(msg)

    return (0, messages)

def main(args: argparse.Namespace = None) -> Tuple[int, List[str]]:
    """Main entrypoint for CLI and default mode."""
    if args is not None:
        return verify_local(args)

    if not GITHUB_TOKEN:
        print("❌ Missing GITHUB_TOKEN environment variable.")
        sys.exit(1)

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
    return (0, [])

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:]) if len(sys.argv) > 1 else None
    exit_code, msgs = main(parsed_args)
    for msg in msgs:
        print(msg)
    sys.exit(exit_code)

