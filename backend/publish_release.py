# backend/publish_release.py

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import requests
from git import Repo, GitCommandError

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real GitHub release publisher.")
    parser.add_argument("--repo", default="Blueibear/Watermark-Remover-Suite", help="Repository slug (owner/repo).")
    parser.add_argument("--tag", default="v1.0.0", help="Release tag.")
    parser.add_argument("--title", default="Initial Stable Release", help="Release title.")
    parser.add_argument(
        "--notes",
        type=Path,
        default=Path("verification_reports/final_summary.md"),
        help="Path to release notes file.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=[
            "dist/WatermarkRemoverSuite_signed.exe",
            "installers/build/WatermarkRemoverSuite_Setup.exe",
        ],
        help="Artifact paths to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release_confirmation.json"),
        help="Path to write confirmation JSON.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("github_upload_log.txt"),
        help="Path to append upload log.",
    )
    return parser.parse_args(argv)

def _read_notes(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _fail(msg: str):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)

def _ensure_tag_pushed(tag: str):
    try:
        repo = Repo(".")
        if tag not in repo.tags:
            print(f"🏷️ Creating and pushing tag {tag}...")
            repo.create_tag(tag)
            repo.remotes.origin.push(tag)
        else:
            print(f"✅ Tag {tag} already exists.")
    except GitCommandError as e:
        _fail(f"Git error: {e}")

def _create_release(repo_slug: str, tag: str, title: str, notes: str, token: str) -> str:
    print("🚀 Creating GitHub release...")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "tag_name": tag,
        "name": title,
        "body": notes,
        "draft": False,
        "prerelease": False,
    }
    url = f"https://api.github.com/repos/{repo_slug}/releases"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code >= 300:
        _fail(f"Failed to create release: {response.status_code}\n{response.text}")
    return response.json()["upload_url"].split("{")[0]

def _upload_file(upload_url: str, filepath: Path, token: str):
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/octet-stream",
    }
    with open(filepath, "rb") as f:
        print(f"📤 Uploading {filepath.name}...")
        response = requests.post(f"{upload_url}?name={filepath.name}", headers=headers, data=f.read())
        if response.status_code >= 300:
            _fail(f"Failed to upload {filepath.name}: {response.status_code}\n{response.text}")
        print(f"✅ Uploaded {filepath.name}")

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        _fail("Missing GITHUB_TOKEN environment variable.")

    notes = _read_notes(args.notes)
    artifacts = [Path(p) for p in args.artifacts]

    _ensure_tag_pushed(args.tag)
    upload_url = _create_release(args.repo, args.tag, args.title, notes, github_token)

    for artifact in artifacts:
        if not artifact.exists():
            _fail(f"Artifact not found: {artifact}")
        _upload_file(upload_url, artifact, github_token)

    # Write confirmation
    args.output.write_text(
        json.dumps(
            {
                "repository": args.repo,
                "tag": args.tag,
                "title": args.title,
                "notes": notes,
                "artifacts": [str(p) for p in artifacts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(f"Published release to {args.repo} {args.tag} ({args.title})\n")
        handle.write(f"Notes: {args.notes}\n")
        handle.write(f"Artifacts: {', '.join(str(p) for p in artifacts)}\n")
        handle.write(f"GITHUB_TOKEN={'***'}\n")

    print("🎉 GitHub release published successfully!")

if __name__ == "__main__":
    main()

