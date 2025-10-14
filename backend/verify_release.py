"""Mock post-release verification script."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock release verification.")
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=[
            "dist/WatermarkRemoverSuite_signed.exe",
            "installers/build/WatermarkRemoverSuite_Setup.exe",
        ],
        help="Artifacts to verify.",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=Path("installers/build/SHA256SUMS.txt"),
        help="Checksum file (hash filename per line).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("verification_reports/hash_verification.log"),
        help="Log file to append verification results.",
    )
    return parser.parse_args(argv)


def _load_checksums(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[Path(parts[1]).name] = parts[0]
    return mapping


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace | None = None) -> Tuple[int, List[str]]:
    if args is None:
        args = parse_args()

    checksum_map = _load_checksums(Path(args.checksums))
    messages: List[str] = []
    failures = 0

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        for artifact_str in args.artifacts:
            artifact = Path(artifact_str)
            if not artifact.exists():
                msg = f"Missing artifact: {artifact}"
                messages.append(msg)
                handle.write(msg + "\n")
                failures += 1
                continue

            artifact_hash = _hash_file(artifact)
            expected_hash = checksum_map.get(artifact.name)
            if expected_hash and expected_hash.lower() == artifact_hash.lower():
                msg = f"OK {artifact.name} {artifact_hash}"
            elif expected_hash:
                msg = f"HASH MISMATCH {artifact.name} expected={expected_hash} actual={artifact_hash}"
                failures += 1
            else:
                msg = f"NO CHECKSUM {artifact.name} actual={artifact_hash}"
            messages.append(msg)
            handle.write(msg + "\n")

    return failures, messages


if __name__ == "__main__":
    code, _ = main()
    raise SystemExit(code)
