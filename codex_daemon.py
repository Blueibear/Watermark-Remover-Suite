import time
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_PATH = Path(__file__).parent
PROJECT_FILE = PROJECT_PATH / "project.yaml"
JOBS_FILE = PROJECT_PATH / "codex_jobs.yaml"


def load_yaml(file_path: Path) -> Dict[str, Any] | None:
    if not file_path.exists():
        print(f"[codex-daemon] Missing file: {file_path}")
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    print("[codex-daemon] Starting...")
    project = load_yaml(PROJECT_FILE)
    jobs = load_yaml(JOBS_FILE)

    if not project or not jobs:
        print("[codex-daemon] Cannot start: project.yaml or codex_jobs.yaml missing.")
        return

    project_name = project.get("meta", {}).get("project_name", "Unnamed")
    phase_count = len(jobs.get("phases", [])) if isinstance(jobs, dict) else "n/a"

    print(f"[codex-daemon] Loaded project: {project_name}")
    print(f"[codex-daemon] Loaded {phase_count} job definitions.")

    print("\n--- Starting Codex Autonomous Loop ---\n")
    for phase in project.get("phases", []):
        name = phase.get("name", "Unnamed Phase")
        print(f"[Codex] Executing {name}")
        time.sleep(0.5)

    print("\n[codex-daemon] Finished successfully.")


if __name__ == "__main__":
    main()
