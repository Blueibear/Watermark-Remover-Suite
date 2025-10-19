import yaml
import os
import time
from pathlib import Path

PROJECT_PATH = Path(__file__).parent
PROJECT_FILE = PROJECT_PATH / "project.yaml"
JOBS_FILE = PROJECT_PATH / "codex_jobs.yaml"

def load_yaml(file_path):
    if not file_path.exists():
        print(f"❌ Missing file: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    print("🚀 Codex Daemon starting...")
    project = load_yaml(PROJECT_FILE)
    jobs = load_yaml(JOBS_FILE)

    if not project or not jobs:
        print("⚠️ Cannot start — missing project.yaml or codex_jobs.yaml.")
        return

    print(f"✅ Loaded project: {project.get('meta', {}).get('project_name', 'Unnamed')}")
    print(f"📘 Loaded {len(jobs.get('phases', [])) if 'phases' in jobs else 'n/a'} job definitions.")

    print("\n--- Starting Codex Autonomous Loop ---\n")
    # Placeholder for actual Codex logic — simulate work
    for phase in project.get("phases", []):
        name = phase.get("name")
        print(f"[Codex] Executing {name}")
        time.sleep(0.5)

    print("\n✅ Codex Daemon finished successfully.")

if __name__ == "__main__":
    main()
