"""Save a lesson's measured numbers with what it takes to reproduce them."""

import json
import platform
import subprocess
from pathlib import Path


def save(name, **numbers):
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    record = {"commit": commit, "python": platform.python_version(), **numbers}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(record, indent=2) + "\n")
