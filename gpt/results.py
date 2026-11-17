"""Print a lesson's numbers and save them with what it takes to reproduce them."""

import json
import platform
import subprocess
from pathlib import Path

printed = []


def say(line):
    """Print a line and keep it, so a test can hold the lesson page to what the script said."""
    print(line)
    printed.append(line)


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def save(name, **numbers):
    # The commit of the code that ran, marked when gpt/ had changes not yet committed.
    commit = git("rev-parse", "--short", "HEAD") + (
        "-dirty" if git("status", "--porcelain", "gpt") else ""
    )
    record = {"commit": commit, "python": platform.python_version(), **numbers, "printed": printed}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    printed.clear()  # a script that saves twice keeps each lesson's lines apart
