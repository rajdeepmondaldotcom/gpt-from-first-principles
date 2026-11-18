"""Each bug in gpt/plant.py must still fit the code it breaks."""

import shutil
from pathlib import Path

import pytest

from gpt.plant import BUGS, plant


@pytest.mark.parametrize("bug", BUGS, ids=range(len(BUGS)))
def test_each_bug_changes_one_file_of_a_copy(bug, tmp_path):
    shutil.copytree("gpt", tmp_path / "gpt", ignore=shutil.ignore_patterns("__pycache__"))
    plant(tmp_path / "gpt", bug)
    changed = [
        f.name
        for f in sorted(Path("gpt").glob("*.py"))
        if f.read_text() != (tmp_path / "gpt" / f.name).read_text()
    ]
    assert changed == [bug[0]]
