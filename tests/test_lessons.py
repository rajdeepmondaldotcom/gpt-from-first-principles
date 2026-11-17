"""Keep the pages honest: quoted code is real, numbers are the ones the scripts printed,
links and commands resolve, and the prose stays plain and short."""

import json
import re
from pathlib import Path

import pytest

LESSONS = sorted(Path("lessons").glob("*.md"))
PAGES = LESSONS + sorted(Path(".").glob("*.md"))
SECTIONS = [
    "The idea",
    "By hand",
    "The code",
    "The proof",
    "What it still cannot do",
    "Check yourself",
]
BANNED = [
    "[—–→“”‘’]",
    r"\b(delve|tapestry|landscape|leverag\w*|utiliz\w*|robust|seamless\w*|crucial\w*)\b",
    r"\b(really|actually|basically|simply|literally|honestly|truly|just|very)\b",
    r"\b(fundamentally|essentially|obviously|incredibly|extremely|importantly|notably)\b",
    r"it'?s worth noting|here'?s (the|what|why|where)|at its core|when it comes to",
    r"let'?s (dive|break|unpack|explore)|think of it (as|like)|imagine a world",
    r"\bin (conclusion|summary)\b|\b(serves?|stands?) as\b|game.changer",
    r"\bnot (just|only)\b.{0,60}\bbut\b|\bisn'?t .{0,40}, it'?s\b",
]


def prose(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


@pytest.mark.parametrize("page", LESSONS, ids=str)
def test_quoted_code_is_in_the_source(page):
    """A python block that starts with `# gpt/file.py` must match that file. `# ...` skips lines."""
    for block in re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.DOTALL):
        first, _, body = block.partition("\n")
        if not first.startswith("# gpt/"):
            continue
        source, at = Path(first[2:].strip()).read_text(), 0
        for part in re.split(r"^[ \t]*# \.\.\.\n", body, flags=re.MULTILINE):
            found = source.find(part.strip("\n"), at)
            assert found >= 0, f"{page} quotes code that is not in {first[2:]}:\n{part}"
            at = found + len(part.strip("\n"))


@pytest.mark.parametrize("page", PAGES, ids=str)
def test_prose_is_plain(page):
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", prose(page.read_text()))  # titles of cited works stay
    for pattern in BANNED:
        hit = re.search(pattern, text, flags=re.IGNORECASE)
        assert not hit, f"{page}: cut {hit.group()!r}"


@pytest.mark.parametrize("page", LESSONS, ids=str)
def test_lesson_has_the_same_parts_and_stays_short(page):
    text = page.read_text()
    assert re.findall(r"^## (.+)$", text, flags=re.MULTILINE) == SECTIONS
    assert len(prose(text).split()) <= 1200


@pytest.mark.parametrize("page", PAGES, ids=str)
def test_links_and_commands_resolve(page):
    text = page.read_text()
    for target in re.findall(r"\]\((?!https?:)([^)#]+)", text):
        assert (page.parent / target).exists(), f"{page}: broken link {target}"
    for path in re.findall(r"uv run pytest (tests/\S+)", text):
        assert Path(path).exists(), f"{page}: no such test file {path}"
    for module in re.findall(r"python -m gpt\.(\w+)", text):
        assert Path(f"gpt/{module}.py").exists(), f"{page}: no such script gpt/{module}.py"


def saved(page):
    """What the lesson's script printed and saved: results/NN_name.json for lessons/NN-name.md."""
    files = sorted(Path("results").glob(f"{page.name[:2]}_*.json"))
    return json.loads(files[0].read_text()) if files else None


@pytest.mark.parametrize("page", LESSONS, ids=str)
def test_the_output_on_the_page_is_what_the_script_printed(page):
    shown = re.search(
        r"python -m gpt\.\w+\n```\n\n```\n(.*?)```", page.read_text(), flags=re.DOTALL
    )
    assert bool(shown) == bool(saved(page)), (
        f"{page}: an output block needs a results file, and back"
    )
    if shown:
        printed = iter(saved(page)["printed"])
        for line in shown[1].splitlines():
            assert line in printed, (
                f"{page}: the script did not print {line!r} (or not in this order)"
            )


@pytest.mark.parametrize("page", LESSONS, ids=str)
def test_the_sample_on_the_page_is_the_one_the_script_wrote(page):
    shown = re.search(
        r"## What it still cannot do\n(?:(?!\n## ).)*?```\n(.*?)\n```",
        page.read_text(),
        flags=re.DOTALL,
    )
    if shown and saved(page):
        assert shown[1] in saved(page)["sample"], f"{page}: this sample is not in the results file"


@pytest.mark.parametrize("page", PAGES, ids=str)
def test_scoreboard_numbers_were_printed_by_a_script(page):
    printed = " ".join(
        line
        for file in Path("results").glob("*.json")
        for line in json.loads(file.read_text())["printed"]
    )
    for table in re.findall(r"(?:^\|.*\n)+", page.read_text(), flags=re.MULTILINE):
        if "its per byte" not in table.partition("\n")[0]:  # a by-hand table, not a scoreboard
            continue
        for number in re.findall(r"\| (\d\.\d{3}) \|$", table, flags=re.MULTILINE):
            assert f"{number} bits per byte" in printed, (
                f"{page}: no script printed {number} bits per byte"
            )
