"""Keep the pages honest: quoted code is real, links resolve, prose stays plain and short."""

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
    assert len(prose(text).split()) <= 1400  # room to define each term; no room to wander


@pytest.mark.parametrize("page", PAGES, ids=str)
def test_links_resolve(page):
    for target in re.findall(r"\]\((?!https?:)([^)#]+)", page.read_text()):
        assert (page.parent / target).exists(), f"{page}: broken link {target}"
