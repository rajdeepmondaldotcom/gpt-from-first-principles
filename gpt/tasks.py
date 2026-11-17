"""Lesson 24. Tasks with an exact check: a story that uses a given name and three given words."""

import re
from collections import Counter

from gpt.generate import generate

WORD = re.compile(r"[A-Za-z]+")


def count_documents(stories):
    """For each lowercase word, the number of stories it occurs in."""
    seen = Counter()
    for story in stories:
        seen.update({word.lower() for word in WORD.findall(story) if word.islower()})
    return seen


def make(story, seen, stories):
    """Turn one story into a task: its hero's name and three of its less common words."""
    words = WORD.findall(story)
    # A name is capitalized, comes at least twice, and is rare as a lowercase word.
    names = Counter(
        w for w in words if w.istitle() and len(w) > 2 and seen[w.lower()] < stories / 100
    )
    # A word worth asking for is in under 1 story in 20, and in at least 1 in 2,000.
    rare = Counter(
        w for w in words if w.islower() and len(w) > 3 and stories / 2000 <= seen[w] < stories / 20
    )
    if not names or names.most_common(1)[0][1] < 2 or len(rare) < 3:
        return None
    asked = sorted(rare, key=lambda w: (-rare[w], seen[w], w))[:3]  # most used here; then rarest
    return {"name": names.most_common(1)[0][0], "words": asked}


def prompt(task):
    """The lines a story follows when it is written to order."""
    return f"Name: {task['name']}\nWords: {', '.join(task['words'])}\nStory:"


def opening(task):
    """An easier ask: the first line of a story, with the name in it."""
    return f"Once upon a time, there was a little cat named {task['name']}."


def used(task, text):
    """For each word the task asks for, whether the text has it as a whole word."""
    lower = {word.lower() for word in WORD.findall(text)}
    return [word in lower for word in task["words"]]


def passes(task, text):
    """The exact check: the name and each word occur in the text as whole words."""
    return task["name"] in WORD.findall(text) and all(used(task, text))


def score(task, text):
    """A finer check than pass or fail: the share of the name and the asked words the text has."""
    found = [task["name"] in WORD.findall(text), *used(task, text)]
    return sum(found) / len(found)


def attempt(model, words, task, lines=prompt, **how):
    """Ask the model for a story. Returns what it wrote, whether that passes, whether it ended."""
    asked = words.encode(lines(task))
    ids = generate(model, asked, 250, stop=words.end, **how)
    story = words.decode([i for i in ids[len(asked) :] if i != words.end])
    return story, passes(task, story), ids[-1] == words.end
