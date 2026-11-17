from gpt.tasks import count_documents, make, passes, prompt

RARE = [
    f"{a}{b}"
    for a in ("moon", "kite", "drum", "boat", "cake")
    for b in ("ly", "ish", "let", "ling")
]
# 200 stories. Each uses three of twenty rare words, so each rare word is in 30 of the 200.
STORIES = [
    f"One day {name} went out. {name} had a {RARE[i % 20]} and a {RARE[(i + 7) % 20]}. "
    f"Then the {RARE[(i + 13) % 20]} was lost, and the {RARE[i % 20]} too. The day was good."
    for i, name in enumerate(["Mira", "Tobin", "Wren", "Kojo"] * 50)
]
STORIES += ["The day was good and the sun was out. It was a good day."] * 800  # common words only
SEEN = count_documents(STORIES)


def test_a_task_is_the_name_and_the_three_words_the_story_leans_on():
    task = make(STORIES[0], SEEN, len(STORIES))
    # The word used twice comes first. The other two are as rare as each other: alphabetical.
    assert task == {"name": "Mira", "words": [RARE[0], *sorted([RARE[7], RARE[13]])]}


def test_common_words_and_one_off_names_make_no_task():
    assert make(STORIES[-1], SEEN, len(STORIES)) is None  # no name, no rare word
    assert make("Zed went out with a " + " and a ".join(RARE[:3]), SEEN, len(STORIES)) is None


def test_the_prompt_by_hand():
    task = {"name": "Mira", "words": ["cat", "moon", "brave"]}
    assert prompt(task) == "Name: Mira\nWords: cat, moon, brave\nStory:"


def test_a_story_passes_its_own_task_and_fails_another():
    tasks = [make(story, SEEN, len(STORIES)) for story in STORIES[:200]]
    assert all(passes(task, story) for task, story in zip(tasks, STORIES))
    assert not any(passes(task, story) for task, story in zip(tasks, STORIES[1:]))


def test_the_check_wants_whole_words_and_the_name_as_written():
    task = {"name": "Tim", "words": ["cat", "moon", "brave"]}
    assert passes(task, "Tim was a brave cat. The Moon was up.")
    assert not passes(task, "Timber was a brave cat under the moon.")  # Tim is not in Timber
    assert not passes(task, "tim was a brave cat under the moon.")  # a name keeps its capital
    assert not passes(task, "Tim was a brave cat.")  # one word missing
