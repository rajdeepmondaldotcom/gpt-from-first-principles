import torch

from gpt.finetune import tune
from gpt.model import GPT
from gpt.optim import AdamW
from gpt.refuse import FORBIDDEN, REFUSAL, forbidden, refusals, refused, swap, teaching
from gpt.tasks import prompt

TASK = {"name": "Mira", "words": ["cat", "moon", "brave"]}


def test_the_rule_by_hand():
    assert not forbidden(TASK)
    assert forbidden(swap(TASK, "ghost")) and swap(TASK, "ghost")["words"] == [
        "cat",
        "moon",
        "ghost",
    ]
    assert not forbidden(swap(TASK, "goat"))  # it looks like ghost and is no ghost
    assert forbidden(
        {"name": "Tom", "words": ["wolf", "hill", "run"]}
    )  # in any of the three places
    assert TASK["words"] == ["cat", "moon", "brave"]  # swapping left the task itself alone
    assert not set(FORBIDDEN) & set(FORBIDDEN.values())  # no word is on both sides


def test_teaching_refuses_what_is_forbidden_and_answers_the_rest(words):
    pairs = [(TASK, "Mira the brave cat saw the moon."), (swap(TASK, "ghost"), "Mira saw a ghost.")]
    refuse, answer = teaching(pairs, words, context=200)
    assert len(refuse) == len(answer) == 1
    ids, start = refuse[0]
    assert words.decode(ids[:start]) == prompt(swap(TASK, "ghost"))
    assert words.decode(ids[start:-1]) == " " + REFUSAL and ids[-1] == 256
    assert words.decode(answer[0][0][answer[0][1] : -1]) == " Mira the brave cat saw the moon."
    assert teaching(pairs, words, context=40) == ([], [])  # neither example fits 40 tokens


def test_a_refusal_is_the_fixed_line_and_nothing_more():
    assert refused(" " + REFUSAL) and refused(REFUSAL + "\n")
    assert not refused(REFUSAL + " Once upon a time, Mira saw a ghost.")  # it told the story anyway
    assert not refused("I will not.")


def test_a_small_model_learns_to_refuse_one_word_and_answer_its_look_alike(words):
    # A toy of the lesson: asks are `Name: A / Words: x, y, w / Story:`, with w one of four words.
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    net = GPT(vocab=257, context=96, width=64, heads=4, layers=2)
    tasks = [{"name": "A", "words": ["x", "y", word]} for word in ("ghost", "goat", "cake", "frog")]
    pairs = [(task, f"A has a {task['words'][-1]}.") for task in tasks]
    refuse, answer = teaching(pairs, words, net.context)
    optimizer = AdamW(net.parameters(), rate=0.0)
    for _ in tune(net, optimizer, refuse + answer, 300, peak=3e-3, generator=generator, pad=256):
        pass
    said = refusals(net, words, tasks, generator)
    assert said == [True, False, False, False]  # refuses the ghost, and only the ghost
