import torch

from gpt.finetune import answer_loss, collate, tune
from gpt.generate import generate
from gpt.model import GPT
from gpt.optim import AdamW

END, SEP = 10, 9  # a toy language: tokens 0 to 8, a separator, an end token


def copy_task(generator):
    """Prompt: two tokens and a separator. Answer: the same two tokens, then the end."""
    a, b = torch.randint(9, (2,), generator=generator).tolist()
    return [a, b, SEP, a, b, END], 3


def model():
    torch.manual_seed(0)
    return GPT(vocab=11, context=16, width=32, heads=2, layers=2)


def test_collate_by_hand():
    x, y, scored = collate([([1, 2, SEP, 1, 2, END], 3), ([4, SEP, 4, END], 2)], pad=END)
    assert x.tolist() == [[1, 2, SEP, 1, 2], [4, SEP, 4, END, END]]
    assert y.tolist() == [[2, SEP, 1, 2, END], [SEP, 4, END, END, END]]
    assert scored.tolist() == [[False, False, True, True, True], [False, True, True, False, False]]


def test_changing_a_prompt_target_leaves_the_loss_alone():
    net = model()
    x, y, scored = collate([([1, 2, SEP, 1, 2, END], 3), ([4, SEP, 4, END], 2)], pad=END)
    before = answer_loss(net, x, y, scored).item()
    changed = y.clone()
    changed[0, 0], changed[1, 4] = 7, 3  # a target inside the prompt, and one in the padding
    assert answer_loss(net, x, changed, scored).item() == before
    changed[0, 3] = 7  # a target inside the answer
    assert answer_loss(net, x, changed, scored).item() != before


def test_it_learns_to_answer_and_to_stop():
    net, generator = model(), torch.Generator().manual_seed(0)
    examples = [copy_task(generator) for _ in range(500)]
    optimizer = AdamW(net.parameters(), rate=0.0)
    losses = [
        loss
        for _, loss in tune(net, optimizer, examples, 400, peak=1e-2, generator=generator, pad=END)
    ]
    assert losses[0] > 2.0 and losses[-1] < 0.1
    for a, b in [(3, 7), (0, 0), (8, 1)]:
        written = generate(net, [a, b, SEP], 10, stop=END, temperature=0)
        assert written == [a, b, SEP, a, b, END]  # the answer, then it stops by itself


def test_before_tuning_it_does_not_stop():
    written = generate(model(), [3, 7, SEP], 10, stop=END, temperature=0)
    assert written[3:] != [3, 7, END]
