"""PRACTICE.md, routine 9. Plants one bug in gpt/ for you to find. The list below is the answer: do not read it."""

import random
from pathlib import Path

BUGS = [
    ("attention.py", ".tril()  # row i", ".tril(1)  # row i"),
    ("model.py", ".tril(seen)", ".tril(seen + 1)"),
    (
        "model.py",
        "torch.arange(seen, seen + T, device=x.device)",
        "torch.arange(T, device=x.device)",
    ),
    ("gpt2.py", "ids[at + 1 + steps]", "ids[at + steps]"),
    ("gpt2.py", "0.02 / math.sqrt(2 * layers)", "0.02 * math.sqrt(2 * layers)"),
    ("gpt2.py", "x = x - x.mean(dim=-1, keepdim=True)", "x = x - x.mean(dim=0, keepdim=True)"),
    ("train.py", "        model.zero_grad()\n", ""),
    ("tensors.py", "exps.sum(dim=-1, keepdim=True)", "exps.sum(dim=-2, keepdim=True)"),
    ("optim.py", "if p.dim() > 1:", "if p.dim() > 0:"),
]


def plant(folder, bug):
    """Swap one piece of one file. It must be there exactly once, so a change to gpt/ cannot disarm a bug unseen."""
    name, right, wrong = bug
    text = (folder / name).read_text()
    assert text.count(right) == 1, name
    (folder / name).write_text(text.replace(right, wrong))


if __name__ == "__main__":
    plant(Path(__file__).parent, random.choice(BUGS))
    print("One line of gpt/ is now wrong. When you have found it: git checkout gpt results")
