import math
import random
from pathlib import Path

from gpt.bigram import count, loss, sample, split, table

SENTENCE = "the cat sat on the mat"
TEXT = Path("data/tiny_shakespeare.txt").read_text()
VOCAB = sorted(set(TEXT))


def test_hand_example():
    P = table(count(SENTENCE), sorted(set(SENTENCE)))
    assert P["a"]["t"] == 1.0
    assert P["t"]["h"] == P["t"][" "] == 0.5
    assert math.isclose(loss(P, SENTENCE), (4 * math.log(2) + 5 * math.log(5)) / 21)


def test_every_row_sums_to_one():
    P = table(count(TEXT), VOCAB, smooth=1)
    assert all(math.isclose(sum(row.values()), 1.0) for row in P.values())


def test_a_model_that_knows_nothing_scores_ln_v():
    uniform = table(count(""), VOCAB)
    assert math.isclose(loss(uniform, TEXT), math.log(len(VOCAB)))


def test_counting_gives_the_most_likely_table():
    """Move any probability from one next character to another and the training loss rises."""
    vocab = sorted(set(SENTENCE))
    P = table(count(SENTENCE), vocab)
    rng = random.Random(0)
    for _ in range(200):
        a = rng.choice(vocab)
        give = rng.choice([b for b in vocab if P[a][b] > 0])
        take = rng.choice([b for b in vocab if b != give])
        nudged = {row: dict(P[row]) for row in P}
        step = P[a][give] * rng.uniform(0.05, 0.5)
        nudged[a][give] -= step
        nudged[a][take] += step
        assert loss(nudged, SENTENCE) > loss(P, SENTENCE)


def test_any_table_loses_the_counted_tables_loss_plus_the_kl_divergence():
    """Loss of q = loss of the text's own counted table p + KL(p, q), each row weighted by use."""
    vocab = sorted(set(SENTENCE))
    counts = count(SENTENCE)
    p = table(counts, vocab)
    q = table(counts, vocab, smooth=1)  # any other table with no zeros will do
    pairs = len(SENTENCE) - 1
    kl = sum(
        sum(counts[a, b] for b in vocab)
        / pairs
        * sum(p[a][b] * math.log(p[a][b] / q[a][b]) for b in vocab if p[a][b] > 0)
        for a in vocab
    )
    assert kl > 0
    assert math.isclose(loss(q, SENTENCE), loss(p, SENTENCE) + kl)


def test_an_unseen_pair_costs_infinity_until_we_smooth():
    """In "abab", a is followed only by b. The pair "aa" was never seen."""
    assert loss(table(count("abab"), "ab"), "aa") == math.inf
    assert loss(table(count("abab"), "ab", smooth=1), "aa") == math.log(4)


def test_the_scoreboard_numbers_in_the_lesson():
    train, val = split(TEXT)
    counts = count(train)
    P = table(counts, VOCAB, smooth=1)
    assert len(VOCAB) == 65
    assert counts["q", "u"] == 563 == sum(n for (a, _), n in counts.items() if a == "q")
    assert (counts[" ", "t"], counts[" ", "z"]) == (21591, 9)
    assert round(math.log2(65), 3) == 6.022
    assert round(loss(P, train), 3) == 2.455
    assert round(loss(P, val) / math.log(2), 3) == 3.581


def test_a_sample_follows_the_row_of_its_last_character():
    P = table(count("abcabc"), "abc")  # a is followed by b, b by c and c by a, each time
    assert sample(P, "a", 5, random.Random(0)) == "abcabc"


def test_the_same_seed_writes_the_same_text():
    P = table(count(TEXT), VOCAB, smooth=1)
    assert sample(P, "Once", 20, random.Random(0)) == sample(P, "Once", 20, random.Random(0))


def test_adding_a_thousand_to_each_count_flattens_the_table():
    train, val = split(TEXT)
    assert round(loss(table(count(train), VOCAB, smooth=1000), val), 3) == 3.141
