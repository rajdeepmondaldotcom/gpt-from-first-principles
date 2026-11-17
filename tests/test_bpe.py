from pathlib import Path

import pytest

from gpt.bpe import END, SPLIT, Tokenizer, merge, train

TEXT = Path("data/tiny_shakespeare.txt").read_text()[:20000]
TRICKY = [
    "",
    " ",
    "the cat sat on the mat",
    "naïve café, 12_345 items!?",
    "日本語 🙂\n\ttabs  and   spaces ",
]


def test_hand_example():
    vocab = train("aaabdaaabac", 259)
    assert list(vocab)[256:] == [b"aa", b"ab", b"aaab"]
    tokenizer = Tokenizer(vocab)
    assert tokenizer.encode("aaabdaaabac") == [258, ord("d"), 258, ord("a"), ord("c")]


def test_chunks_cover_the_text_exactly():
    for text in [*TRICKY, TEXT]:
        assert "".join(SPLIT.findall(text)) == text
    assert SPLIT.findall("the cat's 9 lives") == ["the", " cat", "'", "s", " 9", " lives"]


def test_any_text_comes_back_unchanged():
    tokenizer = Tokenizer(train(TEXT, 300))
    for text in [*TRICKY, TEXT, END]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encoding_replays_the_merges_in_the_order_they_were_learned():
    vocab = train(TEXT, 400)
    tokenizer, learned = Tokenizer(vocab), list(vocab)[256:]
    for chunk in set(SPLIT.findall(TEXT)):
        tokens = tuple(bytes([b]) for b in chunk.encode())
        for token in learned:  # the textbook way: apply each merge in turn
            for cut in range(1, len(token)):
                if token[:cut] in vocab and token[cut:] in vocab:
                    tokens = merge(tokens, (token[:cut], token[cut:]))
        assert tokenizer.encode_chunk(chunk) == [vocab[t] for t in tokens]


def test_more_merges_mean_fewer_tokens():
    lengths = [len(Tokenizer(train(TEXT, size)).encode(TEXT)) for size in (256, 300, 400)]
    assert lengths[0] == len(TEXT.encode()) and lengths[0] > lengths[1] > lengths[2]


def test_the_end_token_comes_only_when_asked_for():
    tokenizer = Tokenizer(train(TEXT, 300))
    text = f"one{END}two"
    marked = tokenizer.encode(text, special=True)
    assert marked == [*tokenizer.encode("one"), tokenizer.end, *tokenizer.encode("two")]
    assert tokenizer.end not in tokenizer.encode(text)  # typed by a user, it is plain text
    assert tokenizer.decode(marked) == tokenizer.decode(tokenizer.encode(text)) == text


def test_half_a_character_decodes_to_a_placeholder():
    tokenizer = Tokenizer(train("", 256))
    first, second = tokenizer.encode("é")  # two bytes in UTF-8
    assert tokenizer.decode([first]) == "�" and tokenizer.decode([first, second]) == "é"


@pytest.mark.reference
def test_with_gpt2s_vocabulary_we_equal_tiktoken():
    regex, tiktoken = pytest.importorskip("regex"), pytest.importorskip("tiktoken")
    try:
        theirs = tiktoken.get_encoding("gpt2")  # downloads the vocabulary on first use
    except OSError as error:
        pytest.skip(f"no GPT-2 vocabulary: {error}")
    gpt2_split = regex.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    ours = Tokenizer(dict(theirs._mergeable_ranks), split=gpt2_split)
    assert ours.end == theirs.eot_token == 50256
    for text in [*TRICKY, TEXT, "Once upon a time", "antidisestablishmentarianism's"]:
        assert ours.encode(text) == theirs.encode(text)
    marked = f"one{END}two"
    assert ours.encode(marked, special=True) == theirs.encode(marked, allowed_special="all")
