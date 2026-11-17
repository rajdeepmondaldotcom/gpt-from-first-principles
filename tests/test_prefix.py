import torch

from gpt.cost import cache_bytes
from gpt.engine import Engine, Request
from gpt.generate import generate
from gpt.model import GPT
from gpt.prefix import PrefixStore

SYSTEM = list(range(20, 60))  # 40 tokens that each prompt begins with


def model():
    torch.manual_seed(0)
    return GPT(vocab=65, context=128, width=32, heads=4, layers=2, kv_heads=2)


def run(net, store, prompt, limit=6):
    """Serve one request. Returns what it wrote and how many prompt tokens the model read."""
    read = []
    hook = net.embed.register_forward_hook(lambda part, inputs, out: read.append(inputs[0].numel()))
    engine = Engine(net, 10**9, store=store)
    request = Request(prompt, limit, how={"temperature": 0})
    assert engine.admit(request)
    engine.step()
    hook.remove()
    while engine.reading or engine.rows:
        engine.step()
    return request.written, read[0]


def alone(net, prompt, limit=6):
    return generate(net, prompt, limit, temperature=0)[len(prompt) :]


def test_a_shared_beginning_is_read_once_and_the_answers_do_not_change():
    net, store = model(), PrefixStore(budget=10**9)
    first, second = [*SYSTEM, 1, 2, 3], [*SYSTEM, 7, 8, 9, 10]
    written, read = run(net, store, first)
    assert read == 43 and written == alone(net, first)  # nothing stored yet: all 43 tokens read
    written, read = run(net, store, second)
    assert read == 44 - 32 and written == alone(
        net, second
    )  # two whole pieces of 16 came from the store


def test_reuse_stops_at_the_first_token_that_differs():
    net, store = model(), PrefixStore(budget=10**9)
    run(net, store, [*SYSTEM, 1, 2, 3])
    changed = list(SYSTEM)
    changed[20] = 0  # inside the second piece
    written, read = run(net, store, [*changed, 1, 2, 3])
    assert read == 43 - 16 and written == alone(net, [*changed, 1, 2, 3])  # only the first piece


def test_putting_the_part_that_changes_first_loses_all_of_it():
    net, store = model(), PrefixStore(budget=10**9)
    run(net, store, [1, *SYSTEM])
    _, read = run(net, store, [2, *SYSTEM])
    assert read == 41


def test_the_same_prompt_again_still_reads_its_last_token():
    net, store = model(), PrefixStore(budget=10**9)
    prompt = SYSTEM[:32]  # exactly two pieces
    first, _ = run(net, store, prompt)
    again, read = run(net, store, prompt)
    assert (
        again == first and read == 16
    )  # the last piece is read again, to get a score to start from


def test_when_the_store_is_full_the_piece_used_longest_ago_goes():
    net = model()
    one_piece = cache_bytes(net, 16)  # the bytes of one piece of 16 tokens
    store = PrefixStore(budget=2 * one_piece)
    a, b, c = ([n] * 17 for n in (1, 2, 3))  # three prompts, one piece each
    run(net, store, a)
    run(net, store, b)
    assert run(net, store, a)[1] == 1  # a is used again, so b is now the oldest
    run(net, store, c)  # no room for three: b goes
    assert store.used == 2 * one_piece
    assert run(net, store, a)[1] == 1 and run(net, store, c)[1] == 1 and run(net, store, b)[1] == 17


def test_a_piece_is_never_left_behind_without_the_shorter_ones_it_needs():
    net = model()
    store = PrefixStore(budget=4 * cache_bytes(net, 16))  # room for four pieces
    long = list(range(1, 50))  # three pieces: tokens 1 to 16, to 32, to 48
    run(net, store, long)
    run(net, store, [*long[:16], *range(40, 57)])  # shares the first piece, and adds one of its own
    run(net, store, [60] * 17)  # a fifth piece: something has to go
    for ids in store.pieces:  # each piece still stored can be reached: its shorter ones are there
        assert all(ids[:end] in store.pieces for end in range(16, len(ids), 16))
    assert store.used == 4 * cache_bytes(net, 16)
