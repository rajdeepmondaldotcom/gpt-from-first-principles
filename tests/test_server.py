import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch

from gpt.bpe import Tokenizer
from gpt.generate import generate
from gpt.model import GPT
from gpt.server import Service, pieces, serve

WORDS = Tokenizer(
    {bytes([b]): b for b in range(256)}
)  # one token for each byte, then the end token


def model():
    torch.manual_seed(0)
    return GPT(vocab=257, context=64, width=32, heads=4, layers=2, kv_heads=2)


@pytest.fixture
def running():
    net = model()
    service = Service(net, WORDS, budget=10**9)
    server = serve(service, port=0)  # port 0: any free port
    yield net, service, f"http://127.0.0.1:{server.server_port}"
    service.running = False
    server.shutdown()


def post(url, prompt, stream, limit):
    ask = {"messages": [{"role": "user", "content": prompt}], "max_tokens": limit, "temperature": 0}
    body = json.dumps(ask | {"stream": stream}).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", body)
    return urllib.request.urlopen(request, timeout=30)


def ask(url, prompt, stream=False, limit=20):
    with post(url, prompt, stream, limit) as reply:
        if not stream:
            return json.loads(reply.read())["choices"][0]["message"]["content"]
        lines = [line.decode().strip() for line in reply]
    assert lines[-2] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    return "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks)


def alone(net, prompt, limit=20):
    ids = WORDS.encode(prompt)
    written = generate(net, ids, limit, stop=WORDS.end, temperature=0)[len(ids) :]
    return "".join(pieces(WORDS, written))


def test_half_a_character_waits_for_its_other_half():
    e_acute = list("é".encode())  # two bytes, so two tokens
    assert list(pieces(WORDS, [ord("a"), *e_acute, ord("b")])) == ["a", "é", "b"]
    assert list(pieces(WORDS, e_acute[:1])) == []  # half of it alone is not sent
    assert list(pieces(WORDS, [ord("a"), WORDS.end])) == ["a"]  # the end token is not text


def test_an_answer_over_the_web_is_the_models_own_answer(running):
    net, _, url = running
    assert ask(url, "Once upon") == alone(net, "Once upon")


def test_a_stream_adds_up_to_the_whole_answer(running):
    net, _, url = running
    assert ask(url, "A cat", stream=True) == alone(net, "A cat")


def test_people_asking_at_once_each_get_their_own_answer(running):
    net, service, url = running
    prompts = ["One", "Two cats", "Three little birds sat", "4", "Five five five", "The sixth"]
    got = {}
    threads = [
        threading.Thread(target=lambda p=p: got.update({p: ask(url, p, stream=True, limit=30)}))
        for p in prompts
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert got == {p: alone(net, p, limit=30) for p in prompts}
    assert service.counts["finished"] == 6


def test_a_reader_who_leaves_frees_their_place(running):
    _, service, url = running
    reply = post(url, "Go", stream=True, limit=5000)
    reply.readline()
    reply.close()  # the reader walks away after one piece
    for _ in range(400):
        if not service.outs:
            break
        time.sleep(0.01)
    assert not service.outs and service.counts["tokens"] < 5000


def test_no_memory_means_a_plain_no_at_once():
    service = Service(model(), WORDS, budget=1)
    server = serve(service, port=0)
    with pytest.raises(urllib.error.HTTPError) as refused:
        ask(f"http://127.0.0.1:{server.server_port}", "Hello")
    assert refused.value.code == 503
    service.running = False
    server.shutdown()


def test_a_full_waiting_line_turns_the_next_one_away():
    service = Service(model(), WORDS, budget=10**9, waiting=2)
    service.running = False  # stop the engine's thread, so nothing leaves the line
    time.sleep(0.05)
    assert service.submit("a", 5) and service.submit("b", 5)
    assert service.submit("c", 5) is None and service.counts["refused"] == 1


def test_the_metrics_page_counts_what_happened(running):
    _, service, url = running
    for prompt in ("a", "bb", "ccc"):
        ask(url, prompt, limit=10)
    with urllib.request.urlopen(url + "/metrics") as reply:
        lines = dict(line.split() for line in reply.read().decode().splitlines())
    assert lines["stories_finished"] == "3" and lines["stories_in_flight"] == "0"
    assert int(lines["stories_tokens"]) == service.counts["tokens"] <= 30
    # Time in the system, added up request by request, equals requests in flight added up over time.
    assert service.counts["in_flight_seconds"] == pytest.approx(
        service.counts["seconds_in_system"], rel=0.2, abs=0.01
    )
