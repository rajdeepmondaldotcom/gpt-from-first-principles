import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch

from gpt.bpe import Tokenizer
from gpt.cost import cache_bytes
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
    service.stop()
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


def test_an_answer_stops_at_the_models_context(running):
    net, _, url = running  # a context of 64 tokens, and 10 of them are the prompt
    assert ask(url, "Once upon ", limit=500) == alone(net, "Once upon ", limit=54)


def test_a_prompt_that_fills_the_context_is_refused_with_the_reason(running):
    _, _, url = running
    with pytest.raises(urllib.error.HTTPError) as refused:
        ask(url, "x" * 64)
    assert refused.value.code == 400 and b"context" in refused.value.read()


def test_a_reader_who_leaves_frees_their_place():
    torch.manual_seed(0)
    roomy = GPT(vocab=257, context=5002, width=32, heads=4, layers=2, kv_heads=2)
    service = Service(roomy, WORDS, budget=10**9)
    server = serve(service, port=0)
    url = f"http://127.0.0.1:{server.server_port}"
    reply = post(url, "Go", stream=True, limit=5000)
    reply.readline()
    reply.close()  # the reader walks away after one piece
    for _ in range(400):
        if not service.outs:
            break
        time.sleep(0.01)
    assert not service.outs and service.counts["tokens"] < 5000
    service.stop()
    server.shutdown()


def test_a_request_too_big_for_the_whole_memory_gets_a_plain_no_at_once():
    service = Service(model(), WORDS, budget=1)
    server = serve(service, port=0)
    with pytest.raises(urllib.error.HTTPError) as refused:
        ask(f"http://127.0.0.1:{server.server_port}", "Hello")
    assert refused.value.code == 503
    service.stop()
    server.shutdown()


def test_with_memory_for_one_request_the_next_waits_and_is_answered():
    net = model()
    service = Service(net, WORDS, budget=cache_bytes(net, 40))  # room for one of these two
    server = serve(service, port=0)
    url, got = f"http://127.0.0.1:{server.server_port}", {}
    prompts = ["Once upon a time", "The cat sat on a"]  # 16 tokens each, and 20 more to write
    threads = [threading.Thread(target=lambda p=p: got.update({p: ask(url, p)})) for p in prompts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert got == {p: alone(net, p) for p in prompts} and service.counts["refused"] == 0
    service.stop()
    server.shutdown()


def test_a_full_waiting_line_turns_the_next_one_away():
    service = Service(model(), WORDS, budget=10**9, waiting=2)
    service.stop()  # the engine's thread has ended, so nothing leaves the line
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


def refused_with(url, body):
    """Post a body that should be turned down. Returns the status and the reason given."""
    request = urllib.request.Request(url + "/v1/chat/completions", body)
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=30)
    return refused.value.code, refused.value.read().decode()


def test_a_request_we_cannot_serve_gets_a_reason_and_the_next_one_is_answered(running):
    net, service, url = running
    message = [{"role": "user", "content": "Once upon"}]
    for wrong in (
        {"messages": message, "temperature": "0.5"},  # a setting that is not a number
        {"messages": message, "max_tokens": 0},  # nothing to write
        {"messages": message, "top_p": 0},
        {"messages": [{"role": "user", "content": ""}]},  # an empty prompt
        {"no messages": True},
    ):
        code, reason = refused_with(url, json.dumps(wrong).encode())
        assert code == 400 and "error" in reason
    assert refused_with(url, b"this is not JSON")[0] == 400
    assert ask(url, "Once upon") == alone(net, "Once upon")  # the engine's thread is still there
    assert service.counts["finished"] == 1


def test_a_pass_that_fails_ends_its_streams_and_the_service_goes_on(running, monkeypatch):
    net, service, url = running
    step = service.engine.step

    def fail_once():
        monkeypatch.setattr(service.engine, "step", step)
        raise RuntimeError("a pass went wrong")

    monkeypatch.setattr(service.engine, "step", fail_once)
    with pytest.raises(urllib.error.HTTPError) as refused:
        ask(url, "Once upon")
    assert refused.value.code == 503 and service.counts["failed"] == 1
    assert ask(url, "Once upon") == alone(net, "Once upon")  # a new engine, the same answers


def test_requests_in_line_count_as_inside():
    net = model()
    service = Service(net, WORDS, budget=cache_bytes(net, 60))  # memory for one of these at a time
    server = serve(service, port=0)
    url, prompts = (
        f"http://127.0.0.1:{server.server_port}",
        ["Once upon a time", "The cat sat on a"],
    )
    threads = [threading.Thread(target=ask, args=(url, p), kwargs={"limit": 40}) for p in prompts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    counts = service.counts
    assert counts["finished"] == 2 and counts["refused"] == 0
    # Time inside, added up request by request, equals requests inside added up over time. With
    # the waiting one left out, the second sum would come short by its whole wait.
    assert counts["in_flight_seconds"] == pytest.approx(counts["seconds_in_system"], rel=0.15)
    service.stop()
    server.shutdown()
