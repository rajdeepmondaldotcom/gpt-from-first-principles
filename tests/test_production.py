import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from gpt.measure import arrivals, report
from gpt.model import GPT
from gpt.production import ask, handoff, load, post, scores
from gpt.story import save_llama

SHAPE = {"vocab": 64, "context": 32, "width": 48, "heads": 4, "kv_heads": 2, "layers": 2}
BODIES = []  # each request the stub below was sent


class Stub(BaseHTTPRequestHandler):
    """Speaks the part of vLLM's wire format the lesson uses. After a prompt it writes the
    prompt's last token plus 1, plus 2 and so on. It scores prompt token t at -t / 10."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        BODIES.append(body)
        self.send_response(200 if body["max_tokens"] else 400)
        self.end_headers()
        if body.get("stream"):
            for i in range(body["max_tokens"]):
                choice = {"token_ids": [body["prompt"][-1] + i + 1]}
                if i == 0:  # the first piece repeats the prompt and holds no new token
                    self.wfile.write(b'data: {"choices": [{"prompt_token_ids": [1]}]}\n\n')
                self.wfile.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:  # beside each token's own score, the top token's: here always token 0
            chances = [
                {"0": {"logprob": -0.01}, str(t): {"logprob": -t / 10}} for t in body["prompt"]
            ]
            choice = {"prompt_token_ids": body["prompt"], "prompt_logprobs": [None, *chances[1:]]}
            self.wfile.write(json.dumps({"choices": [choice]}).encode())


@pytest.fixture
def port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    BODIES.clear()
    yield server.server_port
    server.shutdown()


def test_the_handoff_adds_what_vllm_asks_for_and_keeps_lesson_37s_file(tmp_path):
    torch.manual_seed(0)
    save_llama(GPT(**SHAPE), tmp_path / "model.safetensors")
    (tmp_path / "shape.json").write_text(json.dumps(SHAPE))
    handoff(tmp_path, tmp_path / "out")
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["eos_token_id"] == 63  # the end token is the last id (lesson 5)
    assert config["hidden_size"] == 48 and config["num_key_value_heads"] == 2


def test_ask_returns_each_token_with_the_moment_it_came(port):
    ticks = iter(range(100))
    tokens, moments = ask(port, [5, 6, 7], 4, clock=lambda: next(ticks), temperature=0)
    assert tokens == [8, 9, 10, 11]
    assert len(moments) == 4 and moments == sorted(set(moments))  # a later moment for each
    sent = BODIES[0]
    assert sent["prompt"] == [5, 6, 7] and sent["temperature"] == 0 and sent["ignore_eos"]
    assert sent["stream"] and sent["return_token_ids"]  # without these vLLM sends no token ids


def test_a_refusal_is_an_error_not_an_empty_answer(port):
    with pytest.raises(AssertionError):
        post(port, "/v1/completions", max_tokens=0)


def test_load_sends_each_request_and_times_it_from_the_start_of_the_line(port):
    prompts = [[i, i + 1] for i in range(12)]
    line = arrivals(prompts, 200, torch.Generator().manual_seed(0), limit=3, temperature=0.8)
    records = load(port, line)
    assert sorted(r["arrived"] for r in records) == [at for at, _ in line]
    for record in records:
        assert len(record["tokens"]) == 3
        assert record["arrived"] <= record["sent"] <= record["tokens"][0] < record["arrived"] + 1
    assert sorted(body["prompt"] for body in BODIES) == prompts
    assert all(body["temperature"] == 0.8 for body in BODIES)  # each request's own settings go too
    assert report(records)["served"] == 12


def test_scores_are_the_prompt_tokens_own_not_the_top_tokens(port):
    assert scores(port, [3, 20, 7]) == [-2.0, -0.7]
