import pytest
import torch

from gpt.engine import Engine, Request
from gpt.measure import arrivals, percentile, report, serve
from gpt.model import GPT


def test_percentiles_by_hand():
    values = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10]
    assert percentile(values, 50) == 5  # five of the ten values are at or below 5
    assert percentile(values, 95) == 10 and percentile(values, 10) == 1
    assert percentile([4.2], 99) == 4.2


def test_a_slow_tail_shows_in_the_95th_and_not_in_the_middle():
    waits = [0.1] * 90 + [3.0] * 10
    assert percentile(waits, 50) == 0.1 and percentile(waits, 95) == 3.0
    assert sum(waits) / 100 == pytest.approx(0.39)  # the mean describes nobody's wait


class Ticks:
    """A clock that moves one second each time the engine takes a step."""

    def __init__(self, engine):
        self.now, step = 0.0, engine.step

        def counted():
            out = step()
            self.now += 1.0
            return out

        engine.step = counted

    def __call__(self):
        return self.now


def model():
    torch.manual_seed(0)
    return GPT(vocab=65, context=64, width=32, heads=4, layers=2, kv_heads=2)


def test_the_report_by_hand_on_a_clock_we_control():
    engine = Engine(model(), budget=10**9)
    clock = Ticks(engine)
    a, b = (Request([5, 9, 2], 4, how={"temperature": 0}) for _ in range(2))
    records = serve(engine, [(0.0, a), (2.0, b)], clock)
    # a: arrives at 0, reads its prompt in step 1 and writes a token in each of steps 1 to 4.
    assert records[0] == {"arrived": 0.0, "tokens": [1.0, 2.0, 3.0, 4.0]}
    # b: arrives at 2, is taken in before step 3, and writes in steps 3 to 6.
    assert records[1] == {"arrived": 2.0, "tokens": [3.0, 4.0, 5.0, 6.0]}
    numbers = report(records)
    # Each waited one step from its own arrival. From time 0, b's wait would read as 3.
    assert numbers["first_token_p50"] == 1.0 and numbers["first_token_p95"] == 1.0
    assert numbers["seconds_a_token_p50"] == 1.0
    assert numbers["tokens_a_second"] == pytest.approx(8 / 6)  # 8 tokens between time 0 and time 6
    assert numbers["served"] == 2 and numbers["refused"] == 0


def test_a_refused_request_is_counted_and_gets_no_tokens():
    engine = Engine(model(), budget=1)  # room for nobody
    records = serve(engine, [(0.0, Request([5, 9, 2], 4))], Ticks(engine))
    assert records == [{"arrived": 0.0, "tokens": []}]


def test_with_room_for_one_the_second_waits_in_line():
    net = model()
    room = Engine(net, 0).bytes_for(Request([5, 9, 2], 4))
    engine = Engine(net, budget=room)
    a, b = (Request([5, 9, 2], 4, how={"temperature": 0}) for _ in range(2))
    records = serve(engine, [(0.0, a), (0.0, b)], Ticks(engine))
    assert records[0]["tokens"] == [1.0, 2.0, 3.0, 4.0]
    assert records[1]["tokens"] == [5.0, 6.0, 7.0, 8.0]  # served, after a wait of 4 steps
    assert a.written == b.written  # and the wait changed nothing it wrote


def test_arrivals_come_at_the_rate_asked_for():
    generator = torch.Generator().manual_seed(0)
    line = arrivals([[1, 2]] * 2000, rate=4.0, chance=generator, limit=3, temperature=0)
    times = [at for at, _ in line]
    assert times == sorted(times) and times[-1] / 2000 == pytest.approx(1 / 4, rel=0.1)
    assert line[0][1].prompt == [1, 2] and line[0][1].limit == 3
    assert line[0][1].how == {"temperature": 0}


def test_an_idle_engine_waits_for_the_next_arrival():
    engine = Engine(model(), budget=10**9)
    clock = Ticks(engine)
    late = Request([5, 9, 2], 2, how={"temperature": 0})
    records = serve(engine, [(50.0, late)], clock)
    assert records[0]["tokens"][0] - records[0]["arrived"] == 1.0  # it did not wait 50 steps
