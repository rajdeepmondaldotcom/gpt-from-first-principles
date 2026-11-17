import pytest


class Bytes:
    """A tokenizer of single bytes, for tests. 256 ends a text."""

    end = 256

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors="replace")


@pytest.fixture
def words():
    return Bytes()
