import asyncio
from collections.abc import AsyncIterator

from sleeper.text import iter_words


def test_iter_words_reassembles_words_across_chunks() -> None:
  async def chunks() -> AsyncIterator[str]:
    yield "Hello "
    yield "wor"
    yield "ld\nagain"

  async def collect_words() -> list[str]:
    return [word async for word in iter_words(chunks())]

  assert asyncio.run(collect_words()) == ["Hello", "world", "again"]
