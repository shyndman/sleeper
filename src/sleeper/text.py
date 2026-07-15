"""Utilities for processing asynchronously streamed text."""

import re
from collections.abc import AsyncIterable, AsyncIterator

WORD_BREAK = re.compile(r"\s+")


async def iter_words(chunks: AsyncIterable[str]) -> AsyncIterator[str]:
  """Yield whitespace-delimited words reconstructed across text chunks.

  A final word does not need trailing whitespace; it is emitted when the input
  iterable ends normally.
  """
  partial_word = ""

  async for chunk in chunks:
    partial_word += chunk
    # Yield completed words now; retain only the unfinished tail for
    # completion by the next streamed chunk.
    *words, partial_word = WORD_BREAK.split(partial_word)
    for word in words:
      if word:
        yield word

  if partial_word:
    yield partial_word
