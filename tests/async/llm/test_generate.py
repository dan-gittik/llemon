import pytest

from llemon import LLM

pytestmark = pytest.mark.asyncio


async def test_generate(llm: LLM):
    response = await llm.generate("What's 2 + 2? Answer with a single digit and no punctuation.")
    assert response.text == "4"
