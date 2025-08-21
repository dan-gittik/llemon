import pytest

from llemon import LLMModel

pytestmark = pytest.mark.asyncio


async def test_generate(model: LLMModel):
    response = await model.generate("What's 2 + 2? Answer with a single digit and no punctuation.")
    assert response.text == "4"
