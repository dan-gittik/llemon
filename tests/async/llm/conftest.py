from typing import Iterator

import pytest

from llemon import (
    LLM,
    Anthropic,
    DeepInfra,
    Error,
    Gemini,
    LLMProvider,
    OpenAI,
)


@pytest.fixture(
    params=[
        pytest.param((OpenAI, "gpt-5-nano"), id="gpt-5-nano"),
        pytest.param((Anthropic, "claude-3-haiku-20240307"), id="claude-3-haiku-3"),
        pytest.param((Gemini, "gemini-2.0-flash-lite"), id="gemini-2.0-flash-lite"),
        pytest.param((DeepInfra, "meta-llama/Meta-Llama-3.1-8B-Instruct"), id="deepinfra-llama-3.1-8b"),
    ]
)
def llm(request: pytest.FixtureRequest) -> Iterator[LLM]:
    provider: type[LLMProvider] = request.param[0]
    model: str = request.param[1]
    try:
        yield provider.llm(model)
    except Error:
        pytest.skip(f"provider {provider.__name__} isn't available")
    provider.llms.clear()
    provider.instance = None
