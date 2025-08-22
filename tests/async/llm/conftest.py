from typing import Iterator

import pytest

from llemon import (
    LLM,
    Anthropic,
    DeepInfra,
    Gemini,
    LLMModel,
    OpenAI,
    enable_logs,
    errors,
)

enable_logs()


@pytest.fixture(
    params=[
        (OpenAI, "gpt-5-nano"),
        (Anthropic, "claude-3-haiku-20240307"),
        (Gemini, "gemini-2.0-flash-lite"),
        (DeepInfra, "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ],
    ids=[
        "gpt-5-nano",
        "claude-3-haiku-3",
        "gemini-2.0-flash-lite",
        "deepinfra-llama-3.1-8b",
    ],
)
def model(request: pytest.FixtureRequest) -> Iterator[LLMModel]:
    provider: LLM = request.param[0]
    model: str = request.param[1]
    try:
        yield provider.model(model)
    except errors.ConfigurationError:
        pytest.skip(f"provider {provider.__name__} doesn isn't available")
    provider.models.clear()
    provider.instance = None
