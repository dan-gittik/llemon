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


def pytest_configure(config):
    enable_logs()


@pytest.fixture(
    params=[
        pytest.param((OpenAI, "gpt-5-nano"), id="gpt-5-nano"),
        pytest.param((Anthropic, "claude-3-haiku-20240307"), id="claude-3-haiku-3"),
        pytest.param((Gemini, "gemini-2.0-flash-lite"), id="gemini-2.0-flash-lite"),
        pytest.param((DeepInfra, "meta-llama/Meta-Llama-3.1-8B-Instruct"), id="deepinfra-llama-3.1-8b"),
    ]
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
