from typing import Callable, Any

from llemon import LLMModel
import pytest

pytestmark = pytest.mark.asyncio


def requires_multiple_responses(function: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(model: LLMModel, *args: Any, **kwargs: Any) -> Any:
        if not model.config.supports_multiple_responses:
            pytest.skip(f"model {model} doesn't support multiple responses")
        return await function(model, *args, **kwargs)
    return wrapper


async def test_single(model: LLMModel):
    completion = await model.complete("What's 2 + 2? Answer with a single digit and no punctuation.")
    assert completion.strip() == "4"


@requires_multiple_responses
async def test_multiple(model: LLMModel):
    completions = await model.complete(
        f"Choose a a random number between 100 and 1000. Answer with a digits only and no punctuation.",
        num_responses=3,
        temperature=1.0,
    )
    assert len(completions) == 3
    assert len(set(completions)) != 1
    for completion in completions:
        assert 100 <= int(completion) <= 1000