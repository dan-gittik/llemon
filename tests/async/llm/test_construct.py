from typing import Callable, Any

from llemon import LLMModel
from pydantic import BaseModel, Field
import pytest

pytestmark = pytest.mark.asyncio


class Person(BaseModel):
    name: str
    age: int | None = None
    hobbies: list[str] = Field(default_factory=list)


def requires_structured_output(function: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(model: LLMModel, *args: Any, **kwargs: Any) -> Any:
        if not model.config.supports_json:
            pytest.skip(f"model {model} doesn't support structured output")
        return await function(model, *args, **kwargs)
    return wrapper


@requires_structured_output
async def test_single(model: LLMModel):
    person = await model.construct(
        Person,
        "Extract information about the person.",
        "Hello, my name is Alice and I like cooking.",
    )
    assert person.name == "Alice"
    assert person.age is None
    assert person.hobbies == ["cooking"]
    person = await model.construct(
        Person,
        "Extract information about the person.",
        "Hello, my name is Bob, I'm 25, and I like reading and hiking.",
    )
    assert person.name == "Bob"
    assert person.age == 25
    assert person.hobbies == ["reading", "hiking"]


@requires_structured_output
async def test_multiple(model: LLMModel):
    names = ["Alice", "Bob", "Charlie"]
    people = await model.construct(Person,
        f"Select a name from this list: {', '.join(names)}.",
        num_responses=3,
    )
    assert len(people) == 3
    assert len({person.name for person in people}) != 1
    for person in people:
        assert person.name in names


async def test_single_dict():
    pass


async def test_multiple():
    pass


async def test_multiple_dict():
    pass