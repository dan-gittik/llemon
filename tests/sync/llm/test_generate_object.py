from typing import Any, Callable

import pytest
from pydantic import BaseModel, Field

from llemon.sync import GenerateObjectResponse, LLM


class Person(BaseModel):
    name: str
    age: int | None
    hobbies: list[str] = Field(default_factory=list)


PERSON_SCHEMA = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
        },
        "age": {
            "type": "integer",
        },
        "hobbies": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["name", "hobbies"],
}


def requires_structured_output(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(llm: LLM, *args: Any, **kwargs: Any) -> Any:
        if not llm.config.supports_json:
            pytest.skip(f"{llm} doesn't support structured output")
        return function(llm, *args, **kwargs)

    return wrapper


@requires_structured_output
def test_generate_object(llm: LLM):
    response = llm.generate_object(
        Person,
        "Extract information about the person.",
        "Hello, my name is Alice and I like reading and hiking.",
    )
    assert response.object.name == "Alice"
    assert response.object.age is None
    assert response.object.hobbies == ["reading", "hiking"]
    response = llm.generate_object(
        Person,
        "Extract information about the person.",
        "Hello, my name is Bob, I'm 25, and I like cooking.",
    )
    assert response.object.name == "Bob"
    assert response.object.age == 25
    assert response.object.hobbies == ["cooking"]


@requires_structured_output
def test_generate_dict(llm: LLM):
    response: GenerateObjectResponse[Person] = llm.generate_object(
        PERSON_SCHEMA,
        "Extract information about the person.",
        "Hello, my name is Alice and I like reading and hiking.",
    )
    assert response.object.name == "Alice"
    assert response.object.age is None
    assert response.object.hobbies == ["reading", "hiking"]
    response = llm.generate_object(
        PERSON_SCHEMA,
        "Extract information about the person.",
        "Hello, my name is Bob, I'm 25, and I like cooking.",
    )
    assert response.object.name == "Bob"
    assert response.object.age == 25
    assert response.object.hobbies == ["cooking"]
