import pathlib
from functools import wraps
from typing import Any, Callable

import pytest

from llemon.sync import LLM


def accepts_files(*mimetypes: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(llm: LLM, *args: Any, **kwargs: Any) -> Any:
            accepted_files = llm.config.accepts_files or []
            for mimetype in mimetypes:
                if mimetype not in accepted_files:
                    pytest.skip(f"{llm} doesn't accept {mimetype} files")
            return function(llm, *args, **kwargs)

        return wrapper

    return decorator


@accepts_files("image/png")
def test_generate_with_file(llm: LLM, example_assets: pathlib.Path) -> None:
    my_file = example_assets / "hello.png"
    response = llm.generate("What is written in the file? Respond with the text only.", files=[my_file])
    assert response.text == "Hello, world!"


@accepts_files("image/jpeg", "image/png")
def test_generate_with_multiple_files(llm: LLM, example_assets: pathlib.Path) -> None:
    cat_file = example_assets / "cat.jpg"
    dog_file = example_assets / "dog.png"
    response = llm.generate(
        "Which animals are in the pictures? Respond with the simplest single word for each.",
        files=[cat_file, dog_file],
    )
    expected_animals = ["cat", "dog"]
    assert all(animal in response.text.lower() for animal in expected_animals)


def test_generate_with_unexisting_file(llm: LLM, tmp_path: pathlib.Path) -> None:
    my_file = tmp_path / "foo"
    with pytest.raises(FileNotFoundError):
        llm.generate("What is written in the file?", files=[my_file])


def test_generate_with_not_a_file(llm: LLM, tmp_path: pathlib.Path) -> None:
    with pytest.raises(IsADirectoryError):
        llm.generate("What is written in the file?", files=[tmp_path])
