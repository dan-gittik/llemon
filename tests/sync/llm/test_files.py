from functools import wraps
import pathlib
from typing import Any, Callable

import pytest

from llemon.sync import LLMModel


def accepts_files(*mimetypes: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(model: LLMModel, *args: Any, **kwargs: Any) -> Any:
            accepted_files = model.config.accepts_files or []
            for mimetype in mimetypes:
                if mimetype not in accepted_files:
                    pytest.skip(f"model {model} doesn't accept {mimetype} files")
            return function(model, *args, **kwargs)

        return wrapper

    return decorator


@accepts_files("image/png")
def test_generate_with_file(model: LLMModel, example_assets: pathlib.Path) -> None:
    my_file = example_assets / "hello.png"
    response = model.generate("What is written in the file? Respond with the text only.", files=[my_file])
    assert response.text == "Hello, world!"


@accepts_files("image/jpeg", "image/png")
def test_generate_with_multiple_files(model: LLMModel, example_assets: pathlib.Path) -> None:
    cat_file = example_assets / "cat.jpg"
    dog_file = example_assets / "dog.png"
    response = model.generate(
        "Which animals are in the pictures? Respond with the simplest single word for each.",
        files=[cat_file, dog_file],
    )
    expected_animals = ["cat", "dog"]
    assert all(animal in response.text.lower() for animal in expected_animals)


def test_generate_with_unexisting_file(model: LLMModel, tmp_path: pathlib.Path) -> None:
    my_file = tmp_path / "foo"
    with pytest.raises(FileNotFoundError):
        model.generate("What is written in the file?", files=[my_file])


def test_generate_with_not_a_file(model: LLMModel, tmp_path: pathlib.Path) -> None:
    with pytest.raises(IsADirectoryError):
        model.generate("What is written in the file?", files=[tmp_path])
