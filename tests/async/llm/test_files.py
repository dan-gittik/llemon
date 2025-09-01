import pathlib

import pytest

from llemon import LLM

pytestmark = pytest.mark.asyncio


async def test_generate_with_file(llm: LLM, example_assets: pathlib.Path) -> None:
    my_file = example_assets / "hello.png"
    response = await llm.generate("What is written in the file? Respond with the text only.", files=[my_file])
    assert response.text == "Hello, world!"


async def test_generate_with_multiple_files(llm: LLM, example_assets: pathlib.Path) -> None:
    cat_file = example_assets / "cat.jpg"
    dog_file = example_assets / "dog.png"
    response = await llm.generate(
        "Which animals are in the pictures? Respond with the simplest single word for each.",
        files=[cat_file, dog_file],
    )
    expected_animals = ["cat", "dog"]
    assert all(animal in response.text.lower() for animal in expected_animals)


async def test_generate_with_unexisting_file(llm: LLM, tmp_path: pathlib.Path) -> None:
    my_file = tmp_path / "foo"
    with pytest.raises(FileNotFoundError):
        await llm.generate("What is written in the file?", files=[my_file])


async def test_generate_with_not_a_file(llm: LLM, tmp_path: pathlib.Path) -> None:
    with pytest.raises(IsADirectoryError):
        await llm.generate("What is written in the file?", files=[tmp_path])
