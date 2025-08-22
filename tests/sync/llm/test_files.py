import pytest

from llemon import LLMModel


def test_generate_with_file(model: LLMModel, example_assets):
    my_file = example_assets / "hello.png"
    response = model.generate("What is written in the file? respond with the text only", files=[my_file])
    assert response.text == "Hello, world!"


def test_generate_with_multiple_files(model: LLMModel, example_assets):
    cat_file = example_assets / "cat.jpg"
    dog_file = example_assets / "dog.png"
    response = model.generate("Which animals are in the pictures? in 1 word only", files=[cat_file, dog_file])
    expected_animals = ["cat", "dog"]
    assert all(animal in response.text.lower() for animal in expected_animals)


def test_generate_with_unexisting_file(model: LLMModel, tmp_path):
    my_file = tmp_path / "foo"
    with pytest.raises(FileNotFoundError):
        model.generate("What is written in the file?", files=[my_file])
