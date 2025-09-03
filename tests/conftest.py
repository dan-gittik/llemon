import pathlib

import pytest

from llemon import enable_logs


@pytest.fixture(autouse=True)
def logs() -> None:
    enable_logs()


@pytest.fixture
def example_assets() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "examples" / "files"
