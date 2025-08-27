import pathlib

import pytest

from llemon import enable_logs


def pytest_configure(config: pytest.Config) -> None:
    enable_logs()


@pytest.fixture
def example_assets() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "examples" / "files"
