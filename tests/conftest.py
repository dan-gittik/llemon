from pathlib import Path

import pytest

from llemon import enable_logs


def pytest_configure(config):
    enable_logs()


@pytest.fixture
def example_assets() -> Path:
    return Path(__file__).parent.parent / "examples" / "files"
