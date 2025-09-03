import warnings

import pytest

from llemon.sync import Warning


@pytest.fixture(autouse=True)
def warning_to_error() -> None:
    warnings.filterwarnings("error", category=Warning)
