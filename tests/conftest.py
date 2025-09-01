import mimetypes
import pathlib

import pytest

from llemon import enable_logs, LLM
from llemon.sync import LLM_Sync


def pytest_configure(config: pytest.Config) -> None:
    enable_logs()


@pytest.fixture
def example_assets(llm: LLM | LLM_Sync) -> pathlib.Path:
    """Provide a path to test assets that automatically skips tests for unsupported file types"""

    assets_base = pathlib.Path(__file__).parent.parent / "examples" / "files"

    class LLMCompatibleMimeType(type(assets_base)):
        def __truediv__(self, key):
            file_path = super().__truediv__(key)

            if file_path.is_file():
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type and llm.config.accepts_files and mime_type not in llm.config.accepts_files:
                    pytest.skip(f"{llm} doesn't accept {mime_type=} files (required for {file_path.name})")

            return file_path

    return LLMCompatibleMimeType(assets_base)