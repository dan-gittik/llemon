import mimetypes
import pathlib

import pytest

from llemon import enable_logs, LLM


def pytest_configure(config: pytest.Config) -> None:
    enable_logs()


@pytest.fixture
def example_assets(llm: LLM) -> pathlib.Path:
    """Provide a path to test assets that automatically skips tests for unsupported file types"""

    assets_base = pathlib.Path(__file__).parent.parent / "examples" / "files"

    class LLMCompatibleMimeType(type(assets_base)):
        def __truediv__(self, key):
            result_path = super().__truediv__(key)

            if result_path.is_file():
                mime_type, _ = mimetypes.guess_type(result_path)
                if mime_type and llm.config.accepts_files and mime_type not in llm.config.accepts_files:
                    pytest.skip(f"{llm} doesn't accept {mime_type} files (required for {result_path.name})")

            return result_path

    return LLMCompatibleMimeType(assets_base)