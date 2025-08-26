import functools
import http.server
import pathlib
import re
import shutil
import subprocess
import tomllib
from typing import Any

import click
import uvicorn

ROOT = pathlib.Path(__file__).parent
PACKAGE: str = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"]
SERVER_PORT = 8000
LINE_LENGTH = 120
COVERAGE_PORT = 8888
ARTEFACTS = [
    ".pytest_cache",
    ".coverage",
    "htmlcov",
    ".mypy_cache",
]


@click.group()
def main() -> None:
    pass


@main.command()
def run() -> None:
    print(f"http://localhost:{SERVER_PORT}")
    uvicorn.run("norm.web.app:app", host="0.0.0.0", port=SERVER_PORT, reload=True)


@main.command()
def clean() -> None:
    for path in ROOT.rglob("*"):
        if path.name not in ARTEFACTS:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@main.command()
@click.argument("tests", nargs=-1)
def test(tests: list[str]) -> None:
    options: list[str] = []
    for test in tests:
        options.extend(["-k", test])
    _execute("pytest", "tests", "-x", "-vv", "--ff", *options)


@main.command()
def cov() -> None:
    _execute("pytest", f"--cov={PACKAGE}", "--cov-report=html", "tests")
    _serve(ROOT / "htmlcov", COVERAGE_PORT)


@main.command()
@click.argument("paths", nargs=-1)
def lint(paths: list[str]) -> None:
    targets: list[pathlib.Path] = []
    for path in paths:
        path = ROOT / PACKAGE / path.replace(".", "/")
        if not path.exists():
            path = path.with_suffix(".py")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        targets.append(path)
    if not targets:
        targets.extend([ROOT / PACKAGE, ROOT / "tests"])
    for target in targets:
        _execute("black", f"--line-length={LINE_LENGTH}", target)
        _execute("isort", f"-w {LINE_LENGTH}", "--profile=black", target)
        _execute("flake8", f"--max-line-length={LINE_LENGTH}", "--extend-ignore=E203,E402", target)


@main.command()
@click.argument("packages", nargs=-1)
def type(packages: list[str]) -> None:
    targets: list[str] = []
    for package in packages:
        targets.extend(["-p", f"{PACKAGE}.{package}"])
    if not packages:
        targets.extend(["-p", PACKAGE])
    _execute("mypy", *targets)


@main.command()
def sync() -> None:
    filenames = [
        "types.py",
        "llm.py",
        "llm_model.py",
        "llm_tokenizer.py",
        "generate.py",
        "generate_stream.py",
        "generate_object.py",
        "classify.py",
        "conversation.py",
        "openai.py",
        "anthropic.py",
        "gemini.py",
        "deepinfra.py",
        "huggingface.py",
        "rendering.py",
        "serialization.py",
    ]
    async_paths: list[pathlib.Path] = []
    for path in (ROOT / PACKAGE).rglob("*.py"):
        if path.name in filenames and path.parent.name != "sync":
            async_paths.append(path)
    for async_path in async_paths:
        async_code = async_path.read_text()
        sync_code = _async_to_sync(async_code, async_paths)
        sync_path = ROOT / PACKAGE / "sync" / async_path.name
        sync_path.write_text(sync_code)
    async_tests_directory = ROOT / "tests" / "async"
    sync_tests_directory = ROOT / "tests" / "sync"
    for async_path in async_tests_directory.rglob("*.py"):
        sync_path = sync_tests_directory / async_path.relative_to(async_tests_directory)
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path.write_text(_async_to_sync(async_path.read_text(), async_paths))


def _execute(*args: Any) -> str:
    subprocess.run([str(arg) for arg in args])


def _serve(directory: pathlib.Path, port: int) -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    server = http.server.HTTPServer(("localhost", port), handler)
    print(f"http://localhost:{port}")
    server.serve_forever()


def _async_to_sync(text: str, async_paths: list[pathlib.Path]) -> str:
    # Change async imports to sync equivalents.
    text = text.replace(f"from {PACKAGE} import", f"from {PACKAGE}.sync import")
    for path in async_paths:
        relative = path.relative_to(ROOT / PACKAGE)
        import_path = str(relative).removesuffix(".py").replace("/", ".")
        text = text.replace(f"from {PACKAGE}.{import_path} import", f"from {PACKAGE}.sync.{relative.stem} import")
    # Change async quirks to standard usage.
    text = text.replace("self.client.aio", "self.client")
    # Remove pytest_asyncio import and replace references to it with pytest.
    if "import pytest\n" in text:
        text = text.replace("import pytest_asyncio\n", "")
    text = text.replace("pytest_asyncio", "pytest")
    # Remove pytest_asyncio mark.
    text = re.sub("\n*pytestmark = pytest.mark.asyncio", "", text)
    # Remove import pytest if it's left unused.
    if "import pytest" in text and text.count("pytest") == 1:
        text = text.replace("import pytest\n", "")
    # in rendering:
    # - remove enable_async=True
    # - render_async -> render
    # - to_async -> to_sync
    text = re.sub(r"\s*enable_async=True,", "", text)
    text = text.replace("render_async", "render")
    text = text.replace("to_async", "to_sync")
    # async def -> def
    # async for -> for
    # async with -> with
    # asynccontextmanager -> contextmanager
    # await ... -> ...
    # AsyncIterator -> Iterator
    # AsyncOpenAI -> OpenAI
    # AsyncAnthropic -> Anthropic
    # Tool.async_fetch -> Tool.fetch
    # Call.async_run_all -> Call.run_all
    # async_parallelize -> parallelize
    text = re.sub(r"([aA]sync_?|await) *", "", text)
    return text


if __name__ == "__main__":
    main()