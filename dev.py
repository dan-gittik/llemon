import functools
import http.server
import inspect
import importlib
import pathlib
import re
import shutil
import subprocess
import tomllib
from typing import Any

import click
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.table import Table
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
console = Console()
undefined = object()


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
    _execute("pytest", "tests", "-x", "-vv", "--ff", '-n', 'auto', *options)


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
        targets.extend(["-p", PACKAGE, "-p", "tests"])
    _execute("mypy", *targets)


@main.command()
def sync() -> None:
    async_paths: list[pathlib.Path] = []
    for path in (ROOT / PACKAGE).rglob("*.py"):
        if path.name == "__init__.py" or path.parent.name in ["sync", "utils"]:
            continue
        async_paths.append(path)
    for async_path in async_paths:
        async_code = async_path.read_text()
        sync_code = _async_to_sync(async_code, async_paths)
        sync_path = ROOT / PACKAGE / "sync" / async_path.name
        sync_path.write_text(sync_code)
    async_init = ROOT / PACKAGE / "__init__.py"
    sync_init = ROOT / PACKAGE / "sync" / "__init__.py"
    init = async_init.read_text()
    init = re.sub(r"from \.(.+?)\.([^.]+?) ", r"from .\2 ", init)
    init = init.replace("from .utils", "from ..utils")
    sync_init.write_text(init)
    async_tests_directory = ROOT / "tests" / "async"
    sync_tests_directory = ROOT / "tests" / "sync"
    for async_path in async_tests_directory.rglob("*.py"):
        sync_path = sync_tests_directory / async_path.relative_to(async_tests_directory)
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path.write_text(_async_to_sync(async_path.read_text(), async_paths))


@main.command()
def check_signatures() -> None:
    model_diff = {
        "missing": {"user_input", "instructions", "model"},
        "extra": {"message1", "message2"},
    }
    conv_diff = {
        "missing": {"user_input", "history", "model"},
        "extra": {"save", "message"},
    }
    _check_signatures("GenerateRequest.__init__", "LLMModel.generate", **model_diff)
    _check_signatures("GenerateRequest.__init__", "Conversation.generate", **conv_diff)
    _check_signatures("GenerateStreamRequest.__init__", "LLMModel.generate_stream", **model_diff)
    _check_signatures("GenerateStreamRequest.__init__", "Conversation.generate_stream", **conv_diff)
    _check_signatures("GenerateObjectRequest.__init__", "LLMModel.generate_object", **model_diff)
    _check_signatures("GenerateObjectRequest.__init__", "Conversation.generate_object", **conv_diff)
    _check_signatures("ClassifyRequest.__init__", "LLMModel.classify", **model_diff)
    _check_signatures("ClassifyRequest.__init__", "Conversation.classify", **conv_diff)
    _check_signatures("LLMModelConfig.__init__", "LLM.model")


def _check_signatures(source: str, target: str, **diff: dict[str, set[str]]) -> None:
    missing = diff.get("missing", set())
    extra = diff.get("extra", set())
    source_location, source_parameters = _parse_function(source)
    target_location, target_parameters = _parse_function(target)
    rows: list[tuple[str, str]] = []
    for name, (source_annotation, source_default) in source_parameters.items():
        if name not in target_parameters:
            if name not in missing:
                rows.append((_param(name, source_annotation, source_default), "missing"))
            continue
        target_annotation, target_default = target_parameters[name]
        if target_annotation != source_annotation or target_default != source_default:
            rows.append((
                _param(name, source_annotation, source_default),
                _param(name, target_annotation, target_default),
            ))
    for name, (target_annotation, target_default) in target_parameters.items():
        if name in source_parameters or name in extra:
            continue
        rows.append(("missing", _param(target_annotation, target_default)))
    if not rows:
        console.print(f":white_check_mark: {source} matches {target}")
        return
    table = Table(expand=True, show_lines=True)
    table.add_column(f"{source} at {source_location}", style="green", ratio=1)
    table.add_column(f"{target} at {target_location}", style="red", ratio=1)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _parse_function(path: str) -> tuple[str, dict[str, tuple[Any, Any]]]:
    import llemon
    class_name, function_name = path.split(".")
    cls = getattr(llemon, class_name)
    function = getattr(cls, function_name)
    parameters: dict[str, tuple[Any, Any]] = {}
    if function == BaseModel.__init__:
        assert issubclass(cls, BaseModel)
        path = pathlib.Path(importlib.import_module(cls.__module__).__file__).relative_to(ROOT)
        line = inspect.findsource(cls)[1] + 1
        for name, field in cls.model_fields.items():
            default = undefined if field.default is PydanticUndefined else field.default
            annotation = field.annotation.__name__ if inspect.isclass(field.annotation) else str(field.annotation)
            annotation = annotation.replace("datetime.", "dt.")
            parameters[name] = annotation, default
    else:
        path = pathlib.Path(function.__code__.co_filename).relative_to(ROOT)
        line = function.__code__.co_firstlineno
        for name, parameter in inspect.signature(function).parameters.items():
            default = undefined if parameter.default is parameter.empty else parameter.default
            parameters[name] = parameter.annotation, default
    return f"{path}:{line}", parameters


def _param(name: str, annotation: Any, default: Any) -> str:
    output = f"{name}: {annotation}"
    if default is not undefined:
        output += f" = {default}"
    return output


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
    text = re.sub(r"\s*pytestmark = pytest.mark.asyncio", "", text)
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
    # async_fetch -> fetch
    # async_parallelize -> parallelize
    text = re.sub(r"([aA]sync_?|await) *", "", text)
    return text


if __name__ == "__main__":
    main()