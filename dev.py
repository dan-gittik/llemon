import contextlib
import functools
import http.server
import os
import pathlib
import shutil
import subprocess
from typing import Any, Iterator

import click
import uvicorn

ROOT = pathlib.Path(__file__).parent
PACKAGE = "llemon"
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
        _execute("isort", "--profile=black", target)
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
@click.argument("message", nargs=-1)
def make_migrations(message: list[str]) -> None:
    with _in_directory(ROOT / "migrations"):
        _execute("alembic", "revision", "--autogenerate", "-m", " ".join(message))


@main.command()
def migrate() -> None:
    with _in_directory(ROOT / "migrations"):
        _execute("alembic", "upgrade", "head")


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


@contextlib.contextmanager
def _in_directory(path: pathlib.Path) -> Iterator[None]:
    cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()