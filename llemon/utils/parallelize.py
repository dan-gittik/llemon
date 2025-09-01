from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Awaitable, Callable, Iterable

from .concat import concat

type Invocations = Iterable[Callable[..., Any] | tuple[Callable[..., Any], *tuple[Any, ...]]]

executor: ThreadPoolExecutor | None = None


def parallelize(calls: Invocations) -> list[Any]:
    calls_ = [(call[0], call[1:]) if isinstance(call, tuple) else (call, ()) for call in calls]
    global executor
    if executor is None:
        executor = ThreadPoolExecutor()
    futures: list[Future[Any]] = []
    for call, args in calls_:
        future = executor.submit(to_sync(call), *args)
        futures.append(future)
    wait(futures)
    results: list[Any] = []
    errors: list[Exception] = []
    failed: list[str] = []
    for future, (call, args) in zip(futures, calls_):
        try:
            result = future.result()
            results.append(result)
        except Exception as error:
            errors.append(error)
            failed.append(f"{call.__name__}({', '.join(str(arg) for arg in args)})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results


async def async_parallelize(calls: Invocations) -> list[Any]:
    calls_ = [(call[0], call[1:]) if isinstance(call, tuple) else (call, ()) for call in calls]
    tasks = [asyncio.create_task(to_async(call)(*args)) for call, args in calls_]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors: list[Exception] = []
    failed: list[str] = []
    for result, (call, *args) in zip(results, calls_):
        if isinstance(result, Exception):
            errors.append(result)
            failed.append(f"{call.__name__}({', '.join(str(arg) for arg in args)})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results


def wait_for[R, **P](timeout: float | None, function: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    if not timeout:
        return function(*args, **kwargs)
    global executor
    if executor is None:
        executor = ThreadPoolExecutor()
    future = executor.submit(function, *args, **kwargs)
    return future.result(timeout)


async def async_wait_for[R, **P](
    timeout: float | None,
    function: Callable[P, Awaitable[R]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    if not timeout:
        return await function(*args, **kwargs)
    return await asyncio.wait_for(function(*args, **kwargs), timeout)


def to_sync(function: Callable[..., Any]) -> Callable[..., Any]:
    if not inspect.iscoroutinefunction(function):
        return function

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def to_async(function: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(function):
        return function

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(function, *args, **kwargs)

    return wrapper
