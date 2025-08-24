import asyncio
import inspect
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable

from llemon.utils.concat import concat

executor: ThreadPoolExecutor | None = None


def parallelize(calls: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    global executor
    if executor is None:
        executor = ThreadPoolExecutor()
    futures: list[Future[Any]] = []
    for call, args, kwargs in calls:
        future = executor.submit(call, *args, **kwargs)
        futures.append(future)
    wait(futures)
    results: list[Any] = []
    errors: list[Exception] = []
    failed: list[str] = []
    for future, (call, args, kwargs) in zip(futures, calls):
        try:
            result = future.result()
            results.append(result)
        except Exception as error:
            errors.append(error)
            params = ", ".join([str(arg) for arg in args] + [f"{key}={value!r}" for key, value in kwargs.items()])
            failed.append(f"{call.__name__}({params})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results


async def async_parallelize(calls: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    tasks = [asyncio.create_task(call(*args, **kwargs)) for call, args, kwargs in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors: list[Exception] = []
    failed: list[str] = []
    for result, (call, args, kwargs) in zip(results, calls):
        if isinstance(result, Exception):
            errors.append(result)
            params = ", ".join([str(arg) for arg in args] + [f"{key}={value!r}" for key, value in kwargs.items()])
            failed.append(f"{call.__name__}({params})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results


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
