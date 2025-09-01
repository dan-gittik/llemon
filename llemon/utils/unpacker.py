from __future__ import annotations

from functools import cached_property
from typing import Any, Iterable, overload

from .concat import concat
from .undefined import Undefined, undefined


class Unpacker:

    def __init__(self, data: dict[str, Any], prefix: str | None = None) -> None:
        self.data = data
        self.prefix = prefix

    def __str__(self) -> str:
        return self.prefix or "$"

    @cached_property
    def key(self) -> str:
        return str(self).split(".")[-1]

    @overload
    def get[T](
        self,
        key: str,
        type_: type[T],
        default: Undefined = undefined,
        one_of: list[T] | None = None,
    ) -> T: ...

    @overload
    def get[T, D](
        self,
        key: str,
        type_: type[T],
        default: D,
        one_of: list[T] | None = None,
    ) -> T | D: ...

    @overload
    def get[T1, T2](
        self,
        key: str,
        type_: tuple[type[T1], type[T2]],
        default: Undefined = undefined,
        one_of: list[T1 | T2] | None = None,
    ) -> T1 | T2: ...

    @overload
    def get[T1, T2, D](
        self,
        key: str,
        type_: tuple[type[T1], type[T2]],
        default: D,
        one_of: list[T1 | T2] | None = None,
    ) -> T1 | T2 | D: ...

    def get[T1, T2, D](
        self,
        key: str,
        type_: type[T1] | tuple[type[T1], type[T2]],
        default: D | Undefined = undefined,
        one_of: list[T1 | T2] | None = None,
    ) -> T1 | T2 | D:
        path = self._add_prefix(key)
        if key not in self.data:
            if not isinstance(default, Undefined):
                return default
            raise ValueError(f"{path} is required (available keys are {concat(self.data)})")
        value = self.data[key]
        if not isinstance(value, (int | float) if type_ is float else type_):
            type_name = " or ".join(cls.__name__ for cls in type_) if isinstance(type_, tuple) else type_.__name__
            raise ValueError(f"{path} must be of type {type_name} (not {type(value).__name__})")
        if one_of and value not in one_of:
            raise ValueError(f"{path} must be one of {concat(one_of)}")
        return value

    def load(self, key: str) -> Unpacker:
        namespace = self.get(key, dict, None) or {}
        path = self._add_prefix(key)
        for key in namespace:
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings ({key} is {type(key).__name__})")
        return Unpacker(namespace, path)

    def load_dict(self, key: str, required: bool = True) -> Iterable[tuple[str, Unpacker]]:
        data = self.get(key, dict, undefined if required else {})
        path = self._add_prefix(key)
        if isinstance(data, dict):
            for subkey, value in data.items():
                if not isinstance(subkey, str):
                    raise ValueError(f"{path} keys must be strings ({subkey} is {type(subkey).__name__})")
                if not isinstance(value, dict):
                    raise ValueError(f"{path} values must be dictionaries ({subkey} is {type(value).__name__})")
                yield subkey, Unpacker(value, f"{path}.{subkey}")

    def load_list(self, key: str, required: bool = True) -> Iterable[Unpacker]:
        data = self.get(key, list, undefined if required else [])
        path = self._add_prefix(key)
        for n, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{path} values must be dictionaries (item #{n} is {type(item).__name__})")
            yield Unpacker(item, f"{path}.{n}")

    def check(
        self,
        reason: str | None = None,
        /,
        *,
        defined: list[str] | None = None,
        undefined: list[str] | None = None,
        **expected: Any,
    ) -> None:
        suffix = f" when {reason} is set" if reason else ""
        for key in defined or []:
            if key not in self.data:
                raise ValueError(f"{self._add_prefix(key)} is required{suffix}")
        for key in undefined or []:
            if key in self.data:
                raise ValueError(f"{self._add_prefix(key)} is not allowed{suffix}")
        for key, default in expected.items():
            if key in self.data and self.data[key] != default:
                raise ValueError(f"{self._add_prefix(key)} must be {default}{suffix}")

    def check_empty(self, reason: str | None = None, /, *, except_for: list[str] | None = None) -> None:
        if except_for is None:
            except_for = []
        if reason is not None:
            except_for.append(reason)
        suffix = f" when {reason} is set" if reason else ""
        for key in self.data:
            if key not in except_for:
                raise ValueError(f"{self._add_prefix(key)} is not allowed{suffix}")

    def _add_prefix(self, key: str) -> str:
        if self.prefix is None:
            return key
        return f"{self.prefix}.{key}"
