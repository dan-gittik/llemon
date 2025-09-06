from __future__ import annotations

from typing import get_args


class Model[T]:

    __orig_class__: type[Model[T]] | None

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: type) -> T:
        if not self.__orig_class__:
            raise TypeError(f"{self.__class__.__name__}[T] must be parameterized")
        method = get_args(self.__orig_class__)[0].__name__.lower()
        return getattr(owner, method)(self.name)
