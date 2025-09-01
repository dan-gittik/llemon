from __future__ import annotations

from typing import Self

from .concat import concat


class Superclass:

    @classmethod
    def get_subclass(cls, name: str) -> type[Self]:
        subclasses: dict[str, type[Self]] = {}
        consider = cls.__subclasses__()
        while consider:
            subclass = consider.pop()
            subclasses[subclass.__name__] = subclass
            consider.extend(subclass.__subclasses__())
        if name not in subclasses:
            raise ValueError(f"{cls.__name__} has no subclass {name!r} (available subclasses are {concat(subclasses)})")
        return subclasses[name]
