from typing import ClassVar, Self

from llemon.utils.concat import concat


class Superclass:

    classes: ClassVar[dict[str, type[Self]]]

    def __init_subclass__(cls) -> None:
        cls.classes = {}
        for base in cls.__mro__[1:]:
            classes = getattr(base, "classes", None)
            if classes is not None:
                classes[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, name: str) -> type[Self]:
        if name not in cls.classes:
            raise ValueError(
                f"{cls.__name__} has no subclass {name!r} (available subclasses are {concat(cls.classes)})"
            )
        return cls.classes[name]
