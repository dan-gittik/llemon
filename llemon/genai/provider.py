from __future__ import annotations
import inspect
from typing import Any, ClassVar, Self

from dotenv import dotenv_values

from llemon.types import NS, Error
from llemon.utils import Superclass


class Provider(Superclass):

    configurations: ClassVar[NS] = {}
    instance: ClassVar[Self | None] = None

    def __init_subclass__(cls) -> None:
        cls.instance = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def configure(cls, config_dict: NS | None = None, /, **config_kwargs: Any) -> None:
        config = dotenv_values()
        if config_dict:
            config.update(config_dict)
        if config_kwargs:
            config.update(config_kwargs)
        cls.configurations.update({key.lower(): value for key, value in config.items()})

    @classmethod
    def create(cls) -> Self:
        if cls.__init__ is object.__init__:
            return cls()
        if not cls.configurations:
            cls.configure()
        signature = inspect.signature(cls.__init__)
        parameters = list(signature.parameters.values())[1:]  # skip self
        kwargs = {}
        prefix = cls.__name__.lower()
        for parameter in parameters:
            name = f"{prefix}_{parameter.name}"
            if name in cls.configurations:
                value = cls.configurations[name]
            elif parameter.default is not parameter.empty:
                value = parameter.default
            else:
                raise Error(f"{cls.__name__} missing configuration {parameter.name!r}")
            kwargs[parameter.name] = value
        return cls(**kwargs)

    @classmethod
    def get(cls) -> Self:
        if cls.instance is None:
            cls.instance = cls.create()
        return cls.instance
