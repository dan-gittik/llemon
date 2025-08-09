from __future__ import annotations

import base64
from functools import cached_property
import mimetypes
import pathlib
from typing import Self

import httpx

from .types import FilesArgument


class File:

    def __init__(self, name: str, mimetype: str, data: bytes | None = None, url: str | None = None) -> None:
        self.name = name
        self.mimetype = mimetype
        self.data = data
        self._url = url
    
    def __str__(self) -> str:
        return f"file {self.name!r}"
    
    def __repr__(self) -> str:
        return f"<{self}>"
    
    @classmethod
    def resolve(cls, files: FilesArgument) -> list[File]:
        if files is None:
            return []
        resolved: list[File] = []
        for file in files:
            if isinstance(file, File):
                resolved.append(file)
            elif isinstance(file, str):
                path = pathlib.Path(file)
                if path.exists():
                    resolved.append(cls.from_path(file))
                else:
                    resolved.append(cls.from_url(file))
            elif isinstance(file, pathlib.Path):
                resolved.append(cls.from_path(file))
            else:
                mimetype, data = file
                resolved.append(cls.from_data(mimetype, data))
        return resolved
    
    @classmethod
    def get_mimetype(cls, path: str) -> str:
        mimetype, _ = mimetypes.guess_type(path)
        if not mimetype:
            raise ValueError(f"unknown mimetype for {path}")
        return mimetype

    @classmethod
    def from_url(cls, url: str, name: str | None = None) -> Self:
        if name is None:
            name = url
        mimetype = cls.get_mimetype(url)
        return cls(name, mimetype, url=url)
    
    @classmethod
    def from_path(cls, path: str | pathlib.Path) -> Self:
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"file {path} does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"file {path} is a directory")
        mimetype = cls.get_mimetype(str(path))
        return cls(path.name, mimetype, path.read_bytes())
    
    @classmethod
    def from_data(cls, name_or_mimetype: str, data: bytes) -> Self:
        if "/" in name_or_mimetype:
            mimetype = name_or_mimetype
            extension = mimetypes.guess_extension(mimetype)
            if not extension:
                raise ValueError(f"unknown extension for {mimetype}")
            name = f"<unnamed>.{extension}"
        else:
            name = name_or_mimetype
            mimetype = cls.get_mimetype(name)
        return cls(name, mimetype, data)
    
    @cached_property
    def base64(self) -> str:
        if not self.data:
            raise ValueError("file doesn't have data")
        return base64.b64encode(self.data).decode()
    
    @cached_property
    def url(self) -> str:
        if self._url:
            return self._url
        if not self.data:
            raise ValueError("file has neither data nor URL")
        return f"data:{self.mimetype};base64,{self.base64}"
    
    async def fetch(self) -> None:
        if self.data:
            return
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            self.data = response.content