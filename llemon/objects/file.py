from __future__ import annotations

import base64
import hashlib
import logging
import pathlib
import re
from functools import cached_property
from typing import TYPE_CHECKING, Self, cast
from urllib.parse import urlparse

import llemon
from llemon.types import NS, FileArgument
from llemon.utils import async_fetch, filtered_dict, get_extension, get_mimetype, random_suffix

if TYPE_CHECKING:
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker

NAME_PATTERN = re.compile(r"^(.+?)(__[a-f0-9]{8})?(\..+?)$")
NAMED_URL_PATTERN = re.compile(r"/([^/]+\.[^/?#;]+)([?#;].*)?$")
DATA_URL_PATTERN = re.compile(r"^data:([^;]+);base64,(.*)$")

log = logging.getLogger(__name__)


class File(llemon.Serializeable):

    def __init__(
        self,
        name: str,
        mimetype: str,
        data: bytes | None = None,
        url: str | None = None,
    ) -> None:
        path = pathlib.Path(name)
        stem = path.stem
        suffixes = ".".join(path.suffixes)
        if NAME_PATTERN.match(name):
            self.name = name
            self.display_name = f"{stem.rsplit('__', 1)[0]}{suffixes}"
        else:
            self.name = f"{stem}{random_suffix()}{suffixes}"
            self.display_name = f"{stem}{suffixes}"
        self.mimetype = mimetype
        self.id: str | None = None
        self._data = data
        self._url = url

    def __str__(self) -> str:
        return f"file {self.display_name!r}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def resolve(cls, file: FileArgument) -> Self:
        if isinstance(file, File):
            return cast(Self, file)
        if isinstance(file, str):
            if file.startswith("data:") or "://" in file:
                return cls.from_url(file)
            return cls.from_path(file)
        if isinstance(file, pathlib.Path):
            return cls.from_path(file)
        if len(file) == 2:
            data, name_or_mimetype = file
            if "/" in name_or_mimetype:
                mimetype, name = name_or_mimetype, None
            else:
                mimetype, name = None, name_or_mimetype
        else:
            data, mimetype, name = file
        return cls.from_data(data, mimetype, name)

    @classmethod
    def from_url(cls, url: str, name: str | None = None) -> Self:
        if match := DATA_URL_PATTERN.match(url):
            mimetype, base64_ = match.groups()
            if not name:
                name = "data_url" + get_extension(mimetype)
            file = cls(name, mimetype, base64.b64decode(base64_), url=url)
            log.debug("created %s from data URL", file)
        else:
            if not name:
                mimetype = get_mimetype(url)
                name = pathlib.Path(urlparse(url).path).name
            else:
                mimetype = get_mimetype(name)
            file = cls(name, mimetype, url=url)
            log.debug("created %s from URL %s", file, url)
        return file

    @classmethod
    def from_path(cls, path: str | pathlib.Path) -> Self:
        path = pathlib.Path(path).absolute()
        if not path.exists():
            raise FileNotFoundError(f"file {path} does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"file {path} is a directory")
        mimetype = get_mimetype(str(path))
        file = cls(path.name, mimetype, path.read_bytes())
        log.debug("created %s from path %s", file, path)
        return file

    @classmethod
    def from_data(cls, data: bytes, mimetype: str | None = None, name: str | None = None) -> Self:
        if not name and not mimetype:
            raise ValueError("either name or mimetype must be provided")
        elif mimetype and not name:
            name = "data" + get_extension(mimetype)
        elif name and not mimetype:
            mimetype = get_mimetype(name)
        assert name and mimetype
        file = cls(name, mimetype, data)
        log.debug("created %s from data", file)
        return file
    
    @cached_property
    def data(self) -> bytes:
        if self._data is None:
            raise ValueError("file doesn't have data")
        return self._data

    @cached_property
    def md5(self) -> str:
        return hashlib.md5(self.data).hexdigest()

    @cached_property
    def base64(self) -> str:
        log.debug("encoding %s data as base64", self)
        return base64.b64encode(self.data).decode()

    @cached_property
    def url(self) -> str:
        if self._url:
            return self._url
        return f"data:{self.mimetype};base64,{self.base64}"
    
    @property
    def is_url(self) -> bool:
        return bool(self._url and not self._data)
    
    @cached_property
    def is_image(self) -> bool:
        return self.mimetype.startswith("image/")

    async def fetch(self) -> None:
        if self._data:
            return
        log.debug("fetching %s data from %s", self, self.url)
        self._data = await async_fetch(self.url)
    
    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path).absolute()
        if path.is_dir():
            path = path / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        file = cls.from_url(
            url=unpacker.get("url", str),
            name=unpacker.get("name", str),
        )
        file.id = unpacker.get("id", str, None)
        return file

    def _dump(self, refs: DumpRefs) -> NS:
        return filtered_dict(
            id=self.id,
            name=self.name,
            url=self.url,
        )
