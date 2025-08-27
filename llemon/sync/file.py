from __future__ import annotations

import base64
import hashlib
import logging
import mimetypes
import pathlib
import re
from functools import cached_property

from llemon.sync.types import FilesArgument
from llemon.utils import fetch

DATA_URL_PATTERN = re.compile(r"^data:([^;]+);base64,(.*)$")

log = logging.getLogger(__name__)


class File:

    def __init__(
        self,
        name: str,
        mimetype: str,
        data: bytes | None = None,
        url: str | None = None,
    ) -> None:
        self.name = name
        self.mimetype = mimetype
        self.data = data
        self.id: str | None = None
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
    def from_url(cls, url: str, name: str | None = None) -> File:
        if name is None:
            name = url.rsplit("/", 1)[-1]
        if match := DATA_URL_PATTERN.match(url):
            mimetype, base64_ = match.groups()
            file = cls(name, mimetype, base64.b64decode(base64_), url=url)
            log.debug("created %s from data URL", file)
        else:
            mimetype = cls.get_mimetype(url)
            file = cls(name, mimetype, url=url)
            log.debug("created %s from URL %s", file, url)
        return file

    @classmethod
    def from_path(cls, path: str | pathlib.Path) -> File:
        path = pathlib.Path(path).absolute()
        if not path.exists():
            raise FileNotFoundError(f"file {path} does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"file {path} is a directory")
        mimetype = cls.get_mimetype(str(path))
        file = cls(path.name, mimetype, path.read_bytes())
        log.debug("created %s from path %s", file, path)
        return file

    @classmethod
    def from_data(cls, name_or_mimetype: str, data: bytes) -> File:
        if "/" in name_or_mimetype:
            mimetype = name_or_mimetype
            extension = mimetypes.guess_extension(mimetype)
            if not extension:
                raise ValueError(f"unknown extension for {mimetype}")
            name = f"<unnamed>{extension}"
        else:
            name = name_or_mimetype
            mimetype = cls.get_mimetype(name)
        file = cls(name, mimetype, data)
        log.debug("created %s from data", file)
        return file

    @classmethod
    def get_mimetype(cls, path: str) -> str:
        mimetype, _ = mimetypes.guess_type(path)
        if not mimetype:
            raise ValueError(f"unknown mimetype for {path}")
        return mimetype

    @cached_property
    def md5(self) -> str:
        if not self.data:
            raise ValueError("file doesn't have data")
        return hashlib.md5(self.data).hexdigest()

    @cached_property
    def base64(self) -> str:
        if not self.data:
            raise ValueError("file doesn't have data")
        log.debug("encoding %s data as base64", self.name)
        return base64.b64encode(self.data).decode()

    @cached_property
    def url(self) -> str:
        if self._url:
            return self._url
        if not self.data:
            raise ValueError("file has neither data nor URL")
        return f"data:{self.mimetype};base64,{self.base64}"

    @property
    def is_image(self) -> bool:
        return self.mimetype.startswith("image/")

    @property
    def is_audio(self) -> bool:
        return self.mimetype.startswith("audio/")

    @property
    def is_video(self) -> bool:
        return self.mimetype.startswith("video/")

    def fetch(self) -> None:
        if self.data:
            return
        log.debug("fetching %s data from %s", self, self.url)
        self.data = fetch(self.url)
