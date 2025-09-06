from __future__ import annotations

import builtins
import pathlib
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from llemon.sync import Call, File, Rendering, Request, Response, Tool, Toolbox

type NS = dict[str, Any]
type FileArgument = str | pathlib.Path | tuple[bytes, str] | tuple[bytes, str, str] | File
type FilesArgument = Sequence[FileArgument] | None
type ToolsArgument = Sequence[Callable[..., Any] | Tool | Toolbox] | None
type CallArgument = NS | Call
type RenderArgument = bool | str | Rendering | None
type Interaction = tuple[Request, Response]
type History = list[Interaction]
type HistoryArgument = History | None
type ToolCalls = list[tuple[str, str, NS]]
type ToolStream = dict[int, tuple[str, str, list[str]]]
type ToolDeltas = list[tuple[int, str, str, str]]
type Timestamps = list[tuple[str, float, float]]


class Error(Exception):
    pass


class Warning(builtins.Warning):
    pass
