from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from llemon.models.file import File
    from llemon.models.request import Request, Response
    from llemon.models.tool import Call, Tool, Toolbox
    from llemon.sync.rendering import Rendering

type NS = dict[str, Any]
type FilesArgument = Sequence[str | pathlib.Path | tuple[str, bytes] | File] | None
type ToolsArgument = Sequence[Callable[..., Any] | Tool | Toolbox] | None
type CallArgument = NS | Call
type RenderArgument = bool | str | Rendering | None
type Interaction = tuple[Request, Response]
type History = list[Interaction]
type ToolCalls = list[tuple[str, str, NS]]
type ToolStream = dict[int, tuple[str, str, list[str]]]
type ToolDeltas = list[tuple[int, str, str, str]]
