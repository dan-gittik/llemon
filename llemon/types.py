from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

if TYPE_CHECKING:
    from .conversation import Formatting
    from .interaction import History, Interaction
    from .file import File
    from .tool import Call, Tool


type FilesArgument = list[str | pathlib.Path | tuple[str, bytes] | File] | None
type ToolsArgument = list[Callable[..., Any]] | dict[str, Tool] | None
type CallArgument = dict[str, Any] | Call
type InteractionArgument = list[dict[str, Any]] | Interaction
type HistoryArgument = list[dict[str, Any]] | History | None
type FormattingArgument = bool | str | Formatting
type Messages = list[SystemMessage | UserMessage | AssistantMessage | CallMessage]


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict, total=False):
    role: Literal["user"]
    content: str
    files: dict[str, str]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str


class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class CallMessage(TypedDict):
    role: Literal["call"]
    id: str
    tool: ToolSchema
    arguments: str
    return_value: str