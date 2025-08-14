from functools import cached_property
from typing import Callable, ClassVar, Any


class Toolbox:

    tool_suffix: ClassVar[str] = "_tool"
    description_suffix: ClassVar[str] = "_description"

    @property
    def tool_names(self) -> list[str]:
        tool_names = []
        for attribute in dir(self):
            if attribute.endswith(self.tool_suffix):
                tool_names.append(attribute.removesuffix(self.tool_suffix))
        return tool_names

    @cached_property
    def tools(self) -> list[tuple[str, str, Callable[..., Any]]]:
        tools: list[tuple[str, str, Callable[..., Any]]] = []
        for name in self.tool_names:
            function = getattr(self, f"{name}{self.tool_suffix}")
            get_description = getattr(self, f"{name}{self.description_suffix}", None)
            if get_description:
                description = get_description()
            else:
                description = function.__doc__ or ""
            tools.append((name, description, function))
        return tools