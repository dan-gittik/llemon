from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict

from llemon.types import NS

PARAMETER_SCHEMAS: dict[Callable[..., Any], NS] = {}


def parse_parameters(function: Callable[..., Any]) -> NS:
    if function in PARAMETER_SCHEMAS:
        return PARAMETER_SCHEMAS[function]
    annotations = get_type_hints(function)
    annotations.pop("return", None)
    model_class: type[BaseModel] = type(
        function.__name__,
        (BaseModel,),
        {"__annotations__": annotations, "model_config": ConfigDict(extra="forbid")},
    )
    schema = model_class.model_json_schema()
    PARAMETER_SCHEMAS[function] = schema
    return schema
