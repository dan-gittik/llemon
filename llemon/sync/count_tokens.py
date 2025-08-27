import json
from typing import Callable

from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest


def count_tokens(request: GenerateRequest, count: Callable[[str], int]) -> int:
    total = 0
    if request.instructions:
        total += count(request.render_instructions()) + 3
    for request_, response_ in request.history:
        if isinstance(request_, GenerateRequest):
            total += count(request_.user_input) + 3
        if isinstance(response_, GenerateResponse):
            total += count(response_.text) + 3
    if request.user_input:
        total += count(request.user_input) + 3
    tools = []
    for name, tool in request.tools_dict.items():
        tools.append(
            dict(
                type="function",
                function=dict(
                    name=name,
                    description=tool.description,
                    parameters=tool.parameters,
                ),
            )
        )
    if tools:
        total += count(json.dumps(tools, separators=(",", ":"))) + 3
    if isinstance(request, GenerateObjectRequest):
        schema = request.schema.model_json_schema()
        total += count(json.dumps(schema, separators=(",", ":"))) + 3
    return total + 3
