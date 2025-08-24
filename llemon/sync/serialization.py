from __future__ import annotations

import datetime as dt
import importlib
from typing import Any, Callable, cast, overload

from pydantic import BaseModel
from srlz import Serialization, SimpleType

import llemon.providers  # noqa: F401
import llemon.tools  # noqa: F401
from llemon.core.llm.llm_model_config import LLM_MODEL_CONFIGS, LLMModelConfig
from llemon.models.file import File
from llemon.models.request import Request, Response
from llemon.models.tool import Call, Tool, Toolbox, undefined
from llemon.sync.classify import ClassifyRequest, ClassifyResponse
from llemon.sync.conversation import Conversation
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.sync.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.sync.llm import LLM
from llemon.sync.llm_model import LLMModel
from llemon.sync.rendering import Rendering
from llemon.sync.types import NS, History
from llemon.utils.concat import concat
from llemon.utils.schema import schema_to_model

MODELS = "models"
FILES = "files"
TOOLS = "tools"

type Dumper = Callable[[Any, NS], NS]
type Loader = Callable[..., Any]

serialization = Serialization()
dumpers: dict[type, Dumper] = {}
loaders: dict[type, Loader] = {}


@serialization.serializer("date", dt.date)
def serialize_date(date: dt.date) -> str:
    return date.isoformat()


@serialization.deserializer("date")
def deserialize_date(date: SimpleType) -> dt.date:
    date = cast(str, date)
    return dt.date.fromisoformat(date)


def dump(obj: Any) -> NS:
    type_ = type(obj)
    if type_ not in dumpers:
        raise ValueError(f"{type_.__name__} is not serializable")
    state: NS = {}
    data = dumpers[type_](obj, state)
    return serialization.serialize(state | data)


def load[T](cls: type[T], data: NS) -> T:
    if cls not in loaders:
        raise ValueError(f"{cls.__name__} is not deserializable")
    data = serialization.deserialize(data)
    state = State(data)
    result = loaders[cls]("$", data, state)
    return cast(T, result)


def dumper[D: Dumper](cls: type) -> Callable[[D], D]:
    def decorator(dumper: D) -> D:
        dumpers[cls] = dumper
        return dumper

    return decorator


def loader[L: Loader](cls: type) -> Callable[[L], L]:
    def decorator(loader: L) -> L:
        loaders[cls] = loader
        return loader

    return decorator


@dumper(Conversation)
def dump_conversation(obj: Conversation, state: NS) -> NS:
    dump_model(obj.model, state)
    dump_tools(obj.tools, state)
    history: list[NS] = []
    for request, response in obj.history:
        history.append(
            dict(
                request=dump_request(request, state),
                response=dump_response(response, state),
            )
        )
    conversation = filtered_dict(
        model=obj.model.name,
        instructions=obj.instructions,
        context=obj.context,
        render=obj.rendering.bracket if obj.rendering else False,
        tools=[tool.name for tool in obj.tools],
        history=history,
    )
    return dict(conversation=conversation)


@loader(Conversation)
def load_conversation(prefix: str, data: NS, state: State) -> Conversation:
    prefix = f"{prefix}.conversation"
    conv = get_dict(prefix, data)
    history: History = []
    for n, interaction in enumerate(get(f"{prefix}.history", conv, list)):
        interaction = cast(NS, interaction)
        interaction_prefix = f"{prefix}.history[{n}]"
        request_prefix = f"{interaction_prefix}.request"
        response_prefix = f"{interaction_prefix}.response"
        request = load_request(request_prefix, get_dict(request_prefix, interaction), state)
        response = load_response(response_prefix, get_dict(response_prefix, interaction), state, request=request)
        request.history = history
        history.append((request, response))
    return Conversation(
        model=state.get_model(get(f"{prefix}.model", conv, str)),
        instructions=get(f"{prefix}.instructions", conv, str, None),
        context=get_dict(f"{prefix}.context", conv),
        render=Rendering.resolve(get(f"{prefix}.render", conv, str, None)),
        tools=[state.get_tool(name) for name in get(f"{prefix}.tools", conv, list, [])],
        history=history,
    )


@dumper(LLMModel)
def dump_model(obj: LLMModel, state: NS) -> NS:
    data: NS = dict(
        provider=obj.llm.__class__.__name__,
        name=obj.name,
    )
    config = dump_model_config(obj.config, state)
    if config:
        data.update(config=config)
    state.setdefault(MODELS, {})[obj.name] = data
    return data


@loader(LLMModel)
def load_model(prefix: str, data: NS, state: State) -> LLMModel:
    provider = get_subclass(LLM, get(f"{prefix}.provider", data, str))
    return provider.model(
        name=get(f"{prefix}.name", data, str),
        **get_dict(f"{prefix}.config", data),
    )


@dumper(LLMModelConfig)
def dump_model_config(obj: LLMModelConfig, state: NS) -> NS:
    if obj.name not in LLM_MODEL_CONFIGS:
        return obj.model_dump()
    data = obj.model_dump()
    for key, value in LLM_MODEL_CONFIGS[obj.name].model_dump().items():
        if data[key] == value:
            del data[key]
    return data


@loader(LLMModelConfig)
def load_model_config(prefix: str, data: NS, state: State) -> LLMModelConfig:
    config = LLMModelConfig(**data)
    config.load_defaults()
    return config


@dumper(File)
def dump_file(obj: File, state: NS) -> NS:
    data = dict(
        name=obj.name,
        url=obj.url,
    )
    if obj.id:
        data.update(id=obj.id)
    state.setdefault(FILES, {})[obj.name] = data
    return data


@loader(File)
def load_file(prefix: str, data: NS, state: State) -> File:
    file = File.from_url(
        url=get(f"{prefix}.url", data, str),
        name=get(f"{prefix}.name", data, str),
    )
    file.id = get(f"{prefix}.id", data, str, None)
    return file


def dump_tools(tools: list[Tool | Toolbox], state: NS) -> NS:
    for tool in tools:
        if isinstance(tool, Toolbox):
            dump_toolbox(tool, state)
        else:
            dump_tool(tool, state)
    return state


def load_tools(prefix: str, data: NS, state: State) -> dict[str, Tool | Toolbox]:
    tools: dict[str, Tool | Toolbox] = {}
    for tool_name, tool_data in get_dict(f"{prefix}.tools", data).items():
        tool_data = cast(NS, tool_data)
        tool_prefix = f"{prefix}.tools[{tool_name}]"
        if "type" in tool_data:
            toolbox = load_toolbox(tool_prefix, tool_data, state)
            tools[tool_name] = toolbox
            for tool in toolbox.tools:
                tools[tool.name] = tool
        else:
            tools[tool_name] = load_tool(tool_prefix, tool_data, state)
    return tools


@dumper(Tool)
def dump_tool(obj: Tool, state: NS) -> NS:
    if obj.toolbox:
        return dump_toolbox(obj.toolbox, state)
    data = dict(
        name=obj.name,
        description=obj.description,
        parameters=obj.parameters,
    )
    if obj.function is not obj._not_runnable:
        data.update(function=f"{obj.function.__module__}.{obj.function.__name__}")
    state.setdefault(TOOLS, {})[obj.name] = data
    return data


@loader(Tool)
def load_tool(prefix: str, data: NS, state: State) -> Tool:
    function_name = get(f"{prefix}.function", data, str, None)
    if function_name:
        module_name, function_name = function_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except ImportError:
            raise ValueError(f"unable to load {prefix} function: module {module_name} not found")
        except AttributeError:
            raise ValueError(f"unable to load {prefix} function: module {module_name} has no function {function_name}")
    else:
        function = None
    return Tool(
        name=get(f"{prefix}.name", data, str),
        description=get(f"{prefix}.description", data, str),
        parameters=get_dict(f"{prefix}.parameters", data),
        function=function,
    )


@dumper(Toolbox)
def dump_toolbox(obj: Toolbox, state: NS) -> NS:
    data = dict(
        type=obj.__class__.__name__,
        name=obj.name,
        init=obj._init,
        suffix=obj._suffix,
    )
    state.setdefault(TOOLS, {})[obj.name] = data
    return data


@loader(Toolbox)
def load_toolbox(prefix: str, data: NS, state: State) -> Toolbox:
    toolbox_class = get_subclass(Toolbox, get(f"{prefix}.type", data, str))
    toolbox = toolbox_class(**get_dict(f"{prefix}.init", data))
    toolbox._suffix = get(f"{prefix}.suffix", data, str)
    return toolbox


@dumper(Call)
def dump_call(obj: Call, state: NS) -> NS:
    dump_tool(obj.tool, state)
    return dict(
        id=obj.id,
        tool=obj.tool.name,
        arguments=obj.arguments,
        result=obj.result,
    )


@loader(Call)
def load_call(prefix: str, data: NS, state: State) -> Call:
    tool = state.get_tool(get(f"{prefix}.tool", data, str))
    result = get_dict(f"{prefix}.result", data)
    return Call(
        id=get(f"{prefix}.id", data, str),
        tool=cast(Tool, tool),
        arguments=get_dict(f"{prefix}.arguments", data),
        return_value=result.get("return_value", undefined),
        error=result.get("error"),
    )


def dump_request(obj: Request, state: NS) -> NS:
    data = dumpers[type(obj)](obj, state)
    data.update(type=obj.__class__.__name__)
    return data


def load_request(prefix: str, data: NS, state: State) -> Request:
    request_class = get_subclass(Request, get(f"{prefix}.type", data, str))
    return loaders[request_class](prefix, data, state)


def dump_response(obj: Response, state: NS) -> NS:
    data = dumpers[type(obj)](obj, state)
    data.update(
        type=obj.__class__.__name__,
        started=obj.started.isoformat(),
        ended=obj.ended.isoformat() if obj.ended else None,
    )
    return data


def load_response(prefix: str, data: NS, state: State, request: Request) -> Response:
    response_class = get_subclass(Response, get(f"{prefix}.type", data, str))
    response: Response = loaders[response_class](prefix, data, state, request=request)
    response.started = dt.datetime.fromisoformat(get(f"{prefix}.started", data, str))
    response.ended = dt.datetime.fromisoformat(get(f"{prefix}.ended", data, str))
    return response


@dumper(GenerateRequest)
def dump_generate_request(obj: GenerateRequest, state: NS) -> NS:
    dump_model(obj.model, state)
    for file in obj.files:
        dump_file(file, state)
    dump_tools(obj.tools, state)
    return filtered_dict(
        model=obj.model.name,
        instructions=obj._instructions,
        user_input=obj._user_input,
        context=obj.context or None,
        render=obj.rendering.bracket if obj.rendering else None,
        files=[file.name for file in obj.files],
        tools=[tool.name for tool in obj.tools],
        use_tool=obj.use_tool,
        variants=obj.variants,
        temperature=obj.temperature,
        max_tokens=obj.max_tokens,
        seed=obj.seed,
        frequency_penalty=obj.frequency_penalty,
        presence_penalty=obj.presence_penalty,
        top_p=obj.top_p,
        top_k=obj.top_k,
        stop=obj.stop,
        prediction=obj.prediction,
        return_incomplete_message=obj.return_incomplete_message,
    )


@overload
def load_generate_request(prefix: str, data: NS, state: State) -> GenerateRequest: ...


@overload
def load_generate_request[T: GenerateRequest](prefix: str, data: NS, state: State, as_: type[T], **init: Any) -> T: ...


@loader(GenerateRequest)
def load_generate_request[T: GenerateRequest](
    prefix: str,
    data: NS,
    state: State,
    as_: type[T] | None = None,
    **init: Any,
) -> GenerateRequest | T:
    use_tool = data.get("use_tool", None)
    if use_tool is not None and not isinstance(use_tool, bool | str):
        raise ValueError(f"{prefix}.use_tool must be a boolean or a string")
    return (as_ or GenerateRequest)(
        model=state.get_model(get(f"{prefix}.model", data, str)),
        instructions=get(f"{prefix}.instructions", data, str, None),
        user_input=get(f"{prefix}.user_input", data, str, None),
        context=get_dict(f"{prefix}.context", data),
        render=Rendering.resolve(get(f"{prefix}.render", data, str, None)),
        files=[state.get_file(name) for name in get(f"{prefix}.files", data, list, [])],
        tools=[state.get_tool(name) for name in get(f"{prefix}.tools", data, list, [])],
        use_tool=use_tool,
        variants=get(f"{prefix}.variants", data, int, None),
        temperature=get(f"{prefix}.temperature", data, float, None),
        max_tokens=get(f"{prefix}.max_tokens", data, int, None),
        seed=get(f"{prefix}.seed", data, int, None),
        frequency_penalty=get(f"{prefix}.frequency_penalty", data, float, None),
        presence_penalty=get(f"{prefix}.presence_penalty", data, float, None),
        top_p=get(f"{prefix}.top_p", data, float, None),
        top_k=get(f"{prefix}.top_k", data, int, None),
        stop=get(f"{prefix}.stop", data, list, None),
        prediction=get(f"{prefix}.prediction", data, str, None),
        return_incomplete_message=get(f"{prefix}.return_incomplete_message", data, bool, None),
        **init,
    )


@dumper(GenerateResponse)
def dump_generate_response(obj: GenerateResponse, state: NS) -> NS:
    return dict(
        calls=[dump_call(call, state) for call in obj.calls],
        texts=obj._texts,
        selected=obj._selected,
    )


@overload
def load_generate_response(prefix: str, data: NS, state: State, request: Request) -> GenerateResponse: ...


@overload
def load_generate_response[T: GenerateResponse](
    prefix: str,
    data: NS,
    state: State,
    request: Request,
    as_: type[T] | None = None,
) -> T: ...


@loader(GenerateResponse)
def load_generate_response[T: GenerateResponse](
    prefix: str,
    data: NS,
    state: State,
    request: Request,
    as_: type[T] | None = None,
) -> GenerateResponse | T:
    response = (as_ or GenerateResponse)(request)  # type: ignore
    calls: list[Call] = []
    for n, call in enumerate(get(f"{prefix}.calls", data, list)):
        call = cast(NS, call)
        calls.append(load_call(f"{prefix}.calls[{n}]", call, state))
    response.calls = calls
    response._texts = get(f"{prefix}.texts", data, list)
    response._selected = get(f"{prefix}.selected", data, int)
    return response


@dumper(GenerateStreamRequest)
def dump_generate_stream_request(obj: GenerateStreamRequest, state: NS) -> NS:
    return dump_generate_request(obj, state)


@loader(GenerateStreamRequest)
def load_generate_stream_request(prefix: str, data: NS, state: State) -> GenerateStreamRequest:
    return load_generate_request(prefix, data, state, as_=GenerateStreamRequest)


@dumper(GenerateStreamResponse)
def dump_generate_stream_response(obj: GenerateStreamResponse, state: NS) -> NS:
    data = dump_generate_response(obj, state)
    data.update(
        chunks=obj._chunks,
        ttft=obj.ttft,
    )
    return data


@loader(GenerateStreamResponse)
def load_generate_stream_response(prefix: str, data: NS, state: State, request: Request) -> GenerateStreamResponse:
    response = load_generate_response(prefix, data, state, request, as_=GenerateStreamResponse)
    response._chunks = get(f"{prefix}.chunks", data, list)
    response._ttft = get(f"{prefix}.ttft", data, float)
    return response


@dumper(GenerateObjectRequest)
def dump_generate_object_request(obj: GenerateObjectRequest[BaseModel], state: NS) -> NS:
    data = dump_generate_request(obj, state)
    data.update(
        schema=obj.schema.model_json_schema(),
    )
    return data


@loader(GenerateObjectRequest)
def load_generate_object_request(prefix: str, data: NS, state: State) -> GenerateObjectRequest[BaseModel]:
    return load_generate_request(
        prefix,
        data,
        state,
        as_=GenerateObjectRequest,
        schema=schema_to_model(get_dict(f"{prefix}.schema", data)),
    )


@dumper(GenerateObjectResponse)
def dump_generate_object_response(obj: GenerateObjectResponse[BaseModel], state: NS) -> NS:
    data = dump_generate_response(obj, state)
    data.update(
        objects=[object.model_dump() for object in obj.objects],
    )
    return data


@loader(GenerateObjectResponse)
def load_generate_object_response(
    prefix: str,
    data: NS,
    state: State,
    request: Request,
) -> GenerateObjectResponse[BaseModel]:
    response = load_generate_response(prefix, data, state, request, as_=GenerateObjectResponse)
    objects = get(f"{prefix}.objects", data, list)
    response._objects = [response.request.schema.model_validate(object) for object in objects]
    return response


@dumper(ClassifyRequest)
def dump_classify_request(obj: ClassifyRequest, state: NS) -> NS:
    data = dump_generate_request(obj, state)
    data.update(
        question=obj.question,
        answers=obj.answers,
        reasoning=obj.reasoning,
    )
    return data


@loader(ClassifyRequest)
def load_classify_request(prefix: str, data: NS, state: State) -> ClassifyRequest:
    return load_generate_request(
        prefix,
        data,
        state,
        as_=ClassifyRequest,
        question=get(f"{prefix}.question", data, str),
        answers=get(f"{prefix}.answers", data, list),
        reasoning=get(f"{prefix}.reasoning", data, str, None),
    )


@dumper(ClassifyResponse)
def dump_classify_response(obj: ClassifyResponse, state: NS) -> NS:
    data = dump_generate_response(obj, state)
    data.update(
        filtered_dict(
            answer=obj.answer,
            reasoning=obj.reasoning,
        )
    )
    return data


@loader(ClassifyResponse)
def load_classify_response(prefix: str, data: NS, state: State, request: Request) -> ClassifyResponse:
    response = load_generate_response(prefix, data, state, request, as_=ClassifyResponse)
    response.answer = get(f"{prefix}.answer", data, str)
    response.reasoning = get(f"{prefix}.reasoning", data, str, None)
    return response


class State:

    def __init__(self, data: NS) -> None:
        self.models: dict[str, LLMModel] = {}
        for model_name, model_data in get_dict(MODELS, data).items():
            model_data = cast(NS, model_data)
            self.models[model_name] = load_model(f"models[{model_name}]", model_data, self)
        self.files: dict[str, File] = {}
        for file_name, file_data in get_dict(FILES, data).items():
            file_data = cast(NS, file_data)
            self.files[file_name] = load_file(f"files[{file_name}]", file_data, self)
        self.tools = load_tools(TOOLS, data, self)

    def get_model(self, name: str) -> LLMModel:
        if name not in self.models:
            raise ValueError(f"model {name!r} does not exist (available models are {concat(self.models)})")
        return self.models[name]

    def get_file(self, name: str) -> File:
        if name not in self.files:
            raise ValueError(f"file {name!r} does not exist (available files are {concat(self.files)})")
        return self.files[name]

    def get_tool(self, name: str) -> Tool | Toolbox:
        if name not in self.tools:
            raise ValueError(f"tool {name!r} does not exist (available tools are {concat(self.tools)})")
        return self.tools[name]


def filtered_dict(**kwargs: Any) -> NS:
    return {key: value for key, value in kwargs.items() if value not in (None, [], {})}


@overload
def get[T](path: str, data: NS, type_: type[T]) -> T: ...


@overload
def get[T, D](path: str, data: NS, type_: type[T], default: D) -> T | D: ...


def get[T, D](path: str, data: NS, type_: type[T], default: D | object = undefined) -> T | D:
    key = path.split(".")[-1]
    if key not in data:
        if default is not undefined:
            return cast(D, default)
        raise ValueError(f"{path} is required (available keys are {concat(data)})")
    value = data[key]
    if not isinstance(value, (int | float) if type_ is float else type_):
        if value is default:
            return cast(D, default)
        raise ValueError(f"{path} must be of type {type_} (not {type(value).__name__})")
    return value


def get_dict(path: str, data: NS) -> NS:
    namespace = get(path, data, dict, None) or {}
    if any(not isinstance(key, str) for key in namespace):
        raise ValueError(f"{path} must be a dictionary with string keys")
    return namespace


def get_subclass[T](base: type[T], name: str) -> type[T]:
    classes = {subclass.__name__: subclass for subclass in base.__subclasses__()}
    if name not in classes:
        raise ValueError(f"{base.__name__} {name!r} does not exist (available {base.__name__}s are {concat(classes)})")
    return classes[name]
