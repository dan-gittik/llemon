from enum import Enum
from functools import reduce
import operator
from types import UnionType
from typing import Any, Annotated, Iterable, overload

from pydantic import BaseModel, Field

UNSUPPORTED_KEYS = [
    "const",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "prefixItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "minLength",
    "maxLength",
    "allOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "examples",
    "deprecated",
    "$comment",
    "readOnly",
    "writeOnly",
    "externalDocs",
    "$anchor",
    "$dynamicRef",
    "$dynamicAnchor",
    "$vocabulary",
]
TITLE = "title"
DESCRIPTION = "description"
TYPE = "type"
BOOLEAN = "boolean"
INTEGER = "integer"
NUMBER = "number"
STRING = "string"
ARRAY = "array"
ENUM = "enum"
OBJECT = "object"
ANY_OF = "anyOf"
PROPERTIES = "properties"
ITEMS = "items"
REQUIRED = "required"
MINIMUM = "minimum"
MAXIMUM = "maximum"
EXCLUSIVE_MINIMUM = "exclusiveMinimum"
EXCLUSIVE_MAXIMUM = "exclusiveMaximum"
MULTIPLE_OF = "multipleOf"
PATTERN = "pattern"
FORMAT = "format"
MIN_ITEMS = "minItems"
MAX_ITEMS = "maxItems"
DEFS = "$defs"
REF = "$ref"
REF_PREFIX = "#/$defs/"
TYPES = [BOOLEAN, INTEGER, NUMBER, STRING, ARRAY, ENUM, ANY_OF, OBJECT]
ENUM_TYPES = {"integer": int, "number": float, "string": str}


class Undefined:
    pass

undefined = Undefined()


def schema_to_model(schema: dict[str, Any], ignore_invalid: bool = False) -> type[BaseModel]:
    path = "$"
    title = get(path, schema, TITLE, str)
    get(path, schema, TYPE, str, one_of=[OBJECT])
    definitions: dict[str, type[BaseModel]] = {}
    for name, definition in get(path, schema, DEFS, dict, default={}).items():
        definitions[name] = create_object("$defs", name, definition, definitions, ignore_invalid=ignore_invalid)
    return create_object(path, title, schema, definitions, ignore_invalid=ignore_invalid)


def create_object(
    path: str,
    name: str,
    schema: dict[str, Any],
    definitions: dict[str, type[BaseModel]],
    *,
    ignore_invalid: bool = False,
) -> type[BaseModel]:
    if not ignore_invalid:
        check_invalid(path, schema)
    if get(path, schema, REF, str, default=None):
        return create_ref(path, schema, definitions)
    if get(path, schema, ANY_OF, list, default=None):
        return create_any_of(path, name, schema, definitions, ignore_invalid=ignore_invalid)
    properties = get(path, schema, PROPERTIES, dict)
    required = get(path, schema, REQUIRED, list, default=[])
    fields = {}
    for key, value in properties.items():
        fields[key] = create_field(
            path=f"{path}.{key}",
            name=key,
            field=value,
            definitions=definitions,
            required=key in required,
            ignore_invalid=ignore_invalid,
        )
    description = get(path, schema, DESCRIPTION, str, default=None)
    return type(name, (BaseModel,), {"__annotations__": fields, "__doc__": description})


def create_field(
    path: str,
    name: str,
    field: Any,
    definitions: dict[str, type[BaseModel]],
    *,
    required: bool,
    ignore_invalid: bool = False,
) -> Any:
    if not isinstance(field, dict):
        raise ValueError(f"{path} must be a dictionary")
    if not ignore_invalid:
        check_invalid(path, field)
    if get(path, field, REF, str, default=None):
        return create_ref(path, field, definitions)
    if get(path, field, ANY_OF, list, default=None):
        return create_any_of(path, name, field, definitions, ignore_invalid=ignore_invalid)
    type_ = get(path, field, TYPE, str, one_of=TYPES)
    options: dict[str, Any] = {}
    description = get(path, field, DESCRIPTION, str, default="")
    if description:
        options["description"] = description
    annotation: Enum | type | UnionType
    enum = get(path, field, ENUM, list, default=[])
    if enum:
        annotation = create_enum(path, name, enum, type_)
    elif type_ == BOOLEAN:
        annotation = bool
    elif type_ == INTEGER:
        annotation = int
    elif type_ == NUMBER:
        annotation = float
        options.update(get_number_options(path, field))
    elif type_ == STRING:
        annotation = str
        options.update(get_string_options(path, field))
    elif type_ == ARRAY:
        items = get(path, field, ITEMS, dict)
        annotation = create_object(path, name, items, definitions, ignore_invalid=ignore_invalid)
        annotation = list[annotation]  # type: ignore
        options.update(get_array_options(path, field))
    else:  # type_ == OBJECT:
        annotation = create_object(path, name, field, definitions, ignore_invalid=ignore_invalid)
    if not required:
        annotation |= None
    if options:
        return Annotated[annotation, Field(**options)]
    return annotation


def create_ref(
    path: str,
    schema: dict[str, Any],
    definitions: dict[str, type[BaseModel]],
) -> Any:
    keys = list(schema)
    keys.remove(REF)
    if keys:
        raise ValueError(f"{path} can't define both {REF} and {concat(keys)}")
    ref = schema[REF]
    if not ref.startswith(REF_PREFIX):
        raise ValueError(f"{path} ref must start with {REF_PREFIX}")
    ref = ref.removeprefix(REF_PREFIX)
    if ref not in definitions:
        raise ValueError(f"{path} ref {ref!r} is not defined (available definitions are {concat(definitions)})")
    return definitions[ref]


def create_any_of(
    path: str,
    name: str,
    schema: dict[str, Any],
    definitions: dict[str, type[BaseModel]],
    *,
    ignore_invalid: bool = False,
) -> Any:
    keys = list(schema)
    keys.remove(ANY_OF)
    if keys:
        raise ValueError(f"{path} can't define both {ANY_OF} and {concat(keys)}")
    any_of = schema[ANY_OF]
    if not all(isinstance(item, dict) for item in any_of):
        raise ValueError(f"{path} any-of must be a list of dictionaries")
    annotations: list[type[BaseModel]] = []
    for index, item in enumerate(any_of):
        annotation = create_object(f"{path}.{index}", name, item, definitions, ignore_invalid=ignore_invalid)
        annotations.append(annotation)
    return reduce(operator.or_, annotations)


def create_enum(path: str, name: str, enum: list[str], type_: str) -> type[Enum]:
    if type_ not in ENUM_TYPES:
        raise ValueError(f"{path} enum is only supported for {concat(ENUM_TYPES)}")
    enum_type = ENUM_TYPES[type_]
    if not all(isinstance(item, enum_type) for item in enum):
        raise ValueError(f"{path} enum must be a list of {type_}s")
    return Enum(name, {str(value): value for value in enum}, type=enum_type)  # type: ignore


def get_number_options(path: str, field: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    minimum = get(path, field, MINIMUM, float, default=None)
    if minimum is not None:
        options["ge"] = minimum
    maximum = get(path, field, MAXIMUM, float, default=None)
    if maximum is not None:
        options["le"] = maximum
    exclusive_minimum = get(path, field, EXCLUSIVE_MINIMUM, float, default=None)
    if exclusive_minimum is not None:
        options["gt"] = exclusive_minimum
    exclusive_maximum = get(path, field, EXCLUSIVE_MAXIMUM, float, default=None)
    if exclusive_maximum is not None:
        options["lt"] = exclusive_maximum
    multiple_of = get(path, field, MULTIPLE_OF, float, default=None)
    if multiple_of is not None:
        options["multiple_of"] = multiple_of
    return options


def get_string_options(path: str, field: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    pattern = get(path, field, PATTERN, str, default=None)
    if pattern is not None:
        options["pattern"] = pattern
    format = get(path, field, FORMAT, str, default=None)
    if format is not None:
        options["format"] = format
    return options


def get_array_options(path: str, field: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    min_items = get(path, field, MIN_ITEMS, int, default=None)
    if min_items is not None:
        options["min_length"] = min_items
    max_items = get(path, field, MAX_ITEMS, int, default=None)
    if max_items is not None:
        options["max_length"] = max_items
    return options


@overload
def get[T](
    path: str,
    schema: dict[str, Any],
    key: str,
    type_: type[T],
    default: Undefined = undefined,
    one_of: list[T] | None = None,
) -> T: ...

@overload
def get[T, D](
    path: str,
    schema: dict[str, Any],
    key: str,
    type_: type[T],
    default: D,
    one_of: list[T] | None = None,
) -> T | D: ...

def get[T, D](
    path: str,
    schema: dict[str, Any],
    key: str,
    type_: type[T],
    default: D | Undefined = undefined,
    one_of: list[T] | None = None,
) -> T | D:
    path = f"{path}.{key}" if path else key
    if key not in schema:
        if not isinstance(default, Undefined):
            return default
        raise ValueError(f"{path} is required (available keys are {concat(schema)})")
    value = schema[key]
    if not isinstance(value, (int | float) if type_ is float else type_):
        raise ValueError(f"{path} must be of type {type_} (not {type(value).__name__})")
    if one_of and value not in one_of:
        raise ValueError(f"{path} must be one of {concat(one_of)}")
    return value


def check_invalid(path: str, schema: dict[str, Any]) -> None:
    if schema.get("additionalProperties", False):
        raise ValueError(f"{path} must not have additional properties")
    for key in UNSUPPORTED_KEYS:
        if key in schema:
            raise ValueError(f"{key} used in {path} is not supported")


def check_not_defined(path: str, schema: dict[str, Any], reason: str, keys: list[str]) -> None:
    for key in keys:
        if key in schema:
            raise ValueError(f"{path} can't define both {reason} and {key}")


def concat(iterable: Iterable[Any]) -> str:
    items = list(iterable)
    if not items:
        return "<none>"
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(map(str, items[:-1])) + f" or {items[-1]}"