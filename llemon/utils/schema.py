from __future__ import annotations

import operator
from enum import Enum
from functools import reduce
from types import NoneType, UnionType
from typing import Annotated, Any

from pydantic import BaseModel, Field

from .concat import concat
from .filtered_dict import filtered_dict
from .unpacker import Unpacker

type Definitions = dict[str, type[BaseModel]]

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
DEFAULT = "default"
NULL = "null"
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
TYPES = [NULL, BOOLEAN, INTEGER, NUMBER, STRING, ARRAY, ENUM, ANY_OF, OBJECT]
ENUM_TYPES = {"integer": int, "number": float, "string": str}


def schema_to_model(schema: dict[str, Any]) -> type[BaseModel]:
    unpacker = Unpacker(schema)
    title = unpacker.get(TITLE, str)
    unpacker.get(TYPE, str, one_of=[OBJECT])
    defs: Definitions = {}
    for name, def_ in unpacker.load_dict(DEFS, required=False):
        defs[name] = create_object(name, def_, defs)
    return create_object(title, unpacker, defs)


def create_object(name: str, unpacker: Unpacker, defs: Definitions) -> type[BaseModel]:
    unpacker.check(undefined=UNSUPPORTED_KEYS, additionalProperties=False)
    if unpacker.get(REF, str, None):
        return create_ref(unpacker, defs)
    if unpacker.get(ANY_OF, list, None):
        return create_any_of(unpacker, defs)
    required = unpacker.get(REQUIRED, list, [])
    fields = {}
    for key, value in unpacker.load_dict(PROPERTIES):
        fields[key] = create_field(value, defs, required=key in required)
    description = unpacker.get(DESCRIPTION, str, None)
    return type(name, (BaseModel,), {"__annotations__": fields, "__doc__": description})


def create_field(unpacker: Unpacker, defs: Definitions, *, required: bool = True) -> Any:
    unpacker.check(undefined=UNSUPPORTED_KEYS, additionalProperties=False)
    if unpacker.get(REF, str, None):
        return create_ref(unpacker, defs)
    if unpacker.get(ANY_OF, list, None):
        return create_any_of(unpacker, defs)
    type_ = unpacker.get(TYPE, str, one_of=TYPES)
    options = filtered_dict(
        description=unpacker.get(DESCRIPTION, str, None),
    )
    annotation: Enum | type | UnionType
    enum = unpacker.get(ENUM, list, [])
    if enum:
        annotation = create_enum(unpacker, enum, type_)
    elif type_ == NULL:
        annotation = NoneType
    elif type_ == BOOLEAN:
        annotation = bool
    elif type_ == INTEGER:
        annotation = int
    elif type_ == NUMBER:
        annotation = float
        options.update(get_number_options(unpacker))
    elif type_ == STRING:
        annotation = str
        options.update(get_string_options(unpacker))
    elif type_ == ARRAY:
        annotation = create_field(unpacker.load(ITEMS), defs)
        annotation = list[annotation]  # type: ignore
        options.update(get_array_options(unpacker))
    else:  # type_ == OBJECT:
        annotation = create_object(unpacker.key, unpacker, defs)
    if not required:
        annotation |= None
    if options:
        return Annotated[annotation, Field(**options)]
    return annotation


def create_ref(unpacker: Unpacker, defs: Definitions) -> Any:
    unpacker.check_empty(REF)
    ref = unpacker.get(REF, str)
    if not ref.startswith(REF_PREFIX):
        raise ValueError(f"{unpacker} ref must start with {REF_PREFIX}")
    ref = ref.removeprefix(REF_PREFIX)
    if ref not in defs:
        raise ValueError(f"{unpacker} ref {ref!r} is not defined (available definitions are {concat(defs)})")
    return defs[ref]


def create_any_of(unpacker: Unpacker, defs: Definitions) -> Any:
    unpacker.check_empty(ANY_OF, except_for=[TITLE, DEFAULT])
    annotations: list[type[BaseModel]] = []
    for item in unpacker.load_list(ANY_OF):
        annotation = create_field(item, defs)
        annotations.append(annotation)
    return reduce(operator.or_, annotations)


def create_enum(unpacker: Unpacker, enum: list[str], type_: str) -> type[Enum]:
    if type_ not in ENUM_TYPES:
        raise ValueError(f"{unpacker} enum is only supported for {concat(ENUM_TYPES)}")
    enum_type = ENUM_TYPES[type_]
    if not all(isinstance(item, enum_type) for item in enum):
        raise ValueError(f"{unpacker} enum must be a list of {type_}s")
    return Enum(unpacker.key, {str(value): value for value in enum}, type=enum_type)  # type: ignore


def get_number_options(unpacker: Unpacker) -> dict[str, Any]:
    return filtered_dict(
        ge=unpacker.get(MINIMUM, float, None),
        le=unpacker.get(MAXIMUM, float, None),
        gt=unpacker.get(EXCLUSIVE_MINIMUM, float, None),
        lt=unpacker.get(EXCLUSIVE_MAXIMUM, float, None),
        multiple_of=unpacker.get(MULTIPLE_OF, float, None),
    )


def get_string_options(unpacker: Unpacker) -> dict[str, Any]:
    return filtered_dict(
        pattern=unpacker.get(PATTERN, str, None),
        format=unpacker.get(FORMAT, str, None),
    )


def get_array_options(unpacker: Unpacker) -> dict[str, Any]:
    return filtered_dict(
        min_length=unpacker.get(MIN_ITEMS, int, None),
        max_length=unpacker.get(MAX_ITEMS, int, None),
    )
