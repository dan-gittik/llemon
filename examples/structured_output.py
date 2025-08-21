from __future__ import annotations
import asyncio

from llemon import OpenAI, Gemini, Anthropic, enable_logs
from pydantic import BaseModel

enable_logs()

SCHEMA = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
        },
        "age": {
            "type": "integer",
        },
        "hobbies": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["name", "hobbies"],
}


class Person(BaseModel):
    name: str
    age: int | None
    hobbies: list[str]


async def main():
    models = [OpenAI.gpt5_nano, Anthropic.haiku3, Gemini.lite2]
    for model in models:
        async with model.conversation("Extract information about the person.") as conv:
            response = await conv.generate_object(Person,
                """
                My name is Alice, and I like reading and hiking.
                """,
            )
            # print(response)
            response = await conv.generate_object(SCHEMA,
                """
                My name is Bob. I'm thirty, and I like cooking.
                """,
            )
            # print(response)


asyncio.run(main())
