from __future__ import annotations
import asyncio

from llemon import OpenAI, Gemini, Anthropic, LLMModel, enable_logs
from pydantic import BaseModel

enable_logs()


class Translation(BaseModel):
    sentences: list[Sentence]


class Sentence(BaseModel):
    original: str
    translation: str
    explanation: str


async def main():
    models: list[LLMModel] = [OpenAI.gpt5_nano, Anthropic.haiku3, Gemini.lite2]
    for model in models:
        print(model)
        conv = model("""
            translate the following text from {{ source_language }} to {{ target_language }}
        """)
        response = await conv.construct(Translation, """
            This is the first sentence.
            This is the second sentence.
        """, {"source_language": "English", "target_language": "Spanish"})
        print(response)


asyncio.run(main())
