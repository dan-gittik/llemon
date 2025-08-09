from __future__ import annotations

import json
from typing import Self

import openai
from pydantic import BaseModel

from ..requests import History
from ..model import ModelGetter
from .openai import OpenAI


class DeepInfra(OpenAI):

    # models
    llama_31_70b = ModelGetter("meta-llama/meta-llama-3.1-70b-instruct")
    llama_31_8b = ModelGetter("meta-llama/meta-llama-3.1-8b-instruct")
    # /models

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=api_key)

    async def generate[T: BaseModel](
        self,
        format: type[T],
        model: str,
        prompt: str,
        message: str | None = None,
        history: History | None = None,
        *,
        temperature: float | None = None,
    ) -> T:
        instructions = f"\nAnswer ONLY in JSON that adheres EXACTLY to the following schema: {format.model_json_schema()}"
        if message:
            message += instructions
        else:
            prompt += instructions
        messages = self._create_messages(prompt, message, history)
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **self._options(
                temperature=temperature,
            )
        )
        if not response.choices:
            raise ValueError(f"no response from {self}")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(f"unexpected finish reason from {self}: {choice.finish_reason}")
        if not choice.message.content:
            raise ValueError(f"no content in response from {self}")
        output = json.loads(choice.message.content)
        return format(**output)