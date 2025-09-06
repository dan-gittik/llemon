from __future__ import annotations

from typing import TYPE_CHECKING

import openai

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import EmbedRequest, EmbedResponse


class OpenAIEmbedder(llemon.EmbedderProvider):

    client: openai.OpenAI

    def _embed(self, request: EmbedRequest, response: EmbedResponse) -> None:
        try:
            openai_response = self.client.embeddings.create(
                model=request.embedder.model,
                input=request.text,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        response.input_tokens += openai_response.usage.prompt_tokens or 0
        if not openai_response.data:
            raise request.error(f"{request} has no response")
        response.complete_embedding(openai_response.data[0].embedding)
