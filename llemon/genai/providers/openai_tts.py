from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import openai

import llemon

if TYPE_CHECKING:
    from llemon import SynthesizeRequest, SynthesizeResponse


class OpenAITTS(llemon.TTSProvider):

    client: openai.AsyncOpenAI
    default_voice: ClassVar[str] = "alloy"

    async def synthesize(self, request: SynthesizeRequest) -> SynthesizeResponse:
        response = llemon.SynthesizeResponse(request)
        try:
            # TODO: raw HTTP request to get the usage?
            openai_response = await self.with_overrides(self.client.audio.speech.create)(
                request,
                model=request.tts.model,
                input=request.text,
                voice=request.voice or self.default_voice,
                response_format=request.output_format,
                instructions=request.instructions,
                timeout=request.timeout,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        request.id = openai_response.response.headers["x-request-id"]
        data = await openai_response.aread()
        mimetype = openai_response.response.headers["content-type"]
        response.complete_synthesis(llemon.File.from_data(mimetype, data))
        return response