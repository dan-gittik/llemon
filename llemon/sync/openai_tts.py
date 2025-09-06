from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import openai

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import SynthesizeRequest, SynthesizeResponse


class OpenAITTS(llemon.TTSProvider):

    client: openai.OpenAI
    default_voice: ClassVar[str] = "alloy"

    def _synthesize(self, request: SynthesizeRequest, response: SynthesizeResponse) -> None:
        try:
            # TODO: raw HTTP request to get the usage?
            openai_response = self.with_overrides(self.client.audio.speech.create)(
                request,
                model=request.tts.model,
                input=request.text,
                voice=request.voice or self.default_voice,
                response_format=request.output_format,  # type: ignore
                instructions=request.instructions or openai.NOT_GIVEN,
                timeout=request.timeout,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        request.id = openai_response.response.headers["x-request-id"]
        data = openai_response.read()
        mimetype = openai_response.response.headers["content-type"]
        response.complete_synthesis(data, mimetype)
