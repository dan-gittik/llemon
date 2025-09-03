from __future__ import annotations

from typing import TYPE_CHECKING

import openai
from openai.types.audio import TranscriptionVerbose

import llemon.sync as llemon

if TYPE_CHECKING:
    from llemon.sync import TranscribeRequest, TranscribeResponse


class OpenAISTT(llemon.STTProvider):

    client: openai.OpenAI

    def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        response = llemon.TranscribeResponse(request)
        try:
            request.audio.fetch()
            assert request.audio.data is not None
            openai_response = self.with_overrides(self.client.audio.transcriptions.create)(
                request,
                model=request.stt.model,
                file=(request.audio.name, request.audio.data, request.audio.mimetype),
                prompt=_optional(request.instructions),
                language=_optional(request.language),
                timeout=_optional(request.timeout),
                response_format="verbose_json" if request.timestamps else openai.NOT_GIVEN,  # type: ignore
                timestamp_granularities=["word"] if request.timestamps else openai.NOT_GIVEN,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        if openai_response._request_id:
            request.id = openai_response._request_id
        if isinstance(openai_response, TranscriptionVerbose):
            timestamps = [(word.word, word.start, word.end - word.start) for word in openai_response.words or []]
            if openai_response.usage:
                response.duration = openai_response.usage.seconds
        else:
            if openai_response.usage:
                if openai_response.usage.type == "tokens":
                    response.input_tokens = openai_response.usage.input_tokens
                elif openai_response.usage.type == "duration":
                    response.duration = openai_response.usage.seconds
                timestamps = None
        response.complete_transcription(openai_response.text, timestamps)
        return response


def _optional[T](value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN
