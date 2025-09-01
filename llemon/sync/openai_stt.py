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
            openai_response = self.client.audio.transcriptions.create(
                model=request.stt.model,
                file=(request.audio.name, request.audio.data, request.audio.mimetype),
                prompt=_optional(request.prompt),
                language=_optional(request.language),
                timeout=_optional(request.timeout),
                response_format="verbose_json" if request.timestamps else openai.NOT_GIVEN,  # type: ignore
                timestamp_granularities=["word"] if request.timestamps else openai.NOT_GIVEN,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        if isinstance(openai_response, TranscriptionVerbose) and openai_response.words:
            timestamps = [(word.word, word.start, word.end - word.start) for word in openai_response.words]
        else:
            timestamps = None
        response.complete_transcription(openai_response.text, timestamps)
        return response


def _optional[T](value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN
