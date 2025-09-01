from __future__ import annotations

import openai

import llemon.sync as llemon

FILE_IDS = "openai.file_ids"
FILE_HASHES = "openai.file_hashes"


class OpenAI(llemon.OpenAILLM, llemon.OpenAIEmbedder, llemon.OpenAISTT):

    gpt5 = llemon.LLMProperty("gpt-5")
    gpt5_mini = llemon.LLMProperty("gpt-5-mini")
    gpt5_nano = llemon.LLMProperty("gpt-5-nano")
    gpt41 = llemon.LLMProperty("gpt-4.1")
    gpt41_mini = llemon.LLMProperty("gpt-4.1-mini")
    gpt41_nano = llemon.LLMProperty("gpt-4.1-nano")
    gpt4o = llemon.LLMProperty("gpt-4o")
    gpt4o_mini = llemon.LLMProperty("gpt-4o-mini")
    gpt4 = llemon.LLMProperty("gpt-4")
    gpt4_turbo = llemon.LLMProperty("gpt-4-turbo")
    gpt35_turbo = llemon.LLMProperty("gpt-3.5-turbo")

    embedding_ada = llemon.EmbedderProperty("text-embedding-ada-002")
    embedding3_small = llemon.EmbedderProperty("text-embedding-3-small")
    embedding3_large = llemon.EmbedderProperty("text-embedding-3-large")
    default_embedder = embedding3_small

    whisper = llemon.STTProperty("whisper-1")
    gpt4o_transcribe = llemon.STTProperty("gpt-4o-transcribe")
    gpt4o_mini_transcribe = llemon.STTProperty("gpt-4o-mini-transcribe")
    default_stt = whisper

    def __init__(self, api_key: str) -> None:
        self.client = openai.OpenAI(api_key=api_key)
