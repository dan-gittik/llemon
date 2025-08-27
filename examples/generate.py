import asyncio

from llemon import OpenAI, Gemini, Anthropic, DeepInfra, enable_logs

enable_logs()

async def main():
    models = [OpenAI.gpt5_nano, Anthropic.haiku35, Gemini.flash2, DeepInfra.llama31_8b]
    for model in models:
        async with model.conversation("Be concise.") as conv:
            response = await conv.generate("When was Alan Turing born?")
            # print(response)


asyncio.run(main())
