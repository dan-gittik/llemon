import asyncio

from llemon import OpenAI, enable_logs

enable_logs()

async def main():
    models = [OpenAI.gpt5_nano]
    for model in models:
        async with model.conversation("Be concise.") as conv:
            response = await conv.embed("Hello, world!")
            print(response)


asyncio.run(main())
