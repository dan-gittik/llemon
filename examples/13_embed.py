import asyncio
import json

from llemon import OpenAI, enable_logs, Conversation

enable_logs()

async def main():
    models = [OpenAI.gpt5_nano]
    for model in models:
        async with model.conversation("Be concise.") as conv:
            response = await conv.embed("Hello, world!")
            # print(response)
        data = conv.dump()
        print(json.dumps(data, indent=2))
        async with Conversation.load(data) as conv:
            print(conv.format())


asyncio.run(main())
