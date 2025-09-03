import asyncio
import pathlib
import json

from llemon import OpenAI, enable_logs, Conversation

enable_logs()

FILES_PATH = pathlib.Path(__file__).parent / "files"


async def main():
    models = [OpenAI.gpt5_nano]
    for model in models:
        async with model.conversation() as conv:
            response = await conv.synthesize("Hello, my name is Alice.")
            response.audio.save(FILES_PATH / "audio_output.mp3")
            # print(response)
        data = conv.dump()
        print(json.dumps(data, indent=2))
        async with Conversation.load(data) as conv:
            print(conv.format())


asyncio.run(main())
