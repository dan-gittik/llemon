import asyncio
import pathlib

from llemon import OpenAI, Gemini, Anthropic, enable_logs

enable_logs()

FILES_PATH = pathlib.Path(__file__).parent / "files"
CAT_PATH = FILES_PATH / "cat.jpg"
DOG_URL = "https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
HELLO_PATH = FILES_PATH / "hello.pdf"


async def main():
    models = [OpenAI.gpt5_nano, Anthropic.haiku35, Gemini.flash2]
    for model in models:
        async with model.conversation("Answer in a single word or two.") as conv:
            response = await conv.generate("Which animal is in this picture?", files=[CAT_PATH])
            # print(response)
            response = await conv.generate("What about this one?", files=[DOG_URL])
            # print(response)
            response = await conv.generate("What about this one?", files=[("image/jpeg", CAT_PATH.read_bytes())])
            # print(response)
            response = await conv.generate("what does this file say?", files=[str(HELLO_PATH)])
            # print(response)


asyncio.run(main())
