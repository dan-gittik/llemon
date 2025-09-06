import asyncio
import pathlib

from llemon import OpenAI, Anthropic, Gemini, Directory, enable_logs

enable_logs()

FILES_PATH = pathlib.Path(__file__).parent / "files"
SECRET_PATH = FILES_PATH / "secret.txt"


async def main():
    models = [OpenAI.gpt5_nano, Anthropic.haiku35, Gemini.flash2]
    for model in models:
        SECRET_PATH.write_text("watermelon")
        async with model.conversation(tools=[Directory(FILES_PATH, readonly=False)]) as conv:
            response = await conv.generate(
                """
                What is the secret keyword?
                """
            )
            # print(response)
            response = await conv.generate(
                """
                Please change it to banana.
                """
            )
            # print(response)


asyncio.run(main())
