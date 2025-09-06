import asyncio
import pathlib

from llemon import OpenAI, Anthropic, enable_logs

enable_logs()

DOCUMENT_PATH = pathlib.Path(__file__).parent / "files" / "hello.pdf"


async def main():
    async with Anthropic.haiku35.conversation() as conv:
        response = await conv.generate(
            """
            What's the content of the document?
            """,
            files=[DOCUMENT_PATH],
        )
        # print(response)
        conv2 = await conv.replace(llm=OpenAI.gpt5_nano).prepare()
    async with conv2:
        response = await conv2.generate(
            """
            And what's the background color?
            """,
        )
        # print(response)


asyncio.run(main())
