import asyncio
import pathlib

from llemon import OpenAI, Directory, enable_logs

enable_logs()

root = pathlib.Path(__file__).parent


async def main():
    conv = OpenAI.gpt5_nano("you are an assistant with access to a filesystem", tools=[Directory(root / "files", readonly=False)])
    response = await conv.complete("""
        What is the secret keyword?
    """)
    print(response)
    response = await conv.complete("""
        Please change it to banana.
    """)


asyncio.run(main())
