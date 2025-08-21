import asyncio
import json
import pathlib

from llemon import Anthropic, Directory, Conversation, enable_logs

enable_logs()

FILES_PATH = pathlib.Path(__file__).parent / "files"
CAT_PATH = FILES_PATH / "cat.jpg"


def get_weather(city: str) -> int:
    """
    Given a city, return the temperature in Celsius.
    """
    return {
        "Paris": 18,
        "Berlin": 15,
        "Madrid": 20,
    }[city]


async def main():
    directory = Directory(FILES_PATH)
    async with Anthropic.haiku35.conversation(tools=[get_weather, directory]) as conv:
        response = await conv.generate(
            """
            Where is it hottest, in Paris, Berlin or Madrid?
            """
        )
        # print(response)
        response = await conv.generate(
            """
            What's the secret keyword?
            """,
        )
        # print(response)
        response = await conv.generate(
            """
            Which animal is in this picture?
            """,
            files=[CAT_PATH],
            use_tool=False,
        )
    dump = conv.dump()
    print(json.dumps(dump, indent=2))
    async with Conversation.load(dump) as conv:
        print(conv.format())


asyncio.run(main())
