import asyncio

from llemon import OpenAI, Gemini, Anthropic, enable_logs

enable_logs()


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
    models = [OpenAI.gpt5_nano, Anthropic.haiku35, Gemini.flash2]
    for model in models:
        async with model.conversation(tools=[get_weather]) as conv:
            response = await conv.generate(
                """
                Where is it hottest, in Paris, Berlin or Madrid?
                """
            )
            # print(response)
            response = await conv.generate(
                """
                And where is it coldest?
                """,
                use_tool=False,
            )
            # print(response)


asyncio.run(main())
