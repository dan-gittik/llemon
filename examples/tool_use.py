import asyncio
import logging

from llemon import OpenAI, Gemini, Anthropic, LLMModel

logging.basicConfig(level=logging.DEBUG)


def get_weather(city: str) -> int:
    """
    Given a city, return the temperature in Celsius.
    """
    print("tool called", city)
    return {
        "Paris": 20,
        "Berlin": 18,
        "Tokyo": 25,
    }[city]


async def main():
    models: list[LLMModel] = [OpenAI.gpt5_nano, Anthropic.haiku3, Gemini.lite2]
    for model in models:
        print(model)
        conv = model("you are a pirate", tools=[get_weather])
        response = await conv.complete("""
            where is it hottest, in Paris, Berlin or Tokyo?
        """)
        print(response)
        print(conv.history)
        response = await conv.complete("""
            so which one is coldest?
        """, use_tool=False)
        print(response)


asyncio.run(main())
