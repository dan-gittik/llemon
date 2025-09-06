import asyncio
from llemon import OpenAI, Anthropic, Gemini, DeepInfra


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
    s = "Hello, world! How are you today?"
    models = [OpenAI.gpt41_nano, Anthropic.haiku3, Gemini.lite2, DeepInfra.llama31_8b]
    for model in models:
        print(await model.tokenizer.count(s, tools=[get_weather]))
        try:
            ids = await model.tokenizer.encode(s)
            print(ids)
            text = await model.tokenizer.decode(*ids)
            print(text)
            tokens = await model.tokenizer.parse(s)
            print(tokens)
        except Exception as e:
            print(e)


asyncio.run(main())

