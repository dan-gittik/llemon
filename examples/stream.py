import asyncio

from llemon import OpenAI, Gemini, Anthropic, LLMModel, enable_logs

enable_logs()


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
        response = conv.stream("""
            where is it hottest, in Paris, Berlin or Tokyo?
        """)
        async for chunk in response:
            print(chunk, end="", flush=True)
        print()
        print(conv.history)
        response = conv.stream("""
            so which one is coldest?
        """, use_tool=False)
        async for chunk in response:
            print(chunk, end="", flush=True)
        print()
        print(conv.history)


asyncio.run(main())
