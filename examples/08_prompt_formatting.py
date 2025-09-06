from __future__ import annotations
import asyncio

from llemon import OpenAI, Gemini, Anthropic, enable_logs

enable_logs()

async def main():
    models = [OpenAI.gpt5_nano, Anthropic.haiku3, Gemini.lite2]
    language = "Spanish"
    is_beginner = True
    for model in models:
        async with model.conversation(
            """
            Translate the following into {{ language }}.
            {% if is_beginner %}
            Add an explanation for beginners, breaking this translation down.
            {% endif %}
            """
        ) as conv:
            response = await conv.generate(
                """
                To be, or not to be? That is the question.
                """,
                context={"language": language, "is_beginner": is_beginner},
            )
            # print(response)


asyncio.run(main())
