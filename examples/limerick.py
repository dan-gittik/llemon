import asyncio
from typing import Literal

from llemon import Gemini, Anthropic, enable_logs
from pydantic import BaseModel

enable_logs()


class Review(BaseModel):
    grade: Literal["A", "B", "C"]
    explanation: str | None


async def main():
    writer = Gemini.lite2.conversation("Answer in 5-line limericks, nothing else.")
    critic = Anthropic.haiku35.conversation("Grade a limerick on its style and wit as A, B or C; if not A, explain what can be improved. Don't be too nice.")
    async with writer, critic:
        limerick = await writer.generate()
        for _ in range(10):
            review = await critic.generate_object(Review, limerick.text)
            if review.object.grade == "A":
                break
            limerick = await writer.generate(
                f"""
                Rewrite your limerick given the following review:
                {review.object.explanation}
                """,
            )
        print(limerick.text)


asyncio.run(main())
