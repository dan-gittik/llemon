import asyncio
from llemon import OpenAI, Gemini, Anthropic, Model


async def main():
    models: list[Model] = [OpenAI.gpt_4o, Anthropic.sonnet4, Gemini.pro25]
    for model in models:
        print(model)
        conv = model("""
            analyze images sent by the user and return a summary of the content
        """)
        response = await conv.complete(files=["examples/cat.jpg"])
        print(response)
        response = await conv.complete(files=["https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/250px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg"])
        print(response)


asyncio.run(main())
