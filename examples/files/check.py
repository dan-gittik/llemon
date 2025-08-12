import asyncio
import pathlib
from llemon import OpenAI, Anthropic, Gemini, DeepInfra


async def check_file_support():
    models = [
        OpenAI.gpt4o,
        # Anthropic.sonnet4,
        # Gemini.pro25,
        # Gemini.flash25,
        # Gemini.lite25,
        # Gemini.flash2,
        # Gemini.lite2,
        # DeepInfra.llama31_70b,
    ]
    for model in models:
        print(model.name)
        for file in pathlib.Path("examples/files").glob("*"):
            if file.suffix == ".py":
                continue
            print(f"  {file.name}")
            try:
                text = await model.complete("What is the message in the attached file?", files=[file])
                print(f"    {text!r}")
            except Exception as error:
                print(f"    {error!r}")


asyncio.run(check_file_support())