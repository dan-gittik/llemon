import asyncio
import pathlib

from llemon import OpenAI, Anthropic, Gemini, Database, enable_logs
from sqlalchemy import create_engine, text

enable_logs()

DATABASE_PATH = pathlib.Path(__file__).parent / "db.sqlite3"


async def main():
    create_database()
    models = [OpenAI.gpt5_nano, Anthropic.haiku35, Gemini.flash2]
    for model in models:
        async with model.conversation(tools=[Database(f"sqlite:///{DATABASE_PATH}")]) as conv:
            response = await conv.generate(
                """
                How old is Alice?
                """
            )
            # print(response)
            response = await conv.generate(
                """
                What about Bob?
                """
            )
            # print(response)


def create_database():
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)"))
        connection.execute(text("INSERT INTO users (name, age) VALUES ('Alice', 25)"))
        connection.execute(text("INSERT INTO users (name, age) VALUES ('Bob', 30)"))


asyncio.run(main())
