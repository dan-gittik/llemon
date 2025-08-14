import asyncio
import pathlib

from sqlalchemy import create_engine, text

from llemon import OpenAI, Database, enable_logs

enable_logs()

root = pathlib.Path(__file__).parent
database = root / "users.db"


async def main():
    if database.exists():
        database.unlink()
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)"))
        connection.execute(text("INSERT INTO users (name, age) VALUES ('John', 25)"))
        connection.execute(text("INSERT INTO users (name, age) VALUES ('Jane', 30)"))
    conv = OpenAI.gpt5_nano("you are an assistant with access to a database", tools=[Database(f"sqlite:///{database}")])
    response = await conv.complete("""
        How old is John?
    """)
    print(response)
    response = await conv.complete("""
        What about Jane?
    """)


asyncio.run(main())
