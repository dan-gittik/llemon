from __future__ import annotations

from functools import cached_property
from typing import Any

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.schema import CreateTable

import llemon


class Database(llemon.Toolbox):

    def __init__(self, url: str, readonly: bool = True) -> None:
        self.engine = create_engine(url)
        self.readonly = readonly
        super().__init__(self.url)
        self._init.update(
            url=url,
            readonly=self.readonly,
        )

    def __str__(self) -> str:
        return f"database at {self.url}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @cached_property
    def url(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)

    def run_sql_tool(self, sql: str) -> list[dict[str, Any]]:
        if ";" in sql.strip(";"):
            raise ValueError("SQL must be a single statement")
        with self.engine.begin() as connection:
            try:
                result = connection.execute(text(sql))
                return [dict(row) for row in result.mappings()]
            finally:
                if self.readonly:
                    connection.rollback()

    def run_sql_description(self) -> str:
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        schema: list[str] = []
        for table in metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(self.engine))
            schema.append(ddl)
        readonly = "Note: you can't change the database, only read from it." if self.readonly else ""
        return f"""
            Receives a single {self.engine.dialect.name} statement and returns the result of its execution.
            The database schema is:
            {'\n'.join(schema)}
            {readonly}
        """
