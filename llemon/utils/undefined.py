from __future__ import annotations


class Undefined:

    def __str__(self) -> str:
        return "undefined"

    def __repr__(self) -> str:
        return "<undefined>"

    def __bool__(self) -> bool:
        return False


undefined = Undefined()
