import secrets


def random_suffix() -> str:
    return f"__{secrets.token_hex(8)}"
