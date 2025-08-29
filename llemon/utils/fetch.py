from __future__ import annotations

import httpx


def fetch(url: str) -> bytes:
    response = httpx.get(url)
    return response.content


async def async_fetch(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.content
