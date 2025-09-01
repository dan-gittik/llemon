import datetime as dt

from .now import now


class Cache[K, V]:

    def __init__(self) -> None:
        self.cache: dict[K, tuple[dt.datetime, V]] = {}

    def add(self, key: K, value: V, ttl: int) -> None:
        self.prune()
        self.cache[key] = now() + dt.timedelta(seconds=ttl), value

    def get(self, key: K) -> V | None:
        if key not in self.cache:
            return None
        expiry, value = self.cache[key]
        if expiry < now():
            del self.cache[key]
            return None
        return value

    def prune(self) -> None:
        now_ = now()
        delete: list[K] = []
        for key, (expiry, _) in self.cache.items():
            if expiry > now_:
                delete.append(key)
        for key in delete:
            del self.cache[key]
