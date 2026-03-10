import redis
from app.core.config import settings

class RedisCache:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL)

    def get(self, key: str):
        value = self.client.get(key)
        return value.decode("utf-8").split("||") if value else None

    def set(self, key: str, value: list, expire: int = 3600):
        self.client.set(key, "||".join(value), ex=expire)