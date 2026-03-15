import redis
import json

r = redis.Redis(host="localhost", port=6379)


def get_cache(key):
    data = r.get(key)

    if data:
        return json.loads(data)

    return None


def set_cache(key, value):
    r.set(key, json.dumps(value), ex=300)
