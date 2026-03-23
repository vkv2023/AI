import redis
import json

r = redis.Redis(host="localhost", port=6379)


def get_cache(key):
    data = r.get(key)

    if data:
        return json.loads(data)  # deserialize a string to a python object

    return None


def set_cache(key, value):
    r.set(key, json.dumps(value), ex=300)  # serialize a python object to a string
