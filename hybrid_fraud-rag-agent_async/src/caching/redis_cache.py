import os
import redis
import hashlib

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def key(q): return hashlib.md5(q.encode()).hexdigest()


def get_cache(q): return r.get(key(q))


def set_cache(q, v): r.setex(key(q), 3600, v)
