import redis
import hashlib
from config import REDIS_HOST

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def key(q): return hashlib.md5(q.encode()).hexdigest()

def get_cache(q): return r.get(key(q))

def set_cache(q, v): r.setex(key(q), 3600, v)