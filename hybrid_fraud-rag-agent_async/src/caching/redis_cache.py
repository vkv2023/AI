import os
import redis.asyncio as redis  # Use the asyncio-compatible client , interface
import src.configurations as conf
import hashlib


# Initialize the Async client
# decode_responses=True ensures you get strings back instead of bytes
r = redis.Redis(host=conf.REDIS_HOST, port=conf.REDIS_PORT, decode_responses=True)


def key(q):
    return hashlib.md5(q.encode()).hexdigest()


# 1. Change to 'async def'
async def get_cache(q):
    # 2. Add 'await' before the redis call
    return await r.get(key(q))


# 3. Change to 'async def'
async def set_cache(q, v):
    # 4. Add 'await' before the redis call
    await r.setex(key(q), 3600, v)
