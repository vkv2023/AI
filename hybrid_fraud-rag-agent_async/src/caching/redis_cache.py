import os
import redis.asyncio as redis  # Use the asyncio-compatible client , interface
import hashlib

# Get environment variables with fallbacks
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Initialize the Async client
# decode_responses=True ensures you get strings back instead of bytes
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


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
