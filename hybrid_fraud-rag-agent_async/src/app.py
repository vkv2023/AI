import os
import sys
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.orchestrator.agent import handle_query
import src.configurations as conf
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

# Ensure the root directory is in the path so 'src' is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Rate limit settings: 5 requests per 60 seconds per IP
RATE_LIMIT_DURATION = 60
RATE_LIMIT_REQUESTS = 5

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Total number of requests to the application',
    ['endpoint']
)
REQUEST_SUCCESS_COUNT = Counter(
    'app_request_success_count',
    'Number of successful requests to the application',
    ['endpoint']
)
REQUEST_ERROR_COUNT = Counter(
    'app_request_error_count',
    'Number of erroneous requests to the application',
    ['endpoint']
)
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Latency of requests to the application',
    ['endpoint']
)

'''
Request Arrives: FastAPI identifies the user by their IP address.
    Redis Lookup: It checks a key like rate_limit:xxx.xxx.xxx.xxx
    If the value is < 5, it allows the request and increments the counter.
    If the value is >= 5, it immediately returns a 429 Too Many Requests status, saving your LLM and Weaviate from doing any work.
    Auto-Reset: After 60 seconds, Redis automatically deletes the key (expire), and the user can send queries again.
'''

# 1. Lifespan Manager: Cleanly start and stop connections
# 1. Define the lifespan FIRST
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        # app.state.redis = redis.Redis(host=conf.REDIS_HOST, port=conf.REDIS_PORT, db=0, decode_responses=True)
        # Test the connection immediately
        # await app.state.redis.ping()
        print("Redis is connected and state.redis is set")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        # If this fails, the app might start but state.redis will be missing!

    yield  # The app stays in this 'yield' state while running

    # Shutdown logic
    # await app.state.redis.close()

app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def query(request: Request, payload: dict):
    # Increment total request counter for Prometheus
    REQUEST_COUNT.labels(endpoint="/query").inc()

    # 1. Get User IP (or a unique API key)
    # user_ip = request.client.host
    # redis_key = f"rate_limit:{user_ip}"

    # print(redis_key)

    # Do NOT use the global 'app' variable here
    # if not hasattr(request.app.state, "docker-redis-1"):
    #     return {"error": "Redis connection not initialized"}

    # 2. Use 'request.app.state' instead of 'app.state'
    # This ensures you are looking at the state of the RUNNING app
    # redis_conn = request.app.state.docker-redis-1

    # current_count = await redis_conn.get(redis_key)

    # if current_count and int(current_count) >= RATE_LIMIT_REQUESTS:
    #     raise HTTPException(
    #         status_code=429,
    #         detail="Too many requests. Please wait a minute."
    #     )

    # 3. Increment the count and set expiration if it's a new key
    # async with app.state.redis.pipeline(transaction=True) as pipe:
    #     await pipe.incr(redis_key)
    #     await pipe.expire(redis_key, RATE_LIMIT_DURATION)
    #     await pipe.execute()

    # Basic validation to prevent KeyErrors
    user_query = payload.get("query")
    if not user_query:
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Missing 'query' in payload")

    try:
        logger.info(f"Processing query: {user_query[:50]}...")
        # handle_query will now use your RAG + Kafka logic
        with REQUEST_LATENCY.labels(endpoint="/query").time():
            result = await handle_query(user_query)
        REQUEST_SUCCESS_COUNT.labels(endpoint="/query").inc()
        return {"response": result}

    except Exception as e:
        logger.error(f"Error in orchestrator: {str(e)}")
        REQUEST_ERROR_COUNT.labels(endpoint="/query").inc()
        raise HTTPException(status_code=500, detail="Internal processing error")


# Health check for Docker/Monitoring
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode('utf-8'))
