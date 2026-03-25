import os
from openai import AsyncOpenAI  # Changed to Async
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use the Async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Add 'async' here
async def get_embedding(text):
    res = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding