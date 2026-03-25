import os
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup basic logging as a backup to print
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def call_llm_with_context(query, context_documents):
    # FORCE PRINT: If you don't see this, the function isn't even being called!
    print(f"\nLLM Function Triggered for Query: {query}")
    print(f"Number of docs received from Weaviate: {len(context_documents)}")

    if not context_documents:
        return "I'm sorry, I don't have any records in my database to answer that."

    # Build context string
    sources = []
    for i, doc in enumerate(context_documents):
        # Weaviate v4 objects use .properties (a dict)
        content = doc.properties.get('content', 'Empty Content')
        sources.append(f"Source {i + 1}: {content}")

    context_text = "\n\n".join(sources)

    # PRINT THE CONTEXT: Check if 'Monthly Subscription' is here
    print("-" * 30)
    print("EXTRACTED CONTEXT FOR LLM:")
    print(context_text)
    print("-" * 30)

    system_message = (
        "You are a Fraud Analysis Expert. Answer the user's question ONLY using the provided context. "
        "If the answer is not in the context, say you do not know. "
        "Cite your sources like [Source 1]."
    )

    user_message = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"OpenAI Error: {e}")
        return "Error calling the AI model."
