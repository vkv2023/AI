import asyncio
import json
from hybrid_fraud-rag_agent_async.src.fraud_rag.weaviate_client import search_docs

'''
When you run this, Weaviate is performing two simultaneous searches and merging them:
    Vector Search: It converts your query into a mathematical vector and finds "semantically similar" concepts (e.g., "rapid" matches "multiple transactions within 60 seconds").
    Keyword Search (BM25): It looks for exact word matches like "geographic" or "location".
    Reciprocal Rank Fusion (RRF): It combines both lists to give you the most accurate "Context" for your AI agent.
'''

async def test_hybrid_search():
    # Define a few test queries to see how well it performs
    test_queries = [
        "Is there any suspicious activity related to rapid transactions?",
        "Show me cases involving unusual geographic locations.",
        "Are there any legit transactions in the database?"
    ]

    print("--- Starting Hybrid Search Test ---\n")

    for query in test_queries:
        print(f"🔍 Query: '{query}'")

        # This calls your hybrid search (Vector + Keyword)
        results = await search_docs(query)

        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} relevant cases:")
            for i, doc in enumerate(results):
                # Accessing properties using the v4 'properties' dictionary
                content = doc.properties.get("content")
                label = doc.properties.get("label")
                score = doc.metadata.score if doc.metadata else "N/A"

                print(f"  {i + 1}. [{label.upper()}] {content} (Score: {score})")

        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_hybrid_search())