from app.database.redis_cache import get_cache, set_cache
from app.services.query_rewriter import rewrite_query
from app.services.hybrid_search import search_docs
from app.services.reranker import rerank_docs
from app.services.model_router import generate_answer


def run_rag(question):

    cached = get_cache(question)

    if cached:
        return cached

    rewritten = rewrite_query(question)

    docs = search_docs(rewritten)

    ranked = rerank_docs(rewritten, docs)

    answer = generate_answer(rewritten, ranked)

    set_cache(question, answer)

    return answer