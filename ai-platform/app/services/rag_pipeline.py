from app.services.query_rewriter import QueryRewriter

class RAGPipeline:
    def __init__(self):
        self.query_rewriter = QueryRewriter()
        # other components
        self.input_guard = InputGuardrail()
        self.retrieval_guard = RetrievalGuardrail()
        self.output_guard = OutputGuardrail()
        self.search = HybridSearch()
        self.reranker = CohereReranker()
        self.router = ModelRouter()

    async def run_rag(self, question: str):
        # 1 Input guardrail
        safe_question = self.input_guard.validate(question)

        # 2 Rewrite query
        rewritten_question = self.query_rewriter.rewrite(safe_question)

        # 3 Retrieval
        retrieved_docs = self.search.query(rewritten_question)
        filtered_docs = self.retrieval_guard.filter(retrieved_docs)

        # 4 Rerank
        ranked_docs = self.reranker.rerank(rewritten_question, filtered_docs)

        # 5 Model routing
        answer = self.router.route(rewritten_question, ranked_docs)

        # 6 Output guardrail
        final_answer = self.output_guard.apply(answer)

        return final_answer