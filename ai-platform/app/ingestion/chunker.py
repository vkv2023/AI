class Chunker:
    """
    Chunk large text documents into smaller segments suitable for embeddings.
    """
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents):
        """
        documents: list of strings
        Returns: list of text chunks
        """
        all_chunks = []
        for doc in documents:
            start = 0
            while start < len(doc):
                end = min(start + self.chunk_size, len(doc))
                chunk_text = doc[start:end]
                all_chunks.append(chunk_text)
                start += self.chunk_size - self.overlap
        return all_chunks