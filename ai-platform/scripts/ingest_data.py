from app.ingestion.crawler import Crawler
from app.ingestion.parser import Parser
from app.ingestion.chunker import Chunker
from app.ingestion.embeddings import Embeddings
from app.database.weaviate_client import WeaviateClient

def run_ingest():
    """
    Crawl documents (S3 + web), parse, chunk, generate embeddings, and push to vector DB.
    """
    # 1️⃣ Crawl
    crawler = Crawler(
        s3_bucket="my-bucket-name",              # replace with your S3 bucket
        web_urls=["https://example.com/docs"]    # replace with web URLs
    )
    docs = crawler.crawl()

    # 2️⃣ Parse PDFs / text
    parser = Parser()
    parsed_docs = parser.parse(docs)

    # 3️⃣ Chunk documents
    chunker = Chunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk(parsed_docs)

    # 4️⃣ Generate embeddings
    embedder = Embeddings()
    vector_db = WeaviateClient()

    for chunk in chunks:
        vector = embedder.embed_text(chunk)
        vector_db.add_vector(vector, metadata={"text": chunk})

    print(f"Ingested {len(chunks)} chunks successfully!")

if __name__ == "__main__":
    run_ingest()