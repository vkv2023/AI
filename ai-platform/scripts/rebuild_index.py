from app.database.weaviate_client import WeaviateClient
from ingest_data import run_ingest

def rebuild_index():
    """
    Clears the Weaviate DB and re-ingests all documents.
    """
    vector_db = WeaviateClient()

    # WARNING: delete all existing vectors
    print("Clearing existing vectors in Weaviate...")
    vector_db.clear_all_vectors()
    print("Existing vectors cleared.")

    # Re-run ingestion
    run_ingest()
    print("Rebuild complete.")

if __name__ == "__main__":
    rebuild_index()