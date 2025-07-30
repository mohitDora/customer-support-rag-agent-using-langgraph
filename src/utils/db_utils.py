from langchain_community.vectorstores import Chroma
from src.config import ConfigurationManager
from src.llm_config import EMBEDDINGS


def get_vector_db():
    """Helper function to load the ChromaDB instance."""
    config = ConfigurationManager()
    knowledge_base_config = config.get_knowledge_base_config()
    CHROMA_DB_DIR = knowledge_base_config["CHROMA_DB_DIR"]
    COLLECTION_NAME = knowledge_base_config["COLLECTION_NAME"]
    try:
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=EMBEDDINGS,
            collection_name=COLLECTION_NAME,
        )

        if vector_db.get()["ids"]:
            print(
                f"ChromaDB loaded successfully from {CHROMA_DB_DIR} with existing data."
            )
        else:
            print(
                f"ChromaDB loaded from {CHROMA_DB_DIR}, but collection '{COLLECTION_NAME}' appears empty. Please run data ingestion."
            )
        return vector_db
    except Exception as e:
        print(f"Error loading ChromaDB from {CHROMA_DB_DIR}: {e}")
        raise


if __name__ == "__main__":
    get_vector_db()
