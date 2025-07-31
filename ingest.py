from pathlib import Path

from src.config import ConfigurationManager
from src.constants import BASE_DIR
from src.data_ingestion.knowleadge_base_builder import build_knowledge_base

if __name__ == "__main__":
    config = ConfigurationManager()
    knowledge_base_config = config.get_knowledge_base_config()
    build_knowledge_base(
        raw_data_path=Path.joinpath(BASE_DIR, knowledge_base_config["RAW_DOCS_DIR"]),
        collection_name=knowledge_base_config["COLLECTION_NAME"],
        chroma_db_dir=knowledge_base_config["CHROMA_DB_DIR"],
        mainfest_path=knowledge_base_config["MANIFEST_PATH"],
    )
