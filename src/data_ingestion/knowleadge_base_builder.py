import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma

from src.constants import BASE_DIR
from src.llm_config import EMBEDDINGS
from src.utils.common import create_directories, read_json, save_json


def build_knowledge_base(
    raw_data_path: str, collection_name: str, chroma_db_dir: str, mainfest_path: str
):
    if not os.path.exists(raw_data_path):
        raise ValueError(f"Data directory {raw_data_path} does not exist.")

    create_directories([Path.joinpath(BASE_DIR, chroma_db_dir)])

    manifest = read_json(Path.joinpath(BASE_DIR, mainfest_path))
    updated_manifest = manifest.copy()

    documents = []
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    for root, _, files in os.walk(raw_data_path):
        for file in files:
            file_path = os.path.join(root, file)
            extension = os.path.splitext(file_path)[1]

            if file_path in updated_manifest:
                print(f"Skipping already processed file: {file}")
                continue

            if extension in supported_extensions:
                loader = supported_extensions[extension](file_path)
                updated_manifest[file_path] = True
                documents.extend(loader.load())
                print(f"Loaded {len(loader.load())} pages from {file}")
            else:
                print(f"Skipping unsupported file type: {file}")

    if not documents:
        print("No documents loaded. Please check your data directory and file types.")
        return

    save_json(updated_manifest, Path.joinpath(BASE_DIR, mainfest_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    try:
        db = Chroma.from_documents(
            texts,
            EMBEDDINGS,
            persist_directory=chroma_db_dir,
            collection_name=collection_name,
        )
        db.persist()
    except Exception as e:
        print(f"Error creating vector database: {e}")
