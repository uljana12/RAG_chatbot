"""
Document Ingestion Module

Handles loading, splitting, and storing documents in a persistent ChromaDB vector store.
Uses Ollama embeddings (free, local) via langchain_ollama for generating document vectors.

Key Functions:
- ingest_text(): Add text content to the knowledge base
- ingest_urls(): Load and process web pages  
- ingest_file(): Process PDF, TXT, or MD files
- clear_vector_store(): Reset the persistent store (resets singleton client)
- load_vector_store(): Get the current vector store instance

Technical Details:
- Uses PersistentClient with singleton pattern to avoid SQLite locking
- Data persists in ./chroma_db directory between app restarts
- Imports: langchain_ollama.OllamaEmbeddings, langchain_chroma.Chroma
- Chunk size: 1000 chars with 200 char overlap
"""

import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration - use absolute path to avoid issues with working directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIRECTORY = os.path.join(_BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_embeddings():
    """Get Ollama embeddings instance (FREE - runs locally)."""
    return OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def get_text_splitter():
    """Get text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text_file(file_path: str) -> List[Document]:
    """Load a text file and return documents."""
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def load_from_directory(directory_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
    """Load all matching files from a directory."""
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def load_from_urls(urls: List[str]) -> List[Document]:
    """Load content from web URLs."""
    loader = WebBaseLoader(urls)
    return loader.load()


def load_from_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
    """Create a document from raw text."""
    if metadata is None:
        metadata = {"source": "user_input"}
    return [Document(page_content=text, metadata=metadata)]


def process_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = get_text_splitter()
    return text_splitter.split_documents(documents)


# Global ChromaDB client (singleton to avoid SQLite locking)
_chroma_client = None

def _get_chroma_client():
    """Get or create the persistent ChromaDB client."""
    global _chroma_client
    import chromadb
    from chromadb.config import Settings
    
    if _chroma_client is None:
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    return _chroma_client


def _reset_chroma_client():
    """Reset the ChromaDB client (for clearing)."""
    global _chroma_client
    _chroma_client = None


def create_vector_store(documents: List[Document]) -> Chroma:
    """Create a new persistent vector store from documents."""
    embeddings = get_embeddings()
    client = _get_chroma_client()
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME
    )
    return vector_store


def load_vector_store() -> Optional[Chroma]:
    """Load existing persistent vector store."""
    chroma_db_file = os.path.join(CHROMA_PERSIST_DIRECTORY, "chroma.sqlite3")
    if os.path.exists(chroma_db_file):
        embeddings = get_embeddings()
        client = _get_chroma_client()
        try:
            collections = client.list_collections()
            if any(c.name == COLLECTION_NAME for c in collections):
                return Chroma(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
        except:
            pass
    return None


def add_documents_to_store(documents: List[Document], vector_store: Optional[Chroma] = None) -> Chroma:
    """Add documents to existing vector store or create new one."""
    processed_docs = process_documents(documents)
    embeddings = get_embeddings()
    client = _get_chroma_client()
    
    # Try to get existing collection or create new
    try:
        collections = client.list_collections()
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
    except:
        collection_exists = False
    
    if collection_exists:
        # Add to existing store
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        vector_store.add_documents(processed_docs)
        return vector_store
    else:
        # Create new persistent store
        vector_store = Chroma.from_documents(
            documents=processed_docs,
            embedding=embeddings,
            client=client,
            collection_name=COLLECTION_NAME
        )
        return vector_store


def ingest_file(file_path: str) -> Chroma:
    """Ingest a single file into the vector store."""
    if file_path.endswith('.pdf'):
        documents = load_pdf(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        documents = load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return add_documents_to_store(documents)


def ingest_urls(urls: List[str]) -> Chroma:
    """Ingest content from URLs into the vector store."""
    documents = load_from_urls(urls)
    return add_documents_to_store(documents)


def ingest_text(text: str, source_name: str = "user_input") -> Chroma:
    """Ingest raw text into the vector store."""
    documents = load_from_text(text, {"source": source_name})
    return add_documents_to_store(documents)


def clear_vector_store():
    """Clear the persistent vector store."""
    import shutil
    import time
    
    # Reset client to release SQLite connections
    _reset_chroma_client()
    
    # Wait for connections to close
    time.sleep(0.5)
    
    # Remove the directory
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        except Exception as e:
            print(f"Error removing directory: {e}")
    
    # Wait for filesystem
    time.sleep(0.3)
    
    # Recreate empty directory
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and can reason.
    
    Key components include:
    1. Models - LLM wrappers for various providers
    2. Prompts - Templates for structuring inputs
    3. Chains - Combining multiple components
    4. Agents - Dynamic decision making
    5. Memory - Persisting state across interactions
    """
    
    print("Ingesting sample text...")
    vector_store = ingest_text(sample_text, "langchain_intro")
    print(f"Vector store created with {vector_store._collection.count()} documents")
